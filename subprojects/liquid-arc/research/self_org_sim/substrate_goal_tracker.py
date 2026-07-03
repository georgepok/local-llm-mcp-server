"""SubstrateGoalTracker: substrate as persistent goal observer.

KEY ARCHITECTURAL SHIFT from v15-v18:
  Previous variants treated the substrate as a stateless predictor (encoder /
  denoiser / gripper head / meta-controller). Each call was independent.

  This module maintains h_goal as the substrate's hidden state ACROSS chunk
  decisions in an episode. The substrate's continuous-time dynamics finally has
  something to evolve continuously — the goal belief — rather than re-evolving
  from random init at every call.

Per-episode lifecycle:
  init_state()          → h_goal_0 (learned prior over belief positions)
  step(h_goal, evid)    → h_goal_1, gripper_logits
  step(h_goal_1, evid)  → h_goal_2, gripper_logits
  ...                     (h_goal carries across turns)
  reset on episode end

Per-position belief roles (semantic init via different projections):
  pos 0: object identity belief (visual features dominant)
  pos 1: target location belief (visual features dominant)
  pos 2: phase belief (state + GR00T chunk dominant)
  pos 3: trust-in-GR00T (chunk consistency)
  pos 4: drift accumulator (mismatch over time)
  pos 5: scratch integration
  pos 6: scratch integration
  pos 7: target readout (output position)

Within a turn, substrate runs n_inner_steps Euler steps to refine the new
evidence into the goal belief. ACROSS turns, h_goal persists.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
from liquid_arc.config import LiquidARCConfig  # type: ignore
from liquid_arc.dynamics import ContinuousDynamics  # type: ignore
from liquid_arc.solver import euler_solve_halting  # type: ignore
from liquid_arc.context_pool import ContextPool  # type: ignore


def make_gt_config(d=128):
    return LiquidARCConfig(
        d_model=d, d_metric=32, d_ffn=128, max_seq_len=8,
        n_ode_steps=4, ode_steps_min=2, ode_steps_max=6,  # FEW per-turn inner steps
        integration_time=0.5,
        tau_min=0.3, tau_max=2.0, t_diffusion_init=0.5,
        routing_mode="metric",
        tau_freeze_steps=1000,
        halting_enabled=True, halting_min_steps=1,
        halting_ponder_lambda=0.0001,
        rezero_enabled=True, rezero_gate_init=-3.0,
        metric_bias_init_std=0.1,
        deep_supervision_enabled=False, ponder_kl_lambda=0.0,
        criticality_loss_enabled=False,
        curvature_diversity_loss_enabled=True,
        curvature_diversity_lambda=0.0001,
        curvature_cv_floor=1.0, curvature_cv_ceiling=8.0,
        tau_quality_loss_enabled=False,
        step_embed_enabled=False,
        step_conditional_operator=False,
        structural_tau_enabled=True, structural_tau_min=0.3, structural_tau_max=3.0,
        # ↑ structural τ critical here: object-identity slot wants LARGE τ (stable),
        #   drift-accumulator wants SMALL τ (fast response).
        norm_ref=20.0, norm_lambda=0.1,
        base_lr=3e-4, structural_lr_ratio=0.1,
        warmup_steps=200, weight_decay=0.01,
        use_torch_compile=False,
    )


class SubstrateGoalTracker(nn.Module):
    """Substrate that maintains persistent goal belief across episode turns.

    Args:
        d_obs: dimension of observation features (e.g. 776-d DINOv2 cat state, or 768-d v10 cond)
        d_state: robot state dimension (default 8 for LIBERO)
        d_chunk: GR00T chunk flattened dimension (16 * 7 = 112)
        d: substrate hidden dimension (default 128)
        K: number of belief positions (default 8)
        action_horizon: number of gripper logits to output per turn (default 16)
    """

    def __init__(self, d_obs=776, d_state=8, d_chunk=112, d=128, K=8,
                 action_horizon=16):
        super().__init__()
        self.d = d
        self.K = K
        self.action_horizon = action_horizon
        self.config = make_gt_config(d=d)

        # Learned initial belief prior (one per K position)
        self.init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.init_belief, std=0.05)

        # Evidence projections (per source)
        self.proj_obs = nn.Linear(d_obs, d)
        self.proj_state = nn.Linear(d_state, d)
        self.proj_chunk = nn.Linear(d_chunk, d)
        self.proj_chunk_diff = nn.Linear(d_chunk, d)  # how much chunk changed vs previous
        for proj in [self.proj_obs, self.proj_state, self.proj_chunk, self.proj_chunk_diff]:
            nn.init.normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)

        # Per-position evidence-to-position injection matrix
        # Each belief position gets a DIFFERENT mixture of evidence sources
        # (encodes the semantic roles described in module docstring)
        self.evidence_mix = nn.Parameter(torch.zeros(K, 4))
        with torch.no_grad():
            # Initialize semantic role mixes
            # cols: [obs, state, chunk, chunk_diff]
            self.evidence_mix[0] = torch.tensor([2.0, 0.0, 0.5, 0.0])  # object: vision-heavy
            self.evidence_mix[1] = torch.tensor([2.0, 0.5, 0.0, 0.0])  # location: vision+state
            self.evidence_mix[2] = torch.tensor([0.5, 1.0, 1.5, 0.0])  # phase: state+chunk
            self.evidence_mix[3] = torch.tensor([0.0, 0.0, 1.0, 1.5])  # trust: chunk+diff
            self.evidence_mix[4] = torch.tensor([0.0, 0.0, 0.0, 2.0])  # drift: diff-only
            self.evidence_mix[5] = torch.tensor([1.0, 1.0, 1.0, 1.0])  # scratch1
            self.evidence_mix[6] = torch.tensor([1.0, 1.0, 1.0, 1.0])  # scratch2
            self.evidence_mix[7] = torch.tensor([0.5, 0.5, 0.5, 0.5])  # readout: balanced

        # Substrate
        self.context_pool = ContextPool(self.config)
        self.dynamics = ContinuousDynamics(self.config)

        # Gripper readout: from position 7 (target/readout) → per-chunk-position gripper logits
        self.gripper_readout = nn.Sequential(
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, action_horizon),
        )

    def init_state(self, batch_size: int, device, dtype=torch.float32):
        """Initial goal belief — learned prior, broadcast across batch."""
        return self.init_belief.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype).contiguous()

    def step(
        self,
        h_goal_prev: torch.Tensor,        # [B, K, d] — previous belief
        obs_features: torch.Tensor,        # [B, d_obs]
        state: torch.Tensor,               # [B, d_state]
        groot_chunk: torch.Tensor,         # [B, action_horizon, action_dim]
        prev_groot_chunk: Optional[torch.Tensor] = None,
        n_steps_override: Optional[int] = None,
    ):
        """Advance goal belief one turn.

        Returns:
            h_goal_new: [B, K, d] — updated belief (persists to next turn)
            gripper_logits: [B, action_horizon] — sigmoid → P(open) per chunk position
            diagnostics dict (metric_cv, halting steps, etc.)
        """
        B = h_goal_prev.shape[0]
        device = h_goal_prev.device

        # Build evidence vectors per source
        e_obs = self.proj_obs(obs_features)              # [B, d]
        e_state = self.proj_state(state)                 # [B, d]
        chunk_flat = groot_chunk.reshape(B, -1)          # [B, K*A]
        e_chunk = self.proj_chunk(chunk_flat)            # [B, d]
        if prev_groot_chunk is not None:
            diff_flat = (groot_chunk - prev_groot_chunk).reshape(B, -1)
            e_diff = self.proj_chunk_diff(diff_flat)
        else:
            e_diff = torch.zeros_like(e_chunk)

        # Stack evidence sources: [B, 4, d]
        evidence_stack = torch.stack([e_obs, e_state, e_chunk, e_diff], dim=1)

        # Per-position injection via evidence_mix:
        # For each belief position k, injection = sum_j (evidence_mix[k, j] * evidence_stack[:, j])
        injection = torch.einsum('kj,bjd->bkd', self.evidence_mix, evidence_stack)  # [B, K, d]

        # Combine with previous belief (additive residual evidence integration)
        h_input = h_goal_prev + injection  # [B, K, d]

        # Substrate evolution: few Euler steps to refine
        context = self.context_pool(h_input, None)
        self.dynamics.set_context(context, mask=None)
        if n_steps_override is not None:
            n_steps = int(n_steps_override)
        elif self.training:
            lo = int(self.config.ode_steps_min)
            hi = int(self.config.ode_steps_max)
            n_steps = int(torch.randint(lo, hi + 1, (1,)).item())
        else:
            n_steps = int(self.config.n_ode_steps)
        self.dynamics.set_n_steps(n_steps)
        T = float(self.config.integration_time)
        out = euler_solve_halting(
            self.dynamics, h_input, (0.0, T), n_steps,
            min_steps=self.config.halting_min_steps,
        )
        if isinstance(out, tuple):
            h_goal_new = out[0]
            ponder = out[1]
        else:
            h_goal_new = out
            ponder = torch.zeros(B, device=device)

        # Readout gripper logits from position 7 (target/readout role)
        gripper_logits = self.gripper_readout(h_goal_new[:, 7])  # [B, action_horizon]

        # Diagnostics
        g = self.dynamics.compute_metric_diag(h_input)
        metric_cv = g.std() / (g.mean() + 1e-8)

        return h_goal_new, gripper_logits, {
            "metric_cv": metric_cv,
            "ponder": ponder.mean(),
            "n_steps": n_steps,
        }

    def detect_drift(self, gripper_logits, groot_chunk):
        """Drift signal: cosine similarity between substrate's gripper prediction
        and GR00T's chunk gripper prediction. Low similarity = high drift.

        Returns scalar per batch element.
        """
        sub_open = torch.sigmoid(gripper_logits)  # [B, K_chunk]
        groot_open = (groot_chunk[..., -1] < 0).float()  # [B, K_chunk]
        # Mean disagreement
        disagreement = (sub_open - groot_open).abs().mean(dim=-1)  # [B]
        return disagreement
