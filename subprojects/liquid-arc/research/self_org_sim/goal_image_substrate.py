"""GoalImageSubstrate: substrate with EXPLICIT goal representation.

Contrast with SubstrateGoalTracker (which had no goal input, was a tactical
gripper predictor misnamed "goal tracker"):
  - This substrate receives goal_image_features as static conditioning,
    representing the TARGET END STATE for the current sub-task.
  - Outputs progress ∈ [0,1] and goal_reached ∈ [0,1] (NOT gripper logits).
  - Trained against expert-derived targets: progress = t/N, goal_reached = (t >= N-16).

Belief positions (8):
  0: current visual state belief (obs-dominant)
  1: target visual state belief (goal-image-dominant)
  2: progress integrator (obs vs goal mismatch)
  3: state/proprio belief
  4: chunk consistency
  5: drift accumulator (chunk_diff)
  6: scratch
  7: readout (balanced)
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
from liquid_arc.config import LiquidARCConfig  # type: ignore
from liquid_arc.dynamics import ContinuousDynamics  # type: ignore
from liquid_arc.solver import euler_solve_halting  # type: ignore
from liquid_arc.context_pool import ContextPool  # type: ignore


def make_gi_config(d=128):
    return LiquidARCConfig(
        d_model=d, d_metric=32, d_ffn=128, max_seq_len=8,
        n_ode_steps=4, ode_steps_min=2, ode_steps_max=6,
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
        norm_ref=20.0, norm_lambda=0.1,
        base_lr=3e-4, structural_lr_ratio=0.1,
        warmup_steps=200, weight_decay=0.01,
        use_torch_compile=False,
    )


class GoalImageSubstrate(nn.Module):
    """Substrate with explicit goal-image conditioning.

    Args:
        d_obs: current observation features dim (e.g. 768 from v10 encoder)
        d_state: robot state dim (8)
        d_chunk: GR00T chunk flattened dim (16 * 7 = 112)
        d_goal: goal-image features dim (384 from DINOv2 ViT-S/14)
        d: substrate hidden dim (128)
        K: belief positions (8)
        action_horizon: kept for compatibility / future per-position progress (16)
    """

    def __init__(self, d_obs=768, d_state=8, d_chunk=112, d_goal=384, d=128,
                 K=8, action_horizon=16, use_chunk=True):
        super().__init__()
        self.d = d
        self.K = K
        self.action_horizon = action_horizon
        self.d_obs = d_obs
        self.d_goal = d_goal
        self.use_chunk = use_chunk
        self.config = make_gi_config(d=d)

        # Learned initial belief prior
        self.init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.init_belief, std=0.05)

        # Evidence projections (5 sources)
        self.proj_obs = nn.Linear(d_obs, d)
        self.proj_state = nn.Linear(d_state, d)
        self.proj_chunk = nn.Linear(d_chunk, d)
        self.proj_chunk_diff = nn.Linear(d_chunk, d)
        self.proj_goal = nn.Linear(d_goal, d)
        for proj in [self.proj_obs, self.proj_state, self.proj_chunk,
                     self.proj_chunk_diff, self.proj_goal]:
            nn.init.normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)

        # Per-position evidence mix (K, 5)
        # cols: [obs, state, chunk, chunk_diff, goal]
        self.evidence_mix = nn.Parameter(torch.zeros(K, 5))
        with torch.no_grad():
            if use_chunk:
                self.evidence_mix[0] = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0])
                self.evidence_mix[1] = torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0])
                self.evidence_mix[2] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
                self.evidence_mix[3] = torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0])
                self.evidence_mix[4] = torch.tensor([0.0, 0.0, 1.5, 0.5, 0.0])
                self.evidence_mix[5] = torch.tensor([0.0, 0.0, 0.0, 2.0, 0.0])
                self.evidence_mix[6] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
                self.evidence_mix[7] = torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0])
            else:
                # No chunk — all evidence is (obs, state, goal). 5-col matrix kept
                # for state-dict compat, but chunk columns zeroed and projection skipped.
                self.evidence_mix[0] = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0])  # current vis
                self.evidence_mix[1] = torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0])  # goal vis
                self.evidence_mix[2] = torch.tensor([1.5, 0.0, 0.0, 0.0, 1.5])  # progress (obs - goal)
                self.evidence_mix[3] = torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0])  # state
                self.evidence_mix[4] = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0])  # integrator
                self.evidence_mix[5] = torch.tensor([1.0, 0.5, 0.0, 0.0, 0.5])  # scratch
                self.evidence_mix[6] = torch.tensor([0.5, 0.5, 0.0, 0.0, 1.0])  # scratch goal-weighted
                self.evidence_mix[7] = torch.tensor([0.5, 0.5, 0.0, 0.0, 1.5])  # readout (goal-heavy)

        self.context_pool = ContextPool(self.config)
        self.dynamics = ContinuousDynamics(self.config)

        # Readouts from position 7
        self.progress_head = nn.Sequential(
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, 1),
        )
        self.goal_reached_head = nn.Sequential(
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, 1),
        )

    def init_state(self, batch_size: int, device, dtype=torch.float32):
        return self.init_belief.unsqueeze(0).expand(batch_size, -1, -1).to(
            device=device, dtype=dtype).contiguous()

    def step(
        self,
        h_goal_prev: torch.Tensor,        # [B, K, d]
        obs_features: torch.Tensor,        # [B, d_obs]
        state: torch.Tensor,               # [B, d_state]
        groot_chunk: torch.Tensor,         # [B, 16, 7]
        goal_features: torch.Tensor,       # [B, d_goal] — STATIC for the sub-task
        prev_groot_chunk: Optional[torch.Tensor] = None,
        n_steps_override: Optional[int] = None,
    ):
        """Advance belief one turn given current obs + STATIC goal_features.

        Returns:
            h_goal_new: [B, K, d]
            progress: [B] in (0,1) (sigmoid)
            goal_reached_logit: [B]
            diagnostics dict
        """
        B = h_goal_prev.shape[0]
        device = h_goal_prev.device

        e_obs = self.proj_obs(obs_features)
        e_state = self.proj_state(state)
        if self.use_chunk:
            chunk_flat = groot_chunk.reshape(B, -1)
            e_chunk = self.proj_chunk(chunk_flat)
            if prev_groot_chunk is not None:
                diff_flat = (groot_chunk - prev_groot_chunk).reshape(B, -1)
                e_diff = self.proj_chunk_diff(diff_flat)
            else:
                e_diff = torch.zeros_like(e_chunk)
        else:
            # Skip chunk projections (avoids feeding inference-OOD chunk distribution)
            e_chunk = torch.zeros(B, self.d, device=device, dtype=e_obs.dtype)
            e_diff = torch.zeros_like(e_chunk)
        e_goal = self.proj_goal(goal_features)

        evidence_stack = torch.stack(
            [e_obs, e_state, e_chunk, e_diff, e_goal], dim=1
        )  # [B, 5, d]
        injection = torch.einsum('kj,bjd->bkd', self.evidence_mix, evidence_stack)
        h_input = h_goal_prev + injection

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

        readout = h_goal_new[:, 7]
        progress_logit = self.progress_head(readout).squeeze(-1)
        goal_reached_logit = self.goal_reached_head(readout).squeeze(-1)
        progress = torch.sigmoid(progress_logit)

        g = self.dynamics.compute_metric_diag(h_input)
        metric_cv = g.std() / (g.mean() + 1e-8)

        return h_goal_new, progress, goal_reached_logit, {
            "metric_cv": metric_cv,
            "ponder": ponder.mean(),
            "n_steps": n_steps,
            "progress_logit": progress_logit.detach(),
        }
