"""GoalImageResidualSubstrate — substrate outputs a CORRECTION to GR00T's chunk.

Contrast with GoalImageSubstrate (variants 2-5, all falsified):
  Those substrates output classifier signals (progress, goal_reached) and an
  inference-time controller decided to advance/bail/replan. The asymmetric cost
  of wrong discrete decisions defeated every variant.

This substrate outputs a [16, 6] delta on GR00T's xyz/rpy actions (NOT
gripper — gripper stays binary {-1,+1}). Trained end-to-end via behavior
cloning: substrate predicts the correction that would bring GR00T's chunk
closer to expert. Inference applies the correction directly. The output IS
the intervention, so train/inference are aligned — no controller pathology.

Bounded output: tanh × max_delta (default 0.05) prevents destabilizing GR00T.
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


def make_residual_config(d=128):
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


class GoalImageResidualSubstrate(nn.Module):
    """Substrate predicting per-step correction to GR00T's chunk.

    Args:
        d_obs: 768 (v10 encoder)
        d_goal: 384 (DINOv2)
        d_chunk: 16 * 7 = 112 (GR00T action chunk flattened)
        d: 128 (substrate hidden)
        K: 8 (belief positions)
        action_horizon: 16 (chunk length)
        action_dim_residual: 6 (xyz/rpy — exclude gripper)
        max_delta: 0.05 (bounded correction magnitude)
    """

    def __init__(self, d_obs=768, d_state=8, d_chunk=112, d_goal=384, d=128,
                 K=8, action_horizon=16, action_dim_residual=6, max_delta=0.05):
        super().__init__()
        self.d = d
        self.K = K
        self.action_horizon = action_horizon
        self.action_dim_residual = action_dim_residual
        self.max_delta = max_delta
        self.d_obs = d_obs
        self.d_goal = d_goal
        self.config = make_residual_config(d=d)

        # Learned initial belief prior
        self.init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.init_belief, std=0.05)

        # Evidence projections (5 sources): obs, state, chunk, chunk_diff, goal
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
        # Semantic init: position 0 is current state, 1 is goal, 2 is chunk-vs-goal
        # mismatch (where correction is needed), 3 is state, 4-6 scratch, 7 readout
        self.evidence_mix = nn.Parameter(torch.zeros(K, 5))
        with torch.no_grad():
            self.evidence_mix[0] = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0])
            self.evidence_mix[1] = torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0])
            self.evidence_mix[2] = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
            self.evidence_mix[3] = torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0])
            self.evidence_mix[4] = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
            self.evidence_mix[5] = torch.tensor([1.0, 0.5, 0.5, 0.5, 0.5])
            self.evidence_mix[6] = torch.tensor([0.5, 1.0, 0.5, 0.5, 0.5])
            self.evidence_mix[7] = torch.tensor([0.5, 0.5, 1.0, 0.5, 1.0])

        self.context_pool = ContextPool(self.config)
        self.dynamics = ContinuousDynamics(self.config)

        # Delta head: from readout (position 7) → [action_horizon, action_dim_residual]
        # tanh × max_delta caps magnitude
        self.delta_head = nn.Sequential(
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, action_horizon * action_dim_residual),
        )
        # Initialize last layer near zero so initial deltas are tiny
        with torch.no_grad():
            self.delta_head[-1].weight.mul_(0.01)
            self.delta_head[-1].bias.zero_()

    def init_state(self, batch_size: int, device, dtype=torch.float32):
        return self.init_belief.unsqueeze(0).expand(batch_size, -1, -1).to(
            device=device, dtype=dtype).contiguous()

    def step(
        self,
        h_goal_prev: torch.Tensor,        # [B, K, d]
        obs_features: torch.Tensor,        # [B, d_obs]
        state: torch.Tensor,               # [B, d_state]
        groot_chunk: torch.Tensor,         # [B, 16, 7]
        goal_features: torch.Tensor,       # [B, d_goal]
        prev_groot_chunk: Optional[torch.Tensor] = None,
        n_steps_override: Optional[int] = None,
    ):
        B = h_goal_prev.shape[0]
        device = h_goal_prev.device

        e_obs = self.proj_obs(obs_features)
        e_state = self.proj_state(state)
        chunk_flat = groot_chunk.reshape(B, -1)
        e_chunk = self.proj_chunk(chunk_flat)
        if prev_groot_chunk is not None:
            diff_flat = (groot_chunk - prev_groot_chunk).reshape(B, -1)
            e_diff = self.proj_chunk_diff(diff_flat)
        else:
            e_diff = torch.zeros_like(e_chunk)
        e_goal = self.proj_goal(goal_features)

        evidence_stack = torch.stack(
            [e_obs, e_state, e_chunk, e_diff, e_goal], dim=1
        )
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

        readout = h_goal_new[:, 7]  # [B, d]
        delta_flat = self.delta_head(readout)  # [B, 16*6]
        delta = torch.tanh(delta_flat).reshape(B, self.action_horizon, self.action_dim_residual)
        delta = delta * self.max_delta  # bounded correction

        g = self.dynamics.compute_metric_diag(h_input)
        metric_cv = g.std() / (g.mean() + 1e-8)

        return h_goal_new, delta, {
            "metric_cv": metric_cv,
            "ponder": ponder.mean(),
            "n_steps": n_steps,
            "delta_norm": delta.abs().mean().detach(),
        }

    def apply_to_chunk(self, groot_chunk, delta):
        """Apply substrate's delta to GR00T's chunk on xyz/rpy dimensions only.

        groot_chunk: [B, 16, 7]  (last dim: xyz3, rpy3, gripper)
        delta:       [B, 16, 6]  (xyz3, rpy3 only)
        Returns:     [B, 16, 7] corrected chunk (gripper unchanged)
        """
        corrected = groot_chunk.clone()
        corrected[..., :6] = corrected[..., :6] + delta
        return corrected
