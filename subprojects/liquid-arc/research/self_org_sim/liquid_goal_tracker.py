"""LiquidGoalTracker — substrate variant #9.

Per user spec:
- Input: ONLY z_vl from GR00T transformer
- Persistent state: h_goal[K, d] carries across GR00T's turns
- Output: small residual added back to z_vl (fed to GR00T's action head)
- Skill: only goal tracking across turns — not vision-language or action knowledge
- Size: small (~100-300K params) — cannot match GR00T's 3B-param knowledge

Architecture:
  z_vl[2048]                          ← from GR00T per turn
       │
       └─→ in_proj[2048→d]
              │
       (h_goal_prev[B,K,d]) ───────── + injection
              │
       ContinuousDynamics (Liquid)    ← persistent state across turns
              │
              ↓
       readout (h_goal[B,K,d] → pooled[B,d])
              │
       out_proj[d→2048] × out_scale   ← bounded residual
              │
              ↓
       residual_zvl[2048]             → added to GR00T's z_vl
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


def make_lgt_config(d=64):
    return LiquidARCConfig(
        d_model=d, d_metric=16, d_ffn=128, max_seq_len=8,
        n_ode_steps=3, ode_steps_min=2, ode_steps_max=4,
        integration_time=0.5,
        tau_min=0.3, tau_max=2.0, t_diffusion_init=0.5,
        routing_mode="metric",
        tau_freeze_steps=500,
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
        norm_ref=10.0, norm_lambda=0.1,
        base_lr=3e-4, structural_lr_ratio=0.1,
        warmup_steps=200, weight_decay=0.01,
        use_torch_compile=False,
    )


class LiquidGoalTracker(nn.Module):
    """Substrate inside GR00T — z_vl in, z_vl residual out.

    Args:
        z_vl_dim: 2048 (GR00T's vl-embed dim)
        d: 64 (substrate hidden dim — SMALL by design)
        K: 4 (belief positions)
        out_scale: 0.5 (cap on residual magnitude; bounded via tanh × out_scale)
    """

    def __init__(self, z_vl_dim=2048, d=64, K=4, out_scale=0.5):
        super().__init__()
        self.z_vl_dim = z_vl_dim
        self.d = d
        self.K = K
        self.out_scale = out_scale
        self.config = make_lgt_config(d=d)

        # Learned initial belief prior
        self.init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.init_belief, std=0.05)

        # Input projector: z_vl → d
        self.in_proj = nn.Linear(z_vl_dim, d)
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.zeros_(self.in_proj.bias)

        # Per-position evidence injection (K, 1) — all positions get z_vl evidence
        # but with different magnitudes (semantic specialization)
        self.evidence_mix = nn.Parameter(torch.ones(K, 1))
        with torch.no_grad():
            self.evidence_mix[0] = 2.0   # "current observation belief"
            self.evidence_mix[1] = 1.0   # "trend"
            self.evidence_mix[2] = 0.5   # "memory"
            self.evidence_mix[3] = 0.5   # "readout"

        self.context_pool = ContextPool(self.config)
        self.dynamics = ContinuousDynamics(self.config)

        # Output projector: pooled readout → z_vl residual
        self.out_proj = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(),
            nn.Linear(d * 2, z_vl_dim),
        )
        # Init last layer near zero so initial residual is tiny
        with torch.no_grad():
            self.out_proj[-1].weight.mul_(0.01)
            self.out_proj[-1].bias.zero_()

    def init_state(self, batch_size: int, device, dtype=torch.float32):
        return self.init_belief.unsqueeze(0).expand(batch_size, -1, -1).to(
            device=device, dtype=dtype).contiguous()

    def step(
        self,
        h_goal_prev: torch.Tensor,        # [B, K, d]
        z_vl: torch.Tensor,                # [B, z_vl_dim]
        n_steps_override: Optional[int] = None,
    ):
        """Process one GR00T turn: update goal belief from z_vl, output residual.

        Returns:
            h_goal_new: [B, K, d] persists to next turn
            residual_zvl: [B, z_vl_dim] bounded ±out_scale — feeds back to GR00T
            diagnostics dict
        """
        B = h_goal_prev.shape[0]
        device = h_goal_prev.device

        # Project z_vl into substrate-space
        e_z = self.in_proj(z_vl)  # [B, d]

        # Per-position injection: each belief position gets evidence_mix[k] * e_z
        # [K, 1] * [B, d] → broadcast → [B, K, d]
        injection = self.evidence_mix.unsqueeze(0) * e_z.unsqueeze(1)
        h_input = h_goal_prev + injection

        # Substrate evolution
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

        # Readout: pool over K positions, project to z_vl dim
        pooled = h_goal_new.mean(dim=1)  # [B, d]
        residual_pre = self.out_proj(pooled)  # [B, z_vl_dim]
        residual = torch.tanh(residual_pre) * self.out_scale

        g = self.dynamics.compute_metric_diag(h_input)
        metric_cv = g.std() / (g.mean() + 1e-8)

        return h_goal_new, residual, {
            "metric_cv": metric_cv,
            "ponder": ponder.mean(),
            "n_steps": n_steps,
            "residual_norm": residual.norm(dim=-1).mean().detach(),
        }
