"""50M-param goal-conditioned JEPA-LGT.

Same architecture as JEPA_LGT_GoalDelta (compact-latent JEPA, in_z + in_goal +
in_delta + in_action evidence injection, Liquid ODE dynamics, scalar-tangent out)
but scaled to ~50M params to test whether the parasitic substrate ceiling at
~60% (L=5 hard) is capacity-driven or architectural.

Scaling:
  d_model:    64    → 896    (14×)
  d_ffn:      128   → 3584   (28×)  scales 4× d
  d_metric:   16    → 224    (14×)  scales d/4
  out_proj:   2-layer → 3-layer (deeper readout)
  K_belief:  variable (default 16)
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


def make_lgt_config_big(d=896):
    return LiquidARCConfig(
        d_model=d,
        d_metric=max(32, d // 4),
        d_ffn=max(256, d * 4),
        max_seq_len=32,
        n_ode_steps=4, ode_steps_min=3, ode_steps_max=5,
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
        warmup_steps=500, weight_decay=0.01,
        use_torch_compile=False,
    )


class JEPA_LGT_Big(nn.Module):
    """~50M JEPA-LGT with same compact-latent prediction signature."""

    def __init__(self, z_vl_dim=2048, action_dim=7, horizon=16,
                 d=1152, K=16, tangent_scale=0.2):
        super().__init__()
        self.z_vl_dim = z_vl_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.d = d
        self.K = K
        self.tangent_scale = tangent_scale
        self.config = make_lgt_config_big(d=d)

        self.init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.init_belief, std=0.05)

        # Four input channels: current state, goal anchor, progress delta, last action
        self.in_z     = nn.Linear(z_vl_dim, d)
        self.in_goal  = nn.Linear(z_vl_dim, d)
        self.in_delta = nn.Linear(z_vl_dim, d)
        self.in_action = nn.Linear(horizon * action_dim, d)
        for layer in (self.in_z, self.in_goal, self.in_delta, self.in_action):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

        # Per-channel learnable gates
        self.action_gate = nn.Parameter(torch.tensor(0.1))
        self.goal_gate   = nn.Parameter(torch.tensor(1.0))
        self.delta_gate  = nn.Parameter(torch.tensor(1.0))

        # Per-position evidence mix
        self.evidence_mix = nn.Parameter(torch.ones(K, 1))
        with torch.no_grad():
            # Spread roles across K positions
            self.evidence_mix[0:K // 4] = 2.0   # current obs
            self.evidence_mix[K // 4:K // 2] = 1.5  # goal-aware planning
            self.evidence_mix[K // 2:3 * K // 4] = 1.5  # progress tracking
            self.evidence_mix[3 * K // 4:] = 0.5  # readout

        # Pre-evidence LN to stabilize the larger scale
        self.in_ln = nn.LayerNorm(d)

        self.context_pool = ContextPool(self.config)
        self.dynamics = ContinuousDynamics(self.config)

        # Deeper out_proj: d → 2d → 4d → 2048 (more capacity in the readout)
        self.out_proj = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(),
            nn.LayerNorm(d * 2),
            nn.Linear(d * 2, d * 4), nn.SiLU(),
            nn.LayerNorm(d * 4),
            nn.Linear(d * 4, z_vl_dim),
        )
        with torch.no_grad():
            self.out_proj[-1].weight.mul_(0.01)
            self.out_proj[-1].bias.zero_()

    def init_state(self, batch_size: int, device, dtype=torch.float32):
        return self.init_belief.unsqueeze(0).expand(batch_size, -1, -1).to(
            device=device, dtype=dtype).contiguous()

    def step(
        self,
        h_goal_prev: torch.Tensor,           # [B, K, d]
        z_t: torch.Tensor,                    # [B, z_vl_dim]
        z_goal: torch.Tensor,                 # [B, z_vl_dim]
        action_chunk_t: torch.Tensor,         # [B, horizon, action_dim]
        n_steps_override: Optional[int] = None,
    ):
        B = h_goal_prev.shape[0]
        device = h_goal_prev.device

        delta = z_goal - z_t                                  # explicit progress

        e_z = self.in_z(z_t)
        e_g = self.in_goal(z_goal) * self.goal_gate
        e_d = self.in_delta(delta) * self.delta_gate
        chunk_flat = action_chunk_t.reshape(B, -1)
        e_a = self.in_action(chunk_flat) * self.action_gate
        e_evidence = self.in_ln(e_z + e_g + e_d + e_a)

        injection = self.evidence_mix.unsqueeze(0) * e_evidence.unsqueeze(1)
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

        pooled = h_goal_new.mean(dim=1)
        tangent_pre = self.out_proj(pooled)
        tangent = torch.tanh(tangent_pre) * self.tangent_scale

        z_pred_next = z_t + tangent

        g = self.dynamics.compute_metric_diag(h_input)
        metric_cv = g.std() / (g.mean() + 1e-8)

        return h_goal_new, z_pred_next, tangent, {
            "metric_cv": metric_cv,
            "ponder": ponder.mean(),
            "n_steps": n_steps,
            "tangent_norm": tangent.norm(dim=-1).mean().detach(),
            "action_gate": self.action_gate.detach(),
            "goal_gate": self.goal_gate.detach(),
            "delta_gate": self.delta_gate.detach(),
        }
