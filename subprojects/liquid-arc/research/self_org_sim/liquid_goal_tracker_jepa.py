"""JEPA-LiquidGoalTracker — action-conditioned latent world model in z_vl space.

JEPA-WM framing (LeCun):
  Encoder    : GR00T backbone           z_t = E(obs_t)               [FROZEN]
  Predictor  : substrate.step(h, z_t, a_t)
                                         h_{t+1}, ẑ_{t+1}             [TRAINED]
  Target     : sg(z_{t+1})  from GR00T at next turn (stop-gradient)
  Loss       : || ẑ_{t+1} - sg(z_{t+1}) ||²

Liquid ODE alignment: out_proj(h_out) is a TANGENT in z-space (what an ODE
produces natively). Adding it to z_t = first-order Euler step → ẑ_{t+1}.

Inference: send (ẑ_{t+1} - z_t) = predicted tangent as a residual via
groot_server's get_action_with_zvl_override. GR00T's action head decodes
from bb_features + tangent → actions correct for the ANTICIPATED state.

Per-turn signature:
  step(h_prev[B,K,d], z_t[B,2048], action_chunk_t[B, horizon, action_dim])
       → h_new[B,K,d], z_pred_next[B,2048], tangent[B,2048], diag
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


class JEPA_LGT(nn.Module):
    """Action-conditioned JEPA world model in z_vl latent space.

    Args:
        z_vl_dim:    2048   GR00T's vl-embed dim
        action_dim:  7      raw action dim (LIBERO Panda: xyz, rpy, gripper)
        horizon:     16     action_chunk length
        d:           64     substrate hidden dim
        K:           4      belief positions
        tangent_scale: 0.2  cap on |ẑ - z| via tanh; analogous to out_scale in v9
    """

    def __init__(self, z_vl_dim=2048, action_dim=7, horizon=16,
                 d=64, K=4, tangent_scale=0.2):
        super().__init__()
        self.z_vl_dim = z_vl_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.d = d
        self.K = K
        self.tangent_scale = tangent_scale
        self.config = make_lgt_config(d=d)

        self.init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.init_belief, std=0.05)

        self.in_z = nn.Linear(z_vl_dim, d)
        nn.init.normal_(self.in_z.weight, std=0.02)
        nn.init.zeros_(self.in_z.bias)

        # Action conditioner: chunk [horizon, action_dim] → d
        self.in_action = nn.Linear(horizon * action_dim, d)
        nn.init.normal_(self.in_action.weight, std=0.02)
        nn.init.zeros_(self.in_action.bias)
        # Gate so action influence starts small and grows during training
        self.action_gate = nn.Parameter(torch.tensor(0.1))

        # Per-position evidence injection
        self.evidence_mix = nn.Parameter(torch.ones(K, 1))
        with torch.no_grad():
            self.evidence_mix[0] = 2.0   # current observation belief
            self.evidence_mix[1] = 1.0   # trend / dynamics
            self.evidence_mix[2] = 0.5   # action memory
            self.evidence_mix[3] = 0.5   # readout

        self.context_pool = ContextPool(self.config)
        self.dynamics = ContinuousDynamics(self.config)

        # Output projector: pooled h → 2048-d TANGENT in z-space
        self.out_proj = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(),
            nn.Linear(d * 2, z_vl_dim),
        )
        with torch.no_grad():
            # Init near zero so first ẑ_{t+1} ≈ z_t (identity-like prior)
            self.out_proj[-1].weight.mul_(0.01)
            self.out_proj[-1].bias.zero_()

    def init_state(self, batch_size: int, device, dtype=torch.float32):
        return self.init_belief.unsqueeze(0).expand(batch_size, -1, -1).to(
            device=device, dtype=dtype).contiguous()

    def step(
        self,
        h_goal_prev: torch.Tensor,           # [B, K, d]
        z_t: torch.Tensor,                    # [B, z_vl_dim]
        action_chunk_t: torch.Tensor,         # [B, horizon, action_dim]  (last executed)
        n_steps_override: Optional[int] = None,
    ):
        """Predict z_{t+1} from (z_t, last_action_chunk_t, h_goal_prev).

        Returns:
            h_goal_new:     [B, K, d]          carries to next turn
            z_pred_next:    [B, z_vl_dim]      ẑ_{t+1} (Euler step from z_t)
            tangent:        [B, z_vl_dim]      ẑ_{t+1} - z_t (the residual for GR00T override)
            diag dict
        """
        B = h_goal_prev.shape[0]
        device = h_goal_prev.device

        # Encode current z and last-executed chunk
        e_z = self.in_z(z_t)                                            # [B, d]
        chunk_flat = action_chunk_t.reshape(B, -1)                       # [B, horizon*action_dim]
        e_a = self.in_action(chunk_flat) * self.action_gate              # [B, d]
        e_evidence = e_z + e_a                                           # [B, d]

        injection = self.evidence_mix.unsqueeze(0) * e_evidence.unsqueeze(1)  # [B, K, d]
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

        # Readout = tangent in z-space (Liquid ODE produces dh/dt natively;
        # project to z-space and you have dz/dt — a velocity that advances z_t one step)
        pooled = h_goal_new.mean(dim=1)                                  # [B, d]
        tangent_pre = self.out_proj(pooled)                              # [B, z_vl_dim]
        tangent = torch.tanh(tangent_pre) * self.tangent_scale           # bounded

        # Euler step: ẑ_{t+1} = z_t + tangent
        z_pred_next = z_t + tangent

        g = self.dynamics.compute_metric_diag(h_input)
        metric_cv = g.std() / (g.mean() + 1e-8)

        return h_goal_new, z_pred_next, tangent, {
            "metric_cv": metric_cv,
            "ponder": ponder.mean(),
            "n_steps": n_steps,
            "tangent_norm": tangent.norm(dim=-1).mean().detach(),
            "action_gate": self.action_gate.detach(),
        }
