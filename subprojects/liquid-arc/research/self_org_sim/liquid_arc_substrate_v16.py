"""v16: substrate AS the flow-matching denoiser. Encoder is flat.

Architectural shift from v15:
  - v15 put substrate in the ENCODER (over GR00T feature positions) → substrate's
    K positions were artificial (denoising depths, single state, latent slots),
    no role differentiation, attention pool collapsed everything to a weighted
    sum. Substrate machinery decorative.

  - v16 puts substrate in the DECODER (over action-chunk positions). The K=16
    flow-matching action timesteps are NATURAL positions with distinct roles
    (t=0 is immediate, t=15 is far-future). The metric learns temporal
    structure (adjacent timesteps diffuse into each other), halting per-position
    is meaningful (early actions need more refinement than late ones), τ varies
    by horizon.

Encoder: flat MLP over (z_bank, z_state, state8) → cond [B, d]. No substrate.

Decoder (substrate-as-denoiser):
  Input: noisy_chunk [B, K=16, A=7], t (flow time), cond [B, d]
  Build h0 = K positions, each = action_in(noisy_chunk[i]) + pos_embed[i] + cond_emb + t_emb
  ContinuousDynamics evolves K positions via Euler ODE with halting + per-pos τ
  Decode: action_out(h_final[i]) → v_pred[i] for each i
  No attention pool — each position outputs its own velocity component.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
from liquid_arc.config import LiquidARCConfig  # type: ignore
from liquid_arc.dynamics import ContinuousDynamics  # type: ignore
from liquid_arc.solver import euler_solve, euler_solve_halting  # type: ignore
from liquid_arc.context_pool import ContextPool  # type: ignore
from liquid_arc.curvature import CurvatureEngine  # type: ignore


def make_v16_config(
    d_model: int = 768,
    d_metric: int = 64,
    d_ffn: int = 512,
    n_ode_steps: int = 16,
    ode_steps_min: int = 12,
    ode_steps_max: int = 20,
    tau_freeze_steps: int = 5000,
    integration_time: float = 2.0,
    cold_start: bool = True,
    K_total: int = 16,
    aux_loss_scale: float = 0.1,
) -> LiquidARCConfig:
    """v16 config — substrate over K=16 action positions.

    K_total is now the action horizon (16) since substrate operates on chunk
    timesteps. tau and metric should organize around temporal proximity.
    """
    return LiquidARCConfig(
        d_model=d_model, d_metric=d_metric, d_ffn=d_ffn, max_seq_len=K_total,
        n_ode_steps=n_ode_steps, ode_steps_min=ode_steps_min,
        ode_steps_max=ode_steps_max, integration_time=integration_time,
        tau_min=0.5, tau_max=1.0, t_diffusion_init=1.0,
        routing_mode="metric", tau_freeze_steps=tau_freeze_steps,
        halting_enabled=True, halting_min_steps=4,
        halting_ponder_lambda=0.001 * aux_loss_scale,
        rezero_enabled=cold_start, rezero_gate_init=-5.0,
        metric_bias_init_std=0.1 if cold_start else 0.0,
        deep_supervision_enabled=True,
        ponder_kl_lambda=0.001 * aux_loss_scale,
        ponder_kl_prior_rate=0.0625,
        criticality_loss_enabled=True,
        criticality_loss_lambda=0.001 * aux_loss_scale,
        criticality_target_ratio=18.0, criticality_D_sq_target=60.0,
        curvature_diversity_loss_enabled=True,
        curvature_diversity_lambda=0.001 * aux_loss_scale,
        curvature_cv_floor=2.0, curvature_cv_ceiling=10.0,
        tau_quality_loss_enabled=True,
        tau_quality_lambda=0.005 * aux_loss_scale,
        tau_mean_target=0.0, tau_log_spread_target=0.6,
        step_embed_enabled=True, n_step_embeds=ode_steps_max,
        step_conditional_operator=True, step_conditional_n_max=ode_steps_max,
        structural_tau_enabled=True, structural_tau_min=0.3, structural_tau_max=3.0,
        norm_ref=50.0, norm_lambda=0.1,
        base_lr=3e-4, structural_lr_ratio=0.1,
        warmup_steps=500, weight_decay=0.01,
        use_torch_compile=False,
    )


class V16Encoder(nn.Module):
    """Flat encoder. NO substrate here — just MLP over GR00T features + state.

    Substrate moved to V16Decoder where K positions correspond to action timesteps.
    """

    def __init__(
        self,
        d: int = 768,
        z_vl_dim: int = 1024,
        z_state_dim: int = 1536,
        K_bank: int = 4,
        state_dim: int = 8,
    ):
        super().__init__()
        self.d = d
        self.K_bank = K_bank

        # z_bank [B, K_bank, z_vl_dim] → flatten → project
        self.zbank_proj = nn.Linear(K_bank * z_vl_dim, d)
        nn.init.normal_(self.zbank_proj.weight, std=0.02)
        nn.init.zeros_(self.zbank_proj.bias)

        self.zstate_proj = nn.Linear(z_state_dim, d)
        nn.init.normal_(self.zstate_proj.weight, std=0.02)
        nn.init.zeros_(self.zstate_proj.bias)

        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        self.fuse = nn.Sequential(
            nn.Linear(3 * d, 2 * d),
            nn.SiLU(),
            nn.Linear(2 * d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.d_out = d

    def forward(self, z_bank: torch.Tensor, z_state: torch.Tensor, state: torch.Tensor):
        # z_bank [B, K_bank, z_vl_dim], z_state [B, z_state_dim], state [B, 8]
        B = state.shape[0]
        b = self.zbank_proj(z_bank.reshape(B, -1))
        s = self.zstate_proj(z_state)
        p = self.state_enc(state)
        cond = self.fuse(torch.cat([b, s, p], dim=-1))
        return cond


class V16Decoder(nn.Module):
    """Substrate AS denoiser. Operates on K=16 action timesteps.

    h0 positions = K=16 action-timestep representations conditioned on (cond, t).
    ContinuousDynamics evolves them via Euler ODE with halting + per-position τ.
    No attention pool — each position outputs its own velocity.
    """

    def __init__(
        self,
        config: LiquidARCConfig,
        action_horizon: int = 16,
        action_dim: int = 7,
        d_t: int = 64,
    ):
        super().__init__()
        self.config = config
        self.d = config.d_model
        self.K = action_horizon
        self.A = action_dim

        # Per-timestep action input projection
        self.action_in = nn.Linear(action_dim, self.d)

        # Per-timestep position embedding (K=16 distinct timestep roles)
        self.pos_embed = nn.Parameter(torch.zeros(action_horizon, self.d))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Flow time embedding
        self.t_embed = nn.Sequential(
            nn.Linear(1, d_t),
            nn.SiLU(),
            nn.Linear(d_t, self.d),
        )

        # Cond projection (per-position injection)
        self.cond_proj = nn.Linear(self.d, self.d)

        # Canonical substrate over K=16 action positions
        self.context_pool = ContextPool(config)
        self.dynamics = ContinuousDynamics(config)
        self.curvature_engine = CurvatureEngine()

        # Per-position velocity readout (zero-init for stable flow start)
        self.action_out = nn.Linear(self.d, action_dim)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def geo_parameters(self):
        params = list(self.context_pool.parameters())
        params.extend(self.dynamics.metric_net_linear1.parameters())
        params.extend(self.dynamics.metric_net_linear2_diag.parameters())
        if hasattr(self.dynamics, "metric_net_linear2_lr"):
            params.extend(self.dynamics.metric_net_linear2_lr.parameters())
        if self.config.channel_gate_enabled:
            params.extend(self.dynamics.gate_net_linear1.parameters())
            params.extend(self.dynamics.gate_net_linear2.parameters())
        else:
            params.extend(self.dynamics.tau_net_linear1.parameters())
            params.extend(self.dynamics.tau_net_linear2.parameters())
        if hasattr(self.dynamics, "structural_tau") and self.dynamics.structural_tau is not None:
            params.append(self.dynamics.structural_tau)
        if self.config.step_embed_enabled and hasattr(self.dynamics, "step_embeds"):
            params.append(self.dynamics.step_embeds)
        if hasattr(self.dynamics, "tau_step_embed"):
            params.extend(self.dynamics.tau_step_embed.parameters())
        if self.config.step_conditional_operator:
            for name in ("metric_film_gamma", "metric_film_beta",
                         "tau_film_gamma", "tau_film_beta",
                         "t_diff_per_step"):
                if hasattr(self.dynamics, name):
                    params.extend(getattr(self.dynamics, name).parameters())
        params.append(self.pos_embed)
        return params

    def other_parameters(self):
        geo_ids = {id(p) for p in self.geo_parameters()}
        return [p for p in self.parameters() if id(p) not in geo_ids]

    def forward(
        self,
        noisy_chunk: torch.Tensor,    # [B, K, A]
        t: torch.Tensor,              # [B]
        cond: torch.Tensor,           # [B, d]
        n_ode_steps_override: Optional[int] = None,
    ):
        B = noisy_chunk.shape[0]

        # Build K=16 substrate positions, each = action-timestep latent
        h_action = self.action_in(noisy_chunk)                          # [B, K, d]
        cond_emb = self.cond_proj(cond) + self.t_embed(t.unsqueeze(-1)) # [B, d]
        h0 = h_action + self.pos_embed.unsqueeze(0) + cond_emb.unsqueeze(1)  # [B, K, d]

        # Substrate setup
        context = self.context_pool(h0, None)
        self.dynamics.set_context(context, mask=None)

        # ODE evolution with halting (per-timestep variable depth)
        if n_ode_steps_override is not None:
            actual_steps = int(n_ode_steps_override)
        elif self.training:
            lo = int(self.config.ode_steps_min)
            hi = int(self.config.ode_steps_max)
            actual_steps = int(torch.randint(lo, hi + 1, (1,)).item())
        else:
            actual_steps = int(self.config.n_ode_steps)
        self.dynamics.set_n_steps(actual_steps)

        T = float(self.config.integration_time)
        if self.config.halting_enabled:
            out = euler_solve_halting(
                self.dynamics, h0, (0.0, T), actual_steps,
                min_steps=self.config.halting_min_steps,
            )
            if isinstance(out, tuple) and len(out) >= 3:
                h, ponder_cost, steps_used = out[0], out[1], out[2]
            else:
                h = out
                ponder_cost = torch.zeros(B, device=h.device)
                steps_used = torch.full((B, self.K), float(actual_steps), device=h.device)
        else:
            h = euler_solve(self.dynamics, h0, (0.0, T), actual_steps)
            ponder_cost = torch.zeros(B, device=h.device)
            steps_used = torch.full((B, self.K), float(actual_steps), device=h.device)

        # Per-position velocity readout (NO pooling)
        v_pred = self.action_out(h)  # [B, K, A]

        # Diagnostics
        g = self.dynamics.compute_metric_diag(h0)
        metric_cv = g.std() / (g.mean() + 1e-8)
        kappa = self.curvature_engine(g)
        if not self.config.channel_gate_enabled:
            tau0 = self.dynamics.compute_tau(h0)
            tau_avg = tau0.mean()
            tau_log_std = tau0.clamp_min(1e-6).log().std()
        else:
            tau0 = None
            tau_avg = torch.tensor(1.0, device=h.device)
            tau_log_std = torch.tensor(0.0, device=h.device)
        t_diff_param = getattr(self.dynamics, "t_diffusion", None)

        return {
            "v_pred": v_pred,
            "h_final": h, "h0": h0, "g": g, "tau": tau0,
            "tau_avg": tau_avg, "tau_log_std": tau_log_std,
            "t_diff_param": t_diff_param,
            "metric_cv": metric_cv, "avg_kappa": kappa.abs().mean(),
            "ponder_cost": ponder_cost, "steps_used": steps_used,
            "actual_steps": actual_steps,
        }
