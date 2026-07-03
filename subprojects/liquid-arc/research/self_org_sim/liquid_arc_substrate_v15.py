"""v15 LIBERO substrate: GR00T features as input, no DINOv2 in Liquid.

Architectural shift from v13:
  - v13 had Liquid encode images via DINOv2 (22M frozen + 7M trainable) → per-modality positions → substrate.
  - v15 USES GR00T's features directly. Substrate operates on:
      * 4 positions from GR00T's depth bank (traj_model_output samples)
      * 1 position from z_state (1536-d, GR00T's state encoding)
      * 1 position from robot proprioception (8-d state)
      * K_latent learnable scratch slots
  - No image processing in Liquid. Liquid is a decoder over GR00T's representation.

Training: uses libero-{suite}-expert-v1/z_vl_bank.dat (4×1024) and z_state.dat (1536).
Inference: query GR00T server → traj_model_output (4×1024 = bank) + z_state. Same dims.

Tradeoff: Liquid is now teacher-coupled (can't run without GR00T) but inherits GR00T's
robotics-aware visual features. Per-suite gap to GR00T should shrink dramatically on
libero_object where DINOv2's general-purpose features failed at object discrimination.
"""
from __future__ import annotations

import sys
import warnings
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


def make_v15_config(
    d_model: int = 768,
    d_metric: int = 64,
    d_ffn: int = 512,
    n_ode_steps: int = 16,
    ode_steps_min: int = 12,
    ode_steps_max: int = 20,
    tau_freeze_steps: int = 5000,
    integration_time: float = 2.0,
    cold_start: bool = True,
    K_total: int = 8,
    aux_loss_scale: float = 0.1,
) -> LiquidARCConfig:
    """v15 config — same as v13.1 (carry over gentler aux scale that didn't NaN)."""
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


class V15Encoder(nn.Module):
    """GR00T features → per-modality positions → canonical LiquidARC substrate."""

    def __init__(
        self,
        config: LiquidARCConfig,
        state_dim: int = 8,
        z_vl_dim: int = 1024,         # per-depth bank entry dim
        z_state_dim: int = 1536,      # GR00T's z_state dim
        K_bank: int = 4,              # number of depth-bank entries
        K_latent: int = 2,            # learnable scratch positions
    ):
        super().__init__()
        self.config = config
        self.d = config.d_model
        self.state_dim = state_dim
        self.z_vl_dim = z_vl_dim
        self.z_state_dim = z_state_dim
        self.K_bank = K_bank
        self.K_latent = K_latent

        # === Per-modality projections ===
        # Each bank entry (1024-d) projects to d. Shared weights across depths
        # (each depth step has same projection but different bank_step_embed).
        self.zbank_proj = nn.Linear(z_vl_dim, self.d)
        nn.init.normal_(self.zbank_proj.weight, std=0.02)
        nn.init.zeros_(self.zbank_proj.bias)

        # Step embedding to differentiate K_bank depth positions (which step in denoising)
        self.bank_step_embed = nn.Parameter(torch.zeros(K_bank, self.d))
        nn.init.normal_(self.bank_step_embed, std=0.02)

        # z_state projection (1536 → d)
        self.zstate_proj = nn.Linear(z_state_dim, self.d)
        nn.init.normal_(self.zstate_proj.weight, std=0.02)
        nn.init.zeros_(self.zstate_proj.bias)

        # Robot state encoder (8 → d via MLP)
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, self.d),
            nn.SiLU(),
            nn.Linear(self.d, self.d),
        )

        # Latent slots — learnable scratch positions for substrate to use
        if K_latent > 0:
            self.latent_slots = nn.Parameter(torch.zeros(K_latent, self.d))
            nn.init.normal_(self.latent_slots, std=0.02)

        self.K = K_bank + 2 + K_latent  # K_bank depths + zstate + state + K_latent

        # Global position embedding (additional differentiator across all K positions)
        self.pos_embed = nn.Parameter(torch.zeros(self.K, self.d))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Canonical substrate
        self.context_pool = ContextPool(config)
        self.dynamics = ContinuousDynamics(config)
        self.curvature_engine = CurvatureEngine()

        # Attention pool aggregation
        self.attn_pool_query = nn.Linear(self.d, 1, bias=True)
        nn.init.zeros_(self.attn_pool_query.weight)
        nn.init.zeros_(self.attn_pool_query.bias)

        self.d_out = self.d

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
        params.append(self.bank_step_embed)
        if self.K_latent > 0:
            params.append(self.latent_slots)
        return params

    def other_parameters(self):
        geo_ids = {id(p) for p in self.geo_parameters()}
        return [p for p in self.parameters() if id(p) not in geo_ids]

    def forward(
        self,
        z_bank: torch.Tensor,       # [B, K_bank, z_vl_dim]
        z_state: torch.Tensor,       # [B, z_state_dim]
        state: torch.Tensor,         # [B, state_dim]
        n_ode_steps_override: Optional[int] = None,
        ablate_substrate: bool = False,
    ):
        B = state.shape[0]

        # === Per-modality positions ===
        # Project each bank step to d, add per-depth step embedding
        bank_pos = self.zbank_proj(z_bank) + self.bank_step_embed.unsqueeze(0)  # [B, K_bank, d]
        zstate_pos = self.zstate_proj(z_state).unsqueeze(1)                       # [B, 1, d]
        state_pos = self.state_enc(state).unsqueeze(1)                            # [B, 1, d]

        positions = [bank_pos, zstate_pos, state_pos]
        if self.K_latent > 0:
            latent_pos = self.latent_slots.unsqueeze(0).expand(B, -1, -1)         # [B, K_latent, d]
            positions.append(latent_pos)

        h0_raw = torch.cat(positions, dim=1)                                       # [B, K, d]
        h0 = h0_raw + self.pos_embed.unsqueeze(0)

        # === Substrate ===
        context = self.context_pool(h0, None)
        self.dynamics.set_context(context, mask=None)

        if ablate_substrate:
            h = h0
            actual_steps = 0
            ponder_cost = torch.zeros(B, device=h.device)
            steps_used = torch.zeros((B, self.K), device=h.device)
        else:
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

        # === Diagnostics ===
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

        # Attention pool
        attn_logits = self.attn_pool_query(h).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=-1)
        cond = (h * attn_weights.unsqueeze(-1)).sum(dim=1)

        return {
            "cond": cond,
            "h_final": h, "h0": h0, "g": g, "tau": tau0,
            "tau_avg": tau_avg, "tau_log_std": tau_log_std,
            "t_diff_param": t_diff_param,
            "metric_cv": metric_cv, "avg_kappa": kappa.abs().mean(),
            "ponder_cost": ponder_cost, "steps_used": steps_used,
            "actual_steps": actual_steps, "attn_weights": attn_weights,
        }
