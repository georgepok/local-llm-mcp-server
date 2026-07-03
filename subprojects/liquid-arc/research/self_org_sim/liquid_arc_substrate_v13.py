"""v13 LIBERO substrate: v10's fan-in + canonical LiquidARC dynamics on K virtual positions.

Architecture insight (from v12 failure analysis):
  - v10's encoder works because it fuses ALL signals (vis CLS, wrist, state, z_vl, goal)
    into a single rich cond [B, d=768] and refines it via simple iterative ODE.
  - v12 broke action prediction by:
    (a) operating substrate on 256 raw vision PATCHES → lossy pool destroys per-patch info
    (b) capping d=256 (vs v10's d=768) — 3× less cond capacity
    (c) z_vl grafted onto cond late, not threaded through ODE

v13's fix:
  - Keep v10's fan-in producing a single 768-d cond_in. ALL signals fused.
  - Expand cond_in into K=8 VIRTUAL positions (cond_in + pos_embed[k]).
  - Substrate operates on these K virtual positions — heat-kernel routing meaningful, MetricNet/TauNet/SoC machinery active.
  - Attention-pool back to single cond [B, 768] — like learned mixture-of-experts, NOT a lossy spatial average.
  - All v12 abilities preserved: halting, structural_τ, SoC stabilizers, step embeds, ReZero, PonderNet, variable-depth.
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

# Canonical LiquidARC modules
_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
from liquid_arc.config import LiquidARCConfig  # type: ignore
from liquid_arc.dynamics import ContinuousDynamics  # type: ignore
from liquid_arc.solver import euler_solve, euler_solve_halting  # type: ignore
from liquid_arc.context_pool import ContextPool  # type: ignore
from liquid_arc.curvature import CurvatureEngine  # type: ignore


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_DIM = 384

_DINOV2_BACKBONE = None


def _get_dinov2_backbone():
    global _DINOV2_BACKBONE
    if _DINOV2_BACKBONE is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _DINOV2_BACKBONE = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", verbose=False
            )
        for p in _DINOV2_BACKBONE.parameters():
            p.requires_grad = False
        _DINOV2_BACKBONE.eval()
    return _DINOV2_BACKBONE


def make_v13_config(
    d_model: int = 768,
    d_metric: int = 64,
    d_ffn: int = 512,
    n_ode_steps: int = 16,
    ode_steps_min: int = 12,
    ode_steps_max: int = 20,
    tau_freeze_steps: int = 5000,
    integration_time: float = 2.0,
    cold_start: bool = True,
    K_virtual: int = 8,
    aux_loss_scale: float = 0.1,
) -> LiquidARCConfig:
    """v13 config: d=768 to match v10's capacity; K=8 virtual positions.

    aux_loss_scale: multiplier on SoC aux loss weights. At 1.0 (canonical),
    d=768 + K=8 + 1000 steps NaN'd at step ~600. Reduce to 0.1 for gentler
    training pressure during early steps.
    """
    return LiquidARCConfig(
        d_model=d_model, d_metric=d_metric, d_ffn=d_ffn,
        max_seq_len=K_virtual,
        n_ode_steps=n_ode_steps, ode_steps_min=ode_steps_min,
        ode_steps_max=ode_steps_max, integration_time=integration_time,
        tau_min=0.5, tau_max=1.0, t_diffusion_init=1.0,
        routing_mode="metric", tau_freeze_steps=tau_freeze_steps,
        halting_enabled=True, halting_min_steps=4,
        halting_ponder_lambda=0.001 * aux_loss_scale,
        rezero_enabled=cold_start, rezero_gate_init=-5.0,
        metric_bias_init_std=0.1 if cold_start else 0.0,  # reduced from 0.5
        deep_supervision_enabled=True,
        ponder_kl_lambda=0.001 * aux_loss_scale,
        ponder_kl_prior_rate=0.0625,
        criticality_loss_enabled=True,
        criticality_loss_lambda=0.001 * aux_loss_scale,  # reduced from 0.01
        criticality_target_ratio=18.0, criticality_D_sq_target=60.0,
        curvature_diversity_loss_enabled=True,
        curvature_diversity_lambda=0.001 * aux_loss_scale,  # reduced from 0.01
        curvature_cv_floor=2.0, curvature_cv_ceiling=10.0,
        tau_quality_loss_enabled=True,
        tau_quality_lambda=0.005 * aux_loss_scale,  # reduced from 0.05
        tau_mean_target=0.0, tau_log_spread_target=0.6,
        step_embed_enabled=True, n_step_embeds=ode_steps_max,
        step_conditional_operator=True, step_conditional_n_max=ode_steps_max,
        structural_tau_enabled=True, structural_tau_min=0.3, structural_tau_max=3.0,
        norm_ref=50.0, norm_lambda=0.1,
        base_lr=3e-4, structural_lr_ratio=0.1,
        warmup_steps=500, weight_decay=0.01,
        use_torch_compile=False,
    )


class DinoCLSExtractor(nn.Module):
    """Frozen DINOv2-small returning ONLY the CLS token (single 384-d global vector).

    v13 uses CLS (DINOv2's own attention-pooled global) instead of raw patches.
    This avoids the lossy-pool-after-substrate problem that broke v12.
    """
    IMAGENET_MEAN = IMAGENET_MEAN
    IMAGENET_STD = IMAGENET_STD

    def __init__(self, backbone=None):
        super().__init__()
        if backbone is None:
            backbone = _get_dinov2_backbone()
        self.backbone = backbone
        self.register_buffer("_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        feats = self.backbone.forward_features(x)
        if isinstance(feats, dict):
            cls = feats.get("x_norm_clstoken")
            if cls is None:
                cls = feats["x_norm_patchtokens"].mean(dim=1)
            return cls
        return feats.mean(dim=1) if feats.ndim == 3 else feats


class V13Encoder(nn.Module):
    """v13.1: per-modality positions. Each of K positions is INITIALIZED from
    a different input signal (vis, wrist, state, z_vl, goal, latent slots).
    Substrate routes BETWEEN MODALITIES via heat-kernel — genuine variety
    drives CV growth without lossy spatial pool.

    K_modalities = number of input signals (5 if all flags on: vis, wrist, state, z_vl, goal)
    K_latent = additional learnable slots for substrate to use as scratch positions
    K_total = K_modalities + K_latent
    """

    def __init__(
        self,
        config: LiquidARCConfig,
        K_virtual: int = 8,  # kept for backward-compat — treated as K_latent budget
        state_dim: int = 8,
        z_vl_dim: int = 1024,
        use_goal_img: bool = True,
        K_latent: int = 3,
    ):
        super().__init__()
        self.config = config
        self.d = config.d_model
        self.state_dim = state_dim
        self.z_vl_dim = z_vl_dim
        self.use_goal_img = use_goal_img

        # === Per-modality projections directly to d (substrate dim) ===
        self.dino = DinoCLSExtractor()
        # Each modality projects to a FULL d-dim position vector — NOT a small d_aux
        self.vis_to_d = nn.Linear(PATCH_DIM, self.d)
        self.wrist_to_d = nn.Linear(PATCH_DIM, self.d)
        self.state_to_d = nn.Sequential(
            nn.Linear(state_dim, self.d), nn.SiLU(), nn.Linear(self.d, self.d),
        )
        n_modalities = 3  # vis, wrist, state
        if z_vl_dim > 0:
            self.z_groot_to_d = nn.Linear(z_vl_dim, self.d)
            nn.init.normal_(self.z_groot_to_d.weight, std=0.02)
            nn.init.zeros_(self.z_groot_to_d.bias)
            n_modalities += 1
        if use_goal_img:
            self.goal_to_d = nn.Linear(PATCH_DIM, self.d)
            n_modalities += 1
        self.n_modalities = n_modalities

        # === Latent slots — learnable scratch positions ===
        # Init small random so they're not zero (which would hurt routing).
        self.K_latent = K_latent
        if K_latent > 0:
            self.latent_slots = nn.Parameter(torch.zeros(K_latent, self.d))
            nn.init.normal_(self.latent_slots, std=0.02)

        self.K = n_modalities + K_latent  # total positions

        # === Position embeddings (small differentiator) ===
        # Each modality slot + each latent slot gets unique embedding.
        # Useful when latent slots want to specialize per-position.
        self.pos_embed = nn.Parameter(torch.zeros(self.K, self.d))
        nn.init.normal_(self.pos_embed, std=0.02)

        # === Canonical LiquidARC substrate (on K virtual positions, NOT patches) ===
        self.context_pool = ContextPool(config)
        self.dynamics = ContinuousDynamics(config)
        self.curvature_engine = CurvatureEngine()

        # === Attention pool over K → single cond ===
        # Init query weights to zero → uniform softmax weights at init → behaves like mean pool.
        # Training learns to concentrate on relevant virtual positions.
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
        if self.K_latent > 0:
            params.append(self.latent_slots)
        return params

    def other_parameters(self):
        geo_ids = {id(p) for p in self.geo_parameters()}
        return [p for p in self.parameters() if id(p) not in geo_ids]

    def forward(
        self,
        img: torch.Tensor,
        wrist_img: torch.Tensor,
        state: torch.Tensor,
        goal_img: Optional[torch.Tensor] = None,
        z_vl: Optional[torch.Tensor] = None,
        n_ode_steps_override: Optional[int] = None,
        ablate_substrate: bool = False,
    ):
        # === Per-modality positions ===
        agent_cls = self.dino(img)                          # [B, 384]
        wrist_cls = self.dino(wrist_img)
        B = state.shape[0]

        # Each modality contributes ONE position vector at full d dim.
        modality_positions = []
        modality_positions.append(self.vis_to_d(agent_cls))      # [B, d]
        modality_positions.append(self.wrist_to_d(wrist_cls))
        modality_positions.append(self.state_to_d(state))
        if self.z_vl_dim > 0:
            if z_vl is None:
                z_vl = torch.zeros(B, self.z_vl_dim, device=state.device, dtype=state.dtype)
            modality_positions.append(self.z_groot_to_d(z_vl))
        if self.use_goal_img:
            if goal_img is not None:
                goal_cls = self.dino(goal_img)
                modality_positions.append(self.goal_to_d(goal_cls))
            else:
                # Use zeros if goal_img path was constructed but not provided this batch
                modality_positions.append(torch.zeros(B, self.d, device=state.device, dtype=state.dtype))

        # Stack modality positions: [B, n_modalities, d]
        modality_stack = torch.stack(modality_positions, dim=1)

        # Add latent slots: expand from [K_latent, d] → [B, K_latent, d]
        if self.K_latent > 0:
            latent_stack = self.latent_slots.unsqueeze(0).expand(B, -1, -1)
            h0_raw = torch.cat([modality_stack, latent_stack], dim=1)   # [B, K, d]
        else:
            h0_raw = modality_stack

        # Add positional embeddings
        h0 = h0_raw + self.pos_embed.unsqueeze(0)                       # [B, K, d]

        # === Canonical substrate (or ablation: skip the ODE entirely) ===
        # Always set context — the dynamics' compute_metric_diag / compute_tau
        # need it for diagnostics, even in ablate mode.
        context = self.context_pool(h0, None)
        self.dynamics.set_context(context, mask=None)

        if ablate_substrate:
            # Hypothesis test: is substrate hurting or helping?
            # Skip ContinuousDynamics integration, use h0 directly. Attention pool
            # will read the pre-substrate per-modality positions (plus learned slots).
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

        # === Attention pool over K virtual positions → cond [B, d] ===
        attn_logits = self.attn_pool_query(h).squeeze(-1)           # [B, K]
        attn_weights = F.softmax(attn_logits, dim=-1)               # [B, K]
        cond = (h * attn_weights.unsqueeze(-1)).sum(dim=1)          # [B, d]

        return {
            "cond": cond,
            "h_final": h,
            "h0": h0,
            "g": g,
            "tau": tau0,
            "tau_avg": tau_avg,
            "tau_log_std": tau_log_std,
            "t_diff_param": t_diff_param,
            "metric_cv": metric_cv,
            "avg_kappa": kappa.abs().mean(),
            "ponder_cost": ponder_cost,
            "steps_used": steps_used,
            "actual_steps": actual_steps,
            "attn_weights": attn_weights,
        }
