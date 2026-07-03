"""v12 LIBERO LiquidARC substrate: DINOv2 patches → canonical ContinuousDynamics → cond.

This is a thin wrapper around the canonical LiquidARC modules
(subprojects/liquid-arc/liquid_arc/). It plugs vision patch tokens into the
ContinuousDynamics + euler_solve_halting pipeline, exposes the full SoC
diagnostics (metric_cv, kappa, tau, D²/4τ), and provides geo_parameters() /
other_parameters() for differential-LR optimizer groups.

The point of v12: the LIBERO Liquid student stops being a vanilla ODE and
becomes a real LiquidARC substrate, capable (in principle) of the phase
transition that produced broad ARC generalization. v11 retrieval can sit
on top unchanged — same DINOv2 features, same blending, same eval pipeline.

Per the v11_findings_catalog.md MUST list, this module includes:
  - canonical MetricNet + TauNet + heat-kernel SDPA (via ContinuousDynamics)
  - euler_solve_halting (ACT-style variable depth, 12-20 randomized steps)
  - canonical SoC stabilizers (criticality, curvature_diversity, tau_quality)
  - cold-start regime (ReZero + PonderNet deep sup + KL prior + metric_bias_init_std)
  - structural τ + recent gradient fix
  - step embeddings + step-conditional FiLM Tier 1
  - 100×-slower geo LR via geo_parameters() / other_parameters() split
  - per-step norm homeostasis (handled inside euler_solve)

Vision substrate: 196 DINOv2-small patches × d_model. Phase transition
operates on these tokens (substrate-B from the design discussion).
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical LiquidARC modules (battle-tested)
_LA_ROOT = Path(__file__).resolve().parents[2]  # subprojects/liquid-arc
sys.path.insert(0, str(_LA_ROOT))
from liquid_arc.config import LiquidARCConfig  # type: ignore
from liquid_arc.dynamics import ContinuousDynamics  # type: ignore
from liquid_arc.solver import euler_solve, euler_solve_halting  # type: ignore
from liquid_arc.context_pool import ContextPool  # type: ignore
from liquid_arc.curvature import CurvatureEngine  # type: ignore


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
N_PATCHES = 256  # DINOv2-small at 224 / patch_size 14 → 16×16=256 patches
PATCH_DIM = 384

_DINOV2_BACKBONE = None


def _get_dinov2_backbone():
    """Lazy-load frozen DINOv2-small (matches v9b/v10 vision encoder)."""
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


def make_v12_config(
    d_model: int = 256,
    d_metric: int = 64,
    d_ffn: int = 512,
    n_ode_steps: int = 16,
    ode_steps_min: int = 12,
    ode_steps_max: int = 20,
    tau_freeze_steps: int = 5000,
    integration_time: float = 2.0,
    cold_start: bool = True,
) -> LiquidARCConfig:
    """Construct LiquidARCConfig with v12-LIBERO MUST findings enabled.

    Defaults match the catalog MUST list: halting + ReZero + PonderNet deep
    sup + KL prior + structural_τ + SoC stabilizers + step-conditional FiLM
    + randomized ODE depth.
    """
    return LiquidARCConfig(
        d_model=d_model,
        d_metric=d_metric,
        d_ffn=d_ffn,
        max_seq_len=N_PATCHES,

        # ODE
        n_ode_steps=n_ode_steps,
        ode_steps_min=ode_steps_min,
        ode_steps_max=ode_steps_max,
        integration_time=integration_time,
        tau_min=0.5, tau_max=1.0,
        t_diffusion_init=1.0,

        # Routing (pure heat kernel)
        routing_mode="metric",

        # Tau freeze
        tau_freeze_steps=tau_freeze_steps,

        # === Halting (ACT variable depth) ===
        halting_enabled=True,
        halting_min_steps=4,
        halting_ponder_lambda=0.01,

        # === Cold-start regime ===
        rezero_enabled=cold_start,
        rezero_gate_init=-5.0,
        metric_bias_init_std=0.5 if cold_start else 0.0,
        deep_supervision_enabled=True,
        ponder_kl_lambda=0.01,
        ponder_kl_prior_rate=0.0625,  # 1/16

        # === SoC stabilizers ===
        criticality_loss_enabled=True,
        criticality_loss_lambda=0.01,
        criticality_target_ratio=18.0,
        criticality_D_sq_target=60.0,

        curvature_diversity_loss_enabled=True,
        curvature_diversity_lambda=0.01,
        curvature_cv_floor=2.0,
        curvature_cv_ceiling=10.0,

        tau_quality_loss_enabled=True,
        tau_quality_lambda=0.05,
        tau_mean_target=0.0,  # auto
        tau_log_spread_target=0.6,

        # === Step embeddings + Tier 1 FiLM ===
        step_embed_enabled=True,
        n_step_embeds=ode_steps_max,
        step_conditional_operator=True,
        step_conditional_n_max=ode_steps_max,

        # === Structural τ (recent gradient fix in canonical code) ===
        structural_tau_enabled=True,
        structural_tau_min=0.3,
        structural_tau_max=3.0,

        # === Norm homeostasis ===
        norm_ref=50.0,
        norm_lambda=0.1,

        # === Optimizer (cold-start: metric 10× slower; distillation would be 0.01) ===
        base_lr=3e-4,
        structural_lr_ratio=0.1,
        warmup_steps=500,
        weight_decay=0.01,

        # Solver choice: plain Euler (SDPA heat kernel doesn't materialize NxN)
        use_torch_compile=False,  # skip compile for first build; enable later if needed
    )


class DinoPatchExtractor(nn.Module):
    """Frozen DINOv2-small that exposes BOTH patch tokens AND CLS.

    Input:  [B, 3, H, W] in [0, 1]
    Returns dict {patches: [B, N, 384], cls: [B, 384]}.

    The CLS is computed by DINOv2's own attention (NOT mean-pool of patches)
    so it preserves spatially-discriminative features. Used in v12+ encoder
    as a bypass to recover gripper-relevant info that mean-pool destroys.
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
            patches = feats.get("x_norm_patchtokens")
            cls = feats.get("x_norm_clstoken")
            if cls is None:
                # Some DINOv2 versions name it differently — fall back to mean
                cls = patches.mean(dim=1)
            return {"patches": patches, "cls": cls}
        # Old-style: features only, no CLS
        return {"patches": feats, "cls": feats.mean(dim=1)}


class LiquidARCVisionEncoder(nn.Module):
    """v12 LIBERO encoder: DINOv2 patches → ContinuousDynamics → cond.

    Architecture (matches the canonical LiquidARCModel pattern but on patches):
      DINOv2 patches [B, 196, 384]
        → patch_proj 384→d
        → + positional embedding
        → ContextPool → set_context
        → euler_solve_halting (canonical, with ACT)
        → patch mean pool
        → concat with wrist + state + (optional goal_img) projections
        → out_proj → cond [B, d_cond]

    geo_parameters() returns MetricNet/TauNet/structural_τ/step_embeds/
    context_pool — these get 100×-slower LR via param-group split.
    """

    def __init__(
        self,
        config: LiquidARCConfig,
        d_aux: int = 256,
        state_dim: int = 8,
        use_goal_img: bool = False,
        z_vl_dim: int = 0,
    ):
        super().__init__()
        self.config = config
        self.d = config.d_model
        self.d_aux = d_aux
        self.use_goal_img = use_goal_img
        self.z_vl_dim = z_vl_dim

        # === Frozen DINOv2 backbones (shared across agent/wrist/goal) ===
        self.dino = DinoPatchExtractor()
        # Project patch tokens 384 -> d
        self.patch_proj = nn.Linear(PATCH_DIM, self.d)
        nn.init.normal_(self.patch_proj.weight, std=0.02)
        nn.init.zeros_(self.patch_proj.bias)

        # Positional embedding for N patches
        self.pos_embed = nn.Parameter(torch.zeros(N_PATCHES, self.d))
        nn.init.normal_(self.pos_embed, std=0.02)

        # === LiquidARC substrate ===
        self.context_pool = ContextPool(config)
        self.dynamics = ContinuousDynamics(config)
        self.curvature_engine = CurvatureEngine()

        # === v12.1 scaffolding fix: attention pool + CLS bypass ===
        # The previous mean-pool over 256 ODE-processed patches destroyed
        # gripper-relevant SMALL-spatial-region info. Attention pool lets
        # the model learn to weight gripper-relevant patches. CLS bypass
        # gives a parallel path to global discriminative features
        # (DINOv2's own attention-pooled CLS) — robustness when the
        # substrate hasn't yet routed task-discriminative features to
        # the pooling stage.
        self.attn_pool_query = nn.Linear(self.d, 1, bias=True)
        nn.init.zeros_(self.attn_pool_query.weight)
        nn.init.zeros_(self.attn_pool_query.bias)
        # Init zero → softmax produces uniform weights → behaves like mean-pool at init.
        # Training drives attn_pool_query to concentrate on relevant patches.

        self.agent_cls_proj = nn.Linear(PATCH_DIM, d_aux)
        nn.init.normal_(self.agent_cls_proj.weight, std=0.02)
        nn.init.zeros_(self.agent_cls_proj.bias)

        # === Auxiliary signals: wrist + state (+ goal_img) ===
        self.wrist_proj = nn.Linear(PATCH_DIM, d_aux)
        # State encoder
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, d_aux),
            nn.SiLU(),
            nn.Linear(d_aux, d_aux),
        )
        if use_goal_img:
            self.goal_proj = nn.Linear(PATCH_DIM, d_aux)

        # v12.2 NEW: z_vl projection (GR00T's vision-language fusion vector).
        # Catalog finding: v10/v9b/no-text-Liquid ALL use z_vl from GR00T.
        # Without it, the substrate has no task-phase signal → gripper collapse.
        # Adding z_vl restores the signal v10 relies on for grasp/release timing.
        if z_vl_dim > 0:
            self.z_groot_proj = nn.Linear(z_vl_dim, d_aux)
            nn.init.normal_(self.z_groot_proj.weight, std=0.02)
            nn.init.zeros_(self.z_groot_proj.bias)

        # Final cond projection: agent_cls + patches_pool + wrist + state (+ z_vl) (+ goal) → d_out
        n_aux = 3 + (1 if use_goal_img else 0) + (1 if z_vl_dim > 0 else 0)
        cond_in = self.d + n_aux * d_aux
        self.cond_out = nn.Linear(cond_in, self.d)

        self.d_out = self.d  # final cond dimension

    def geo_parameters(self):
        """Geometric params: 100×-slower LR vs content."""
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
        # Structural tau
        if hasattr(self.dynamics, "structural_tau") and self.dynamics.structural_tau is not None:
            params.append(self.dynamics.structural_tau)
        # Step embeddings
        if self.config.step_embed_enabled and hasattr(self.dynamics, "step_embeds"):
            params.append(self.dynamics.step_embeds)
        # Tau step embed
        if hasattr(self.dynamics, "tau_step_embed"):
            params.extend(self.dynamics.tau_step_embed.parameters())
        # Step-conditional FiLM (Tier 1)
        if self.config.step_conditional_operator:
            for name in ("metric_film_gamma", "metric_film_beta",
                         "tau_film_gamma", "tau_film_beta",
                         "t_diff_per_step"):
                if hasattr(self.dynamics, name):
                    params.extend(getattr(self.dynamics, name).parameters())
        # Pos embed for vision tokens — part of substrate
        params.append(self.pos_embed)
        return params

    def other_parameters(self):
        """Everything not in geo_parameters."""
        geo_ids = {id(p) for p in self.geo_parameters()}
        return [p for p in self.parameters() if id(p) not in geo_ids]

    def _extract_dino(self, img):
        """[B, 3, H, W] in [0,1] → {patches: [B, N, 384], cls: [B, 384]}."""
        return self.dino(img)

    def forward(
        self,
        img: torch.Tensor,         # [B, 3, H, W]
        wrist_img: torch.Tensor,   # [B, 3, H, W]
        state: torch.Tensor,       # [B, state_dim]
        goal_img: Optional[torch.Tensor] = None,
        z_vl: Optional[torch.Tensor] = None,    # [B, z_vl_dim] from GR00T
        n_ode_steps_override: Optional[int] = None,
    ):
        """Returns dict with cond + full diagnostic suite."""
        # === Agent patches through substrate (with CLS bypass) ===
        agent_dino = self._extract_dino(img)
        agent_patches = agent_dino["patches"]                  # [B, N, 384]
        agent_cls = agent_dino["cls"]                          # [B, 384]
        h0 = self.patch_proj(agent_patches) + self.pos_embed[None]  # [B, N, d]
        B, N, _ = h0.shape

        # Context pool
        ctx = self.context_pool(h0, None)
        self.dynamics.set_context(ctx, mask=None)

        # === ODE integration ===
        # n_ode_steps decision: training=random[min,max], eval=n_ode_steps
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
            # ACT halting solver. Returns (h, ponder_cost, steps_used[, sup])
            out = euler_solve_halting(
                self.dynamics, h0, (0.0, T), actual_steps,
                min_steps=self.config.halting_min_steps,
            )
            if isinstance(out, tuple) and len(out) >= 3:
                h, ponder_cost, steps_used = out[0], out[1], out[2]
            else:
                h = out
                ponder_cost = torch.zeros(B, device=h.device)
                steps_used = torch.full((B, N), float(actual_steps), device=h.device)
        else:
            h = euler_solve(self.dynamics, h0, (0.0, T), actual_steps)
            ponder_cost = torch.zeros(B, device=h.device)
            steps_used = torch.full((B, N), float(actual_steps), device=h.device)

        # === Diagnostics ===
        g = self.dynamics.compute_metric_diag(h0)              # [B, N, d]
        metric_cv = g.std() / (g.mean() + 1e-8)
        kappa = self.curvature_engine(g)
        if not self.config.channel_gate_enabled:
            tau0 = self.dynamics.compute_tau(h0)               # [B, N, 1]
            tau_avg = tau0.mean()
            tau_log_std = tau0.clamp_min(1e-6).log().std()
        else:
            tau0 = None
            tau_avg = torch.tensor(1.0, device=h.device)
            tau_log_std = torch.tensor(0.0, device=h.device)

        # Reference to raw t_diffusion parameter (for criticality loss)
        t_diff_param = getattr(self.dynamics, "t_diffusion", None)

        # === Cond construction (v12.1: attention pool + CLS bypass) ===
        # Attention pool over post-ODE patches — learned weights concentrate
        # on task-relevant patches (e.g., gripper-region, object-region).
        attn_logits = self.attn_pool_query(h).squeeze(-1)      # [B, N]
        attn_weights = F.softmax(attn_logits, dim=-1)          # [B, N]
        patch_pool = (h * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, d]

        # CLS bypass — DINOv2's own attention-pooled CLS, projected to d_aux.
        # Bypasses the ODE→pool path, preserves global discriminative features.
        agent_cls_feat = self.agent_cls_proj(agent_cls)         # [B, d_aux]

        # Auxiliary: wrist CLS (true DINOv2 CLS now, not mean-pool), state, optional goal CLS
        wrist_cls = self._extract_dino(wrist_img)["cls"]        # [B, 384]
        wrist_feat = self.wrist_proj(wrist_cls)
        state_feat = self.state_enc(state)

        cond_parts = [patch_pool, agent_cls_feat, wrist_feat, state_feat]
        # v12.2: inject z_vl (GR00T's vision-language fusion vector) into cond.
        # This is what catalog 14.5 calls "z_vl" — the NO TEXT signal that's still
        # task-discriminative. v9b/v10/no-text-Liquid all use this.
        if self.z_vl_dim > 0:
            if z_vl is None:
                # During smoke/eval without GR00T, fall back to zeros (model sees zero z_vl).
                # This matches Stage 1 "pressure-landscape" pattern (z_groot_drop_prob).
                z_vl_zero = torch.zeros(B, self.z_vl_dim, device=h.device, dtype=h.dtype)
                cond_parts.append(self.z_groot_proj(z_vl_zero))
            else:
                cond_parts.append(self.z_groot_proj(z_vl))
        if self.use_goal_img and goal_img is not None:
            goal_cls = self._extract_dino(goal_img)["cls"]
            cond_parts.append(self.goal_proj(goal_cls))

        cond = F.silu(self.cond_out(torch.cat(cond_parts, dim=-1)))  # [B, d]

        return {
            "cond": cond,
            "h_final": h,
            "h0": h0,
            "g": g,
            "tau": tau0,                  # [B, N, 1] (None if channel_gate)
            "tau_avg": tau_avg,
            "tau_log_std": tau_log_std,
            "t_diff_param": t_diff_param, # raw Parameter ref (pre-softplus)
            "metric_cv": metric_cv,
            "avg_kappa": kappa.abs().mean(),
            "ponder_cost": ponder_cost,
            "steps_used": steps_used,
            "actual_steps": actual_steps,
        }
