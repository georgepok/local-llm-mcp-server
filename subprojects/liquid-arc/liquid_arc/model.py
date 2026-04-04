"""LiquidARCModel — continuous-time geometric model for ARC-AGI.

Architecture:
    Embedding → ContextPool → euler_solve(ContinuousDynamics, h₀, 16 steps) → OutputHead

One shared dynamics module. 16 Euler steps = 16 applications of the same weights.
Emergence from iteration, not from parameter count.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LiquidARCConfig
from .embedding import ARCEmbedding
from .context_pool import ContextPool
from .dynamics import ContinuousDynamics
from .curvature import CurvatureEngine
from .geo_loss import GeometricLoss
from .solver import euler_solve, euler_solve_chunked, invertible_euler_solve, deq_solve


# ARC constants (match fgn/tasks/arc.py)
N_COLORS = 10
PAD_COLOR = 10


class FlatBaselineARC(nn.Module):
    """Flat transformer baseline with same param budget as LiquidARC.

    Standard transformer blocks (LayerNorm → Attention → LayerNorm → FFN)
    with the same embedding and output head. Used for controlled comparison.
    """

    def __init__(self, config: LiquidARCConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        self.embedding = ARCEmbedding(config)

        # Transformer blocks — match param budget (~830K total)
        # Each block: attn (4d² ≈ 260K) + FFN (2×d×d_ffn ≈ 260K) ≈ 520K
        # With d=256, d_ffn=512: ~1 block ≈ 520K params
        # We use 2 blocks to roughly match the dynamics params
        n_blocks = 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=4,
            dim_feedforward=config.d_ffn,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

        self.norm_out = nn.LayerNorm(d)
        self.output_head = nn.Linear(d, N_COLORS, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        colors, xs, ys, roles, sep_mask, sep_types,
        target_mask,
        target_labels=None,
        context_mask=None,
        target_input_colors=None,
        grid_ids=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        device = colors.device

        # Mask test output colors
        colors_masked = colors.clone()
        if target_input_colors is not None:
            colors_masked[target_mask] = target_input_colors[target_mask]
        else:
            colors_masked[target_mask] = PAD_COLOR

        h = self.embedding(colors_masked, xs, ys, roles, sep_mask, sep_types, grid_ids=grid_ids)
        h = self.transformer(h)
        logits = self.output_head(self.norm_out(h))

        result = {"logits": logits}

        # Placeholders for unified logging
        result["metric_cv"] = torch.tensor(0.0, device=device)
        result["avg_kappa"] = torch.tensor(0.0, device=device)

        if target_labels is not None:
            result.update(self._compute_loss(logits, target_labels, target_mask,
                                             target_input_colors, device))

        return result

    def _compute_loss(self, logits, target_labels, target_mask, target_input_colors, device):
        """Compute curriculum-weighted CE loss and accuracy."""
        B = logits.shape[0]
        preds = logits.argmax(dim=-1)

        per_grid_loss = []
        per_grid_acc = []
        tw = self.config.transform_weight
        for b in range(B):
            tgt = target_labels[b]
            valid_b = tgt != -100
            if valid_b.sum() == 0:
                continue
            # Per-cell transform weighting: upweight cells that changed
            per_cell_ce = F.cross_entropy(logits[b][valid_b], tgt[valid_b], reduction='none')
            if target_input_colors is not None:
                inp_b = target_input_colors[b][valid_b]
                changed = (tgt[valid_b] != inp_b)
                cw = self.config.copy_weight
                cell_w = torch.where(changed, tw, cw)
            else:
                cell_w = torch.ones_like(per_cell_ce)
            loss_b = (per_cell_ce * cell_w).sum() / cell_w.sum()
            acc_b = (preds[b][valid_b] == tgt[valid_b]).float().mean()
            per_grid_loss.append(loss_b)
            per_grid_acc.append(acc_b.detach())

        result = {}
        if per_grid_loss:
            grid_losses = torch.stack(per_grid_loss)
            grid_accs = torch.stack(per_grid_acc)
            weights = 3.0 - 2.0 * grid_accs  # upweight grids model is bad at
            ce_loss = (grid_losses * weights).sum() / weights.sum()
        else:
            ce_loss = torch.tensor(0.0, device=device)

        result["ce_loss"] = ce_loss
        result["curv_loss"] = torch.tensor(0.0, device=device)
        result["loss"] = ce_loss

        # Accuracy
        flat_preds = preds.reshape(-1)
        flat_labels = target_labels.reshape(-1)
        valid = flat_labels != -100
        n_valid = valid.sum().clamp(min=1)
        correct_all = (flat_preds[valid] == flat_labels[valid]).sum()
        result["cell_accuracy"] = correct_all.float() / n_valid.float()

        flat_input = target_input_colors.reshape(-1) if target_input_colors is not None else None
        transform = valid & (flat_labels != flat_input) if flat_input is not None else valid
        n_transform = transform.sum().clamp(min=1)
        correct_transform = (flat_preds[transform] == flat_labels[transform]).sum()
        result["transform_accuracy"] = correct_transform.float() / n_transform.float()
        result["n_transform"] = n_transform

        # Raw CE on transform cells only (unweighted, for direct progress tracking)
        flat_logits = logits.reshape(-1, logits.size(-1))
        if transform.sum() > 0:
            result["xform_loss"] = F.cross_entropy(flat_logits[transform], flat_labels[transform])
        else:
            result["xform_loss"] = torch.tensor(0.0, device=device)

        return result


class LiquidARCModel(nn.Module):
    """Continuous-time geometric model for ARC-AGI.

    Embedding → ContextPool(h₀) → euler_solve(dynamics, h₀, 16 steps) → OutputHead

    One ContinuousDynamics module applied repeatedly via ODE integration.
    All computation — metric, diffusion, FFN — inside the dynamics.
    """

    def __init__(self, config: LiquidARCConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        self.embedding = ARCEmbedding(config)
        self.context_pool = ContextPool(config)
        self.dynamics = ContinuousDynamics(config)  # SINGLE instance
        self.curvature_engine = CurvatureEngine()

        # Geometric auxiliary loss (optional)
        if config.geo_loss_enabled:
            self.geo_loss_module = GeometricLoss(config)
        else:
            self.geo_loss_module = None

        self.norm_out = nn.LayerNorm(d)
        self.output_head = nn.Linear(d, N_COLORS, bias=True)

        # Persistent state for temporal continuity across forward passes
        from .persistent_state import PersistentState
        self.persistent = PersistentState(
            alpha=getattr(config, 'persist_alpha', 1.0),
            learnable_alpha=getattr(config, 'persist_learnable_alpha', False),
        )

        # Initialize weights then re-apply special inits
        self.apply(self._init_weights)
        self._reinit_special()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _reinit_special(self):
        """Re-apply special init after _init_weights zeroes biases."""
        with torch.no_grad():
            # Identity metric: Softplus(1.3133) ~ 1.0
            self.dynamics.metric_net_linear2_diag.bias.fill_(math.log(math.e - 1))
            nn.init.normal_(self.dynamics.metric_net_linear2_diag.weight, std=0.05)
            # Low-rank factors: small random init (zero-init = zero gradient trap)
            if hasattr(self.dynamics, 'metric_net_linear2_lr'):
                nn.init.normal_(self.dynamics.metric_net_linear2_lr.weight, std=0.001)
                nn.init.zeros_(self.dynamics.metric_net_linear2_lr.bias)
            if self.config.channel_gate_enabled:
                # Gate: sigmoid(2.0) ≈ 0.88 ≈ current 1/tau_init
                self.dynamics.gate_net_linear2.bias.fill_(2.0)
            else:
                # Tau ~ 1.0 + tau_min
                self.dynamics.tau_net_linear2.bias.fill_(math.log(math.e - 1))
            # Step embeds: already zero-init (additive, no-op at start)
            # W_o and FFN keep normal init — residual pattern (target = h + update)
            # prevents signal destruction; non-zero W_o breaks copy symmetry.

    def forward(
        self,
        colors: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
        roles: torch.Tensor,
        sep_mask: torch.Tensor,
        sep_types: torch.Tensor,
        target_mask: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        target_input_colors: Optional[torch.Tensor] = None,
        grid_ids: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        geo_phase: int = 0,
        boundary_alpha: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        device = colors.device

        # Temporal invariance: use provided n_steps or config default
        actual_steps = n_steps if n_steps is not None else self.config.n_ode_steps

        # Mask test output colors with test input colors (or PAD)
        colors_masked = colors.clone()
        if target_input_colors is not None:
            colors_masked[target_mask] = target_input_colors[target_mask]
        else:
            colors_masked[target_mask] = PAD_COLOR

        # Embed once
        h0_fresh = self.embedding(colors_masked, xs, ys, roles, sep_mask, sep_types,
                                  grid_ids=grid_ids)
        h0 = self.persistent.blend(h0_fresh)  # blend with previous state if active

        # Context computed once from h₀
        context = self.context_pool(h0, context_mask)
        self.dynamics.set_context(context, mask=None)

        # Update FFN amortization divisor to match actual step count
        self.dynamics.set_n_steps(actual_steps)

        # Diagnostics from initial state
        g_init = self.dynamics.compute_metric_diag(h0)
        kappa = self.curvature_engine(g_init)
        metric_cv = g_init.std() / (g_init.mean() + 1e-8)

        if self.config.channel_gate_enabled:
            # Channel-wise gate diagnostics
            gate_init = self.dynamics.compute_gate(h0)  # [B, N, d]
            gate_mean = gate_init.mean()
            gate_pos_std = gate_init.mean(dim=-1).std(dim=1).mean()  # position diversity
            gate_dim_std = gate_init.std(dim=-1).mean()  # dimension diversity (key metric)
            gate_pos_var = gate_init.mean(dim=-1).var(dim=1).mean()  # for loss
            gate_min_val = gate_init.min()
            gate_max_val = gate_init.max()
            # Map to tau-compatible names for backward compat
            tau_avg_val = gate_mean
            tau_std_val = gate_pos_std
            tau_var_val = gate_pos_var
            tau_min_val = gate_min_val
            tau_max_val = gate_max_val
        else:
            tau_init = self.dynamics.compute_tau(h0)
            tau_avg_val = tau_init.mean(dim=1).mean()  # scalar
            tau_flat = tau_init.squeeze(-1)  # [B, N]
            tau_std_val = tau_flat.std(dim=1).mean()
            tau_var_val = tau_flat.var(dim=1).mean()
            tau_min_val = tau_flat.min()
            tau_max_val = tau_flat.max()
            gate_dim_std = None

        # Geometric auxiliary loss at h0 (outside compiled ODE — no torch.compile interaction)
        if (self.geo_loss_module is not None and geo_phase > 0
                and grid_ids is not None and self.config.geo_use_h0):
            h_normed_geo = self.dynamics.norm_geo(h0)
            geo_result = self.geo_loss_module(
                h_normed_geo, g_init,
                xs, ys, grid_ids, sep_mask, colors,
                phase=geo_phase, boundary_alpha=boundary_alpha,
            )
        else:
            geo_result = None  # computed after ODE if geo_use_h0=False

        # Single ODE integration — this IS the model
        T = getattr(self.config, 'integration_time', 1.0)
        if self.config.deq_solver:
            # DEQ: no_grad forward + IFT backward — fastest, O(1) memory
            h = deq_solve(
                self.dynamics, h0, t_span=(0.0, T),
                n_steps=actual_steps,
                n_ift_iters=self.config.deq_ift_iters,
            )
        elif self.config.invertible_solver:
            h = invertible_euler_solve(
                self.dynamics, h0, t_span=(0.0, T),
                n_steps=actual_steps,
                n_fp_iters=self.config.n_fp_iters,
            )
        else:
            # Unrolled Euler: SDPA heat kernel enables FlashAttention,
            # so N*N is never materialized to HBM — no checkpointing needed
            h = euler_solve(
                self.dynamics, h0, t_span=(0.0, T),
                n_steps=actual_steps,
            )

        # Store for persistent state (detached, no grad flow across turns)
        self.persistent.store(h)

        # Output head
        logits = self.output_head(self.norm_out(h))

        result = {
            "logits": logits,
            "h_final": h,
            "metric_cv": metric_cv,
            "avg_kappa": kappa.abs().mean(),
            "tau_avg": tau_avg_val,
            "tau_std": tau_std_val,
            "tau_min": tau_min_val,
            "tau_max": tau_max_val,
        }
        if gate_dim_std is not None:
            result["gate_dim_std"] = gate_dim_std

        # Geo loss at h_final if geo_use_h0=False
        if (self.geo_loss_module is not None and geo_phase > 0
                and grid_ids is not None and not self.config.geo_use_h0
                and geo_result is None):
            g_final = self.dynamics.compute_metric_diag(h)
            h_normed_geo = self.dynamics.norm_geo(h)
            geo_result = self.geo_loss_module(
                h_normed_geo, g_final,
                xs, ys, grid_ids, sep_mask, colors,
                phase=geo_phase, boundary_alpha=boundary_alpha,
            )

        # Geo loss added to result for train.py to assemble
        if geo_result is not None:
            result["geo_loss"] = geo_result["geo_loss"]
            result["geo_mse"] = geo_result["geo_mse"]
        else:
            result["geo_loss"] = torch.tensor(0.0, device=device)
            result["geo_mse"] = torch.tensor(0.0, device=device)

        # Step embed norm diagnostic
        if self.config.step_embed_enabled:
            result["step_embed_norm"] = self.dynamics.step_embeds.norm()

        # Loss + accuracy
        if target_labels is not None:
            result.update(self._compute_loss(
                logits, target_labels, target_mask, target_input_colors,
                kappa, tau_var_val, metric_cv, device,
            ))

        return result

    def _compute_loss(self, logits, target_labels, target_mask,
                      target_input_colors, kappa, tau_var, metric_cv, device):
        """Compute curriculum-weighted CE loss, curvature reg, tau variance penalty, and accuracy."""
        B = logits.shape[0]
        preds = logits.argmax(dim=-1)

        per_grid_loss = []
        per_grid_acc = []
        tw = self.config.transform_weight
        for b in range(B):
            tgt = target_labels[b]
            valid_b = tgt != -100
            if valid_b.sum() == 0:
                continue
            # Per-cell transform weighting: upweight cells that changed
            per_cell_ce = F.cross_entropy(logits[b][valid_b], tgt[valid_b], reduction='none')
            if target_input_colors is not None:
                inp_b = target_input_colors[b][valid_b]
                changed = (tgt[valid_b] != inp_b)
                cw = self.config.copy_weight
                cell_w = torch.where(changed, tw, cw)
            else:
                cell_w = torch.ones_like(per_cell_ce)
            loss_b = (per_cell_ce * cell_w).sum() / cell_w.sum()
            acc_b = (preds[b][valid_b] == tgt[valid_b]).float().mean()
            per_grid_loss.append(loss_b)
            per_grid_acc.append(acc_b.detach())

        result = {}
        if per_grid_loss:
            grid_losses = torch.stack(per_grid_loss)
            grid_accs = torch.stack(per_grid_acc)
            weights = 3.0 - 2.0 * grid_accs  # upweight grids model is bad at
            ce_loss = (grid_losses * weights).sum() / weights.sum()
        else:
            ce_loss = torch.tensor(0.0, device=device)

        # Hard curvature penalty: lambda * |kappa|.mean()
        # Severely punishes curvature growth — forces routing across smooth manifold
        curv_loss = torch.tensor(0.0, device=device)
        if self.config.curvature_lambda > 0:
            curv_loss = self.config.curvature_lambda * kappa.abs().mean()

        # Tau variance maximization: -lambda_tau * Var(tau) across positions
        # Encourages the model to differentiate memory (high tau) from reasoning (low tau)
        tau_var_loss = torch.tensor(0.0, device=device)
        if self.config.tau_var_lambda > 0:
            tau_var_loss = -self.config.tau_var_lambda * tau_var

        # CV floor/ceiling hinge loss: keep metric CV in [floor, ceiling] band
        # Floor prevents plasticity collapse (V1 TTT), ceiling prevents runaway (5M scale)
        cv_floor_loss = torch.tensor(0.0, device=device)
        if self.config.cv_floor_lambda > 0:
            deficit = torch.clamp(self.config.cv_floor_target - metric_cv, min=0.0)
            ceiling = getattr(self.config, 'cv_ceiling_target', 0.0)
            excess = torch.clamp(metric_cv - ceiling, min=0.0) if ceiling > 0 else torch.tensor(0.0, device=device)
            cv_floor_loss = self.config.cv_floor_lambda * (deficit ** 2 + excess ** 2)

        result["ce_loss"] = ce_loss
        result["curv_loss"] = curv_loss
        result["tau_var_loss"] = tau_var_loss
        result["cv_floor_loss"] = cv_floor_loss
        result["loss"] = ce_loss + curv_loss + tau_var_loss + cv_floor_loss

        # Accuracy
        flat_preds = preds.reshape(-1)
        flat_labels = target_labels.reshape(-1)
        valid = flat_labels != -100
        n_valid = valid.sum().clamp(min=1)
        correct_all = (flat_preds[valid] == flat_labels[valid]).sum()
        result["cell_accuracy"] = correct_all.float() / n_valid.float()

        flat_input = target_input_colors.reshape(-1) if target_input_colors is not None else None
        transform = valid & (flat_labels != flat_input) if flat_input is not None else valid
        n_transform = transform.sum().clamp(min=1)
        correct_transform = (flat_preds[transform] == flat_labels[transform]).sum()
        result["transform_accuracy"] = correct_transform.float() / n_transform.float()
        result["n_transform"] = n_transform

        # Raw CE on transform cells only (unweighted, for direct progress tracking)
        flat_logits = logits.reshape(-1, logits.size(-1))
        if transform.sum() > 0:
            result["xform_loss"] = F.cross_entropy(flat_logits[transform], flat_labels[transform])
        else:
            result["xform_loss"] = torch.tensor(0.0, device=device)

        return result

    def geo_parameters(self) -> List[nn.Parameter]:
        """Metric, tau/gate, t_diffusion, context pool, step_embeds (for differentiated LR)."""
        params = list(self.context_pool.parameters())
        params.extend(self.dynamics.metric_net_linear1.parameters())
        params.extend(self.dynamics.metric_net_linear2_diag.parameters())
        if hasattr(self.dynamics, 'metric_net_linear2_lr'):
            params.extend(self.dynamics.metric_net_linear2_lr.parameters())
        if self.config.channel_gate_enabled:
            params.extend(self.dynamics.gate_net_linear1.parameters())
            params.extend(self.dynamics.gate_net_linear2.parameters())
        else:
            params.extend(self.dynamics.tau_net_linear1.parameters())
            params.extend(self.dynamics.tau_net_linear2.parameters())
        params.append(self.dynamics.t_diffusion)
        params.append(self.dynamics.alpha_logit)
        if self.config.step_embed_enabled:
            params.append(self.dynamics.step_embeds)
        return params

    def other_parameters(self) -> List[nn.Parameter]:
        """Everything not in geo_parameters."""
        geo_ids = {id(p) for p in self.geo_parameters()}
        return [p for p in self.parameters() if id(p) not in geo_ids]


def create_model(config: LiquidARCConfig, device: torch.device) -> nn.Module:
    """Create model based on config."""
    if config.model_type == "flat":
        return FlatBaselineARC(config).to(device)
    else:
        return LiquidARCModel(config).to(device)


if __name__ == "__main__":
    print("Testing LiquidARCModel...")

    config = LiquidARCConfig(d_model=64, d_metric=16, d_ffn=128, max_seq_len=128,
                              n_ode_steps=4, tau_var_lambda=0.001)
    model = LiquidARCModel(config)

    n_params = sum(p.numel() for p in model.parameters())
    n_geo = sum(p.numel() for p in model.geo_parameters())
    n_other = sum(p.numel() for p in model.other_parameters())
    print(f"  Total: {n_params:,}")
    print(f"  Geo: {n_geo:,}, Other: {n_other:,}")
    assert n_geo + n_other == n_params, "Parameter groups don't partition"

    B, N = 2, 32
    colors = torch.randint(0, 10, (B, N))
    xs = torch.randint(0, 10, (B, N))
    ys = torch.randint(0, 10, (B, N))
    roles = torch.randint(0, 4, (B, N))
    sep_mask = torch.zeros(B, N, dtype=torch.bool)
    sep_mask[:, [7, 15, 23]] = True
    sep_types = torch.zeros(B, N, dtype=torch.long)
    grid_ids = torch.zeros(B, N, dtype=torch.long)
    grid_ids[:, :8] = 0
    grid_ids[:, 7] = -1
    grid_ids[:, 8:16] = 1
    grid_ids[:, 15] = -1
    grid_ids[:, 16:24] = 2
    grid_ids[:, 23] = -1
    grid_ids[:, 24:] = 3
    target_mask = torch.zeros(B, N, dtype=torch.bool)
    target_mask[:, -4:] = True
    context_mask = ~target_mask
    target_labels = torch.full((B, N), -100, dtype=torch.long)
    target_labels[:, -4:] = torch.randint(0, 10, (B, 4))
    target_input_colors = torch.full((B, N), PAD_COLOR, dtype=torch.long)
    target_input_colors[:, -4:] = torch.randint(0, 10, (B, 4))

    result = model(colors, xs, ys, roles, sep_mask, sep_types,
                   target_mask, target_labels=target_labels,
                   context_mask=context_mask,
                   target_input_colors=target_input_colors,
                   grid_ids=grid_ids)

    assert "loss" in result
    assert "metric_cv" in result
    assert "avg_kappa" in result
    assert "tau_std" in result
    assert "tau_min" in result
    assert "tau_max" in result
    assert "tau_var_loss" in result
    print(f"  Loss: {result['loss'].item():.4f}")
    print(f"  CE: {result['ce_loss'].item():.4f}")
    print(f"  Tau var loss: {result['tau_var_loss'].item():.6f}")
    print(f"  CV: {result['metric_cv']:.4f}, |k|: {result['avg_kappa'].item():.4f}")
    print(f"  Tau: avg={result['tau_avg'].item():.3f}, "
          f"std={result['tau_std'].item():.3f}, "
          f"range=[{result['tau_min'].item():.3f}, {result['tau_max'].item():.3f}]")
    print(f"  Cell acc: {result['cell_accuracy'].item():.4f}")
    print(f"  Xform acc: {result['transform_accuracy'].item():.4f}")

    result["loss"].backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for _ in model.parameters())
    print(f"  Gradients: {has_grad}/{total}")

    # Test flat baseline too
    print("\nTesting FlatBaselineARC...")
    config_flat = LiquidARCConfig(d_model=64, d_ffn=128, max_seq_len=128,
                                   model_type="flat")
    flat_model = FlatBaselineARC(config_flat)
    flat_params = sum(p.numel() for p in flat_model.parameters())
    print(f"  Flat params: {flat_params:,}")

    result_flat = flat_model(colors, xs, ys, roles, sep_mask, sep_types,
                              target_mask, target_labels=target_labels,
                              context_mask=context_mask,
                              target_input_colors=target_input_colors)
    assert "loss" in result_flat
    result_flat["loss"].backward()
    print(f"  Flat loss: {result_flat['loss'].item():.4f}")

    print("\nLiquidARCModel OK")
