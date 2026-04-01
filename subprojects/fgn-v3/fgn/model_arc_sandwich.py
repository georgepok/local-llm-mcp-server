"""SandwichARC — sandwich architecture for ARC-AGI grid reasoning.

Bottom FluidLayers: spatial reasoning within grids (symmetric diffusion OK here)
Middle AttnLayers: asymmetric information routing (demos → test_output)
Top FluidLayers: final spatial processing for output

The key insight: symmetric diffusion is good for spatial reasoning within grids
(adjacency, boundaries, connected regions) but fundamentally cannot route
information asymmetrically from demos to test output. Attention handles that
naturally via separate Q/K projections.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig
from .context_pool import ContextPool
from .fluid_layer import FluidLayer
from .layer_v7_attn import AttnOnlyLayer
from .liquid_layer import LiquidLayer
from .losses import CurvatureRegularization
from .tasks.arc import N_COLORS, MAX_GRID_DIM, PAD_COLOR, N_SEP_TYPES


class ARCEmbedding(nn.Module):
    """Additive embedding for ARC cell-as-token representation.

    h = ColorEmbed(color) + PosX(x) + PosY(y) + RoleEmbed(role) + SepEmbed(sep_type) * is_sep
    Role embedding shared between test_input and test_output (role 3 → 2).
    """

    def __init__(self, config: FGNConfig):
        super().__init__()
        d = config.d_model

        self.color_embed = nn.Embedding(N_COLORS + 1, d)
        self.pos_x_embed = nn.Embedding(MAX_GRID_DIM + 1, d)
        self.pos_y_embed = nn.Embedding(MAX_GRID_DIM + 1, d)
        self.role_embed = nn.Embedding(4, d)
        self.sep_embed = nn.Embedding(N_SEP_TYPES, d)
        self.seq_pos_embed = nn.Embedding(config.max_seq_len, d)

    def forward(self, colors, xs, ys, roles, sep_mask, sep_types):
        B, N = colors.shape
        device = colors.device

        # Share role embedding: test_output (3) → test_input (2)
        roles_shared = roles.clone()
        roles_shared[roles == 3] = 2

        h = (self.color_embed(colors) +
             self.pos_x_embed(xs) +
             self.pos_y_embed(ys) +
             self.role_embed(roles_shared))

        sep_h = self.sep_embed(sep_types)
        h = h + sep_h * sep_mask.unsqueeze(-1).float()

        pos = torch.arange(N, device=device).unsqueeze(0)
        h = h + self.seq_pos_embed(pos)

        return h


class SandwichARC(nn.Module):
    """Sandwich model for ARC-AGI tasks.

    Architecture:
        Bottom FluidLayers (spatial reasoning within grids)
        Middle AttnOnlyLayers (demo → test_output information routing)
        Top FluidLayers (spatial processing for output)
        10-class output head

    All layers bidirectional (mask=None).
    """

    def __init__(self, config: FGNConfig):
        super().__init__()
        self.config = config

        n_bot = config.sandwich_bottom_geo_layers
        n_mid = config.sandwich_middle_attn_layers
        n_top = config.sandwich_top_geo_layers
        self.middle_iters = getattr(config, 'sandwich_middle_iters', 1)

        # Embeddings
        self.embedding = ARCEmbedding(config)

        # Context pooling (for geometric layers)
        self.context_pool = ContextPool(config)

        # Select geometric layer class
        GeoLayer = LiquidLayer if getattr(config, 'liquid_mode', False) else FluidLayer

        # Bottom geometric layers
        self.bottom_geo = nn.ModuleList([
            GeoLayer(config, layer_idx=i)
            for i in range(n_bot)
        ])

        # Middle attention layers
        mid_start = n_bot
        self.middle_attn = nn.ModuleList([
            AttnOnlyLayer(config, layer_idx=mid_start + i)
            for i in range(n_mid)
        ])

        # Top geometric layers
        top_start = n_bot + n_mid
        self.top_geo = nn.ModuleList([
            GeoLayer(config, layer_idx=top_start + i)
            for i in range(n_top)
        ])

        # Output head: 10-class color prediction
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, N_COLORS, bias=True)

        # Curvature regularization — n_layers accounts for refinement iterations
        n_refine = getattr(config, 'n_refine_iters', 1)
        self.curv_reg = CurvatureRegularization(
            n_layers=(n_bot + n_top) * n_refine,
            curvature_lambda=config.curvature_lambda,
            correlation_length_init=config.correlation_length_init,
            curvature_reward_mu=config.curvature_reward_mu,
        )

        # Dropout on embeddings
        self.embed_drop = nn.Dropout(config.dropout)

        # Initialize weights then re-apply FluidLayer special inits
        self.apply(self._init_weights)
        for layer in list(self.bottom_geo) + list(self.top_geo):
            layer.reinit_special()

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
        colors: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
        roles: torch.Tensor,
        sep_mask: torch.Tensor,
        sep_types: torch.Tensor,
        target_mask: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        device = colors.device

        # Replace test output colors with test_input colors at matching (x,y)
        colors_masked = colors.clone()
        target_input_colors = kwargs.get("target_input_colors")
        if target_input_colors is not None:
            colors_masked[target_mask] = target_input_colors[target_mask]
        else:
            colors_masked[target_mask] = PAD_COLOR

        # Embed once
        h = self.embedding(colors_masked, xs, ys, roles, sep_mask, sep_types)
        h = self.embed_drop(h)

        # Latent-space refinement: loop all layers n_refine times
        # Embed once → (geo → attn → geo) × n_refine → logits once
        n_refine = self.config.n_refine_iters
        curvatures = []
        metric_cvs = []
        t_avgs = []

        for _refine in range(n_refine):
            # Recompute context each iteration (h evolves)
            context = self.context_pool(h, context_mask)

            # Bottom geometric layers — spatial reasoning
            for layer in self.bottom_geo:
                h, kappa, m_cv, t_avg = layer(h, context, mask=None)
                curvatures.append(kappa)
                metric_cvs.append(m_cv)
                t_avgs.append(t_avg)

            # Middle attention layers — iterative reasoning
            for _iter in range(self.middle_iters):
                for layer in self.middle_attn:
                    h = layer(h, mask=None)

            # Top geometric layers — output spatial processing
            for layer in self.top_geo:
                h, kappa, m_cv, t_avg = layer(h, context, mask=None)
                curvatures.append(kappa)
                metric_cvs.append(m_cv)
                t_avgs.append(t_avg)

        # Output head
        h_normed = self.norm(h)
        logits = self.output_head(h_normed)

        result = {"logits": logits, "h_final": h}

        # Monitoring stats
        n_bot = len(self.bottom_geo)
        result["metric_cv"] = sum(metric_cvs) / len(metric_cvs)
        result["avg_kappa"] = sum(k.abs().mean() for k in curvatures) / len(curvatures)
        result["bot_metric_cv"] = sum(metric_cvs[:n_bot]) / n_bot
        result["top_metric_cv"] = sum(metric_cvs[n_bot:]) / max(len(metric_cvs) - n_bot, 1)
        result["bot_avg_kappa"] = sum(k.abs().mean() for k in curvatures[:n_bot]) / n_bot
        result["top_avg_kappa"] = sum(
            k.abs().mean() for k in curvatures[n_bot:]
        ) / max(len(curvatures) - n_bot, 1)

        # Timescales
        t_stack = torch.stack(t_avgs)
        t_mean = t_stack.mean(dim=(0, 1))
        if t_mean.shape[0] >= 3:
            result["avg_t_local"] = t_mean[0]
            result["avg_t_medium"] = t_mean[1]
            result["avg_t_global"] = t_mean[2]

        result["structural_energy"] = torch.tensor(0.0, device=device)

        # Loss — curriculum-weighted CE: grids close to solved get upweighted
        # The model discovers how to solve tasks; we just focus its attention
        # on tasks it's close to completing.
        if target_labels is not None:
            B = logits.shape[0]
            flat_input_colors = target_input_colors.reshape(-1) if target_input_colors is not None else None

            # Per-grid CE loss and accuracy (detached weights — no gradient through curriculum)
            per_grid_loss = []
            per_grid_acc = []
            preds = logits.argmax(dim=-1)

            for b in range(B):
                tgt = target_labels[b]
                valid_b = tgt != -100
                n_valid_b = valid_b.sum().clamp(min=1)
                if valid_b.sum() == 0:
                    continue
                loss_b = F.cross_entropy(
                    logits[b][valid_b], tgt[valid_b], reduction='mean'
                )
                acc_b = (preds[b][valid_b] == tgt[valid_b]).float().mean()
                per_grid_loss.append(loss_b)
                per_grid_acc.append(acc_b.detach())  # detach — curriculum weight is a constant

            if per_grid_loss:
                grid_losses = torch.stack(per_grid_loss)
                grid_accs = torch.stack(per_grid_acc)
                # Curriculum weight: near-solved grids get up to 3x weight
                weights = 1.0 + 2.0 * grid_accs
                ce_loss = (grid_losses * weights).sum() / weights.sum()
            else:
                ce_loss = torch.tensor(0.0, device=device)

            # Track transform mask for accuracy reporting
            flat_logits = logits.reshape(-1, N_COLORS)
            flat_labels = target_labels.reshape(-1)
            valid = flat_labels != -100
            if flat_input_colors is not None:
                transform = valid & (flat_labels != flat_input_colors)
            else:
                transform = valid
            n_transform = transform.sum().clamp(min=1)
            result["ce_loss"] = ce_loss

            curv_loss = self.curv_reg(curvatures)
            result["curv_loss"] = curv_loss
            result["loss"] = ce_loss + curv_loss

            # Accuracy: report both overall and transform-only
            flat_preds = preds.reshape(-1)
            n_valid = valid.sum().clamp(min=1)
            correct_all = (flat_preds[valid] == flat_labels[valid]).sum()
            result["cell_accuracy"] = correct_all.float() / n_valid.float()

            correct_transform = (flat_preds[transform] == flat_labels[transform]).sum()
            result["transform_accuracy"] = correct_transform.float() / n_transform.float()
            result["n_transform"] = n_transform

        return result

    def geo_parameters(self) -> List[nn.Parameter]:
        """All geometric parameters: metric, time/tau, curvature, context pool."""
        params = list(self.context_pool.parameters())
        params.extend(self.curv_reg.parameters())
        for layer in list(self.bottom_geo) + list(self.top_geo):
            if isinstance(layer, LiquidLayer):
                # LiquidLayer: dynamics contains metric_net, tau_net, t_diffusion
                params.extend(layer.dynamics.metric_net_linear1.parameters())
                params.extend(layer.dynamics.metric_net_linear2.parameters())
                params.extend(layer.dynamics.tau_net_linear1.parameters())
                params.extend(layer.dynamics.tau_net_linear2.parameters())
                params.append(layer.dynamics.t_diffusion)
                params.extend(layer.curvature_engine.parameters())
            else:
                # FluidLayer: metric_net and time_net at top level
                params.extend(layer.metric_net_linear1.parameters())
                params.extend(layer.metric_net_linear2.parameters())
                params.extend(layer.time_net_linear1.parameters())
                params.extend(layer.time_net_linear2.parameters())
                params.extend(layer.curvature_engine.parameters())
        return params

    def attn_parameters(self) -> List[nn.Parameter]:
        """All attention parameters."""
        params = []
        for layer in self.middle_attn:
            params.extend(layer.attention.parameters())
        return params

    def other_parameters(self) -> List[nn.Parameter]:
        """Everything not in geo or attn parameters."""
        geo_ids = {id(p) for p in self.geo_parameters()}
        attn_ids = {id(p) for p in self.attn_parameters()}
        return [p for p in self.parameters()
                if id(p) not in geo_ids and id(p) not in attn_ids]


def create_arc_model(config: FGNConfig, device: torch.device):
    """Create ARC model based on config."""
    if config.model_type == "flat":
        from .model_arc import FlatTransformerARC
        return FlatTransformerARC(config).to(device)
    elif config.sandwich_mode:
        return SandwichARC(config).to(device)
    else:
        from .model_arc import FluidNetARC
        return FluidNetARC(config).to(device)


if __name__ == "__main__":
    print("Testing SandwichARC...")

    cfg = FGNConfig(
        d_model=64, n_heads=4, n_layers=8,
        d_ff=256, max_seq_len=256,
        d_metric=16, d_ffn_fluid=128, n_scales=3,
        architecture_version="fluid",
        geo_metric_type="learned",
        curvature_lambda=0.0,
        dropout=0.1,
        sandwich_mode=True,
        sandwich_bottom_geo_layers=2,
        sandwich_middle_attn_layers=4,
        sandwich_top_geo_layers=2,
    )

    model = SandwichARC(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    n_geo = sum(p.numel() for p in model.geo_parameters())
    n_attn = sum(p.numel() for p in model.attn_parameters())
    n_other = sum(p.numel() for p in model.other_parameters())
    print(f"  Total: {n_params:,}")
    print(f"  Geo: {n_geo:,}, Attn: {n_attn:,}, Other: {n_other:,}")
    print(f"  Sum: {n_geo + n_attn + n_other:,}")
    assert n_geo + n_attn + n_other == n_params, "Parameter groups don't partition"

    B, N = 2, 32
    colors = torch.randint(0, 10, (B, N))
    xs = torch.randint(0, 10, (B, N))
    ys = torch.randint(0, 10, (B, N))
    roles = torch.randint(0, 4, (B, N))
    sep_mask = torch.zeros(B, N, dtype=torch.bool)
    sep_mask[:, [7, 15, 23]] = True
    sep_types = torch.zeros(B, N, dtype=torch.long)
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
                   target_input_colors=target_input_colors)

    assert "loss" in result
    assert "bot_metric_cv" in result
    assert "top_metric_cv" in result
    print(f"  Loss: {result['loss'].item():.4f}")
    print(f"  Bot CV: {result['bot_metric_cv']:.4f}, Top CV: {result['top_metric_cv']:.4f}")
    print(f"  Bot |k|: {result['bot_avg_kappa'].item():.4f}, Top |k|: {result['top_avg_kappa'].item():.4f}")

    result["loss"].backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for _ in model.parameters())
    print(f"  Gradients: {has_grad}/{total}")

    print("\nSandwichARC OK")
