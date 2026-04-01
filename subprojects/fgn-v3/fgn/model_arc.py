"""FluidNetARC — FluidNet adapted for ARC-AGI grid reasoning.

Cell-as-token architecture:
  h_cell = ColorEmbed(color) + PosEmbed_x(x) + PosEmbed_y(y) + RoleEmbed(role) + SepEmbed(type)

No tokenizer, no vocab, no causal mask. Bidirectional diffusion across full
sequence. Parallel prediction of all test output cells.

Also provides FlatTransformerARC as parameter-matched baseline.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig
from .context_pool import ContextPool
from .fluid_layer import FluidLayer
from .flat_model import FlatTransformerLayer
from .losses import CurvatureRegularization
from .structural_energy import StructuralEnergy
from .tasks.arc import N_COLORS, MAX_GRID_DIM, PAD_COLOR, PAD_COORD, N_SEP_TYPES


class ARCEmbedding(nn.Module):
    """Additive embedding for ARC cell-as-token representation.

    h = ColorEmbed(color) + PosX(x) + PosY(y) + RoleEmbed(role) + SepEmbed(sep_type) * is_sep
    """

    def __init__(self, config: FGNConfig):
        super().__init__()
        d = config.d_model

        # Color: 10 ARC colors + 1 padding
        self.color_embed = nn.Embedding(N_COLORS + 1, d)
        # Position X: 0-29 + 1 padding
        self.pos_x_embed = nn.Embedding(MAX_GRID_DIM + 1, d)
        # Position Y: 0-29 + 1 padding
        self.pos_y_embed = nn.Embedding(MAX_GRID_DIM + 1, d)
        # Role: input_demo / output_demo / test_input / test_output
        self.role_embed = nn.Embedding(4, d)
        # Separator type embeddings
        self.sep_embed = nn.Embedding(N_SEP_TYPES, d)
        # Sequence position (for flat baseline and general ordering)
        self.seq_pos_embed = nn.Embedding(config.max_seq_len, d)

    def forward(self, colors: torch.Tensor, xs: torch.Tensor,
                ys: torch.Tensor, roles: torch.Tensor,
                sep_mask: torch.Tensor, sep_types: torch.Tensor
                ) -> torch.Tensor:
        """Compute additive embeddings.

        Args:
            colors: [B, N] color indices (0-10)
            xs: [B, N] x coordinates (0-30)
            ys: [B, N] y coordinates (0-30)
            roles: [B, N] role indices (0-3)
            sep_mask: [B, N] True for separator positions
            sep_types: [B, N] separator type at each position

        Returns:
            h: [B, N, d_model] embedded representations
        """
        B, N = colors.shape
        device = colors.device

        # Share role embedding between test_input and test_output so the metric
        # has no role-based signal to separate them — prevents geodesic barrier.
        # ROLE_TEST_OUTPUT (3) → ROLE_TEST_INPUT (2)
        roles_shared = roles.clone()
        roles_shared[roles == 3] = 2

        h = (self.color_embed(colors) +
             self.pos_x_embed(xs) +
             self.pos_y_embed(ys) +
             self.role_embed(roles_shared))

        # Add separator embeddings where applicable
        sep_h = self.sep_embed(sep_types)  # [B, N, d]
        h = h + sep_h * sep_mask.unsqueeze(-1).float()

        # Add sequence position
        pos = torch.arange(N, device=device).unsqueeze(0)
        h = h + self.seq_pos_embed(pos)

        return h


class FluidNetARC(nn.Module):
    """FluidNet model for ARC-AGI tasks.

    Uses cell-as-token embeddings, bidirectional FluidLayers (no causal mask),
    and parallel 10-class prediction head for test output cells.
    """

    def __init__(self, config: FGNConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embedding = ARCEmbedding(config)

        # Context pooling
        self.context_pool = ContextPool(config)

        # FluidLayers (bidirectional — no mask passed)
        self.layers = nn.ModuleList([
            FluidLayer(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Output head: 10-class color prediction
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, N_COLORS, bias=True)

        # Curvature regularization
        self.curv_reg = CurvatureRegularization(
            n_layers=config.n_layers,
            curvature_lambda=config.curvature_lambda,
            correlation_length_init=config.correlation_length_init,
            curvature_reward_mu=config.curvature_reward_mu,
        )

        # Structural energy for grid distance alignment
        self.structural_energy = StructuralEnergy(
            max_context_pairs=config.structural_energy_max_pairs,
            mode="positional",  # use positional mode for grid distances
            d_model=config.d_model,
            d_proj=config.structural_energy_d_proj,
            proj_mlp=config.structural_energy_proj_mlp,
        )
        self.lambda_struct = config.structural_energy_lambda

        # Dropout on embeddings
        self.embed_drop = nn.Dropout(config.dropout)

        # Initialize weights then re-apply FluidLayer special inits
        self.apply(self._init_weights)
        for layer in self.layers:
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
        """Forward pass — fully vectorized for torch.compile.

        Args:
            colors: [B, N] color indices (padded to fixed N)
            xs, ys: [B, N] grid coordinates
            roles: [B, N] role indices
            sep_mask: [B, N] separator positions
            sep_types: [B, N] separator type indices
            target_mask: [B, N] True for test output positions
            target_labels: [B, N] target color at target positions, -100 elsewhere
            context_mask: [B, N] True for non-test-output positions

        Returns:
            Dict with logits, loss, and diagnostic metrics.
        """
        B, N = colors.shape
        device = colors.device

        # Replace test output colors with test_input colors at matching (x,y)
        # positions. This makes test_output informationally rich (carrying what
        # the cell looks like in the input) without leaking the answer.
        colors_masked = colors.clone()
        target_input_colors = kwargs.get("target_input_colors")
        if target_input_colors is not None:
            colors_masked[target_mask] = target_input_colors[target_mask]
        else:
            colors_masked[target_mask] = PAD_COLOR

        # Embed
        h = self.embedding(colors_masked, xs, ys, roles, sep_mask, sep_types)
        h = self.embed_drop(h)

        # Context pool from non-target positions
        context = self.context_pool(h, context_mask)

        # FluidLayers — NO causal mask (bidirectional)
        curvatures = []
        metric_cvs = []
        t_avgs = []

        for layer in self.layers:
            h, kappa, m_cv, t_avg = layer(h, context, mask=None)
            curvatures.append(kappa)
            metric_cvs.append(m_cv)
            t_avgs.append(t_avg)

        # Output head on ALL positions (vectorized), loss only on targets
        h_normed = self.norm(h)
        logits = self.output_head(h_normed)  # [B, N, 10]

        result = {"logits": logits}

        # Monitoring stats
        result["metric_cv"] = sum(metric_cvs) / len(metric_cvs)
        result["avg_kappa"] = sum(k.abs().mean() for k in curvatures) / len(curvatures)

        # Timescales
        t_stack = torch.stack(t_avgs)
        t_mean = t_stack.mean(dim=(0, 1))
        if t_mean.shape[0] >= 3:
            result["avg_t_local"] = t_mean[0]
            result["avg_t_medium"] = t_mean[1]
            result["avg_t_global"] = t_mean[2]

        # Structural energy (outside compiled path)
        if self.lambda_struct > 0:
            e_struct = self._compute_grid_structural_energy(
                h, context, xs, ys, kwargs.get("grid_ids"), sep_mask,
                kwargs.get("lengths"))
        else:
            e_struct = torch.tensor(0.0, device=device)
        result["structural_energy"] = e_struct

        # Loss: single vectorized cross_entropy using target_labels
        if target_labels is not None:
            flat_logits = logits.reshape(-1, N_COLORS)  # [B*N, 10]
            flat_labels = target_labels.reshape(-1)       # [B*N]
            n_valid = (flat_labels != -100).sum().clamp(min=1)
            ce_loss = F.cross_entropy(
                flat_logits, flat_labels, ignore_index=-100, reduction='sum'
            ) / n_valid
            result["ce_loss"] = ce_loss

            curv_loss = self.curv_reg(curvatures)
            result["curv_loss"] = curv_loss
            result["loss"] = ce_loss + curv_loss + self.lambda_struct * e_struct

            # Cell accuracy
            preds = logits.argmax(dim=-1)  # [B, N]
            target_positions = target_labels != -100
            correct = (preds[target_positions] == target_labels[target_positions]).sum()
            result["cell_accuracy"] = correct.float() / n_valid.float()

        return result

    @torch.compiler.disable
    def _compute_grid_structural_energy(
        self, h: torch.Tensor,
        context: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
        grid_ids: Optional[torch.Tensor],
        sep_mask: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute structural energy: align geodesic distances with 2D grid distances.

        Computes grid distances on-the-fly from xs/ys (no precomputed N×N matrix).
        Subsamples 64 non-separator cells per batch item within a single grid.
        """
        B, N, _ = h.shape
        device = h.device

        g = self.layers[-1].get_current_metric(h, context)

        energies = []
        for b in range(B):
            n = lengths[b].item() if lengths is not None else N

            # Find non-separator, non-padding positions
            valid = ~sep_mask[b, :n]
            if grid_ids is not None:
                valid = valid & (grid_ids[b, :n] >= 0)
            valid_pos = valid.nonzero(as_tuple=True)[0]

            if valid_pos.shape[0] < 3:
                continue

            # Subsample
            V = valid_pos.shape[0]
            if V > 64:
                perm = torch.randperm(V, device=device)[:64]
                valid_pos = valid_pos[perm]
                V = 64

            # Grid Euclidean distances from xs/ys
            gx = xs[b, valid_pos].float()  # [V]
            gy = ys[b, valid_pos].float()  # [V]
            dx = gx.unsqueeze(1) - gx.unsqueeze(0)
            dy = gy.unsqueeze(1) - gy.unsqueeze(0)
            D_struct = (dx * dx + dy * dy).sqrt()  # [V, V]

            # Mask cross-grid pairs if grid_ids available
            if grid_ids is not None:
                gids = grid_ids[b, valid_pos]
                same_grid = gids.unsqueeze(1) == gids.unsqueeze(0)
                D_struct = D_struct.masked_fill(~same_grid, 0.0)
                pair_valid = same_grid
            else:
                pair_valid = torch.ones(V, V, dtype=torch.bool, device=device)

            # Exclude self-pairs
            pair_valid.fill_diagonal_(False)
            if pair_valid.sum() < 3:
                continue

            D_struct_norm = D_struct / (D_struct.max() + 1e-8)

            # Geodesic distances
            h_sub = h[b, valid_pos]
            g_sub = g[b, valid_pos]
            diff = h_sub.unsqueeze(1) - h_sub.unsqueeze(0)
            g_avg = (g_sub.unsqueeze(1) + g_sub.unsqueeze(0)) / 2
            D_geo = (diff * diff * g_avg).sum(-1)
            D_geo_norm = D_geo / (D_geo.max() + 1e-8)

            energy = ((D_geo_norm - D_struct_norm) ** 2)[pair_valid].mean()
            energies.append(energy)

        if len(energies) == 0:
            return torch.tensor(0.0, device=device)

        return torch.stack(energies).mean()

    def geo_parameters(self) -> List[nn.Parameter]:
        """MetricNet + TimeNet + CurvatureReg + ContextPool + StructuralEnergy parameters."""
        params = list(self.context_pool.parameters())
        params.extend(self.curv_reg.parameters())
        params.extend(self.structural_energy.parameters())
        for layer in self.layers:
            params.extend(layer.metric_net_linear1.parameters())
            params.extend(layer.metric_net_linear2.parameters())
            params.extend(layer.time_net_linear1.parameters())
            params.extend(layer.time_net_linear2.parameters())
            params.extend(layer.curvature_engine.parameters())
        return params

    def other_parameters(self) -> List[nn.Parameter]:
        """Everything not in geo_parameters."""
        geo_ids = {id(p) for p in self.geo_parameters()}
        return [p for p in self.parameters() if id(p) not in geo_ids]


class FlatTransformerARC(nn.Module):
    """Flat transformer baseline for ARC-AGI tasks.

    Same embedding/output structure as FluidNetARC, but uses standard
    dot-product attention instead of geometric diffusion. Bidirectional.
    """

    def __init__(self, config: FGNConfig):
        super().__init__()
        self.config = config

        # Same embeddings
        self.embedding = ARCEmbedding(config)

        # Standard transformer layers
        self.layers = nn.ModuleList([
            FlatTransformerLayer(config, i) for i in range(config.n_layers)
        ])

        # Output head
        self.norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, N_COLORS, bias=True)

        # Dropout
        self.embed_drop = nn.Dropout(config.dropout)

        # Initialize
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
        """Forward pass — fully vectorized for torch.compile."""
        B, N = colors.shape
        device = colors.device

        # Replace test output colors with test_input colors at matching (x,y)
        colors_masked = colors.clone()
        target_input_colors = kwargs.get("target_input_colors")
        if target_input_colors is not None:
            colors_masked[target_mask] = target_input_colors[target_mask]
        else:
            colors_masked[target_mask] = PAD_COLOR

        # Embed
        h = self.embedding(colors_masked, xs, ys, roles, sep_mask, sep_types)
        h = self.embed_drop(h)

        # Bidirectional transformer layers
        for layer in self.layers:
            h = layer(h, mask=None)

        # Output head on ALL positions
        h_normed = self.norm(h)
        logits = self.output_head(h_normed)  # [B, N, 10]

        result = {
            "logits": logits,
            "metric_cv": torch.tensor(0.0, device=device),
            "avg_kappa": torch.tensor(0.0, device=device),
            "structural_energy": torch.tensor(0.0, device=device),
        }

        if target_labels is not None:
            flat_logits = logits.reshape(-1, N_COLORS)
            flat_labels = target_labels.reshape(-1)
            n_valid = (flat_labels != -100).sum().clamp(min=1)
            ce_loss = F.cross_entropy(
                flat_logits, flat_labels, ignore_index=-100, reduction='sum'
            ) / n_valid
            result["ce_loss"] = ce_loss
            result["curv_loss"] = torch.tensor(0.0, device=device)
            result["loss"] = ce_loss

            preds = logits.argmax(dim=-1)
            target_positions = target_labels != -100
            correct = (preds[target_positions] == target_labels[target_positions]).sum()
            result["cell_accuracy"] = correct.float() / n_valid.float()

        return result

    def geo_parameters(self) -> List[nn.Parameter]:
        return []

    def other_parameters(self) -> List[nn.Parameter]:
        return list(self.parameters())


def create_arc_model(config: FGNConfig, device: torch.device):
    """Create ARC model based on config."""
    if config.model_type == "flat":
        return FlatTransformerARC(config).to(device)
    else:
        return FluidNetARC(config).to(device)


if __name__ == "__main__":
    print("Testing FluidNetARC...")

    cfg = FGNConfig(
        d_model=64, n_heads=4, n_layers=2,
        d_ff=256, max_seq_len=256,
        d_metric=16, d_ffn_fluid=128, n_scales=3,
        architecture_version="fluid",
        geo_metric_type="learned",
        curvature_lambda=0.0,
        structural_energy_lambda=0.0,
        dropout=0.1,
    )

    model = FluidNetARC(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    B, N = 2, 32
    N_out = 4

    colors = torch.randint(0, 10, (B, N))
    xs = torch.randint(0, 10, (B, N))
    ys = torch.randint(0, 10, (B, N))
    roles = torch.randint(0, 4, (B, N))
    sep_mask = torch.zeros(B, N, dtype=torch.bool)
    sep_mask[:, [7, 15, 23]] = True
    sep_types = torch.zeros(B, N, dtype=torch.long)
    target_mask = torch.zeros(B, N, dtype=torch.bool)
    target_mask[:, -N_out:] = True
    context_mask = ~target_mask
    target_colors = torch.randint(0, 10, (B, N_out))

    result = model(colors, xs, ys, roles, sep_mask, sep_types,
                   target_mask, target_colors=target_colors,
                   context_mask=context_mask)

    assert "loss" in result
    assert "ce_loss" in result
    assert "cell_accuracy" in result
    print(f"  Loss: {result['loss'].item():.4f}")
    print(f"  CE: {result['ce_loss'].item():.4f}")
    print(f"  Cell acc: {result['cell_accuracy'].item():.4f}")
    print(f"  CV: {result['metric_cv']:.4f}")

    result["loss"].backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for _ in model.parameters())
    print(f"  Gradients: {has_grad}/{total}")

    # Test flat baseline
    print("\nTesting FlatTransformerARC...")
    cfg_flat = FGNConfig(
        d_model=64, n_heads=4, n_layers=2,
        d_ff=256, max_seq_len=256,
        model_type="flat",
        dropout=0.1,
    )
    flat_model = FlatTransformerARC(cfg_flat)
    n_flat = sum(p.numel() for p in flat_model.parameters())
    print(f"  Parameters: {n_flat:,}")

    result_flat = flat_model(colors, xs, ys, roles, sep_mask, sep_types,
                             target_mask, target_colors=target_colors)
    assert "loss" in result_flat
    print(f"  Loss: {result_flat['loss'].item():.4f}")

    result_flat["loss"].backward()
    print("  Gradient flow: OK")

    print("\nFluidNetARC OK")
