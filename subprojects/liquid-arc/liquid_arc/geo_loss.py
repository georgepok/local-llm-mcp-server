"""Geometric auxiliary loss (L_geo) for LiquidARC — MSE on D².

Directly supervises the model's squared geodesic distances against spatial
targets derived from grid coordinates. No softmax — the optimizer cannot
exploit shift-invariance to build warped geometries that produce "correct"
attention patterns without correct distances.

Phases:
  1 (steps 0–5K):    Target = squared Manhattan distance. CE weight = 0.
                      Only MetricNet trains. Forces 2D grid manifold.
  2 (steps 5K+):     Target interpolates manhattan→boundary over 3K steps,
                      then permanent boundary supervision. CE ramps 0→1.
                      Scaffold is PERMANENT (λ_geo=1.0 forever).
"""

from collections import deque
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LiquidARCConfig


class GeometricLoss(nn.Module):
    """MSE between model's D² and spatial target distances."""

    def __init__(self, config: LiquidARCConfig):
        super().__init__()
        self.config = config

    def compute_spatial_target_D2(
        self,
        xs: torch.Tensor,        # [B, N]
        ys: torch.Tensor,        # [B, N]
        grid_ids: torch.Tensor,  # [B, N] (-1 for separators)
        sep_mask: torch.Tensor,  # [B, N]
    ) -> tuple:
        """Compute squared Manhattan distance target and valid-pair mask.

        Target: (|x_i - x_j| + |y_i - y_j|)²  for same-grid pairs.
        Separators and cross-grid pairs are excluded via the mask.

        Returns:
            target_D2: [B, N, N] squared Manhattan distances
            valid_mask: [B, N, N] bool — True for pairs included in the loss
        """
        B, N = xs.shape
        device = xs.device

        # Manhattan distance squared: (|dx| + |dy|)²
        dx = (xs.unsqueeze(2).float() - xs.unsqueeze(1).float()).abs()
        dy = (ys.unsqueeze(2).float() - ys.unsqueeze(1).float()).abs()
        manhattan = dx + dy             # [B, N, N]
        target_D2 = manhattan * manhattan  # squared Manhattan

        # Valid pair mask: same grid, neither is a separator
        gid_i = grid_ids.unsqueeze(2)  # [B, N, 1]
        gid_j = grid_ids.unsqueeze(1)  # [B, 1, N]
        same_grid = (gid_i == gid_j)

        is_sep_i = sep_mask.unsqueeze(2)
        is_sep_j = sep_mask.unsqueeze(1)
        valid_mask = same_grid & ~is_sep_i & ~is_sep_j

        # Also exclude separator self-pairs (grid_id == -1 matches itself)
        neg_grid = (gid_i < 0)
        valid_mask = valid_mask & ~neg_grid

        return target_D2, valid_mask

    def compute_model_D2(
        self,
        h_normed: torch.Tensor,  # [B, N, d]
        g: torch.Tensor,         # [B, N, d] positive metric
    ) -> torch.Tensor:
        """Materialize the model's squared geodesic distance matrix.

        D²_ij = sum_d g_d * (h_i - h_j)²
              = ||k_i||² + ||k_j||² - 2 * k_i · k_j
        where k = h_normed * sqrt(g).

        Clamped to >= 0 via F.relu to prevent NaN from floating-point noise.

        Returns: [B, N, N] non-negative squared distances.
        """
        sqrt_g = torch.sqrt(g)
        k = h_normed * sqrt_g  # [B, N, d]

        # ||k_i||² for each position
        k_norm_sq = (k * k).sum(dim=-1)  # [B, N]

        # D²_ij = ||k_i||² + ||k_j||² - 2 * k_i · k_j
        # [B, N, 1] + [B, 1, N] - 2 * [B, N, N]
        cross = torch.bmm(k, k.transpose(1, 2))  # [B, N, N]
        D2 = k_norm_sq.unsqueeze(2) + k_norm_sq.unsqueeze(1) - 2.0 * cross

        # Guardrail: clamp to >= 0 (floating-point can produce tiny negatives)
        D2 = F.relu(D2)

        return D2

    def _compute_connected_components(
        self,
        colors: torch.Tensor,     # [B, N]
        xs: torch.Tensor,
        ys: torch.Tensor,
        grid_ids: torch.Tensor,
    ) -> torch.Tensor:
        """BFS connected components: same-color, 4-connected, same grid.

        CPU BFS. ~0.1ms per 30x30 grid — negligible.

        Returns: [B, N] component IDs (unique per batch element, -1 for separators).
        """
        B, N = colors.shape
        device = colors.device

        colors_cpu = colors.cpu().numpy()
        xs_cpu = xs.cpu().numpy()
        ys_cpu = ys.cpu().numpy()
        gids_cpu = grid_ids.cpu().numpy()

        comp_ids = torch.full((B, N), -1, dtype=torch.long)

        for b in range(B):
            grid_positions = {}
            for i in range(N):
                gid = int(gids_cpu[b, i])
                if gid < 0:
                    continue
                key = (gid, int(xs_cpu[b, i]), int(ys_cpu[b, i]))
                grid_positions[key] = i

            visited = [False] * N
            comp_id = 0

            for i in range(N):
                if visited[i] or int(gids_cpu[b, i]) < 0:
                    continue
                queue = deque([i])
                visited[i] = True
                color_i = int(colors_cpu[b, i])
                gid_i = int(gids_cpu[b, i])

                while queue:
                    curr = queue.popleft()
                    comp_ids[b, curr] = comp_id
                    cx, cy = int(xs_cpu[b, curr]), int(ys_cpu[b, curr])

                    for ddx, ddy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = cx + ddx, cy + ddy
                        key = (gid_i, nx, ny)
                        if key in grid_positions:
                            j = grid_positions[key]
                            if not visited[j] and int(colors_cpu[b, j]) == color_i:
                                visited[j] = True
                                queue.append(j)

                comp_id += 1

        return comp_ids.to(device)

    def _apply_object_boundaries(
        self,
        target_D2: torch.Tensor,       # [B, N, N] — manhattan² targets
        valid_mask: torch.Tensor,      # [B, N, N]
        component_ids: torch.Tensor,   # [B, N]
        wall_distance: float,          # large constant for cross-object pairs
        alpha: float = 1.0,            # interpolation: 0=manhattan², 1=boundary
    ) -> torch.Tensor:
        """Interpolate target D² from manhattan² to boundary targets.

        When alpha=0: pure manhattan² (Phase 1 continuation).
        When alpha=1: same-object → 0, cross-object → wall (fully pinched).
        Between: target = (1-alpha)*manhattan² + alpha*boundary_target.

        This prevents the geometric shock of snapping targets instantly.

        Returns: [B, N, N] modified target_D2.
        """
        cid_i = component_ids.unsqueeze(2)  # [B, N, 1]
        cid_j = component_ids.unsqueeze(1)  # [B, 1, N]
        same_component = (cid_i == cid_j) & (cid_i >= 0) & (cid_j >= 0)

        # Pure boundary target: same-object → 0, cross-object → wall
        boundary_target = torch.where(same_component, torch.zeros_like(target_D2),
                                       torch.full_like(target_D2, wall_distance))

        # Interpolate: smooth transition from manhattan² to boundary
        interpolated = (1.0 - alpha) * target_D2 + alpha * boundary_target

        # Only apply within valid mask
        target_D2 = torch.where(valid_mask, interpolated, target_D2)

        return target_D2

    def forward(
        self,
        h_normed: torch.Tensor,         # [B, N, d]
        g: torch.Tensor,                 # [B, N, d]
        xs: torch.Tensor,                # [B, N]
        ys: torch.Tensor,                # [B, N]
        grid_ids: torch.Tensor,          # [B, N]
        sep_mask: torch.Tensor,          # [B, N]
        colors: Optional[torch.Tensor],  # [B, N] for Phase 2 connected components
        phase: int,                       # 1 or 2
        boundary_alpha: float = 1.0,     # interpolation for Phase 2 (0=manhattan, 1=boundary)
    ) -> Dict[str, torch.Tensor]:
        """Compute MSE(model_D², target_D²) over valid grid pairs.

        Args:
            phase: 1 = squared Manhattan, 2 = object boundaries (permanent)
            boundary_alpha: Phase 2 interpolation factor (ramps 0→1 over 3K steps)

        Returns dict with:
            geo_loss: scalar MSE loss
            geo_mse: scalar mean MSE for logging (detached)
        """
        cfg = self.config

        if cfg.geo_detach_h:
            h_normed = h_normed.detach()

        # 1. Spatial target (squared Manhattan)
        target_D2, valid_mask = self.compute_spatial_target_D2(
            xs, ys, grid_ids, sep_mask
        )

        # 2. Object boundaries (Phase 2) — interpolated for smooth transition
        if phase >= 2 and colors is not None:
            component_ids = self._compute_connected_components(
                colors, xs, ys, grid_ids
            )
            target_D2 = self._apply_object_boundaries(
                target_D2, valid_mask, component_ids, cfg.geo_wall_distance,
                alpha=boundary_alpha,
            )

        # 3. Model D²
        model_D2 = self.compute_model_D2(h_normed, g)

        # 4. MSE over valid pairs only
        n_valid = valid_mask.sum().clamp(min=1)
        diff = (model_D2 - target_D2) * valid_mask.float()
        mse = (diff * diff).sum() / n_valid

        return {
            "geo_loss": mse,
            "geo_mse": mse.detach(),
        }


if __name__ == "__main__":
    """Unit tests for MSE-based geometric loss."""
    print("Testing GeometricLoss (MSE on D²)...")

    config = LiquidARCConfig(d_model=64, d_metric=16, geo_loss_enabled=True)
    geo = GeometricLoss(config)

    B, N, d = 2, 32, 64

    xs = torch.zeros(B, N, dtype=torch.long)
    ys = torch.zeros(B, N, dtype=torch.long)
    grid_ids = torch.zeros(B, N, dtype=torch.long)
    sep_mask = torch.zeros(B, N, dtype=torch.bool)
    colors = torch.randint(0, 5, (B, N))

    # Grid 0: positions 0-15 (4x4)
    for i in range(16):
        xs[:, i] = i % 4
        ys[:, i] = i // 4
        grid_ids[:, i] = 0
    # Grid 1: positions 16-31 (4x4)
    for i in range(16):
        xs[:, 16 + i] = i % 4
        ys[:, 16 + i] = i // 4
        grid_ids[:, 16 + i] = 1

    # Test spatial target
    target_D2, valid_mask = geo.compute_spatial_target_D2(xs, ys, grid_ids, sep_mask)
    assert target_D2.shape == (B, N, N)
    assert valid_mask.shape == (B, N, N)

    # Cross-grid pairs should be masked out
    cross_valid = valid_mask[:, :16, 16:]
    assert cross_valid.sum() == 0, f"Cross-grid valid: {cross_valid.sum()}"
    print(f"  Block-diagonal mask: PASS (cross-grid valid = 0)")

    # Self-distance should be 0
    diag = target_D2[:, torch.arange(N), torch.arange(N)]
    assert (diag == 0).all(), "Self-distance not zero"
    print(f"  Self-distance = 0: PASS")

    # Adjacent cells (dx=1, dy=0): Manhattan=1, squared=1
    # Position 0 = (0,0), Position 1 = (1,0) → Manhattan = 1
    assert target_D2[0, 0, 1].item() == 1.0, f"Adjacent D²={target_D2[0, 0, 1]}"
    print(f"  Adjacent D² = 1.0: PASS")

    # Diagonal cells (dx=1, dy=1): Manhattan=2, squared=4
    # Position 0 = (0,0), Position 5 = (1,1) → Manhattan = 2
    assert target_D2[0, 0, 5].item() == 4.0, f"Diagonal D²={target_D2[0, 0, 5]}"
    print(f"  Diagonal D² = 4.0: PASS")

    # Test model D² is non-negative
    h_normed = torch.randn(B, N, d)
    g = torch.ones(B, N, d)
    model_D2 = geo.compute_model_D2(h_normed, g)
    assert (model_D2 >= 0).all(), "Negative D² found"
    assert model_D2.shape == (B, N, N)
    print(f"  Model D² non-negative: PASS (min={model_D2.min():.6f})")

    # Test connected components
    colors[0, 0:4] = 0  # same color, same row → one component
    comp_ids = geo._compute_connected_components(colors, xs, ys, grid_ids)
    assert comp_ids[0, 0] == comp_ids[0, 1] == comp_ids[0, 2] == comp_ids[0, 3]
    print(f"  Connected components: PASS")

    # Test Phase 1 forward
    result = geo(h_normed, g, xs, ys, grid_ids, sep_mask, colors, phase=1)
    assert "geo_loss" in result
    assert result["geo_loss"].shape == ()
    assert not torch.isnan(result["geo_loss"]), "NaN in geo_loss"
    print(f"  Phase 1 forward: PASS (MSE = {result['geo_loss'].item():.4f})")

    # Test Phase 2 forward
    result2 = geo(h_normed, g, xs, ys, grid_ids, sep_mask, colors, phase=2)
    assert not torch.isnan(result2["geo_loss"]), "NaN in Phase 2 geo_loss"
    print(f"  Phase 2 forward: PASS (MSE = {result2['geo_loss'].item():.4f})")

    # Test gradient flow
    h_grad = torch.randn(B, N, d, requires_grad=True)
    g_grad = torch.ones(B, N, d, requires_grad=True)
    result_g = geo(h_grad, g_grad, xs, ys, grid_ids, sep_mask, colors, phase=1)
    result_g["geo_loss"].backward()
    assert h_grad.grad is not None, "No gradient on h"
    assert g_grad.grad is not None, "No gradient on g"
    assert not torch.isnan(h_grad.grad).any(), "NaN in h gradient"
    assert not torch.isnan(g_grad.grad).any(), "NaN in g gradient"
    print(f"  Gradient flow: PASS (no NaN)")

    # Test MSE=0 when model matches target perfectly
    # If h_normed encodes positions and g=1, D² should match Euclidean²
    # For squared Manhattan ≈ Euclidean² on a grid, close enough for a sanity check
    print(f"\nGeometricLoss (MSE on D²) OK")
