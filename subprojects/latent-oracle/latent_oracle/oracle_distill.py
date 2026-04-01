"""Oracle distillation loss — MSE between LiquidARC similarity and Qwen oracle.

Teaches LiquidARC's learned heat kernel geometry to match how Qwen groups
grid positions, using precomputed pairwise cosine similarity matrices as
architecture-agnostic distillation targets.

The model's similarity is derived from its own squared geodesic distances:
  S_model = exp(-D²/(4t))
where D² comes from MetricNet and t is the learned diffusion timescale.

The oracle similarity is precomputed cosine similarity between Qwen hidden
states, averaged per grid cell.

Loss: MSE(S_model[cells, cells], S_oracle[cells, cells]) over same-grid pairs.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class OracleDistillLoss(nn.Module):
    """MSE between LiquidARC's learned similarity and Qwen's oracle similarity.

    Reuses the D² computation pattern from geo_loss.py but converts to
    similarity space for comparison with oracle cosine similarities.
    """

    def __init__(self):
        super().__init__()

    def compute_model_D2(
        self,
        h_normed: torch.Tensor,  # [B, N, d]
        g: torch.Tensor,         # [B, N, d] positive metric
    ) -> torch.Tensor:
        """Compute squared geodesic distance matrix (same as geo_loss.py).

        D²_ij = sum_d g_d * (h_i - h_j)²
              = ||k_i||² + ||k_j||² - 2 * k_i · k_j
        where k = h_normed * sqrt(g).

        Returns: [B, N, N] non-negative squared distances.
        """
        sqrt_g = torch.sqrt(g)
        k = h_normed * sqrt_g  # [B, N, d]

        k_norm_sq = (k * k).sum(dim=-1)  # [B, N]

        cross = torch.bmm(k, k.transpose(1, 2))  # [B, N, N]
        D2 = k_norm_sq.unsqueeze(2) + k_norm_sq.unsqueeze(1) - 2.0 * cross

        D2 = F.relu(D2)
        return D2

    def forward(
        self,
        h_normed: torch.Tensor,         # [B, N, d]
        g: torch.Tensor,                 # [B, N, d] positive metric
        t_diffusion: torch.Tensor,       # scalar — model's learned diffusion timescale
        oracle_sim: torch.Tensor,        # [B, N_cells, N_cells] oracle similarity targets
        cell_to_seq: torch.Tensor,       # [B, N_cells] indices mapping cell → sequence position
        valid_mask: torch.Tensor,        # [B, N_cells, N_cells] same-grid pair mask
    ) -> Dict[str, torch.Tensor]:
        """Compute MSE(model_similarity, oracle_similarity) over valid cell pairs.

        Args:
            h_normed: LayerNorm'd hidden states from model
            g: positive metric tensor from MetricNet
            t_diffusion: model's diffusion timescale (shared parameter)
            oracle_sim: precomputed cosine similarity from Qwen [B, N_cells, N_cells]
            cell_to_seq: maps oracle cell index to sequence position [B, N_cells]
            valid_mask: True for same-grid cell pairs [B, N_cells, N_cells]

        Returns dict with:
            distill_loss: scalar MSE loss (for gradient)
            distill_mse: same value, detached (for logging)
            model_sim_mean: mean model similarity over valid pairs (for diagnostics)
            oracle_sim_mean: mean oracle similarity over valid pairs (for diagnostics)
        """
        # 1. Full sequence D²
        D2 = self.compute_model_D2(h_normed, g)  # [B, N, N]

        # 2. Convert to similarity: S = exp(-D²/(4t))
        t = t_diffusion.clamp(min=1e-4)
        model_sim_full = torch.exp(-D2 / (4.0 * t))  # [B, N, N]

        # 3. Extract cell-only submatrix
        B, N_cells = cell_to_seq.shape

        # Clamp sentinel (-1) to 0 for safe gather; valid_mask excludes unmapped cells
        safe_idx = cell_to_seq.clamp(min=0)

        # Gather rows: model_sim_full[b, cell_to_seq[b, i], :] for each cell i
        # Then gather cols: result[b, :, cell_to_seq[b, j]] for each cell j
        idx_row = safe_idx.unsqueeze(2).expand(B, N_cells, D2.shape[2])  # [B, N_cells, N]
        row_selected = torch.gather(model_sim_full, 1, idx_row)  # [B, N_cells, N]

        idx_col = safe_idx.unsqueeze(1).expand(B, N_cells, N_cells)  # [B, N_cells, N_cells]
        model_cell_sim = torch.gather(row_selected, 2, idx_col)  # [B, N_cells, N_cells]

        # 4. MSE over valid (same-grid) pairs
        valid_f = valid_mask.float()
        diff = (model_cell_sim - oracle_sim) * valid_f
        n_valid = valid_f.sum().clamp(min=1)
        mse = (diff * diff).sum() / n_valid

        # Diagnostics
        with torch.no_grad():
            model_mean = (model_cell_sim * valid_f).sum() / n_valid
            oracle_mean = (oracle_sim * valid_f).sum() / n_valid

        return {
            "distill_loss": mse,
            "distill_mse": mse.detach(),
            "model_sim_mean": model_mean,
            "oracle_sim_mean": oracle_mean,
        }


def build_cell_to_seq_map(
    cell_coords: list,
    xs: torch.Tensor,        # [B, N]
    ys: torch.Tensor,        # [B, N]
    grid_ids: torch.Tensor,  # [B, N]
    sep_mask: torch.Tensor,  # [B, N]
) -> torch.Tensor:
    """Map oracle cell coordinates to sequence positions in the model's input.

    For each cell (row, col, grid_id) in the oracle's coordinate system, finds
    the corresponding position in the model's flattened sequence.

    Args:
        cell_coords: list of (row, col, grid_id) per cell (from similarity precompute)
        xs: [B, N] x coordinates in model sequence
        ys: [B, N] y coordinates in model sequence
        grid_ids: [B, N] grid IDs in model sequence
        sep_mask: [B, N] separator mask

    Returns:
        [B, N_cells] long tensor — sequence index for each cell.
        Returns 0 for any cell that can't be mapped (caller should mask these out).
    """
    B, N = xs.shape
    N_cells = len(cell_coords)
    device = xs.device

    cell_to_seq = torch.zeros(B, N_cells, dtype=torch.long, device=device)

    # Pre-convert to CPU for iteration
    xs_cpu = xs.cpu()
    ys_cpu = ys.cpu()
    gids_cpu = grid_ids.cpu()
    sep_cpu = sep_mask.cpu()

    for b in range(B):
        # Build position lookup: (x, y, grid_id) → seq_idx
        pos_map = {}
        for i in range(N):
            if sep_cpu[b, i]:
                continue
            gid = int(gids_cpu[b, i])
            if gid < 0:
                continue
            # In ARC sequences: xs = column, ys = row
            key = (int(ys_cpu[b, i]), int(xs_cpu[b, i]), gid)
            pos_map[key] = i

        for c_idx, (row, col, gid) in enumerate(cell_coords):
            key = (row, col, gid)
            # -1 sentinel for unmapped cells (caller must mask these out)
            seq_idx = pos_map.get(key, -1)
            cell_to_seq[b, c_idx] = seq_idx

    return cell_to_seq


def build_valid_mask(
    cell_coords: list,
) -> torch.Tensor:
    """Build same-grid valid pair mask for cell similarity comparison.

    Args:
        cell_coords: list of (row, col, grid_id) per cell
        n_cells: number of cells

    Returns:
        [N_cells, N_cells] bool tensor — True for same-grid pairs.
    """
    grid_ids = torch.tensor([gid for (_, _, gid) in cell_coords], dtype=torch.long)
    # Same-grid mask: grid_ids[i] == grid_ids[j]
    mask = grid_ids.unsqueeze(1) == grid_ids.unsqueeze(0)
    return mask


if __name__ == "__main__":
    """Unit tests for oracle distillation loss."""
    print("Testing OracleDistillLoss...")

    loss_fn = OracleDistillLoss()

    B, N, d = 2, 32, 64
    N_cells = 16

    # Model inputs
    h_normed = torch.randn(B, N, d)
    g = torch.ones(B, N, d).abs() + 0.1
    t_diffusion = torch.tensor(1.0)

    # Oracle targets (random cosine similarities)
    oracle_sim = torch.randn(B, N_cells, N_cells).sigmoid()
    oracle_sim = (oracle_sim + oracle_sim.transpose(1, 2)) / 2  # symmetric

    # Cell-to-sequence mapping (first N_cells positions)
    cell_to_seq = torch.arange(N_cells).unsqueeze(0).expand(B, -1)

    # Valid mask (two grids of 8 cells each)
    valid_mask = torch.zeros(B, N_cells, N_cells, dtype=torch.bool)
    valid_mask[:, :8, :8] = True
    valid_mask[:, 8:, 8:] = True

    # Forward
    result = loss_fn(h_normed, g, t_diffusion, oracle_sim,
                     cell_to_seq, valid_mask.float())

    assert "distill_loss" in result
    assert result["distill_loss"].shape == ()
    assert not torch.isnan(result["distill_loss"]), "NaN in distill_loss"
    print(f"  Forward: PASS (MSE = {result['distill_loss'].item():.4f})")
    print(f"  Model sim mean: {result['model_sim_mean'].item():.4f}")
    print(f"  Oracle sim mean: {result['oracle_sim_mean'].item():.4f}")

    # Gradient flow
    h_grad = torch.randn(B, N, d, requires_grad=True)
    g_grad = (torch.randn(B, N, d).abs() + 0.1).requires_grad_(True)
    t_grad = torch.tensor(1.0, requires_grad=True)
    result_g = loss_fn(h_grad, g_grad, t_grad, oracle_sim,
                       cell_to_seq, valid_mask.float())
    result_g["distill_loss"].backward()
    assert h_grad.grad is not None, "No gradient on h"
    assert g_grad.grad is not None, "No gradient on g"
    assert t_grad.grad is not None, "No gradient on t"
    assert not torch.isnan(h_grad.grad).any(), "NaN in h gradient"
    assert not torch.isnan(g_grad.grad).any(), "NaN in g gradient"
    print(f"  Gradient flow: PASS (h, g, t all receive gradients)")

    # Test D² computation
    D2 = loss_fn.compute_model_D2(h_normed, g)
    assert (D2 >= 0).all(), "Negative D² found"
    assert D2.shape == (B, N, N)
    print(f"  D² non-negative: PASS (min={D2.min():.6f})")

    # Test build_cell_to_seq_map
    print("\nTesting build_cell_to_seq_map...")
    xs = torch.zeros(B, N, dtype=torch.long)
    ys = torch.zeros(B, N, dtype=torch.long)
    grid_ids = torch.zeros(B, N, dtype=torch.long)
    sep_mask = torch.zeros(B, N, dtype=torch.bool)

    # Set up a 4x4 grid at grid_id=0
    for i in range(16):
        xs[:, i] = i % 4
        ys[:, i] = i // 4
        grid_ids[:, i] = 0

    cell_coords = [(r, c, 0) for r in range(4) for c in range(4)]
    mapping = build_cell_to_seq_map(cell_coords, xs, ys, grid_ids, sep_mask)
    assert mapping.shape == (B, 16)
    # Cell (0,0,0) should map to position 0, (1,0,0) to position 4, etc.
    assert mapping[0, 0].item() == 0, f"Cell (0,0) → pos {mapping[0,0].item()}"
    assert mapping[0, 1].item() == 1, f"Cell (0,1) → pos {mapping[0,1].item()}"
    print(f"  Mapping: PASS")

    # Test unmapped sentinel
    unmapped_coords = [(r, c, 99) for r in range(4) for c in range(4)]  # grid_id=99 not in sequence
    unmapped_map = build_cell_to_seq_map(unmapped_coords, xs, ys, grid_ids, sep_mask)
    assert (unmapped_map == -1).all(), "Unmapped cells should be -1"
    print(f"  Unmapped sentinel: PASS")

    # Test build_valid_mask
    print("\nTesting build_valid_mask...")
    coords_2grid = [(r, c, 0) for r in range(2) for c in range(2)] + \
                   [(r, c, 1) for r in range(2) for c in range(2)]
    mask = build_valid_mask(coords_2grid)
    assert mask[:4, :4].all(), "Same-grid pairs not masked"
    assert not mask[:4, 4:].any(), "Cross-grid pairs should be unmasked"
    print(f"  Valid mask: PASS")

    print("\nOracleDistillLoss OK")
