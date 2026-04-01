"""StructuralEnergy v3 — triplet ranking loss for metric-graph alignment.

Replaces MSE (v2) with margin triplet loss that directly enforces rank
ordering between geodesic distances and graph distances. Uses semi-hard
negative mining and relative margins to focus gradients on informative
triplets and scale separation requirements appropriately.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class StructuralEnergy(nn.Module):
    """Triplet ranking energy between metric geodesic distances and world structure.

    For each anchor room, the nearest graph neighbor is the positive and
    the hardest valid negative (semi-hard mining) provides the gradient.
    Margin scales with the graph distance gap between positive and negative.

    Modes:
      - "graph": Triplet ranking on room_distances from the task.
      - "positional": Legacy MSE mode (unchanged from v2).
    """

    def __init__(self, max_context_pairs: int = 2048, mode: str = "graph",
                 margin_scale: float = 0.1,
                 d_model: int = 0, d_proj: int = 0, proj_mlp: bool = False):
        super().__init__()
        self.max_context_positions = int(math.isqrt(max_context_pairs))
        assert mode in ("graph", "positional"), f"Unknown mode: {mode}"
        self.mode = mode
        self.margin_scale = margin_scale

        # Optional projection head: maps h into space where Euclidean distance
        # can approximate graph distance (bypasses diagonal metric limitation)
        self.proj = None
        if d_proj > 0 and d_model > 0:
            if proj_mlp:
                # Nonlinear MLP: can extract nonlinear spatial features from h
                self.proj = nn.Sequential(
                    nn.Linear(d_model, d_model // 4, bias=False),
                    nn.GELU(),
                    nn.Linear(d_model // 4, d_proj, bias=False),
                )
            else:
                self.proj = nn.Linear(d_model, d_proj, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        g: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        room_distances: Optional[torch.Tensor] = None,
        room_token_positions: Optional[torch.Tensor] = None,
        n_rooms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute structural energy.

        Args:
            h: [B, N, d] hidden states
            g: [B, N, d] metric field (per-position, from Softplus)
            context_mask: [B, N] True for [WORLD] positions (positional mode)
            room_distances: [B, R_max, R_max] normalized graph distances (graph mode)
            room_token_positions: [B, R_max] token position per room, -1=pad (graph mode)
            n_rooms: [B] actual room count per episode (graph mode)

        Returns:
            energy: scalar tensor
        """
        if self.mode == "graph":
            return self._graph_energy(h, g, room_distances,
                                      room_token_positions, n_rooms)
        else:
            return self._positional_energy(h, g, context_mask)

    @torch.compiler.disable
    def _graph_energy(
        self,
        h: torch.Tensor,
        g: torch.Tensor,
        room_distances: Optional[torch.Tensor],
        room_token_positions: Optional[torch.Tensor],
        n_rooms: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Triplet ranking loss with semi-hard mining and relative margins.

        For each anchor room i:
          1. positive = nearest graph neighbor (smallest D_struct[i,j])
          2. negative = semi-hard mined: among rooms farther than positive
             in graph distance, find the one with smallest geodesic distance
             (hardest to distinguish from positive)
          3. margin = margin_scale * (D_struct[i,neg] - D_struct[i,pos])
          4. loss = max(0, D_geo[i,pos] - D_geo[i,neg] + margin)

        Produces O(V) triplets per episode, focused on the most informative
        boundary between adjacent and non-adjacent rooms.
        """
        B, N, _ = h.shape
        device = h.device

        if room_distances is None or room_token_positions is None or n_rooms is None:
            return torch.tensor(0.0, device=device, dtype=h.dtype)

        energies = []
        for b in range(B):
            R = n_rooms[b].item()
            if R < 3:
                continue

            positions = room_token_positions[b, :R]  # [R]
            valid_mask = (positions >= 0) & (positions < N)
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            V = valid_indices.shape[0]

            if V < 3:
                continue

            if V > self.max_context_positions:
                perm = torch.randperm(V, device=device)[:self.max_context_positions]
                valid_indices = valid_indices[perm]
                V = self.max_context_positions

            tok_pos = positions[valid_indices]  # [V]

            h_rooms = h[b, tok_pos]   # [V, d]

            if self.proj is not None:
                # Project into space where Euclidean ~ graph distance
                h_proj = self.proj(h_rooms)  # [V, d_proj]
                diff = h_proj.unsqueeze(1) - h_proj.unsqueeze(0)  # [V, V, d_proj]
                D_geo = (diff * diff).sum(-1)                      # [V, V]
            else:
                g_rooms = g[b, tok_pos]   # [V, d]
                diff = h_rooms.unsqueeze(1) - h_rooms.unsqueeze(0)          # [V, V, d]
                g_avg = (g_rooms.unsqueeze(1) + g_rooms.unsqueeze(0)) / 2   # [V, V, d]
                D_geo = (diff * diff * g_avg).sum(-1)                       # [V, V]

            # Normalize to [0,1] so margin_scale is interpretable
            D_geo_norm = D_geo / (D_geo.max() + 1e-8)

            # Graph distances for valid room pairs
            D_struct = room_distances[b][valid_indices][:, valid_indices]  # [V, V]

            # Semi-hard triplet mining per anchor
            triplet_losses = []
            for i in range(V):
                # Graph distances from anchor (mask self)
                d_struct_i = D_struct[i].clone()
                d_struct_i[i] = float('inf')

                # Positive: nearest graph neighbor
                pos_idx = d_struct_i.argmin()
                d_struct_pos = d_struct_i[pos_idx]
                d_geo_pos = D_geo_norm[i, pos_idx]

                # Valid negatives: farther than positive in graph distance
                neg_mask = d_struct_i > d_struct_pos
                neg_mask[i] = False
                if neg_mask.sum() == 0:
                    continue

                # Semi-hard mining: among valid negatives, pick the one
                # with smallest geodesic distance (hardest to separate)
                d_geo_neg_candidates = D_geo_norm[i].clone()
                d_geo_neg_candidates[~neg_mask] = float('inf')
                neg_idx = d_geo_neg_candidates.argmin()

                d_geo_neg = D_geo_norm[i, neg_idx]
                d_struct_neg = D_struct[i, neg_idx]

                # Relative margin: scales with graph distance gap
                margin = self.margin_scale * (d_struct_neg - d_struct_pos)

                # Triplet hinge loss
                loss_t = torch.clamp(d_geo_pos - d_geo_neg + margin, min=0.0)
                triplet_losses.append(loss_t)

            if len(triplet_losses) == 0:
                continue

            energies.append(torch.stack(triplet_losses).mean())

        if len(energies) == 0:
            return torch.tensor(0.0, device=device, dtype=h.dtype)

        return torch.stack(energies).mean()

    @torch.compiler.disable
    def _positional_energy(
        self,
        h: torch.Tensor,
        g: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Legacy positional-proxy structural energy (v0.1 behavior)."""
        B, _, _ = h.shape

        if context_mask is None:
            return torch.tensor(0.0, device=h.device, dtype=h.dtype)

        energies = []
        for b in range(B):
            mask_b = context_mask[b]
            ctx_indices = mask_b.nonzero(as_tuple=True)[0]
            C = ctx_indices.shape[0]

            if C < 2:
                continue

            if C > self.max_context_positions:
                perm = torch.randperm(C, device=h.device)[:self.max_context_positions]
                ctx_indices = ctx_indices[perm]
                C = self.max_context_positions

            h_ctx = h[b, ctx_indices]
            g_ctx = g[b, ctx_indices]

            diff = h_ctx.unsqueeze(1) - h_ctx.unsqueeze(0)
            g_avg = (g_ctx.unsqueeze(1) + g_ctx.unsqueeze(0)) / 2
            D_geo = (diff * diff * g_avg).sum(-1)

            D_geo_max = D_geo.max()
            D_geo_norm = D_geo / (D_geo_max + 1e-8)

            pos = ctx_indices.float()
            pos_diff = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
            ctx_len = pos.max() - pos.min() + 1
            D_struct = pos_diff / (ctx_len + 1e-8)

            energy_b = ((D_geo_norm - D_struct) ** 2).mean()
            energies.append(energy_b)

        if len(energies) == 0:
            return torch.tensor(0.0, device=h.device, dtype=h.dtype)

        return torch.stack(energies).mean()


if __name__ == "__main__":
    print("Testing StructuralEnergy v3 (semi-hard triplet)...")

    B, N, d = 2, 64, 64
    R = 10

    # --- Test graph mode (triplet) ---
    se_graph = StructuralEnergy(max_context_pairs=2048, mode="graph", margin_scale=0.1)

    h = torch.randn(B, N, d)
    g = torch.ones(B, N, d)

    room_distances = torch.rand(B, R, R)
    room_distances = (room_distances + room_distances.transpose(1, 2)) / 2
    for b in range(B):
        for i in range(R):
            room_distances[b, i, i] = 0.0

    room_token_positions = torch.arange(R).unsqueeze(0).expand(B, -1) * 5
    n_rooms = torch.full((B,), R, dtype=torch.long)

    energy = se_graph(h, g, room_distances=room_distances,
                      room_token_positions=room_token_positions, n_rooms=n_rooms)
    print(f"  Random h: energy={energy.item():.6f} (should be > 0)")
    assert energy.item() > 0.0

    # Gradient flow
    h_grad = torch.randn(B, N, d, requires_grad=True)
    g_grad = torch.ones(B, N, d, requires_grad=True)
    energy_grad = se_graph(h_grad, g_grad, room_distances=room_distances,
                           room_token_positions=room_token_positions, n_rooms=n_rooms)
    energy_grad.backward()
    assert h_grad.grad is not None and h_grad.grad.abs().sum() > 0, "No grad for h"
    assert g_grad.grad is not None and g_grad.grad.abs().sum() > 0, "No grad for g"
    print("  Gradient flow: OK")

    # Perfect alignment → low energy
    # Place rooms on a line proportional to their distance from room 0
    h_aligned = torch.zeros(B, N, d)
    for b in range(B):
        for i in range(R):
            pos = i * 5
            h_aligned[b, pos, 0] = room_distances[b, 0, i] * 10.0
    g_aligned = torch.ones(B, N, d)
    energy_aligned = se_graph(h_aligned, g_aligned, room_distances=room_distances,
                              room_token_positions=room_token_positions, n_rooms=n_rooms)
    print(f"  Aligned h: energy={energy_aligned.item():.6f} (should be < random)")
    assert energy_aligned.item() < energy.item(), "Aligned should have lower energy"

    # Missing data → zero energy
    energy_none = se_graph(h, g)
    assert energy_none.item() == 0.0
    print("  No data: OK")

    # Too few rooms → zero energy
    n_rooms_small = torch.full((B,), 2, dtype=torch.long)
    energy_small = se_graph(h, g, room_distances=room_distances,
                            room_token_positions=room_token_positions, n_rooms=n_rooms_small)
    assert energy_small.item() == 0.0
    print("  Too few rooms: OK")

    # --- Test projection mode ---
    print("\n  --- Projection mode (d_proj=16) ---")
    se_proj = StructuralEnergy(max_context_pairs=2048, mode="graph", margin_scale=0.1,
                                d_model=d, d_proj=16)
    assert se_proj.proj is not None, "Projection head should exist"

    energy_proj = se_proj(h, g, room_distances=room_distances,
                          room_token_positions=room_token_positions, n_rooms=n_rooms)
    print(f"  Random h (projected): energy={energy_proj.item():.6f} (should be > 0)")
    assert energy_proj.item() > 0.0

    # Gradient flow through projection
    h_grad2 = torch.randn(B, N, d, requires_grad=True)
    energy_proj2 = se_proj(h_grad2, None, room_distances=room_distances,
                           room_token_positions=room_token_positions, n_rooms=n_rooms)
    energy_proj2.backward()
    assert h_grad2.grad is not None and h_grad2.grad.abs().sum() > 0, "No grad through proj"
    # Verify projection weights get gradients
    assert se_proj.proj.weight.grad is not None, "No grad for proj weights"
    print("  Projection gradient flow: OK")

    # --- Test positional mode (legacy) ---
    se_pos = StructuralEnergy(max_context_pairs=2048, mode="positional")
    context_mask = torch.zeros(B, N, dtype=torch.bool)
    context_mask[:, :20] = True
    energy_pos = se_pos(h, g, context_mask=context_mask)
    print(f"  Positional mode: energy={energy_pos.item():.6f}")

    print("\nStructuralEnergy v3 OK")
