"""Parallel transport and holonomy detection.

Phase 1a: Identity transport (returns vectors unchanged).
Phase 2: Low-rank antisymmetric transport with subspace exponential.

NOTE (Phase 2 interface change): The current forward() takes (v, g_i, g_j)
with shapes [B, N, d], transporting all N positions from a single source
metric to a single target metric. Phase 2 attention needs PAIRWISE transport:
for each query position i, transport value vectors from every key position j.
This requires a batched interface: (v, g_target, g_source) where g_target and
g_source are [B, N_q, N_k, d] (or computed on-the-fly from [B, N, d] metrics
for each pair). Plan the API change before Phase 2 implementation.
"""

import torch
import torch.nn as nn

from .config import FGNConfig


class ParallelTransport(nn.Module):
    """Low-rank antisymmetric transport operator.

    Phase 1a: Returns input unchanged (identity transport).
    Phase 2: Full implementation with subspace trick.

    U: learned d x r basis for rotation subspace
    S_ij = 0.5 * (M - M^T) where M = W_S * (g_i - g_j)
    Pi = exp(U @ S @ U^T)

    Subspace trick: since U*S*U^T has rank <= 2r, compute exp in the
    r-dimensional subspace: Pi = I + U*(exp(S)-I)*U^T
    """

    def __init__(self, config: FGNConfig):
        super().__init__()
        self.enabled = False  # Phase 1a: disabled
        self.d_model = config.d_model
        self.rank = config.transport_rank

        # Phase 2 parameters (allocated but unused in Phase 1a)
        self.U = nn.Parameter(torch.empty(config.d_model, config.transport_rank))
        nn.init.orthogonal_(self.U)

        self.W_S = nn.Linear(config.d_model, config.transport_rank * config.transport_rank)

    def forward(self, v: torch.Tensor, g_i: torch.Tensor,
                g_j: torch.Tensor) -> torch.Tensor:
        """Transport vector v from position j to position i.

        Args:
            v: [B, N, d] vectors to transport
            g_i: [B, N, d] metric at target
            g_j: [B, N, d] metric at source

        Returns:
            [B, N, d] transported vectors
        """
        if not self.enabled:
            return v

        # Phase 2 implementation
        return self._transport_phase2(v, g_i, g_j)

    def _transport_phase2(self, v: torch.Tensor, g_i: torch.Tensor,
                          g_j: torch.Tensor) -> torch.Tensor:
        """Full transport computation (Phase 2)."""
        B, N, d = v.shape
        r = self.rank

        # Generate antisymmetric matrix from metric difference
        diff = g_i - g_j  # [B, N, d]
        M_raw = self.W_S(diff).view(B, N, r, r)  # [B, N, r, r]
        S = 0.5 * (M_raw - M_raw.transpose(-2, -1))  # Antisymmetric

        # Subspace exponential: Pi = I + U*(exp(S)-I)*U^T
        transported = self._apply_subspace_exp(v, S)
        return transported

    def _apply_subspace_exp(self, v: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Apply exp(U @ S @ U^T) to v via subspace trick.

        Cost: O(r^3 + d*r^2) instead of O(d^3).
        """
        B, N, d = v.shape

        # Project v into subspace: v_sub = U^T @ v
        U = self.U  # [d, r]
        v_sub = torch.einsum("dr,bnd->bnr", U, v)  # [B, N, r]

        # Compute exp(S) in r x r subspace
        exp_S = torch.matrix_exp(S)  # [B, N, r, r]

        # Apply in subspace: (exp(S) - I) @ v_sub
        I_r = torch.eye(self.rank, device=S.device, dtype=S.dtype)
        delta = exp_S - I_r
        v_rotated = torch.einsum("bnrs,bns->bnr", delta, v_sub)

        # Project back: v + U @ v_rotated
        v_out = v + torch.einsum("dr,bnr->bnd", U, v_rotated)
        return v_out


class HolonomyDetector(nn.Module):
    """Ordered product holonomy measurement.

    H_ijk = Pi_ki * Pi_jk * Pi_ij
    Phase 1a: Must return EXACTLY zero (correctness check).
    Phase 2: Measures ||H - I||_F to establish epsilon noise floor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, transport: ParallelTransport, g: torch.Tensor,
                triplets: torch.Tensor) -> torch.Tensor:
        """Compute holonomy for given triplets.

        Args:
            transport: ParallelTransport module
            g: [B, N, d] metric tensor
            triplets: [M, 3] indices (i, j, k) into sequence

        Returns:
            [M] holonomy norms ||H - I||_F per triplet
        """
        M = triplets.shape[0]
        B, N, d = g.shape

        # Use identity matrix as test vector set
        v = torch.eye(d, device=g.device, dtype=g.dtype).unsqueeze(0).expand(B, -1, -1)
        # v: [B, d, d]

        results = []
        for m in range(M):
            i, j, k = triplets[m]

            g_i = g[:, i:i+1].expand(-1, d, -1)  # [B, d, d]
            g_j = g[:, j:j+1].expand(-1, d, -1)
            g_k = g[:, k:k+1].expand(-1, d, -1)

            # Pi_ij: transport from j to i
            v1 = transport(v, g_i, g_j)
            # Pi_jk: transport from k to j
            v2 = transport(v1, g_j, g_k)
            # Pi_ki: transport from i to k
            v3 = transport(v2, g_k, g_i)

            # Holonomy = v3 - v (should be identity for closed loop)
            h_norm = (v3 - v).norm(dim=(-2, -1)).mean()  # Average over batch
            results.append(h_norm)

        return torch.stack(results)


if __name__ == "__main__":
    cfg = FGNConfig(d_model=64, n_heads=4, transport_rank=8)
    transport = ParallelTransport(cfg)
    detector = HolonomyDetector()

    B, N, d = 2, 16, 64
    g = torch.ones(B, N, d)  # Identity metric
    v = torch.randn(B, N, d)

    # Phase 1a: identity transport
    v_out = transport(v, g, g)
    assert torch.allclose(v_out, v), "Phase 1a transport must be identity"

    # Phase 1a holonomy must be exactly zero
    triplets = torch.tensor([[0, 4, 8], [1, 5, 9]])
    h_norms = detector(transport, g, triplets)
    assert torch.allclose(h_norms, torch.zeros_like(h_norms)), \
        f"Phase 1a holonomy must be 0, got {h_norms}"

    print("ParallelTransport (Phase 1a) OK")
    print("HolonomyDetector OK")
