"""CurvatureEngine — differentiable 1D tensor trace curvature from metric field.

kappa_i = (1/d) sum_k (1/g_i^k^2) * (g_i^k * D2g_i^k - 0.5 * (Dg_i^k)^2)

Uses direct tensor slicing for finite differences with mirror padding.
Fully differentiable for regularization. No learnable parameters.
"""

import torch
import torch.nn as nn


class CurvatureEngine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """Compute curvature from metric field.

        Args:
            g: [B, N, d] positive metric components

        Returns:
            kappa: [B, N] scalar curvature per position
        """
        B, N, d = g.shape

        # Mirror pad: g[-1] = g[1], g[N] = g[N-2]
        g_padded = torch.cat([
            g[:, 1:2, :],   # left pad
            g,               # original
            g[:, -2:-1, :],  # right pad
        ], dim=1)  # [B, N+2, d]

        # Central first difference: dg[i] = g[i+1] - g[i-1]
        dg = g_padded[:, 2:, :] - g_padded[:, :-2, :]  # [B, N, d]

        # Second difference: d2g[i] = g[i+1] - 2*g[i] + g[i-1]
        d2g = g_padded[:, 2:, :] - 2.0 * g_padded[:, 1:-1, :] + g_padded[:, :-2, :]

        # Component-wise curvature
        eps = 1e-6
        g_sq = g * g + eps
        term = (g * d2g - 0.5 * dg * dg) / g_sq

        # Average over dimensions
        kappa = term.mean(dim=-1)  # [B, N]
        return kappa


if __name__ == "__main__":
    engine = CurvatureEngine()

    # Constant metric → ~0 curvature
    g_const = torch.ones(2, 16, 32)
    k = engine(g_const)
    assert k.shape == (2, 16)
    assert k.abs().max() < 1e-5

    # Varying metric → non-zero curvature
    g_vary = torch.ones(2, 16, 32)
    g_vary[:, 8, :] = 3.0
    k2 = engine(g_vary)
    assert k2[:, 8].abs().mean() > k2[:, 0].abs().mean()

    # Gradient flow
    g_vary.requires_grad_(True)
    k3 = engine(g_vary)
    k3.sum().backward()
    assert g_vary.grad is not None and g_vary.grad.abs().sum() > 0

    print("CurvatureEngine OK")
