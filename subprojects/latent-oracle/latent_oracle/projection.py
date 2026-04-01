"""Oracle Projection Head — maps oracle embeddings to LiquidARC's geometry.

Two outputs:
  1. z_context [B, d_model]: injected into MetricNet via set_context()
  2. kappa_target [B, 1]: scalar curvature supervision signal (Softplus ≥ 0)

Design: kappa_target is the clean scalar signal — "how curved should the manifold
be for this task?" Tau is derived from dynamics (not independently controllable),
and g_target per-position requires knowing N (varies per task). Oracle knowledge
enters the metric indirectly via z_context → MetricNet takes [h_normed || context].
"""

import math

import torch
import torch.nn as nn


class OracleProjectionHead(nn.Module):
    """Projects oracle embeddings to context vector + curvature target.

    Args:
        oracle_dim: dimension of oracle embeddings (e.g. 4096 for Qwen3.5-9B)
        d_model: LiquidARC hidden dimension (e.g. 768 for 5M model)
        d_hidden: bottleneck dimension for both projections
    """

    def __init__(self, oracle_dim: int = 4096, d_model: int = 768, d_hidden: int = 1024):
        super().__init__()

        # Context projection: oracle_dim → d_hidden → d_model
        self.context_proj = nn.Sequential(
            nn.Linear(oracle_dim, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_model),
            nn.LayerNorm(d_model),
        )

        # Curvature target head: oracle_dim → d_hidden/4 → 1
        self.kappa_head = nn.Sequential(
            nn.Linear(oracle_dim, d_hidden // 4),
            nn.GELU(),
            nn.Linear(d_hidden // 4, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize so kappa_head output ≈ 0.05 (typical |κ| in trained 5M model)."""
        with torch.no_grad():
            # Softplus(x) ≈ x for large x, Softplus(x) → 0 for x << 0
            # We want Softplus(bias) ≈ 0.05 → bias = log(e^0.05 - 1) ≈ -2.97
            kappa_linear = self.kappa_head[2]  # the Linear before Softplus
            kappa_linear.bias.fill_(math.log(math.exp(0.05) - 1))

    def forward(self, oracle_emb: torch.Tensor):
        """Project oracle embeddings.

        Args:
            oracle_emb: [B, oracle_dim] oracle embeddings (bf16/fp32)

        Returns:
            z_context: [B, d_model] context vector for dynamics.set_context()
            kappa_target: [B, 1] non-negative curvature target
        """
        z_context = self.context_proj(oracle_emb)       # [B, d_model]
        kappa_target = self.kappa_head(oracle_emb)       # [B, 1]
        return z_context, kappa_target


if __name__ == "__main__":
    print("Testing OracleProjectionHead...")

    head = OracleProjectionHead(oracle_dim=4096, d_model=768, d_hidden=1024)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"  Parameters: {n_params:,}")

    # Forward pass
    B = 4
    oracle_emb = torch.randn(B, 4096)
    z_ctx, kappa_t = head(oracle_emb)
    assert z_ctx.shape == (B, 768), f"z_context shape: {z_ctx.shape}"
    assert kappa_t.shape == (B, 1), f"kappa_target shape: {kappa_t.shape}"
    assert (kappa_t >= 0).all(), "kappa_target must be non-negative (Softplus)"

    print(f"  z_context: {z_ctx.shape}, norm={z_ctx.norm(dim=-1).mean():.4f}")
    print(f"  kappa_target: {kappa_t.shape}, mean={kappa_t.mean():.4f}")

    # Gradient flow
    oracle_emb_g = torch.randn(B, 4096, requires_grad=True)
    z, k = head(oracle_emb_g)
    (z.sum() + k.sum()).backward()
    assert oracle_emb_g.grad is not None
    print(f"  Gradient flows: OK")

    print("OracleProjectionHead OK")
