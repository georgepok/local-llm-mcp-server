"""MetricNetwork — maps hidden states to positive-definite diagonal metric tensors.

Shared across all attention heads in a layer. This prevents projection
absorption where heads learn to undo the metric.

Phase 1: g_i = Softplus(MLP(h_i)) -> R^d, all components > 0
Phase 2: g_i = diag(D_i) + L_i @ L_i^T (low-rank upgrade)

v6: Supports optional context conditioning from ContextPool episode vector.
When context is provided, uses additive feature combination before metric output.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig


class MetricNetwork(nn.Module):
    def __init__(self, config: FGNConfig):
        super().__init__()
        d = config.d_model
        bottleneck = d // 4

        # v3/v4/v5 path: standard MLP
        self.net = nn.Sequential(
            nn.Linear(d, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d),
        )

        # v6 path: context-conditioned metric
        self.h_proj = nn.Linear(d, bottleneck)
        self.context_proj = nn.Linear(d, bottleneck)
        self.out_linear = nn.Linear(bottleneck, d)

        # Initialize final bias so Softplus output ~= 1.0 at init (identity metric).
        # Softplus^{-1}(1.0) = log(e^1 - 1) ~= 1.3133
        with torch.no_grad():
            # v3/v4/v5 path
            self.net[-1].bias.fill_(math.log(math.e - 1))  # ~1.3133
            # Moderate init so metric can develop variation from identity.
            # At input=1.31: Softplus'(1.31)=sigma(1.31)~=0.79, so metric
            # variation ~ 0.79 * std * activation_scale. std=0.05 gives CV~0.04.
            nn.init.normal_(self.net[-1].weight, std=0.05)

            # v6 path: same initialization for identity metric at init
            self.out_linear.bias.fill_(math.log(math.e - 1))  # ~1.3133
            nn.init.normal_(self.out_linear.weight, std=0.05)

    def forward(self, h: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute metric tensor for each position.

        Args:
            h: Hidden states [B, N, d]
            context: Optional episode-level context vector [B, d] from ContextPool (v6)

        Returns:
            Metric tensor [B, N, d], all components > 0

        v3/v4/v5: When context is None, uses standard MLP path (backward compatible).
        v6: When context is provided, uses additive conditioning with context vector.
        """
        if context is None:
            # v3/v4/v5 path: unchanged behavior for backward compatibility
            return F.softplus(self.net(h))
        else:
            # v6 path: context-conditioned metric
            # context: [B, d] -> [B, bottleneck]
            # h: [B, N, d] -> [B, N, bottleneck]
            # combined: [B, N, bottleneck] via broadcasting
            combined = self.h_proj(h) + self.context_proj(context).unsqueeze(1)
            activated = F.gelu(combined)
            return F.softplus(self.out_linear(activated))


if __name__ == "__main__":
    cfg = FGNConfig(d_model=64, n_heads=4)
    net = MetricNetwork(cfg)
    h = torch.randn(2, 16, 64)

    # Test v3/v4/v5 path (no context)
    print("Testing v3/v4/v5 path (no context)...")
    g = net(h)
    assert g.shape == h.shape
    assert (g > 0).all(), "Metric must be positive"
    print(f"g mean={g.mean():.4f}, std={g.std():.4f}")
    assert abs(g.mean().item() - 1.0) < 0.3, "Init should be ~1.0"

    # Gradient flow test
    loss = g.sum()
    loss.backward()
    for name, p in net.named_parameters():
        if "context_proj" not in name and "out_linear" not in name and "h_proj" not in name:
            assert p.grad is not None and p.grad.abs().sum() > 0, f"No grad for {name}"
    print("v3/v4/v5 path OK")

    # Test v6 path (with context)
    print("\nTesting v6 path (with context)...")
    net.zero_grad()
    context = torch.randn(2, 64)
    g_ctx = net(h, context=context)
    assert g_ctx.shape == h.shape, f"Shape mismatch: {g_ctx.shape} vs {h.shape}"
    assert (g_ctx > 0).all(), "Context-conditioned metric must be positive"
    print(f"g_ctx mean={g_ctx.mean():.4f}, std={g_ctx.std():.4f}")
    assert abs(g_ctx.mean().item() - 1.0) < 0.3, "Init should be ~1.0 with context too"

    # Gradient flow test for v6 path
    loss_ctx = g_ctx.sum()
    loss_ctx.backward()
    for name, p in net.named_parameters():
        if "net.0" not in name and "net.2" not in name:  # Skip v3/v4/v5 path layers
            assert p.grad is not None and p.grad.abs().sum() > 0, f"No grad for {name} in v6 path"
    print("v6 path OK")

    print("\nMetricNetwork OK (both paths validated)")
