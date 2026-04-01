"""GeoOnlyLayer — pure geometric routing layer for v7 sandwich architecture.

No attention pathway. Metric + GeoRoute + FFN only.
Used for bottom and top layers in the sandwich.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import FGNConfig
from .metric import MetricNetwork
from .curvature import CurvatureEngine
from .geo_route import GeoRoute


class GeoOnlyLayer(nn.Module):
    def __init__(self, config: FGNConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_learned_metric = (config.geo_metric_type == "learned")

        # Pre-norm layers
        self.norm_geo = nn.LayerNorm(config.d_model)
        self.norm_ff = nn.LayerNorm(config.d_model)

        # Metric (optional — flat mode uses g=1)
        if self.use_learned_metric:
            self.norm_metric = nn.LayerNorm(config.d_model)
            self.metric = MetricNetwork(config)
            self.curvature = CurvatureEngine()

        # Geometric routing (v6/v7 Q/K path)
        self.geo_route = GeoRoute(config)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, h: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass — pure geometric routing + FFN.

        Returns:
            (h, kappa [B, N], metric_cv scalar, avg_entropy scalar)
        """
        B, N, _ = h.shape

        # Step 1: Compute metric
        if self.use_learned_metric:
            g = self.metric(self.norm_metric(h), context=context)
            kappa = self.curvature(g)
            metric_cv = g.std() / g.mean()
        else:
            g = torch.ones(B, N, h.shape[2], device=h.device, dtype=h.dtype)
            kappa = torch.zeros(B, N, device=h.device, dtype=h.dtype)
            metric_cv = torch.tensor(0.0, device=h.device, dtype=h.dtype)

        # Step 2: Geometric routing (full residual)
        h_geo, geo_weights = self.geo_route(
            self.norm_geo(h), g, mask=mask, return_weights=True)
        h = h + self.resid_drop(h_geo)

        # Step 3: Entropy for diagnostics
        eps = 1e-8
        entropy = -(geo_weights * torch.log(geo_weights + eps)).sum(dim=-1)
        positions = torch.arange(1, N + 1, device=h.device, dtype=h.dtype)
        max_entropy = torch.log(positions + eps).unsqueeze(0)
        normalized_entropy = entropy / (max_entropy + eps)
        avg_entropy = normalized_entropy.mean()

        # Step 4: FFN
        h = h + self.resid_drop(self.ffn(self.norm_ff(h)))

        return h, kappa, metric_cv, avg_entropy


if __name__ == "__main__":
    print("=== GeoOnlyLayer Self-Test ===\n")

    # Test 1: Basic forward pass
    print("--- Test 1: Basic forward pass ---")
    cfg = FGNConfig(d_model=64, n_heads=4, d_ff=256, geo_heads=4,
                    architecture_version="v7", geo_metric_type="learned")
    layer = GeoOnlyLayer(cfg, layer_idx=0)

    B, N = 2, 16
    h = torch.randn(B, N, 64)
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)

    out, kappa, m_cv, avg_ent = layer(h, mask=mask)
    assert out.shape == (B, N, 64), f"Output shape: {out.shape}"
    assert kappa.shape == (B, N), f"Kappa shape: {kappa.shape}"
    print(f"metric_cv={m_cv.item():.4f}, |kappa|={kappa.abs().mean().item():.4f}, entropy={avg_ent.item():.4f}")
    print("PASS\n")

    # Test 2: No attention components
    print("--- Test 2: No attention components ---")
    assert not hasattr(layer, 'attention'), "Should not have attention"
    assert not hasattr(layer, 'norm_attn'), "Should not have norm_attn"
    print("PASS\n")

    # Test 3: Context conditioning
    print("--- Test 3: Context conditioning ---")
    context = torch.randn(B, 64)
    out_ctx, _, _, _ = layer(h, mask=mask, context=context)
    assert out_ctx.shape == (B, N, 64)
    print("PASS\n")

    # Test 4: Gradient flow
    print("--- Test 4: Gradient flow ---")
    layer_g = GeoOnlyLayer(cfg, layer_idx=0)
    out_g, kappa_g, _, _ = layer_g(h, mask=mask)
    loss = out_g.sum() + kappa_g.sum()
    loss.backward()
    for name, p in layer_g.metric.named_parameters():
        assert p.grad is not None, f"No grad for metric.{name}"
        assert p.grad.abs().sum() > 0, f"Zero grad for metric.{name}"
    for name, p in layer_g.geo_route.named_parameters():
        assert p.grad is not None, f"No grad for geo_route.{name}"
    for name, p in layer_g.ffn.named_parameters():
        assert p.grad is not None, f"No grad for ffn.{name}"
    print("PASS\n")

    # Test 5: Flat metric mode
    print("--- Test 5: Flat metric mode ---")
    cfg_flat = FGNConfig(d_model=64, n_heads=4, d_ff=256, geo_heads=4,
                         architecture_version="v7", geo_metric_type="flat")
    layer_flat = GeoOnlyLayer(cfg_flat, layer_idx=0)
    _, kappa_f, m_cv_f, _ = layer_flat(h, mask=mask)
    assert kappa_f.abs().sum().item() == 0.0, "Flat should have zero curvature"
    assert m_cv_f.item() == 0.0, "Flat should have zero CV"
    print("PASS\n")

    n_params = sum(p.numel() for p in layer.parameters())
    print(f"Parameters: {n_params:,}")
    print("=== All GeoOnlyLayer tests PASSED ===")
