"""FGNv4Layer — dual-pathway transformer layer with geometric routing + standard attention.

Architecture (pre-norm):
  g = MetricNetwork(LayerNorm(h))
  h_geo = GeoRoute(LayerNorm(h), g)              ← geometric routing (structural)
  h = h + gate * h_geo
  h_attn = StandardAttention(LayerNorm(h))        ← content routing (semantic)
  h = h + (1 - gate) * h_attn
  h = h + FFN(LayerNorm(h))

The gate is a per-layer learned scalar controlling the balance between pathways.
"""

from typing import Optional

import torch
import torch.nn as nn

from .config import FGNConfig
from .metric import MetricNetwork
from .curvature import CurvatureEngine
from .geo_route import GeoRoute
from .standard_attention import StandardAttention


class FGNv4Layer(nn.Module):
    def __init__(self, config: FGNConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_learned_metric = (config.geo_metric_type == "learned")

        # Pre-norm layers
        self.norm_geo = nn.LayerNorm(config.d_model)
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.norm_ff = nn.LayerNorm(config.d_model)

        # Core geometric components
        if self.use_learned_metric:
            self.norm_metric = nn.LayerNorm(config.d_model)
            self.metric = MetricNetwork(config)
            self.curvature = CurvatureEngine()
        self.geo_route = GeoRoute(config)

        # Standard attention (content routing)
        self.attention = StandardAttention(config)

        # Geometric gate: sigmoid(gate_geo_raw) controls balance
        # Init to config.gate_init (default 3.0 → sigmoid ≈ 0.95, geo-dominant)
        self.gate_geo_raw = nn.Parameter(torch.tensor(config.gate_init))

        # FFN with dropout
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.resid_drop = nn.Dropout(config.dropout)

        # Diagnostics (not read in compiled path)
        self._last_curvature: Optional[torch.Tensor] = None
        self._last_metric: Optional[torch.Tensor] = None

    def forward(self, h: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            h: [B, N, d_model]
            mask: [N, N] causal mask

        Returns:
            (h, kappa [B, N], metric_cv scalar, gate_value scalar)
        """
        # Compute metric (or use flat g=1)
        if self.use_learned_metric:
            g = self.metric(self.norm_metric(h))
            kappa = self.curvature(g)
            metric_cv = g.std() / g.mean()
            self._last_curvature = kappa
            self._last_metric = g
        else:
            g = torch.ones(h.shape[0], h.shape[1], h.shape[2],
                           device=h.device, dtype=h.dtype)
            kappa = torch.zeros(h.shape[0], h.shape[1],
                                device=h.device, dtype=h.dtype)
            metric_cv = torch.tensor(0.0, device=h.device, dtype=h.dtype)

        # Gate value
        gate = torch.sigmoid(self.gate_geo_raw)

        # Step 1: Geometric routing
        h_normed_geo = self.norm_geo(h)
        h_geo, _ = self.geo_route(h_normed_geo, g, mask=mask)

        # Gated geometric residual
        h = h + gate * self.resid_drop(h_geo)

        # Step 2: Standard attention on geo-processed representations
        h_attn = self.attention(self.norm_attn(h), mask=mask)
        h = h + (1.0 - gate) * self.resid_drop(h_attn)

        # FFN
        h = h + self.resid_drop(self.ffn(self.norm_ff(h)))

        return h, kappa, metric_cv, gate

    @property
    def last_curvature(self) -> Optional[torch.Tensor]:
        return self._last_curvature

    @property
    def last_metric(self) -> Optional[torch.Tensor]:
        return self._last_metric


if __name__ == "__main__":
    for metric_type in ["learned", "flat"]:
        print(f"\n--- geo_metric_type={metric_type} ---")
        cfg = FGNConfig(d_model=64, n_heads=4, d_ff=256, geo_heads=1,
                        architecture_version="v4", gate_init=3.0,
                        geo_metric_type=metric_type)
        layer = FGNv4Layer(cfg, layer_idx=0)

        B, N = 2, 16
        h = torch.randn(B, N, 64)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)

        out, kappa, m_cv, gate_val = layer(h, mask=mask)
        assert out.shape == (B, N, 64)
        assert kappa.shape == (B, N)

        print(f"gate={gate_val.item():.4f} (raw={layer.gate_geo_raw.item():.2f})")
        print(f"metric_cv={m_cv.item():.4f}")
        print(f"|kappa|={kappa.abs().mean().item():.4f}")

        loss = out.sum()
        loss.backward()
        if hasattr(layer, 'metric'):
            for name, p in layer.metric.named_parameters():
                assert p.grad is not None, f"No grad for metric.{name}"
        assert layer.gate_geo_raw.grad is not None, "No grad for gate"

        n_params = sum(p.numel() for p in layer.parameters())
        print(f"Parameters: {n_params:,}")
        print(f"FGNv4Layer ({metric_type}) OK")
