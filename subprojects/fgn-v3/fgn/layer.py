"""FGNTransformerLayer — single transformer layer with Riemannian attention.

Architecture (pre-norm):
  g = MetricNetwork(LayerNorm(h))
  h = h + HeatKernelAttention(LayerNorm(h), g)
  h = h + FFN(LayerNorm(h))
"""

from typing import Optional

import torch
import torch.nn as nn

from .config import FGNConfig
from .metric import MetricNetwork
from .curvature import CurvatureEngine
from .attention import HeatKernelAttention


class FGNTransformerLayer(nn.Module):
    def __init__(self, config: FGNConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-norm layers
        self.norm_metric = nn.LayerNorm(config.d_model)
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.norm_ff = nn.LayerNorm(config.d_model)

        # Core components
        self.metric = MetricNetwork(config)
        self.curvature = CurvatureEngine()
        self.attention = HeatKernelAttention(config)

        # FFN with dropout
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.resid_drop = nn.Dropout(config.dropout)

        # Cached curvature for regularization
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
            (h [B, N, d_model], kappa [B, N], metric_cv scalar, scale_entropy scalar)
        """
        # Compute metric from pre-normed hidden states
        h_normed = self.norm_metric(h)
        g = self.metric(h_normed)  # [B, N, d] — shared across heads

        # Curvature
        kappa = self.curvature(g)

        # Metric CV (coefficient of variation) — cheap scalar stat
        metric_cv = g.std() / g.mean()

        # Cache for diagnostic scripts (not read in compiled forward path)
        self._last_curvature = kappa
        self._last_metric = g

        # Attention with geometric metric + residual dropout
        attn_out, scale_entropy = self.attention(self.norm_attn(h), g, mask=mask)
        h = h + self.resid_drop(attn_out)

        # FFN + residual dropout
        h = h + self.resid_drop(self.ffn(self.norm_ff(h)))

        return h, kappa, metric_cv, scale_entropy

    @property
    def last_curvature(self) -> Optional[torch.Tensor]:
        return self._last_curvature

    @property
    def last_metric(self) -> Optional[torch.Tensor]:
        return self._last_metric


if __name__ == "__main__":
    cfg = FGNConfig(d_model=64, n_heads=4, d_ff=256)
    layer = FGNTransformerLayer(cfg, layer_idx=0)

    B, N = 2, 16
    h = torch.randn(B, N, 64)
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)

    out, kappa, m_cv, s_ent = layer(h, mask=mask)
    assert out.shape == (B, N, 64)
    assert kappa.shape == (B, N)
    assert m_cv.shape == ()
    assert s_ent.shape == ()
    assert layer.last_curvature is not None
    assert layer.last_metric is not None

    loss = out.sum() + s_ent
    loss.backward()
    for name, p in layer.metric.named_parameters():
        assert p.grad is not None, f"No grad for metric.{name}"
    print("FGNTransformerLayer OK")
