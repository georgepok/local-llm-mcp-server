"""FGNv5Layer — hierarchical escalation layer.

Architecture:
  g = MetricNetwork(LayerNorm(h))
  h_geo, geo_weights = GeoRoute(LayerNorm(h), g)     ← always runs, full residual
  h = h + h_geo
  entropy = -sum(w * log(w)) per query position       ← computed from geo_weights
  escalate_weights = sigmoid((entropy - threshold) * sharpness)
  h_attn = Attention(LayerNorm(h))                     ← runs on all tokens
  h = h + escalate_weights * h_attn                    ← soft-gated additive correction
  h = h + FFN(LayerNorm(h))

No blend gate. GeoRoute is always the foundation. Attention provides
targeted corrections where GeoRoute's entropy indicates uncertainty.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig
from .metric import MetricNetwork
from .curvature import CurvatureEngine
from .geo_route import GeoRoute
from .standard_attention import StandardAttention


class FGNv5Layer(nn.Module):
    def __init__(self, config: FGNConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_learned_metric = (config.geo_metric_type == "learned")
        self.escalation_mode = config.escalation_mode

        # Pre-norm layers
        self.norm_geo = nn.LayerNorm(config.d_model)
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.norm_ff = nn.LayerNorm(config.d_model)

        # Metric (optional — flat mode uses g=1)
        if self.use_learned_metric:
            self.norm_metric = nn.LayerNorm(config.d_model)
            self.metric = MetricNetwork(config)
            self.curvature = CurvatureEngine()

        # Geometric routing (always runs)
        self.geo_route = GeoRoute(config)

        # Standard attention (fires on escalated tokens via soft gating)
        self.attention = StandardAttention(config)

        # Escalation threshold
        if self.escalation_mode == "soft":
            # Learned threshold via sigmoid(threshold_raw), init at 0.7
            # sigmoid(0.847) ≈ 0.7
            self.threshold_raw = nn.Parameter(torch.tensor(0.847))
            # Sharpness is set externally by training loop during annealing
            self.sharpness = config.escalation_sharpness_init
        else:
            # Fixed threshold, no learned parameter
            self.fixed_threshold = config.escalation_threshold

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
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            h: [B, N, d_model]
            mask: [N, N] causal mask

        Returns:
            (h, kappa [B, N], metric_cv scalar, avg_entropy scalar,
             escalation_rate scalar)
        """
        B, N, _ = h.shape

        # Step 1: Compute metric (or flat g=1)
        if self.use_learned_metric:
            g = self.metric(self.norm_metric(h))
            kappa = self.curvature(g)
            metric_cv = g.std() / g.mean()
        else:
            g = torch.ones(B, N, h.shape[2], device=h.device, dtype=h.dtype)
            kappa = torch.zeros(B, N, device=h.device, dtype=h.dtype)
            metric_cv = torch.tensor(0.0, device=h.device, dtype=h.dtype)

        # Step 2: Geometric routing (ALWAYS runs, full residual)
        h_geo, geo_weights = self.geo_route(
            self.norm_geo(h), g, mask=mask, return_weights=True)
        h = h + self.resid_drop(h_geo)

        # Step 3: Compute entropy from GeoRoute weights
        eps = 1e-8
        # geo_weights: [B, N, N]
        entropy = -(geo_weights * torch.log(geo_weights + eps)).sum(dim=-1)  # [B, N]

        # Normalize by max possible entropy: log(i+1) for causal position i
        positions = torch.arange(1, N + 1, device=h.device, dtype=h.dtype)
        max_entropy = torch.log(positions + eps).unsqueeze(0)  # [1, N]
        normalized_entropy = entropy / (max_entropy + eps)  # [B, N] in ~[0, 1]

        # Step 4: Compute escalation weights (soft gating)
        if self.escalation_mode == "soft":
            threshold = torch.sigmoid(self.threshold_raw)
            escalate_weights = torch.sigmoid(
                (normalized_entropy - threshold) * self.sharpness)  # [B, N]
        else:
            threshold = self.fixed_threshold
            escalate_weights = (normalized_entropy > threshold).float()  # [B, N]

        # Step 5: Attention (dense_masked — compute for all, scale by escalation)
        h_attn = self.attention(self.norm_attn(h), mask=mask)  # [B, N, d]
        h = h + escalate_weights.unsqueeze(-1) * self.resid_drop(h_attn)

        # Step 6: FFN
        h = h + self.resid_drop(self.ffn(self.norm_ff(h)))

        # Stats
        avg_entropy = normalized_entropy.mean()
        escalation_rate = escalate_weights.mean()

        return h, kappa, metric_cv, avg_entropy, escalation_rate


if __name__ == "__main__":
    for metric_type in ["learned", "flat"]:
        print(f"\n--- geo_metric_type={metric_type} ---")
        cfg = FGNConfig(d_model=64, n_heads=4, d_ff=256, geo_heads=1,
                        architecture_version="v5", geo_metric_type=metric_type,
                        escalation_mode="soft")
        layer = FGNv5Layer(cfg, layer_idx=0)

        B, N = 2, 16
        h = torch.randn(B, N, 64)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)

        out, kappa, m_cv, avg_ent, esc_rate = layer(h, mask=mask)
        assert out.shape == (B, N, 64)
        assert kappa.shape == (B, N)

        print(f"metric_cv={m_cv.item():.4f}")
        print(f"|kappa|={kappa.abs().mean().item():.4f}")
        print(f"avg_entropy={avg_ent.item():.4f}")
        print(f"escalation_rate={esc_rate.item():.4f}")

        if hasattr(layer, 'threshold_raw'):
            print(f"threshold={torch.sigmoid(layer.threshold_raw).item():.4f}")

        loss = out.sum()
        loss.backward()
        if hasattr(layer, 'metric'):
            for name, p in layer.metric.named_parameters():
                assert p.grad is not None, f"No grad for metric.{name}"
        if hasattr(layer, 'threshold_raw'):
            assert layer.threshold_raw.grad is not None, "No grad for threshold"

        n_params = sum(p.numel() for p in layer.parameters())
        print(f"Parameters: {n_params:,}")
        print(f"FGNv5Layer ({metric_type}) OK")

    # Test fixed mode
    print("\n--- escalation_mode=fixed ---")
    cfg_fixed = FGNConfig(d_model=64, n_heads=4, d_ff=256, geo_heads=1,
                          architecture_version="v5", geo_metric_type="learned",
                          escalation_mode="fixed", escalation_threshold=0.5)
    layer_fixed = FGNv5Layer(cfg_fixed, layer_idx=0)
    out_f, _, _, ent_f, esc_f = layer_fixed(h, mask=mask)
    assert out_f.shape == (B, N, 64)
    print(f"Fixed threshold: esc_rate={esc_f.item():.4f}")
    print("FGNv5Layer (fixed) OK")
