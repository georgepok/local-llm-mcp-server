"""FGNv6Layer — budget-based attention escalation layer.

Architecture:
  g = MetricNetwork(LayerNorm(h), context)    ← context-conditioned metric
  h_geo, geo_weights = GeoRoute(LayerNorm(h), g)  ← always runs
  h = h + h_geo
  entropy = -sum(w * log(w)) / log(position+1)   ← normalized entropy
  escalate_mask = top-k(entropy, budget * N)       ← HARD budget, not soft threshold
  h_attn = Attention(LayerNorm(h))
  h = h + escalate_mask * h_attn                   ← binary mask, not soft weight
  h = h + FFN(LayerNorm(h))

Key differences from v5:
  1. Budget-based escalation: Fixed budget fraction (0.0-1.0) determines exactly
     how many positions get attention. Top-k selection by entropy, binary mask.
  2. Context conditioning: MetricNetwork receives optional context parameter from
     ContextPool episode vector for cross-token metric adaptation.
  3. No threshold_raw, no sharpness, no escalation_mode — removed entirely.
  4. When budget=0.0, StandardAttention is not instantiated (pure geometry).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from .config import FGNConfig
from .metric import MetricNetwork
from .curvature import CurvatureEngine
from .geo_route import GeoRoute
from .standard_attention import StandardAttention


class FGNv6Layer(nn.Module):
    def __init__(self, config: FGNConfig, layer_idx: int, budget: float):
        """Initialize FGNv6Layer with budget-based escalation.

        Args:
            config: FGN configuration
            layer_idx: Layer index in the stack
            budget: Fraction of positions to escalate to attention [0.0, 1.0].
                    0.0 = pure geometry (no attention), 1.0 = all positions.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.budget = budget
        self.use_learned_metric = (config.geo_metric_type == "learned")

        # Pre-norm layers
        self.norm_geo = nn.LayerNorm(config.d_model)
        self.norm_ff = nn.LayerNorm(config.d_model)

        # Only create attention norm if we need attention
        if self.budget > 0.0:
            self.norm_attn = nn.LayerNorm(config.d_model)

        # Metric (optional — flat mode uses g=1)
        if self.use_learned_metric:
            self.norm_metric = nn.LayerNorm(config.d_model)
            self.metric = MetricNetwork(config)
            self.curvature = CurvatureEngine()

        # Geometric routing (always runs)
        self.geo_route = GeoRoute(config)

        # Standard attention (only if budget > 0.0)
        if self.budget > 0.0:
            self.attention = StandardAttention(config)

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
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor]:
        """Forward pass with budget-based escalation.

        Args:
            h: Hidden states [B, N, d_model]
            mask: Optional causal mask [N, N]
            context: Optional episode context vector [B, d_model] from ContextPool

        Returns:
            (h, kappa [B, N], metric_cv scalar, avg_entropy scalar,
             escalation_rate scalar)
        """
        B, N, _ = h.shape

        # Step 1: Compute metric (context-conditioned or flat)
        if self.use_learned_metric:
            g = self.metric(self.norm_metric(h), context=context)
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

        # Step 3: Budget-based escalation
        if self.budget > 0.0:
            # Compute normalized entropy from GeoRoute weights
            eps = 1e-8
            # geo_weights: [B, N, N]
            entropy = -(geo_weights * torch.log(geo_weights + eps)).sum(dim=-1)  # [B, N]

            # Normalize by max possible entropy: log(i+1) for causal position i
            positions = torch.arange(1, N + 1, device=h.device, dtype=h.dtype)
            max_entropy = torch.log(positions + eps).unsqueeze(0)  # [1, N]
            normalized_entropy = entropy / (max_entropy + eps)  # [B, N] in ~[0, 1]

            # Top-k budget selection (hard mask)
            k = max(1, int(self.budget * N))
            _, top_idx = normalized_entropy.topk(k, dim=-1)  # [B, k]
            escalate_mask = torch.zeros(B, N, device=h.device, dtype=h.dtype)
            escalate_mask.scatter_(1, top_idx, 1.0)  # [B, N], binary 0/1

            # Attention on all tokens, apply mask
            h_attn = self.attention(self.norm_attn(h), mask=mask)
            h = h + escalate_mask.unsqueeze(-1) * self.resid_drop(h_attn)

            # Stats
            avg_entropy = normalized_entropy.mean()
            escalation_rate = escalate_mask.mean()
        else:
            # Budget=0: pure geometry, skip attention entirely
            avg_entropy = torch.tensor(0.0, device=h.device, dtype=h.dtype)
            escalation_rate = torch.tensor(0.0, device=h.device, dtype=h.dtype)

        # Step 4: FFN
        h = h + self.resid_drop(self.ffn(self.norm_ff(h)))

        return h, kappa, metric_cv, avg_entropy, escalation_rate


if __name__ == "__main__":
    print("=== FGNv6Layer Self-Test ===\n")

    # Test 1: Budget=0.0 (pure geometry, no attention)
    print("--- Test 1: budget=0.0 (pure geometry) ---")
    cfg_zero = FGNConfig(d_model=64, n_heads=4, d_ff=256, geo_heads=1,
                         architecture_version="v6", geo_metric_type="learned")
    layer_zero = FGNv6Layer(cfg_zero, layer_idx=0, budget=0.0)

    B, N = 2, 16
    h = torch.randn(B, N, 64)
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)

    # Verify attention is not instantiated
    assert not hasattr(layer_zero, 'attention'), "budget=0.0 should not instantiate attention"
    assert not hasattr(layer_zero, 'norm_attn'), "budget=0.0 should not create norm_attn"

    out, kappa, m_cv, avg_ent, esc_rate = layer_zero(h, mask=mask)
    assert out.shape == (B, N, 64), f"Output shape mismatch: {out.shape}"
    assert kappa.shape == (B, N), f"Kappa shape mismatch: {kappa.shape}"
    assert avg_ent.item() == 0.0, "avg_entropy should be 0.0 when budget=0.0"
    assert esc_rate.item() == 0.0, "escalation_rate should be 0.0 when budget=0.0"

    print(f"metric_cv={m_cv.item():.4f}")
    print(f"|kappa|={kappa.abs().mean().item():.4f}")
    print(f"avg_entropy={avg_ent.item():.4f}")
    print(f"escalation_rate={esc_rate.item():.4f}")
    print("PASS: Pure geometry mode works correctly\n")

    # Test 2: Budget=0.3 (30% escalation)
    print("--- Test 2: budget=0.3 (30% escalation) ---")
    cfg_budget = FGNConfig(d_model=64, n_heads=4, d_ff=256, geo_heads=1,
                           architecture_version="v6", geo_metric_type="learned")
    layer_budget = FGNv6Layer(cfg_budget, layer_idx=0, budget=0.3)

    # Verify attention is instantiated
    assert hasattr(layer_budget, 'attention'), "budget=0.3 should instantiate attention"
    assert hasattr(layer_budget, 'norm_attn'), "budget=0.3 should create norm_attn"

    out_b, kappa_b, m_cv_b, avg_ent_b, esc_rate_b = layer_budget(h, mask=mask)
    assert out_b.shape == (B, N, 64), f"Output shape mismatch: {out_b.shape}"
    assert kappa_b.shape == (B, N), f"Kappa shape mismatch: {kappa_b.shape}"

    # Verify escalation rate is approximately 0.3
    expected_rate = 0.3
    actual_rate = esc_rate_b.item()
    tolerance = 0.05
    assert abs(actual_rate - expected_rate) < tolerance, \
        f"Escalation rate {actual_rate:.4f} not within {tolerance} of {expected_rate}"

    print(f"metric_cv={m_cv_b.item():.4f}")
    print(f"|kappa|={kappa_b.abs().mean().item():.4f}")
    print(f"avg_entropy={avg_ent_b.item():.4f}")
    print(f"escalation_rate={esc_rate_b.item():.4f} (target=0.30)")
    print("PASS: Budget-based escalation works correctly\n")

    # Test 3: Flat metric mode
    print("--- Test 3: flat metric mode ---")
    cfg_flat = FGNConfig(d_model=64, n_heads=4, d_ff=256, geo_heads=1,
                         architecture_version="v6", geo_metric_type="flat")
    layer_flat = FGNv6Layer(cfg_flat, layer_idx=0, budget=0.2)

    assert not hasattr(layer_flat, 'metric'), "Flat mode should not instantiate metric"
    assert not hasattr(layer_flat, 'norm_metric'), "Flat mode should not create norm_metric"

    out_f, kappa_f, m_cv_f, avg_ent_f, esc_rate_f = layer_flat(h, mask=mask)
    assert kappa_f.abs().sum().item() == 0.0, "Flat metric should have zero curvature"
    assert m_cv_f.item() == 0.0, "Flat metric should have zero metric_cv"

    print(f"metric_cv={m_cv_f.item():.4f}")
    print(f"|kappa|={kappa_f.abs().mean().item():.4f}")
    print(f"avg_entropy={avg_ent_f.item():.4f}")
    print(f"escalation_rate={esc_rate_f.item():.4f}")
    print("PASS: Flat metric mode works correctly\n")

    # Test 4: Context conditioning
    print("--- Test 4: context conditioning ---")
    cfg_ctx = FGNConfig(d_model=64, n_heads=4, d_ff=256, geo_heads=1,
                        architecture_version="v6", geo_metric_type="learned")
    layer_ctx = FGNv6Layer(cfg_ctx, layer_idx=0, budget=0.25)

    context = torch.randn(B, 64)
    out_ctx, kappa_ctx, m_cv_ctx, avg_ent_ctx, esc_rate_ctx = \
        layer_ctx(h, mask=mask, context=context)

    assert out_ctx.shape == (B, N, 64), f"Output shape mismatch: {out_ctx.shape}"
    assert kappa_ctx.shape == (B, N), f"Kappa shape mismatch: {kappa_ctx.shape}"

    print(f"metric_cv={m_cv_ctx.item():.4f}")
    print(f"|kappa|={kappa_ctx.abs().mean().item():.4f}")
    print(f"avg_entropy={avg_ent_ctx.item():.4f}")
    print(f"escalation_rate={esc_rate_ctx.item():.4f}")
    print("PASS: Context conditioning works correctly\n")

    # Test 5: Gradient flow
    print("--- Test 5: gradient flow ---")
    layer_grad = FGNv6Layer(cfg_budget, layer_idx=0, budget=0.3)
    out_g, kappa_g, _, _, _ = layer_grad(h, mask=mask)

    loss = out_g.sum() + kappa_g.sum()
    loss.backward()

    # Check metric gradients
    if hasattr(layer_grad, 'metric'):
        for name, p in layer_grad.metric.named_parameters():
            assert p.grad is not None, f"No grad for metric.{name}"
            assert p.grad.abs().sum() > 0, f"Zero grad for metric.{name}"

    # Check attention gradients
    if hasattr(layer_grad, 'attention'):
        for name, p in layer_grad.attention.named_parameters():
            assert p.grad is not None, f"No grad for attention.{name}"

    # Check FFN gradients
    for name, p in layer_grad.ffn.named_parameters():
        assert p.grad is not None, f"No grad for ffn.{name}"

    n_params = sum(p.numel() for p in layer_grad.parameters())
    print(f"Parameters: {n_params:,}")
    print("PASS: Gradient flow verified\n")

    # Test 6: Budget edge cases
    print("--- Test 6: budget edge cases ---")

    # Budget=1.0 (all positions)
    layer_all = FGNv6Layer(cfg_budget, layer_idx=0, budget=1.0)
    _, _, _, _, esc_all = layer_all(h, mask=mask)
    assert abs(esc_all.item() - 1.0) < 0.01, f"Budget=1.0 should give esc_rate~1.0, got {esc_all.item():.4f}"
    print(f"budget=1.0: escalation_rate={esc_all.item():.4f} (target=1.00)")

    # Very small budget
    layer_small = FGNv6Layer(cfg_budget, layer_idx=0, budget=0.01)
    _, _, _, _, esc_small = layer_small(h, mask=mask)
    assert 0.0 < esc_small.item() <= 0.1, f"Budget=0.01 should give small esc_rate, got {esc_small.item():.4f}"
    print(f"budget=0.01: escalation_rate={esc_small.item():.4f}")
    print("PASS: Budget edge cases work correctly\n")

    print("=== All FGNv6Layer tests PASSED ===")
