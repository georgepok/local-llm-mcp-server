"""AttnOnlyLayer — pure transformer attention layer for v7 sandwich architecture.

No geometric components. StandardAttention + FFN only.
Used for middle layers in the sandwich.
"""

from typing import Optional

import torch
import torch.nn as nn

from .config import FGNConfig
from .standard_attention import StandardAttention


class AttnOnlyLayer(nn.Module):
    def __init__(self, config: FGNConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-norm layers
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.norm_ff = nn.LayerNorm(config.d_model)

        # Standard attention
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
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass — pure attention + FFN.

        Returns:
            h [B, N, d_model]
        """
        h = h + self.resid_drop(self.attention(self.norm_attn(h), mask=mask))
        h = h + self.resid_drop(self.ffn(self.norm_ff(h)))
        return h


if __name__ == "__main__":
    print("=== AttnOnlyLayer Self-Test ===\n")

    cfg = FGNConfig(d_model=64, n_heads=4, d_ff=256, architecture_version="v7")

    # Test 1: Basic forward pass
    print("--- Test 1: Basic forward pass ---")
    layer = AttnOnlyLayer(cfg, layer_idx=0)
    B, N = 2, 16
    h = torch.randn(B, N, 64)
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    out = layer(h, mask=mask)
    assert out.shape == (B, N, 64), f"Output shape: {out.shape}"
    print("PASS\n")

    # Test 2: No geometric components
    print("--- Test 2: No geometric components ---")
    assert not hasattr(layer, 'metric'), "Should not have metric"
    assert not hasattr(layer, 'geo_route'), "Should not have geo_route"
    assert not hasattr(layer, 'curvature'), "Should not have curvature"
    print("PASS\n")

    # Test 3: Gradient flow
    print("--- Test 3: Gradient flow ---")
    layer_g = AttnOnlyLayer(cfg, layer_idx=0)
    out_g = layer_g(h, mask=mask)
    out_g.sum().backward()
    for name, p in layer_g.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert p.grad.abs().sum() > 0, f"Zero grad for {name}"
    print("PASS\n")

    # Test 4: Causal masking
    print("--- Test 4: Causal masking ---")
    layer_c = AttnOnlyLayer(cfg, layer_idx=0)
    h_test = torch.randn(1, 4, 64)
    mask4 = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)

    # Forward with full sequence
    out_full = layer_c(h_test, mask=mask4)

    # Modify position 3 and check position 0 is unchanged
    h_mod = h_test.clone()
    h_mod[0, 3] += 100.0
    out_mod = layer_c(h_mod, mask=mask4)

    diff_pos0 = (out_full[0, 0] - out_mod[0, 0]).abs().max().item()
    assert diff_pos0 < 1e-5, f"Position 0 changed by {diff_pos0} when future modified"
    print("PASS\n")

    n_params = sum(p.numel() for p in layer.parameters())
    print(f"Parameters: {n_params:,}")
    print("=== All AttnOnlyLayer tests PASSED ===")
