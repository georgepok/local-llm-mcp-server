"""
Context pooling module for FGN v6.

Implements attention-weighted pooling over world-description positions to extract
a per-episode context vector that can be used for decision-making.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig


class ContextPool(nn.Module):
    """
    Attention-weighted pooling over context positions.

    Computes attention scores for each position in the input sequence,
    optionally masked to focus on world-description positions, then
    produces a weighted sum to create a fixed-size context vector.
    """

    def __init__(self, config: FGNConfig):
        """
        Initialize the context pooling module.

        Args:
            config: FGN configuration containing d_model dimension
        """
        super().__init__()
        self.config = config
        d = config.d_model

        # Attention pooling network: computes scalar score per position
        # d -> d//4 -> 1 with tanh nonlinearity
        self.attn_pool = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.Tanh(),
            nn.Linear(d // 4, 1)
        )

        # Output projection for the pooled context vector
        self.out_proj = nn.Linear(d, d)

    def forward(
        self,
        h: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool context from hidden states using attention weighting.

        Args:
            h: Hidden states of shape [B, N, d]
            context_mask: Optional boolean mask [B, N] where True indicates
                         context positions to pool over. If None, uses first
                         25% of sequence as heuristic fallback.

        Returns:
            Pooled context vector of shape [B, d]
        """
        B, N, _ = h.shape

        # Compute attention scores for each position: [B, N, d] -> [B, N, 1] -> [B, N]
        scores = self.attn_pool(h).squeeze(-1)

        # Apply context mask if provided, otherwise use heuristic fallback
        if context_mask is not None:
            # Mask out non-context positions with -inf before softmax
            scores = scores.masked_fill(~context_mask, float('-inf'))
        else:
            # Heuristic fallback: use first 25% of sequence as context
            mask = torch.zeros(B, N, dtype=torch.bool, device=h.device)
            mask[:, :N // 4] = True
            scores = scores.masked_fill(~mask, float('-inf'))

        # Compute attention weights via softmax: [B, N]
        alpha = F.softmax(scores, dim=-1)

        # Weighted sum over positions: [B, N, 1] * [B, N, d] -> [B, N, d] -> [B, d]
        pooled = (alpha.unsqueeze(-1) * h).sum(dim=1)

        # Project to output space
        return self.out_proj(pooled)


if __name__ == "__main__":
    print("Testing ContextPool...")

    # Create test configuration
    class TestConfig:
        d_model = 64

    config = TestConfig()

    # Create module
    pool = ContextPool(config)

    # Test parameters
    B, N, d = 2, 16, 64

    # Test 1: With explicit context mask (first 4 of 16 positions)
    print(f"\nTest 1: Explicit context_mask (first 4 of {N} positions)")
    h = torch.randn(B, N, d)
    context_mask = torch.zeros(B, N, dtype=torch.bool)
    context_mask[:, :4] = True

    output = pool(h, context_mask)
    print(f"  Input shape: {h.shape}")
    print(f"  Context mask: {context_mask[0].tolist()}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (B, d), f"Expected shape ({B}, {d}), got {output.shape}"

    # Test 2: With context_mask=None (fallback heuristic)
    print(f"\nTest 2: context_mask=None (fallback to first 25% = {N//4} positions)")
    output_fallback = pool(h, context_mask=None)
    print(f"  Output shape: {output_fallback.shape}")
    assert output_fallback.shape == (B, d), f"Expected shape ({B}, {d}), got {output_fallback.shape}"

    # Test 3: Verify gradient flow
    print("\nTest 3: Gradient flow")
    h_grad = torch.randn(B, N, d, requires_grad=True)
    output_grad = pool(h_grad, context_mask)
    loss = output_grad.sum()
    loss.backward()
    assert h_grad.grad is not None, "Gradient not flowing through ContextPool"
    print(f"  Gradient shape: {h_grad.grad.shape}")
    print(f"  Gradient norm: {h_grad.grad.norm().item():.4f}")

    # Test 4: Different context masks
    print("\nTest 4: Variable context regions")
    context_mask_var = torch.zeros(B, N, dtype=torch.bool)
    context_mask_var[0, :3] = True  # First batch: 3 positions
    context_mask_var[1, :8] = True  # Second batch: 8 positions
    output_var = pool(h, context_mask_var)
    print(f"  Batch 0 context: {context_mask_var[0].sum().item()} positions")
    print(f"  Batch 1 context: {context_mask_var[1].sum().item()} positions")
    print(f"  Output shape: {output_var.shape}")
    assert output_var.shape == (B, d), f"Expected shape ({B}, {d}), got {output_var.shape}"

    print("\n" + "="*50)
    print("ContextPool OK")
    print("="*50)
