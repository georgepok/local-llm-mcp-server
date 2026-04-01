"""Context pooling — attention-weighted pooling over context positions.

Computes a fixed-size per-episode context vector from input sequence positions.
Used to condition the ODE dynamics on the full input context.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LiquidARCConfig


class ContextPool(nn.Module):
    """Attention-weighted pooling: h [B, N, d] → context [B, d]."""

    def __init__(self, config: LiquidARCConfig):
        super().__init__()
        d = config.d_model

        # Attention scoring: d → d//4 → 1 with tanh bottleneck
        self.attn_pool = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.Tanh(),
            nn.Linear(d // 4, 1),
        )

        self.out_proj = nn.Linear(d, d)

    def forward(
        self,
        h: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool context from hidden states.

        Args:
            h: [B, N, d] hidden states
            context_mask: [B, N] bool — True = context position. None = first 25%.

        Returns:
            [B, d] pooled context vector
        """
        B, N, _ = h.shape

        scores = self.attn_pool(h).squeeze(-1)  # [B, N]

        if context_mask is not None:
            scores = scores.masked_fill(~context_mask, float('-inf'))
        else:
            mask = torch.zeros(B, N, dtype=torch.bool, device=h.device)
            mask[:, :N // 4] = True
            scores = scores.masked_fill(~mask, float('-inf'))

        alpha = F.softmax(scores, dim=-1)  # [B, N]
        pooled = (alpha.unsqueeze(-1) * h).sum(dim=1)  # [B, d]

        return self.out_proj(pooled)


if __name__ == "__main__":
    print("Testing ContextPool...")
    config = LiquidARCConfig(d_model=64)
    pool = ContextPool(config)

    B, N, d = 2, 16, 64
    h = torch.randn(B, N, d)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :4] = True

    out = pool(h, mask)
    assert out.shape == (B, d)

    # Gradient flow
    h_g = torch.randn(B, N, d, requires_grad=True)
    pool(h_g, mask).sum().backward()
    assert h_g.grad is not None

    print("ContextPool OK")
