"""StandardAttention — multi-head dot-product attention for v4 content routing.

Identical to FlatAttention in flat_model.py. Separated into its own module
so v4 layers can import it independently.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig


class StandardAttention(nn.Module):
    """Standard multi-head dot-product attention with causal masking."""

    def __init__(self, config: FGNConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_drop = nn.Dropout(config.dropout)
        self.scale = config.d_head ** -0.5

    def forward(self, h: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard scaled dot-product attention.

        Args:
            h: [B, N, d_model] hidden states
            mask: [N, N] causal mask (True = masked positions)

        Returns:
            attn_output [B, N, d_model]
        """
        B, N, _ = h.shape

        Q = self.W_q(h).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = self.W_k(h).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.W_v(h).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        scores = (Q @ K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        attn_out = attn_weights @ V
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, self.d_model)
        return self.W_o(attn_out)


if __name__ == "__main__":
    cfg = FGNConfig(d_model=64, n_heads=4, max_seq_len=32)
    attn = StandardAttention(cfg)

    B, N = 2, 16
    h = torch.randn(B, N, 64)
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)

    out = attn(h, mask=mask)
    assert out.shape == (B, N, 64), f"Got {out.shape}"

    loss = out.sum()
    loss.backward()
    for name, p in attn.named_parameters():
        assert p.grad is not None, f"No grad for {name}"

    n_params = sum(p.numel() for p in attn.parameters())
    print(f"Parameters: {n_params:,}")
    print("StandardAttention OK")
