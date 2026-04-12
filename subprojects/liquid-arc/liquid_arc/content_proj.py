"""Content Projection — maps ODE state to soft prompt tokens for Qwen3.

The attention bias tells Qwen3 WHERE to attend.
The content projection tells Qwen3 WHAT the ODE has learned.
Both channels operate simultaneously during generation.

The ODE state h [1, N, d_ode] holds N token positions in LiquidARC's
Riemannian space. ContentProjection pools these down to n_prefix summary
positions via attention-weighted pooling, then projects to Qwen3's
embedding dimension d_llm.

These n_prefix soft tokens are prepended to Qwen3's input embeddings
alongside the text prompt. The bias handles routing geometry; the prefix
handles semantic content. Both are differentiable if online learning
is enabled.
"""

import torch
import torch.nn as nn


class ContentProjection(nn.Module):
    """Project ODE state positions to Qwen3-compatible soft prompt tokens.

    Compresses N ODE positions into n_prefix summary positions via learned
    attention-weighted pooling, then projects to LLM embedding dimension.

    Args:
        d_ode: ODE state dimension (e.g. 768 for 5M model)
        d_llm: Qwen3 hidden size (e.g. 2048 or 4096)
        n_prefix: number of soft prompt tokens to produce (default 8)
    """

    def __init__(self, d_ode: int, d_llm: int, n_prefix: int = 8):
        super().__init__()
        self.d_ode = d_ode
        self.d_llm = d_llm
        self.n_prefix = n_prefix

        # Attention pooling: produces n_prefix attention weights over N positions
        # Output: [B, n_prefix, N] after softmax
        self.pool = nn.Linear(d_ode, n_prefix)

        # Project pooled ODE representations to LLM embedding space
        self.proj = nn.Linear(d_ode, d_llm)

        # LayerNorm for scale matching with Qwen3's embedding table output
        self.norm = nn.LayerNorm(d_llm)

        # Initialize proj to near-zero so prefix starts as near-neutral
        # (avoids disturbing Qwen3 at the start of training/inference)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.uniform_(self.pool.weight, -0.1, 0.1)
        nn.init.zeros_(self.pool.bias)

        print(f"  ContentProjection: d_ode={d_ode} → n_prefix={n_prefix} × d_llm={d_llm}")

    def forward(self, h_ode: torch.Tensor) -> torch.Tensor:
        """Project ODE state to soft prompt tokens.

        Args:
            h_ode: [1, N, d_ode] ODE state (N token positions)

        Returns:
            prefix: [1, n_prefix, d_llm] soft prompt tokens ready to prepend
                    to Qwen3's input embeddings

        Example:
            proj = ContentProjection(d_ode=768, d_llm=2048, n_prefix=8)
            prefix = proj(h_ode)  # [1, 8, 2048]
            combined = torch.cat([prefix, text_embeds], dim=1)  # [1, 8+T, 2048]
        """
        # Attention-weighted pooling: compress N positions → n_prefix summaries
        # pool outputs [1, N, n_prefix], transpose to [1, n_prefix, N]
        attn_logits = self.pool(h_ode).transpose(1, 2)           # [1, n_prefix, N]
        attn_weights = torch.softmax(attn_logits, dim=-1)        # [1, n_prefix, N]
        pooled = torch.bmm(attn_weights, h_ode)                  # [1, n_prefix, d_ode]

        # Project to LLM embedding space with LayerNorm for scale alignment
        return self.norm(self.proj(pooled))                       # [1, n_prefix, d_llm]

    def extra_repr(self) -> str:
        return (f"d_ode={self.d_ode}, d_llm={self.d_llm}, "
                f"n_prefix={self.n_prefix}")
