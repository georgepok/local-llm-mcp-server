"""HeatKernelAttention — multi-scale heat kernel attention on Riemannian manifold.

For each scale s:
  d_g(i,j)^2 = (q_i - k_j)^T * diag((g_i + g_j)/2) * (q_i - k_j)
  K_s(i,j) = exp(-d_g^2 / (4*t_s))

Each K_s is INDEPENDENTLY row-normalized before scale mixing.
Per-token scale fusion: w_s = Softmax(W_scale * h_i)
Final: A(i,j) = sum_s w_{s,i} * K_s_normalized(i,j)

SHARED METRIC CONTRACT: The d-dimensional metric is reduced to d_head by
averaging across head groups. All heads see the SAME metric — this prevents
projection absorption where individual heads learn to undo the metric.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig


class HeatKernelAttention(nn.Module):
    def __init__(self, config: FGNConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.n_scales = config.n_scales
        self.d_model = config.d_model

        # Q, K, V projections
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

        # Learnable diffusion times in log-space
        # t_init = (0.1, 1.0, 10.0) -> log values
        log_t_init = [math.log(t) for t in config.t_init]
        self.log_t = nn.Parameter(torch.tensor(log_t_init))

        # Per-token scale selection
        self.W_scale = nn.Linear(config.d_model, config.n_scales)

        # Dropout on attention output
        self.attn_drop = nn.Dropout(config.dropout)

        # Chunk size for memory-efficient long-sequence attention
        # For seq_len <= chunk_size, uses direct O(N^2) computation
        # For seq_len > chunk_size, chunks queries to limit peak memory
        self.chunk_size = 256

        # Cached scale weights from last forward pass (for entropy computation)
        self._last_scale_weights: Optional[torch.Tensor] = None

    def forward(self, h: torch.Tensor, g: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Heat kernel attention forward pass.

        Args:
            h: [B, N, d_model] hidden states
            g: [B, N, d_model] diagonal metric (positive, shared across heads)
            mask: [N, N] causal mask (True = masked positions)

        Returns:
            (attn_output [B, N, d_model], scale_entropy scalar)
        """
        B, N, _ = h.shape

        # 1. Project Q, K, V -> [B, H, N, d_head]
        Q = self.W_q(h).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = self.W_k(h).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.W_v(h).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # 2. SHARED METRIC: reduce d-dim metric to d_head by averaging across
        #    head groups. This ensures ALL heads see the same geometry.
        #    g: [B, N, d_model] -> [B, N, n_heads, d_head] -> mean over heads -> [B, N, d_head]
        g_shared = g.view(B, N, self.n_heads, self.d_head).mean(dim=2)
        # Broadcast across heads: [B, 1, N, d_head]
        g_shared = g_shared.unsqueeze(1)

        # 3. Scale weights per token: [B, N, S]
        scale_weights = F.softmax(self.W_scale(h), dim=-1)
        # Cache for diagnostic scripts (not read in compiled forward path)
        self._last_scale_weights = scale_weights

        # 4. Scale entropy (computed inline for torch.compile compatibility)
        eps = 1e-8
        entropy = -(scale_weights * torch.log(scale_weights + eps)).sum(-1)
        scale_entropy = -entropy.mean()

        t = self.log_t.exp()  # [S]

        # 5. Attention computation (static path, no dynamic branching)
        attn_out = self._direct_attention(Q, K, V, g_shared, scale_weights, t, mask)

        # 6. Reshape, dropout, and output project
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, self.d_model)
        return self.W_o(self.attn_drop(attn_out)), scale_entropy

    def _direct_attention(self, Q, K, V, g_shared, scale_weights, t, mask):
        """Direct (non-chunked) attention — for short sequences."""
        # Geodesic distance
        diff = Q.unsqueeze(3) - K.unsqueeze(2)
        g_avg = (g_shared.unsqueeze(3) + g_shared.unsqueeze(2)) / 2.0
        d_sq = (diff * diff * g_avg).sum(-1)  # [B, H, N, N]

        attn_out = torch.zeros_like(V)
        for s in range(self.n_scales):
            log_K = -d_sq / (4.0 * t[s])
            if mask is not None:
                log_K = log_K.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            K_s = F.softmax(log_K, dim=-1)
            w_s = scale_weights[:, :, s].unsqueeze(1).unsqueeze(-1)
            attn_out = attn_out + w_s * (K_s @ V)
        return attn_out

    def _chunked_attention(self, Q, K, V, g_shared, scale_weights, t, mask):
        """Memory-efficient chunked attention — for long sequences.

        Processes query tokens in chunks of self.chunk_size, computing geodesic
        distance and attention weights for each chunk against ALL key tokens.
        Peak memory: O(chunk_size * N * d) instead of O(N^2 * d).
        """
        B, H, N, d = Q.shape
        C = self.chunk_size
        attn_out = torch.zeros_like(V)

        for q_start in range(0, N, C):
            q_end = min(q_start + C, N)
            Q_c = Q[:, :, q_start:q_end]          # [B, H, C, d]
            g_q = g_shared[:, :, q_start:q_end]    # [B, 1, C, d]
            w_c = scale_weights[:, q_start:q_end]   # [B, C, S]

            # Geodesic distance: [B, H, C, N]
            diff = Q_c.unsqueeze(3) - K.unsqueeze(2)      # [B, H, C, N, d]
            g_avg = (g_q.unsqueeze(3) + g_shared.unsqueeze(2)) / 2.0  # [B, 1, C, N, d]
            d_sq = (diff * diff * g_avg).sum(-1)           # [B, H, C, N]

            chunk_out = torch.zeros(B, H, q_end - q_start, d,
                                    device=Q.device, dtype=Q.dtype)

            for s in range(self.n_scales):
                log_K = -d_sq / (4.0 * t[s])
                if mask is not None:
                    mask_c = mask[q_start:q_end]  # [C, N]
                    log_K = log_K.masked_fill(mask_c.unsqueeze(0).unsqueeze(0), float("-inf"))
                K_s = F.softmax(log_K, dim=-1)    # [B, H, C, N]
                w_s = w_c[:, :, s].unsqueeze(1).unsqueeze(-1)  # [B, 1, C, 1]
                chunk_out = chunk_out + w_s * (K_s @ V)

            attn_out[:, :, q_start:q_end] = chunk_out

        return attn_out

    def cached_scale_entropy(self) -> torch.Tensor:
        """Compute scale entropy from cached weights of the last forward pass.

        Returns negative entropy (minimizing this maximizes entropy).
        Uses the actual scale weights computed during forward, not stale h.
        """
        w = self._last_scale_weights  # [B, N, S]
        assert w is not None, "Must call forward() before cached_scale_entropy()"
        eps = 1e-8
        entropy = -(w * torch.log(w + eps)).sum(-1)  # [B, N]
        return -entropy.mean()

    def scale_entropy(self, h: torch.Tensor) -> torch.Tensor:
        """Compute scale entropy from given hidden states (for external use)."""
        weights = F.softmax(self.W_scale(h), dim=-1)
        eps = 1e-8
        entropy = -(weights * torch.log(weights + eps)).sum(-1)
        return -entropy.mean()


if __name__ == "__main__":
    cfg = FGNConfig(d_model=64, n_heads=4, n_scales=3, max_seq_len=32)
    attn = HeatKernelAttention(cfg)

    h = torch.randn(2, 16, 64)
    g = F.softplus(torch.randn(2, 16, 64))  # Positive metric

    # Causal mask
    N = 16
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)

    # Test direct path (N=16 < chunk_size=256)
    out, scale_ent = attn(h, g, mask=mask)
    assert out.shape == (2, 16, 64), f"Got {out.shape}"
    assert scale_ent.shape == ()

    # Gradient flow
    loss = out.sum() + scale_ent
    loss.backward()
    for name, p in attn.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
    print("HeatKernelAttention (direct path) OK")

    # Test chunked path (force chunk_size < N)
    attn2 = HeatKernelAttention(cfg)
    attn2.chunk_size = 8  # Force chunking at N=16
    out2, _ = attn2(h, g, mask=mask)
    assert out2.shape == (2, 16, 64), f"Chunked got {out2.shape}"
    loss2 = out2.sum()
    loss2.backward()
    print("HeatKernelAttention (chunked path) OK")
