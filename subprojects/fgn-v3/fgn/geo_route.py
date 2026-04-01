"""GeoRoute — geometric information aggregation based on manifold distance.

Architecture versions:
  - v4/v5: No Q/K projections. Routes purely based on where information sits
    on the learned Riemannian manifold. Single temperature, shared metric.
  - v6: Q/K projections for geometric routing. Per-head temperature
    (linspace from short-range to long-range). Per-head metric slicing.

Key differences from v3 HeatKernelAttention:
  - Distance computed between raw representations (v4/v5) or Q/K projections (v6)
  - No content matching — that's StandardAttention's job
  - Single learned temperature (v4/v5) or per-head temperatures (v6)
  - 1 head by default (optionally more, sharing same metric in v4/v5)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig


class GeoRoute(nn.Module):
    def __init__(self, config: FGNConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.geo_heads
        self.d_head = config.d_model // config.geo_heads
        self.arch_version = config.architecture_version

        assert config.d_model % config.geo_heads == 0, \
            f"d_model={config.d_model} not divisible by geo_heads={config.geo_heads}"

        # Value and output projections
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

        # v6/v7: Q/K projections and per-head temperature
        if self.arch_version in ("v6", "v7"):
            self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
            self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
            # Per-head temperature: linspace from short-range to long-range
            self.log_t = nn.Parameter(torch.linspace(-1.0, 2.0, self.n_heads))
        else:
            # Single learnable temperature (log-space, init=log(1.0)=0)
            self.log_t = nn.Parameter(torch.zeros(1))

        # Dropout on attention output
        self.attn_drop = nn.Dropout(config.dropout)

        # Chunk size for memory-efficient long-sequence attention
        self.chunk_size = 256

    def forward(self, h_normed: torch.Tensor, g: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_weights: bool = False,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Geometric routing forward pass.

        Args:
            h_normed: [B, N, d_model] layer-normed hidden states
            g: [B, N, d_model] diagonal metric (positive)
            mask: [N, N] causal mask (True = masked positions)
            return_weights: if True, return attention weights as second output

        Returns:
            (h_geo [B, N, d_model], geo_weights [B, N, N] or None)
        """
        B, N, _ = h_normed.shape

        # Value projection
        V = self.W_v(h_normed)  # [B, N, d_model]

        if self.arch_version in ("v6", "v7"):
            # v6/v7: Q/K projections, per-head metric and temperature
            Q = self.W_q(h_normed).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
            K = self.W_k(h_normed).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
            V = V.view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
            g_heads = g.view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
            t = self.log_t.exp()  # [H]

            if N <= self.chunk_size:
                out, w_geo = self._direct_v6(Q, K, g_heads, V, t, mask)
            else:
                out, w_geo = self._chunked_v6(Q, K, g_heads, V, t, mask)

            # out: [B, H, N, d_h] -> [B, N, d_model]
            out = out.permute(0, 2, 1, 3).reshape(B, N, self.d_model)

            # weights: [B, H, N, N] -> average across heads -> [B, N, N]
            weights_out = None
            if return_weights:
                weights_out = w_geo.mean(dim=1)

            return self.W_o(self.attn_drop(out)), weights_out
        else:
            # v4/v5: raw distance path (existing code)
            if self.n_heads > 1:
                V = V.view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
                # [B, H, N, d_head]

            t = self.log_t.exp()  # scalar temperature

            if N <= self.chunk_size:
                out, w_geo = self._direct(h_normed, g, V, t, mask)
            else:
                out, w_geo = self._chunked(h_normed, g, V, t, mask)

            # Reshape from multi-head back to d_model
            if self.n_heads > 1:
                out = out.permute(0, 2, 1, 3).reshape(B, N, self.d_model)

            # For multi-head, average weights across heads for entropy computation
            weights_out = None
            if return_weights:
                if self.n_heads > 1:
                    # w_geo is [B, H, N, N] — average across heads
                    weights_out = w_geo.mean(dim=1)  # [B, N, N]
                else:
                    weights_out = w_geo  # [B, N, N]

            return self.W_o(self.attn_drop(out)), weights_out

    def _direct(self, h_normed, g, V, t, mask):
        """Direct O(N^2) geodesic distance + attention."""
        B, N, d = h_normed.shape

        # Pairwise geodesic distance on raw representations
        # diff[i,j] = h_normed[i] - h_normed[j]
        diff = h_normed.unsqueeze(2) - h_normed.unsqueeze(1)  # [B, N, N, d]
        g_avg = (g.unsqueeze(2) + g.unsqueeze(1)) / 2.0       # [B, N, N, d]
        d_sq = (diff * diff * g_avg).sum(-1)                   # [B, N, N]

        # Geometric attention weights
        log_w = -d_sq / (4.0 * t)

        if mask is not None:
            log_w = log_w.masked_fill(mask.unsqueeze(0), float('-inf'))

        w_geo = F.softmax(log_w, dim=-1)  # [B, N, N]

        if self.n_heads > 1:
            # V is [B, H, N, d_head], w_geo is [B, N, N]
            # Expand w_geo for heads: [B, 1, N, N] @ [B, H, N, d_head]
            return w_geo.unsqueeze(1) @ V, w_geo  # [B, H, N, d_head], [B, N, N]
        else:
            return w_geo @ V, w_geo  # [B, N, d_model], [B, N, N]

    def _chunked(self, h_normed, g, V, t, mask):
        """Memory-efficient chunked geodesic attention for long sequences."""
        B, N, _ = h_normed.shape
        C = self.chunk_size

        if self.n_heads > 1:
            out = torch.zeros_like(V)  # [B, H, N, d_head]
        else:
            out = torch.zeros_like(V)  # [B, N, d_model]

        all_weights = torch.zeros(B, N, N, device=h_normed.device, dtype=h_normed.dtype)

        for q_start in range(0, N, C):
            q_end = min(q_start + C, N)
            h_q = h_normed[:, q_start:q_end]  # [B, C, d]
            g_q = g[:, q_start:q_end]          # [B, C, d]

            diff = h_q.unsqueeze(2) - h_normed.unsqueeze(1)    # [B, C, N, d]
            g_avg = (g_q.unsqueeze(2) + g.unsqueeze(1)) / 2.0  # [B, C, N, d]
            d_sq = (diff * diff * g_avg).sum(-1)                # [B, C, N]

            log_w = -d_sq / (4.0 * t)

            if mask is not None:
                mask_c = mask[q_start:q_end]  # [C, N]
                log_w = log_w.masked_fill(mask_c.unsqueeze(0), float('-inf'))

            w_geo = F.softmax(log_w, dim=-1)  # [B, C, N]

            all_weights[:, q_start:q_end] = w_geo

            if self.n_heads > 1:
                out[:, :, q_start:q_end] = w_geo.unsqueeze(1) @ V
            else:
                out[:, q_start:q_end] = w_geo @ V

        return out, all_weights

    def _direct_v6(self, Q, K, g_heads, V, t, mask):
        """Direct O(N^2) geodesic distance + attention for v6 (Q/K mode)."""
        B, H, N, d_h = Q.shape

        # Pairwise geodesic distance on Q/K projections
        diff = Q.unsqueeze(3) - K.unsqueeze(2)  # [B, H, N, N, d_h]
        g_avg = (g_heads.unsqueeze(3) + g_heads.unsqueeze(2)) / 2.0  # [B, H, N, N, d_h]
        d_sq = (diff * diff * g_avg).sum(-1)  # [B, H, N, N]

        # Per-head temperature: t is [H], reshape to [1, H, 1, 1]
        t_bcast = t.view(1, H, 1, 1)
        log_w = -d_sq / (4.0 * t_bcast)

        if mask is not None:
            log_w = log_w.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        w_geo = F.softmax(log_w, dim=-1)  # [B, H, N, N]
        out = w_geo @ V  # [B, H, N, d_h]
        return out, w_geo

    def _chunked_v6(self, Q, K, g_heads, V, t, mask):
        """Memory-efficient chunked geodesic attention for v6 (Q/K mode)."""
        B, H, N, d_h = Q.shape
        C = self.chunk_size

        out = torch.zeros_like(V)  # [B, H, N, d_h]
        all_weights = torch.zeros(B, H, N, N, device=Q.device, dtype=Q.dtype)

        t_bcast = t.view(1, H, 1, 1)

        for q_start in range(0, N, C):
            q_end = min(q_start + C, N)
            Q_c = Q[:, :, q_start:q_end]        # [B, H, C, d_h]
            g_q = g_heads[:, :, q_start:q_end]   # [B, H, C, d_h]

            diff = Q_c.unsqueeze(3) - K.unsqueeze(2)    # [B, H, C, N, d_h]
            g_avg = (g_q.unsqueeze(3) + g_heads.unsqueeze(2)) / 2.0  # [B, H, C, N, d_h]
            d_sq = (diff * diff * g_avg).sum(-1)          # [B, H, C, N]

            log_w = -d_sq / (4.0 * t_bcast)

            if mask is not None:
                mask_c = mask[q_start:q_end]  # [C, N]
                log_w = log_w.masked_fill(mask_c.unsqueeze(0).unsqueeze(0), float('-inf'))

            w_geo = F.softmax(log_w, dim=-1)  # [B, H, C, N]
            all_weights[:, :, q_start:q_end] = w_geo
            out[:, :, q_start:q_end] = w_geo @ V  # [B, H, C, d_h]

        return out, all_weights


if __name__ == "__main__":
    cfg = FGNConfig(d_model=64, n_heads=4, geo_heads=1, max_seq_len=32,
                    architecture_version="v4")
    geo = GeoRoute(cfg)

    B, N = 2, 16
    h = torch.randn(B, N, 64)
    g = F.softplus(torch.randn(B, N, 64))
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)

    out, w = geo(h, g, mask=mask, return_weights=True)
    assert out.shape == (B, N, 64), f"Got {out.shape}"
    assert w is not None, "Expected weights with return_weights=True"
    assert w.shape == (B, N, N), f"Weights shape {w.shape}, expected {(B, N, N)}"

    # Test without returning weights
    out_nw, w_nw = geo(h, g, mask=mask)
    assert out_nw.shape == (B, N, 64)
    assert w_nw is None, "Expected no weights without return_weights"

    # Gradient flow
    loss = out.sum()
    loss.backward()
    for name, p in geo.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
    print(f"GeoRoute (1 head) OK, log_t={geo.log_t.item():.3f}")

    # Test with multiple geo heads
    cfg2 = FGNConfig(d_model=64, n_heads=4, geo_heads=2, max_seq_len=32,
                     architecture_version="v4")
    geo2 = GeoRoute(cfg2)
    out2, w2 = geo2(h, g, mask=mask, return_weights=True)
    assert out2.shape == (B, N, 64), f"Multi-head got {out2.shape}"
    assert w2.shape == (B, N, N), f"Multi-head weights {w2.shape}, expected {(B, N, N)}"
    out2.sum().backward()
    print(f"GeoRoute (2 heads) OK")

    # Test chunked path
    geo3 = GeoRoute(cfg)
    geo3.chunk_size = 8
    out3, w3 = geo3(h, g, mask=mask, return_weights=True)
    assert out3.shape == (B, N, 64), f"Chunked got {out3.shape}"
    assert w3.shape == (B, N, N), f"Chunked weights {w3.shape}"
    print("GeoRoute (chunked) OK")

    # Test v6 mode with Q/K projections and per-head temperature
    cfg_v6 = FGNConfig(d_model=64, n_heads=4, geo_heads=4, max_seq_len=32,
                       architecture_version="v6")
    geo_v6 = GeoRoute(cfg_v6)
    out_v6, w_v6 = geo_v6(h, g, mask=mask, return_weights=True)
    assert out_v6.shape == (B, N, 64), f"v6 got {out_v6.shape}"
    assert w_v6.shape == (B, N, N), f"v6 weights {w_v6.shape}"
    out_v6.sum().backward()
    for name, p in geo_v6.named_parameters():
        assert p.grad is not None, f"v6: No grad for {name}"
    print(f"GeoRoute v6 (4 heads, Q/K) OK, log_t={geo_v6.log_t.data.tolist()}")

    # Test v6 chunked
    geo_v6c = GeoRoute(cfg_v6)
    geo_v6c.chunk_size = 8
    out_v6c, w_v6c = geo_v6c(h, g, mask=mask, return_weights=True)
    assert out_v6c.shape == (B, N, 64)
    assert w_v6c.shape == (B, N, N)
    print("GeoRoute v6 (chunked) OK")

    n_params = sum(p.numel() for p in geo.parameters())
    print(f"Parameters: {n_params:,}")
    n_params_v6 = sum(p.numel() for p in geo_v6.parameters())
    print(f"Parameters (v6): {n_params_v6:,}")
    print("GeoRoute OK")
