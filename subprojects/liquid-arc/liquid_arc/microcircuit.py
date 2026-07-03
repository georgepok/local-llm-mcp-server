"""Microcircuit substrate — Phase 1 experiment.

Hypothesis: compressing token sequences into a small set of "microcircuit
slots" (M=32 or so), running dynamics on the compressed state, then
expanding back, preserves representational capacity while:
  - Making the substrate's inner loop M² not T² (e.g. 32² vs 512²)
  - Creating explicit "slots" that can later specialize (Phase 1b+)
  - Matching the cortical-column structure (fixed-size local circuits)
  - Staying hardware-friendly (all dense matmul)

This first variant uses FIXED slot queries (learned parameters) and
full cross-attention in/out. Later phases will add:
  - Learned sparse routing between slots (instead of full self-attn
    inside the dynamics)
  - Multiple microcircuit groups with block-sparse connectivity
  - Specialization via different receptive fields
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroCircuitWrapper(nn.Module):
    """Wraps an existing dynamics module + solver.

    Forward path:
      h0 [B, T, d]
        → compress via cross-attention → slots [B, M, d]
        → dynamics for K ODE steps (operates on M slots, not T tokens)
        → expand via cross-attention → h_out [B, T, d]

    The dynamics module is whatever ContinuousDynamics instance is
    passed in. MicroCircuitWrapper only changes the IO shape.
    """

    def __init__(self, d: int, M: int, dynamics: nn.Module,
                 n_ode_steps: int = 16):
        super().__init__()
        self.d = d
        self.M = M
        self.n_ode_steps = n_ode_steps
        self.dynamics = dynamics

        # Learnable slot queries. These become the initial microcircuit states
        # (in Phase 1a, a linear combination of input token features).
        self.slot_queries = nn.Parameter(torch.randn(M, d) * 0.02)

        # Compress: each slot queries the input sequence
        # (standard transformer cross-attention: Q = slots, K = V = input)
        self.compress_q = nn.Linear(d, d, bias=False)
        self.compress_k = nn.Linear(d, d, bias=False)
        self.compress_v = nn.Linear(d, d, bias=False)
        self.compress_o = nn.Linear(d, d, bias=False)
        self.compress_norm = nn.LayerNorm(d)

        # Expand: each token queries the slots
        self.expand_q = nn.Linear(d, d, bias=False)
        self.expand_k = nn.Linear(d, d, bias=False)
        self.expand_v = nn.Linear(d, d, bias=False)
        self.expand_o = nn.Linear(d, d, bias=False)
        self.expand_norm = nn.LayerNorm(d)

        # Zero-init output projections so initial pass is near-identity
        with torch.no_grad():
            nn.init.xavier_uniform_(self.compress_q.weight)
            nn.init.xavier_uniform_(self.compress_k.weight)
            nn.init.xavier_uniform_(self.compress_v.weight)
            nn.init.xavier_uniform_(self.compress_o.weight, gain=0.01)
            nn.init.xavier_uniform_(self.expand_q.weight)
            nn.init.xavier_uniform_(self.expand_k.weight)
            nn.init.xavier_uniform_(self.expand_v.weight)
            nn.init.xavier_uniform_(self.expand_o.weight, gain=0.01)

    def _sdpa(self, q, k, v, mask=None):
        """Standard scaled dot-product attention."""
        d = q.shape[-1]
        return F.scaled_dot_product_attention(
            q / (d ** 0.5), k, v, attn_mask=mask,
        )

    def compress(self, h: torch.Tensor) -> torch.Tensor:
        """Compress [B, T, d] → [B, M, d] via cross-attention."""
        B, T, d = h.shape
        slot_init = self.slot_queries.unsqueeze(0).expand(B, -1, -1)  # [B, M, d]
        h_norm = self.compress_norm(h)
        q = self.compress_q(slot_init)           # [B, M, d]
        k = self.compress_k(h_norm)               # [B, T, d]
        v = self.compress_v(h_norm)               # [B, T, d]
        out = self._sdpa(q, k, v)                 # [B, M, d]
        out = self.compress_o(out)
        # Residual with slot_init (zero-init output → near-identity at start)
        return slot_init + out

    def expand(self, slots: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """Expand [B, M, d] → [B, T, d] by token-queries over slots.

        h0 is used as the residual stream — each token starts from its
        original embedding and adds context from the microcircuit slots.
        """
        h_norm = self.expand_norm(h0)
        q = self.expand_q(h_norm)     # [B, T, d]
        k = self.expand_k(slots)       # [B, M, d]
        v = self.expand_v(slots)       # [B, M, d]
        out = self._sdpa(q, k, v)      # [B, T, d]
        out = self.expand_o(out)
        # Residual with h0 (so compression isn't forced lossy)
        return h0 + out

    def forward(self, h0: torch.Tensor, context: torch.Tensor,
                mask=None, euler_solve_fn=None) -> torch.Tensor:
        """Run microcircuit substrate on input tokens.

        Args:
            h0: [B, T, d] input token representations
            context: [B, d] pooled context (from ContextPool)
            mask: unused here (microcircuits don't use token mask)
            euler_solve_fn: function(dynamics, h0, t_span, n_steps) -> h

        Returns:
            h_out: [B, T, d] updated token representations
        """
        # 1. Compress T tokens → M slots
        slots = self.compress(h0)   # [B, M, d]

        # 2. Run ODE dynamics on slots (M=32, not T=512)
        #    Mask is None — microcircuits don't have token-causal structure
        self.dynamics.set_context(context, mask=None)
        self.dynamics.set_n_steps(self.n_ode_steps)
        slots_evolved = euler_solve_fn(
            self.dynamics, slots, t_span=(0.0, 2.0),
            n_steps=self.n_ode_steps,
        )
        if isinstance(slots_evolved, tuple):
            slots_evolved = slots_evolved[0]

        # 3. Expand M slots → T tokens via token-queries
        h_out = self.expand(slots_evolved, h0)  # [B, T, d]
        return h_out


class ChunkedMicroCircuitWrapper(nn.Module):
    """Phase 1a.2 — local-first chunked microcircuit substrate.

    Rationale: the global-latent MicroCircuitWrapper violated the
    biological invariant of *local-first integration*. Cortical columns
    process a local spatial patch; long-range information arrives via
    *sparse* routing of column summaries, not a giant global bottleneck.

    This wrapper:
      1. Splits T tokens into M contiguous chunks of L = T/M tokens.
      2. Runs the shared dynamics module INDEPENDENTLY on each chunk
         (batch-parallel via reshape to [B*M, L, d]).
      3. Optionally routes per-chunk summaries to later chunks via
         chunk-level causal attention (one vector per chunk).

    This preserves causality naturally at two levels (token-causal
    within a chunk, chunk-causal across chunks) and tests whether
    local integration + sparse long-range routing is sufficient for NTP.

    If `inter_chunk_routing=False`, this is the pure locality ablation:
    no information crosses chunk boundaries except through shared
    weights. A strong collapse indicates long-range routing is
    load-bearing; survival indicates local structure dominates NTP.
    """

    def __init__(self, d: int, M: int, dynamics: nn.Module,
                 n_ode_steps: int = 16,
                 inter_chunk_routing: bool = True):
        super().__init__()
        self.d = d
        self.M = M
        self.n_ode_steps = n_ode_steps
        self.dynamics = dynamics
        self.inter_chunk_routing = inter_chunk_routing

        if inter_chunk_routing:
            self.summary_q = nn.Linear(d, d, bias=False)
            self.summary_k = nn.Linear(d, d, bias=False)
            self.summary_v = nn.Linear(d, d, bias=False)
            self.summary_o = nn.Linear(d, d, bias=False)
            self.summary_norm = nn.LayerNorm(d)
            with torch.no_grad():
                nn.init.xavier_uniform_(self.summary_q.weight)
                nn.init.xavier_uniform_(self.summary_k.weight)
                nn.init.xavier_uniform_(self.summary_v.weight)
                # Zero-init output so inter-chunk routing starts inactive
                nn.init.xavier_uniform_(self.summary_o.weight, gain=0.01)

    def _route_summaries(self, h_chunks: torch.Tensor,
                         B: int, M: int, L: int) -> torch.Tensor:
        """Route chunk-level summaries via causal attention.

        Input: h_chunks [B*M, L, d] (post-dynamics per chunk).
        Output: residual [B, T, d] to add to the flattened h_out.
        """
        # Summary = last token of each chunk (carries causal state)
        h_4d = h_chunks.view(B, M, L, self.d)
        summaries = h_4d[:, :, -1, :]                 # [B, M, d]

        s_norm = self.summary_norm(summaries)
        q = self.summary_q(s_norm)
        k = self.summary_k(s_norm)
        v = self.summary_v(s_norm)

        # Chunk-level causal mask: chunk m can only attend to 0..m
        d = self.d
        chunk_causal = torch.triu(
            torch.ones(M, M, dtype=torch.bool, device=h_chunks.device),
            diagonal=1,
        )
        routed = F.scaled_dot_product_attention(
            q / (d ** 0.5), k, v, attn_mask=~chunk_causal,
        )                                              # [B, M, d]
        routed = self.summary_o(routed)

        # Broadcast: same routed vector added to every token in the chunk
        routed_bcast = routed.unsqueeze(2).expand(B, M, L, d)
        return routed_bcast.reshape(B, M * L, d)

    def forward(self, h0: torch.Tensor, context: torch.Tensor,
                mask=None, euler_solve_fn=None) -> torch.Tensor:
        """Chunked local dynamics + optional sparse summary routing.

        Args:
            h0: [B, T, d]
            context: [B, d] pooled context (used as fallback if local
                chunk contexts are degenerate — here we recompute per-chunk).
            mask: upstream causal mask (unused — we build a local [L, L]
                mask per chunk).
            euler_solve_fn: function(dynamics, h, t_span, n_steps).

        Returns:
            h_out: [B, T, d]
        """
        B, T, d = h0.shape
        M = self.M
        assert T % M == 0, (
            f"ChunkedMicroCircuit: seq_len T={T} must be divisible by "
            f"M={M} chunks")
        L = T // M

        # Reshape to batch-parallel chunks
        h_chunks = h0.view(B, M, L, d).reshape(B * M, L, d)

        # Per-chunk context: pool each chunk's own tokens
        chunk_context = h_chunks.mean(dim=1)          # [B*M, d]

        # Local token-level causal mask [L, L]
        local_causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=h0.device),
            diagonal=1,
        )

        self.dynamics.set_context(chunk_context, mask=local_causal_mask)
        self.dynamics.set_n_steps(self.n_ode_steps)

        h_evolved = euler_solve_fn(
            self.dynamics, h_chunks,
            t_span=(0.0, 2.0), n_steps=self.n_ode_steps,
        )
        if isinstance(h_evolved, tuple):
            h_evolved = h_evolved[0]

        # Flatten back: [B*M, L, d] → [B, T, d]
        h_out = h_evolved.view(B, M, L, d).reshape(B, T, d)

        if self.inter_chunk_routing:
            h_out = h_out + self._route_summaries(h_evolved, B, M, L)

        return h_out
