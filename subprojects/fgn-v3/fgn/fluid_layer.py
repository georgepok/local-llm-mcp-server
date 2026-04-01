"""FluidLayer — pure geometric computation layer for FluidNet.

No attention. The diffusion kernel derived from learned Riemannian geometry
IS the only information routing mechanism.

Single-shot (n_diffusion_iters=1):
  1. MetricNet(concat(LN(h), ctx)) → g [B,N,d]  per-position metric
  2. geodesic_distance(h, g)       → D² [B,N,N]  pairwise squared distances
  3. TimeNet(LN(h))                → t [B,N,3]   per-position diffusion times
  4. softmax(-D²/(4t+ε))           → K [B,N,N]   per-scale diffusion kernel
  5. K @ V → propagate             → [B,N,d]     three-scale value aggregation
  6. FFN(LN(h))                    → [B,N,d]     local nonlinear transform

Iterative diffusion (n_diffusion_iters>1):
  1. MetricNet(h, ctx) → g          compute metric once from initial h
  2. For k in 1..K:
     a. geodesic_distance(h, g) → D²   recompute distances (h evolves, g fixed)
     b. heat_kernel(D², t)      → K     diffusion kernel
     c. h = h + W_o(K @ V)             residual diffusion step
  3. FFN(h) after convergence

Same metric weights, iterated — geometry and state co-evolve.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig
from .curvature import CurvatureEngine


class FluidLayer(nn.Module):
    def __init__(self, config: FGNConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        d = config.d_model
        d_met = config.d_metric
        n_scales = config.n_scales

        # Pre-norm layers
        self.norm_geo = nn.LayerNorm(d)
        self.norm_time = nn.LayerNorm(d)
        self.norm_val = nn.LayerNorm(d)
        self.norm_ff = nn.LayerNorm(d)

        # MetricNet: concat-based conditioning with context
        # [h || ctx] → bottleneck → metric
        self.metric_net_linear1 = nn.Linear(2 * d, d_met)
        self.metric_net_linear2 = nn.Linear(d_met, d)

        # Init for identity metric at start: Softplus(1.3133) ≈ 1.0
        with torch.no_grad():
            self.metric_net_linear2.bias.fill_(math.log(math.e - 1))
            nn.init.normal_(self.metric_net_linear2.weight, std=0.05)

        # TimeNet: per-position, per-scale diffusion time
        self.time_net_linear1 = nn.Linear(d, d_met)
        self.time_net_linear2 = nn.Linear(d_met, n_scales)

        # Init for well-separated timescales: softplus_inv(0.1), softplus_inv(1.0), softplus_inv(10.0)
        with torch.no_grad():
            t_init = torch.tensor([0.1, 1.0, 10.0])
            bias_init = torch.log(torch.exp(t_init) - 1.0)  # softplus_inv
            self.time_net_linear2.bias.copy_(bias_init)

        # Value projections (3 scales, split d_model)
        d_v_base = d // n_scales  # 85 for d=256, n=3
        d_v_first = d - d_v_base * (n_scales - 1)  # 86 — absorb remainder
        self.d_v_sizes = [d_v_first] + [d_v_base] * (n_scales - 1)

        self.W_v = nn.ModuleList([
            nn.Linear(d, dv, bias=False) for dv in self.d_v_sizes
        ])
        self.W_o = nn.Linear(d, d, bias=False)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d, config.d_ffn_fluid),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ffn_fluid, d),
        )

        # Curvature (diagnostic)
        self.curvature_engine = CurvatureEngine()

        # Dropout
        self.resid_drop = nn.Dropout(config.dropout)

        # Chunking threshold
        self.chunk_size = 256
        self.n_scales = n_scales
        self.n_diffusion_iters = getattr(config, 'n_diffusion_iters', 1)
        self.diffusion_floor = getattr(config, 'diffusion_floor', 0.0)

    def reinit_special(self):
        """Re-apply special initialization after model._init_weights zeroes biases."""
        with torch.no_grad():
            # Identity metric init: Softplus(1.3133) ≈ 1.0
            self.metric_net_linear2.bias.fill_(math.log(math.e - 1))
            nn.init.normal_(self.metric_net_linear2.weight, std=0.05)
            # Well-separated timescales
            t_init = torch.tensor([0.1, 1.0, 10.0])
            bias_init = torch.log(torch.exp(t_init) - 1.0)
            self.time_net_linear2.bias.copy_(bias_init)

    def get_current_metric(
        self, h: torch.Tensor, context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute metric field without full forward pass. For structural energy."""
        B, N, d = h.shape
        h_normed = self.norm_geo(h)
        ctx_exp = context.unsqueeze(1).expand(B, N, d)
        cat_input = torch.cat([h_normed, ctx_exp], dim=-1)
        g = F.softplus(self.metric_net_linear2(
            F.gelu(self.metric_net_linear1(cat_input))
        ))
        return g

    def forward(
        self,
        h: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with optional iterative diffusion.

        Args:
            h: [B, N, d] hidden states
            context: [B, d] episode-level context from ContextPool
            mask: [N, N] causal mask (True = masked future positions)

        Returns:
            (h, kappa, metric_cv, t_avg)
            h: [B, N, d] updated hidden states
            kappa: [B, N] scalar curvature per position
            metric_cv: tensor, coefficient of variation of metric
            t_avg: [B, 3] average diffusion times per scale
        """
        B, N, d = h.shape

        # 1. Compute metric once from initial h (shared across iterations)
        h_normed = self.norm_geo(h)
        ctx_exp = context.unsqueeze(1).expand(B, N, d)
        cat_input = torch.cat([h_normed, ctx_exp], dim=-1)  # [B, N, 2d]
        g = F.softplus(self.metric_net_linear2(
            F.gelu(self.metric_net_linear1(cat_input))
        ))  # [B, N, d]

        # 2. Curvature (diagnostic only, from initial metric)
        kappa = self.curvature_engine(g)  # [B, N]
        metric_cv = g.std() / (g.mean() + 1e-8)  # keep as tensor (no .item())

        # 3. Per-position diffusion times (computed once)
        t = F.softplus(self.time_net_linear2(
            F.gelu(self.time_net_linear1(self.norm_time(h)))
        ))  # [B, N, n_scales]
        t_avg = t.mean(dim=1)  # [B, n_scales] for logging

        # 4. Iterative diffusion: same g and t, h evolves each iteration
        #    D² is recomputed each iteration from evolving h
        eps = 1e-6
        for _iter in range(self.n_diffusion_iters):
            # Recompute distances from current h (geometry fixed, state evolves)
            h_for_dist = self.norm_geo(h) if _iter > 0 else h_normed
            if N <= self.chunk_size:
                D_sq = self._direct_distance(h_for_dist, g)
            else:
                D_sq = self._chunked_distance(h_for_dist, g)

            # Three diffusion kernels from same geometry
            kernels = []
            for s in range(self.n_scales):
                t_s = t[:, :, s:s+1]  # [B, N, 1]
                log_K = -D_sq / (4.0 * t_s + eps)  # [B, N, N]
                if mask is not None:
                    log_K = log_K.masked_fill(mask.unsqueeze(0), float('-inf'))
                K = F.softmax(log_K, dim=-1)
                if self.diffusion_floor > 0:
                    alpha = self.diffusion_floor
                    K = (1.0 - alpha) * K + alpha / N
                kernels.append(K)

            # Value projections (recomputed from current h each iteration)
            h_val = self.norm_val(h)
            values = [wv(h_val) for wv in self.W_v]

            # Propagate + concat + project
            propagated = torch.cat(
                [K @ V for K, V in zip(kernels, values)],
                dim=-1,
            )  # [B, N, d]
            h = h + self.resid_drop(self.W_o(propagated))

        # 5. FFN (after diffusion converges)
        h = h + self.resid_drop(self.ffn(self.norm_ff(h)))

        return h, kappa, metric_cv, t_avg

    def _direct_distance(
        self, h: torch.Tensor, g: torch.Tensor,
    ) -> torch.Tensor:
        """Direct O(N^2) geodesic distance computation.

        D²[i,j] = sum_k((h_i^k - h_j^k)^2 * g_avg_ij^k)

        Causal mask is NOT applied here — it's applied to log_K in forward()
        to avoid inf gradients.
        """
        diff = h.unsqueeze(2) - h.unsqueeze(1)          # [B, N, N, d]
        g_avg = (g.unsqueeze(2) + g.unsqueeze(1)) / 2.0  # [B, N, N, d]
        D_sq = (diff * diff * g_avg).sum(-1)              # [B, N, N]
        return D_sq

    def _chunked_distance(
        self, h: torch.Tensor, g: torch.Tensor,
    ) -> torch.Tensor:
        """Memory-efficient chunked geodesic distance for long sequences."""
        B, N, _ = h.shape
        C = self.chunk_size
        D_sq = torch.zeros(B, N, N, device=h.device, dtype=h.dtype)

        for q_start in range(0, N, C):
            q_end = min(q_start + C, N)
            h_q = h[:, q_start:q_end]  # [B, C, d]
            g_q = g[:, q_start:q_end]  # [B, C, d]

            diff = h_q.unsqueeze(2) - h.unsqueeze(1)       # [B, C, N, d]
            g_avg = (g_q.unsqueeze(2) + g.unsqueeze(1)) / 2.0
            chunk_D = (diff * diff * g_avg).sum(-1)          # [B, C, N]
            D_sq[:, q_start:q_end] = chunk_D

        return D_sq


if __name__ == "__main__":
    from .config import FGNConfig

    print("Testing FluidLayer...")
    cfg = FGNConfig(d_model=64, n_heads=4, d_ff=256, n_layers=6,
                    vocab_size=100, max_seq_len=32,
                    d_metric=16, d_ffn_fluid=128, n_scales=3,
                    architecture_version="fluid")
    layer = FluidLayer(cfg, layer_idx=0)

    B, N = 2, 16
    h = torch.randn(B, N, 64)
    ctx = torch.randn(B, 64)
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)

    # Forward pass
    h_out, kappa, metric_cv, t_avg = layer(h, ctx, mask=mask)
    assert h_out.shape == (B, N, 64), f"h shape: {h_out.shape}"
    assert kappa.shape == (B, N), f"kappa shape: {kappa.shape}"
    assert t_avg.shape == (B, 3), f"t_avg shape: {t_avg.shape}"
    print(f"  h_out shape: {h_out.shape}")
    print(f"  kappa shape: {kappa.shape}, mean |k|={kappa.abs().mean():.4f}")
    print(f"  metric_cv: {metric_cv:.4f}")
    print(f"  t_avg: {t_avg[0].tolist()}")

    # Gradient flow
    loss = h_out.sum()
    loss.backward()
    for name, p in layer.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"No grad for {name}"
    print("  Gradient flow: OK")

    # Test chunked path
    layer2 = FluidLayer(cfg, layer_idx=0)
    layer2.chunk_size = 8
    h_chunked, _, _, _ = layer2(h, ctx, mask=mask)
    assert h_chunked.shape == (B, N, 64)
    print("  Chunked path: OK")

    # Test iterative diffusion
    cfg3 = FGNConfig(d_model=64, n_heads=4, d_ff=256, n_layers=6,
                     vocab_size=100, max_seq_len=32,
                     d_metric=16, d_ffn_fluid=128, n_scales=3,
                     architecture_version="fluid", n_diffusion_iters=3)
    layer3 = FluidLayer(cfg3, layer_idx=0)
    h3 = torch.randn(B, N, 64)
    h3_out, k3, cv3, t3 = layer3(h3, ctx, mask=mask)
    assert h3_out.shape == (B, N, 64)
    loss3 = h3_out.sum()
    loss3.backward()
    for name, p in layer3.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"No grad for {name} in iterative mode"
    print(f"  Iterative diffusion (K=3): OK, |k|={k3.abs().mean():.4f}")

    # Verify timescale initialization
    t_init = F.softplus(layer.time_net_linear2.bias)
    print(f"  Initial timescales: {t_init.tolist()}")
    assert t_init[0] < t_init[1] < t_init[2], "Timescales should be ordered"

    # Value split verification
    total_v = sum(layer.d_v_sizes)
    assert total_v == 64, f"Value dims don't sum to d_model: {total_v}"
    print(f"  Value split: {layer.d_v_sizes} (sum={total_v})")

    n_params = sum(p.numel() for p in layer.parameters())
    print(f"  Parameters: {n_params:,}")

    print("FluidLayer OK")
