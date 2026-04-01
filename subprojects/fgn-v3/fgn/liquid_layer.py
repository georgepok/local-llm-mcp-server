"""LiquidLayer — continuous-time geometric computation via Neural ODE.

True Liquid Time-Constant dynamics:
    dh/dt = -(1/tau(h)) * [h - tanh(W_o(K_g(h) @ W_v(h)))]

This is a contraction toward a diffusion target:
    - K_g(h) = heat kernel from learned metric: softmax(-D^2 / (4*t_diff))
    - The "target" = tanh(W_o(K @ V)) is where diffusion wants h to be
    - tau(h) = input-dependent time constant: softplus(tau_net(h)) + tau_min
    - Small tau => fast snap to target. Large tau => resist change (fading memory).

Integrated via fixed-step Euler with torch.compile. The adjoint method (torchdiffeq)
gives O(1) memory but recomputes O(N^2) distances in the backward ODE — 30x slower.
Euler with 4 steps matches FluidLayer's memory and gives ~500 tok/s.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig
from .curvature import CurvatureEngine
from .ode_solver import euler_solve


class LiquidDynamics(nn.Module):
    """Computes dh/dt for the Liquid Time-Constant ODE.

    True LTC: dh/dt = -(1/tau) * [h - target(h)]
    where target(h) = tanh(W_o(K_g(h) @ W_v(h)))

    This creates exponential decay toward the diffusion target with
    input-dependent time constant. States naturally forget unless
    continuously driven — fading memory.
    """

    def __init__(self, config: FGNConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        d = config.d_model
        d_met = config.d_metric

        # Pre-norm
        self.norm_geo = nn.LayerNorm(d)
        self.norm_val = nn.LayerNorm(d)

        # MetricNet: [LN(h) || ctx] -> bottleneck -> g via Softplus
        self.metric_net_linear1 = nn.Linear(2 * d, d_met)
        self.metric_net_linear2 = nn.Linear(d_met, d)

        # Init for identity metric: Softplus(1.3133) ~ 1.0
        with torch.no_grad():
            self.metric_net_linear2.bias.fill_(math.log(math.e - 1))
            nn.init.normal_(self.metric_net_linear2.weight, std=0.05)

        # TauNet: per-position adaptive time constant
        self.tau_net_linear1 = nn.Linear(d, d_met)
        self.tau_net_linear2 = nn.Linear(d_met, 1)
        self.tau_min = config.tau_min

        # Init tau to softplus_inv(1.0) so initial tau ~ 1.0 + tau_min
        with torch.no_grad():
            self.tau_net_linear2.bias.fill_(math.log(math.e - 1))

        # Single value projection (replaces 3 per-scale in FluidLayer)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)

        # Learned diffusion timescale (single scalar)
        self.t_diffusion = nn.Parameter(torch.tensor(config.t_diffusion_init))

        # Chunking threshold
        self.chunk_size = 256

        # Stored context (set before ODE integration, read during dynamics)
        self._context: Optional[torch.Tensor] = None
        self._mask: Optional[torch.Tensor] = None

    def set_context(self, context: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Set episode context and mask before ODE integration."""
        self._context = context
        self._mask = mask

    def forward(self, t_ode, h: torch.Tensor) -> torch.Tensor:
        """Compute dh/dt at ODE time t_ode.

        True LTC dynamics: dh/dt = -(1/tau) * [h - target(h)]
        where target = tanh(W_o(K @ V))

        Args:
            t_ode: current ODE time (unused — autonomous system)
            h: [B, N, d] current hidden states

        Returns:
            dh_dt: [B, N, d] time derivative
        """
        B, N, d = h.shape
        context = self._context  # [B, d]
        mask = self._mask

        # 1. Compute metric from current h (truly liquid — g evolves with h)
        h_normed = self.norm_geo(h)
        ctx_exp = context.unsqueeze(1).expand(B, N, d)
        cat_input = torch.cat([h_normed, ctx_exp], dim=-1)  # [B, N, 2d]
        g = F.softplus(self.metric_net_linear2(
            F.gelu(self.metric_net_linear1(cat_input))
        ))  # [B, N, d]

        # 2. Geodesic distances
        if N <= self.chunk_size:
            D_sq = self._direct_distance(h_normed, g)
        else:
            D_sq = self._chunked_distance(h_normed, g)

        # 3. Heat kernel (single scale — ODE integration replaces multi-scale)
        t_diff = F.softplus(self.t_diffusion)  # ensure positive
        eps = 1e-6
        log_K = -D_sq / (4.0 * t_diff + eps)  # [B, N, N]
        if mask is not None:
            log_K = log_K.masked_fill(mask.unsqueeze(0), float('-inf'))
        K = F.softmax(log_K, dim=-1)  # [B, N, N]

        # 4. Diffusion target: tanh(W_o(K @ V))
        h_val = self.norm_val(h)
        V = self.W_v(h_val)  # [B, N, d]
        target = torch.tanh(self.W_o(K @ V))  # [B, N, d]

        # 5. Liquid time constant (per-position)
        tau = F.softplus(self.tau_net_linear2(
            F.gelu(self.tau_net_linear1(h_normed))
        )) + self.tau_min  # [B, N, 1]

        # 6. LTC dynamics: decay toward diffusion target
        dh_dt = -(1.0 / tau) * (h - target)

        return dh_dt

    def _direct_distance(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Direct O(N^2) geodesic distance: D^2[i,j] = sum_k((h_i-h_j)^2 * g_avg)."""
        diff = h.unsqueeze(2) - h.unsqueeze(1)           # [B, N, N, d]
        g_avg = (g.unsqueeze(2) + g.unsqueeze(1)) / 2.0  # [B, N, N, d]
        D_sq = (diff * diff * g_avg).sum(-1)               # [B, N, N]
        return D_sq

    def _chunked_distance(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Memory-efficient chunked geodesic distance for long sequences."""
        B, N, _ = h.shape
        C = self.chunk_size
        D_sq = torch.zeros(B, N, N, device=h.device, dtype=h.dtype)

        for q_start in range(0, N, C):
            q_end = min(q_start + C, N)
            h_q = h[:, q_start:q_end]
            g_q = g[:, q_start:q_end]

            diff = h_q.unsqueeze(2) - h.unsqueeze(1)
            g_avg = (g_q.unsqueeze(2) + g.unsqueeze(1)) / 2.0
            chunk_D = (diff * diff * g_avg).sum(-1)
            D_sq[:, q_start:q_end] = chunk_D

        return D_sq


class LiquidLayer(nn.Module):
    """Continuous-time geometric layer: ODE integration + FFN.

    Wraps LiquidDynamics in Euler solver. Same interface as FluidLayer:
    forward(h, context, mask) -> (h, kappa, metric_cv, tau_diagnostic)
    """

    def __init__(self, config: FGNConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        d = config.d_model

        # ODE dynamics
        self.dynamics = LiquidDynamics(config, layer_idx)

        # FFN (after ODE integration)
        self.norm_ff = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, config.d_ffn_fluid),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ffn_fluid, d),
        )

        # Curvature (diagnostic — computed on initial metric)
        self.curvature_engine = CurvatureEngine()

        # Dropout
        self.resid_drop = nn.Dropout(config.dropout)

        # ODE config
        self.n_ode_steps = config.n_ode_steps

    def reinit_special(self):
        """Re-apply special initialization after model._init_weights zeroes biases."""
        with torch.no_grad():
            # Identity metric: Softplus(1.3133) ~ 1.0
            self.dynamics.metric_net_linear2.bias.fill_(math.log(math.e - 1))
            nn.init.normal_(self.dynamics.metric_net_linear2.weight, std=0.05)
            # Tau ~ 1.0 + tau_min
            self.dynamics.tau_net_linear2.bias.fill_(math.log(math.e - 1))

    def forward(
        self,
        h: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: Euler ODE integration + FFN.

        Args:
            h: [B, N, d] hidden states
            context: [B, d] episode-level context
            mask: [N, N] causal mask (True = masked)

        Returns:
            (h, kappa, metric_cv, tau_avg)
            h: [B, N, d] updated hidden states
            kappa: [B, N] scalar curvature per position
            metric_cv: tensor, coefficient of variation of metric
            tau_avg: [B, 1] average tau (replaces t_avg in FluidLayer)
        """
        B, N, d = h.shape

        # Compute initial metric for diagnostics
        h_normed = self.dynamics.norm_geo(h)
        ctx_exp = context.unsqueeze(1).expand(B, N, d)
        cat_input = torch.cat([h_normed, ctx_exp], dim=-1)
        g_init = F.softplus(self.dynamics.metric_net_linear2(
            F.gelu(self.dynamics.metric_net_linear1(cat_input))
        ))

        kappa = self.curvature_engine(g_init)
        metric_cv = g_init.std() / (g_init.mean() + 1e-8)

        # Compute initial tau for diagnostics
        tau_init = F.softplus(self.dynamics.tau_net_linear2(
            F.gelu(self.dynamics.tau_net_linear1(h_normed))
        )) + self.dynamics.tau_min  # [B, N, 1]
        tau_avg = tau_init.mean(dim=1)  # [B, 1]

        # Set context for ODE dynamics
        self.dynamics.set_context(context, mask)

        # Euler integration: 4 steps = 4 distance matrices ≈ FluidLayer's 3 scales
        h_integrated = euler_solve(
            self.dynamics, h, t_span=(0.0, 1.0), n_steps=self.n_ode_steps,
        )

        # Residual connection
        h = h + self.resid_drop(h_integrated - h)

        # FFN
        h = h + self.resid_drop(self.ffn(self.norm_ff(h)))

        return h, kappa, metric_cv, tau_avg


if __name__ == "__main__":
    from .config import FGNConfig

    print("Testing LiquidLayer...")
    cfg = FGNConfig(
        d_model=64, n_heads=4, d_ff=256, n_layers=6,
        vocab_size=100, max_seq_len=32,
        d_metric=16, d_ffn_fluid=128,
        architecture_version="fluid",
        liquid_mode=True, n_ode_steps=4,
        tau_min=0.1, t_diffusion_init=1.0,
    )
    layer = LiquidLayer(cfg, layer_idx=0)

    B, N = 2, 16
    h = torch.randn(B, N, 64)
    ctx = torch.randn(B, 64)
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)

    # Forward pass
    h_out, kappa, metric_cv, tau_avg = layer(h, ctx, mask=mask)
    assert h_out.shape == (B, N, 64), f"h shape: {h_out.shape}"
    assert kappa.shape == (B, N), f"kappa shape: {kappa.shape}"
    assert tau_avg.shape == (B, 1), f"tau_avg shape: {tau_avg.shape}"
    print(f"  h_out shape: {h_out.shape}")
    print(f"  kappa shape: {kappa.shape}, mean |k|={kappa.abs().mean():.4f}")
    print(f"  metric_cv: {metric_cv:.4f}")
    print(f"  tau_avg: {tau_avg[0].tolist()}")

    # Gradient flow
    loss = h_out.sum()
    loss.backward()
    grads_ok = True
    for name, p in layer.named_parameters():
        if p.requires_grad:
            if p.grad is None or p.grad.abs().sum() == 0:
                print(f"  WARNING: No grad for {name}")
                grads_ok = False
    assert grads_ok, "Missing gradients"
    print("  Gradient flow: OK")

    # Test without mask
    layer.zero_grad()
    h2 = torch.randn(B, N, 64)
    h2_out, k2, cv2, tau2 = layer(h2, ctx, mask=None)
    assert h2_out.shape == (B, N, 64)
    print("  No-mask forward: OK")

    # Test chunked path
    layer3 = LiquidLayer(cfg, layer_idx=0)
    layer3.dynamics.chunk_size = 8
    h3 = torch.randn(B, N, 64)
    h3_out, _, _, _ = layer3(h3, ctx, mask=mask)
    assert h3_out.shape == (B, N, 64)
    print("  Chunked path: OK")

    # Verify tau is meaningful (not all at tau_min)
    tau_vals = tau_avg.detach()
    print(f"  tau values: mean={tau_vals.mean():.4f}, min={tau_vals.min():.4f}, max={tau_vals.max():.4f}")
    assert tau_vals.mean() > cfg.tau_min, f"Tau should be > tau_min={cfg.tau_min}"

    # Verify |kappa| is bounded
    print(f"  |kappa| max: {kappa.abs().max():.4f}")

    n_params = sum(p.numel() for p in layer.parameters())
    print(f"  Parameters: {n_params:,}")

    print("LiquidLayer OK")
