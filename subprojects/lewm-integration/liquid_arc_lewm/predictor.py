"""LiquidARCPredictor — drop-in replacement for LeWM's ARPredictor.

Signature matches upstream ARPredictor exactly:
    forward(emb[B, T, D], act_emb[B, T, A]) -> [B, T, D]

The replacement trades flat causal self-attention for an ODE on a learned
Riemannian manifold. Each timestep is a node; the causal mask enforces
past->present information flow; the heat kernel K = softmax(-D²_g / 4t)
routes information along geodesics of the learned metric g(h, action).

Action conditioning: mean-pool action embeddings over the history window,
project to d_model, pass as ContinuousDynamics.set_context. MetricNet sees
action as global context → different actions produce different metrics.
Per-position action conditioning is a future ablation (requires extending
set_context, which today assumes a single [B, D] context vector).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from liquid_arc.config import LiquidARCConfig
from liquid_arc.dynamics import ContinuousDynamics
from liquid_arc.solver import euler_solve


class LiquidARCPredictor(nn.Module):
    def __init__(self, input_dim: int, action_emb_dim: int,
                 ode_config: LiquidARCConfig, output_dim: int | None = None,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.action_emb_dim = action_emb_dim
        self.cfg = ode_config
        d_model = ode_config.d_model

        self.dynamics = ContinuousDynamics(ode_config)

        self.proj_in = (nn.Linear(input_dim, d_model)
                        if input_dim != d_model else nn.Identity())
        self.proj_out = (nn.Linear(d_model, self.output_dim)
                         if self.output_dim != d_model else nn.Identity())
        self.action_proj = nn.Linear(action_emb_dim, d_model)

        self.dropout = nn.Dropout(dropout)
        self._causal_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular bool mask, True = BLOCKED (matches dynamics.py)."""
        key = (T, device)
        m = self._causal_mask_cache.get(key)
        if m is None:
            m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device),
                           diagonal=1)
            self._causal_mask_cache[key] = m
        return m

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        T = emb.shape[1]
        h0 = self.proj_in(emb)  # [B, T, d_model]

        causal = self._causal_mask(T, h0.device)
        action_ctx = self.action_proj(act_emb.mean(dim=1))  # [B, d_model]
        self.dynamics.set_context(action_ctx, mask=causal)

        n_steps = self.cfg.n_ode_steps
        self.dynamics.set_n_steps(n_steps)

        y = euler_solve(
            self.dynamics, h0,
            t_span=(0.0, self.cfg.integration_time),
            n_steps=n_steps,
        )

        return self.dropout(self.proj_out(y))
