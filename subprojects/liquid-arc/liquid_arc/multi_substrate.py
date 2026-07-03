"""Multi-substrate ContinuousDynamics — K parallel dynamics with lateral
coupling at each ODE step.

Validated mechanism (research/self_org_sim/multi_substrate_toy.py):
    K=1 isolated MSE 0.32 → K=2 coupled 3-step MSE 0.09 (71% reduction)
    Substrates differentiate (cos_sim=0.38, ablation_spread=1.58)
    Coupling + multiple inner iterations are both required.

LiquidARC port: K parallel ContinuousDynamics, each at d_model wide, with
each receiving a lateral context augmentation (mean of other substrates'
hidden states) at every ODE step. The solver sees a SINGLE module that
takes/returns concatenated states of shape [B, N, K*d].
"""

from typing import Optional

import torch
import torch.nn as nn

from .config import LiquidARCConfig
from .dynamics import ContinuousDynamics


class MultiSubstrateDynamics(nn.Module):
    """K parallel weight-untied ContinuousDynamics with lateral coupling.

    State carried by the solver: h_combined [B, N, K*d_per]
        — concatenation of K per-substrate states
    Each forward call:
        1. Split h_combined into K substrate states [B, N, d_per]
        2. Compute per-substrate lateral context: mean of OTHERS' h
        3. For each substrate k: augment its context with lateral, run dynamics
        4. Concatenate the K dh values back to [B, N, K*d_per]

    The substrate's MetricNet/W_o etc. are independent across substrates —
    self-organization happens through differential gradient flow (each
    substrate's contribution to the output gets its own gradient signal).
    """

    def __init__(self, base_config: LiquidARCConfig, K: int = 2,
                 lateral_weight: float = 0.5):
        super().__init__()
        self.K = K
        self.d_per = base_config.d_model
        self.lateral_weight = lateral_weight
        # K independent ContinuousDynamics — share template, separate parameters.
        self.substrates = nn.ModuleList([
            ContinuousDynamics(base_config) for _ in range(K)
        ])
        # Default halting flag (mirrors single substrate so solver detects)
        self.halting_enabled = base_config.halting_enabled

    def set_context(self, context: torch.Tensor,
                     mask: Optional[torch.Tensor] = None):
        """Distribute base context to all substrates. Lateral augmentation
        added per-step in forward()."""
        for sub in self.substrates:
            sub.set_context(context, mask=mask)
        self._base_context = context
        self._mask = mask

    def set_n_steps(self, n_steps: int):
        for sub in self.substrates:
            sub.set_n_steps(n_steps)

    def set_step_index(self, step_idx: int, n_steps: int):
        for sub in self.substrates:
            if hasattr(sub, 'set_step_index'):
                sub.set_step_index(step_idx, n_steps)

    def set_step_embed(self, step_idx: int, n_steps: int):
        for sub in self.substrates:
            if hasattr(sub, 'set_step_embed'):
                sub.set_step_embed(step_idx, n_steps)

    def reset_fast_weights(self, B, device, dtype):
        for sub in self.substrates:
            if hasattr(sub, 'reset_fast_weights'):
                sub.reset_fast_weights(B, device, dtype)

    def reset_id_history(self, B, N, device, dtype):
        for sub in self.substrates:
            if hasattr(sub, 'reset_id_history'):
                sub.reset_id_history(B, N, device, dtype)

    def forward(self, t_ode, h_combined: torch.Tensor):
        """Compute combined dh/dt from K substrate dynamics + lateral coupling.

        Args:
            t_ode: scalar (unused — autonomous)
            h_combined: [B, N, K*d_per]

        Returns:
            If halting enabled: (dh_combined, p_halt_mean)
                dh_combined: [B, N, K*d_per]
                p_halt_mean: [B, N, 1] — mean halt probability across substrates
            Else: dh_combined
        """
        B, N, total_d = h_combined.shape
        assert total_d == self.K * self.d_per, \
            f"h_combined dim {total_d} != K*d_per ({self.K * self.d_per})"
        h_list = list(h_combined.split(self.d_per, dim=-1))
        # h_stack: [K, B, N, d_per]
        h_stack = torch.stack(h_list, dim=0)
        if self.K > 1:
            sum_all = h_stack.sum(dim=0, keepdim=True)         # [1, B, N, d]
            others_sum = sum_all - h_stack                      # [K, B, N, d]
            others_mean = others_sum / float(self.K - 1)        # [K, B, N, d]
        else:
            others_mean = torch.zeros_like(h_stack)

        dh_list = []
        p_halt_list = []
        base_ctx = self._base_context
        for k, sub in enumerate(self.substrates):
            # Lateral context augmentation: pool others' state across N
            lateral_ctx = others_mean[k].mean(dim=1)            # [B, d_per]
            if base_ctx is not None:
                aug_ctx = base_ctx + self.lateral_weight * lateral_ctx
            else:
                aug_ctx = self.lateral_weight * lateral_ctx
            # Temporarily install augmented context for this substrate's call.
            saved_ctx = sub._context
            sub._context = aug_ctx
            try:
                result = sub(t_ode, h_list[k])
            finally:
                sub._context = saved_ctx
            if isinstance(result, tuple):
                dh_k, p_halt_k = result
                p_halt_list.append(p_halt_k)
            else:
                dh_k = result
            dh_list.append(dh_k)

        dh_combined = torch.cat(dh_list, dim=-1)               # [B, N, K*d_per]
        if p_halt_list:
            # Combine halt probs: average across substrates (any substrate
            # halting indicates the position is "settling")
            p_halt_combined = torch.stack(p_halt_list, dim=0).mean(dim=0)
            return dh_combined, p_halt_combined
        return dh_combined
