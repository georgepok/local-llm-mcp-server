"""Persistent ODE state — temporal continuity across forward passes.

Stores h_final from each forward pass and blends it into h₀ of the next.
Operates ENTIRELY OUTSIDE the compiled ODE graph — no recompilation triggered.

The LTC contraction guarantee bounds temporal drift:
  At tau=0.65, information decays by exp(-1/0.65) per unit time.
  After ~5τ, old state is attenuated by >95%.
"""

import torch
import torch.nn as nn
from typing import Optional


class PersistentState(nn.Module):
    """Manages temporal state persistence across forward passes.

    Usage in model.forward():
        h0 = self.embedding(...)
        h0 = self.persistent.blend(h0)
        h_final = euler_solve(...)
        self.persistent.store(h_final)
    """

    def __init__(self, alpha: float = 0.7, learnable_alpha: bool = False):
        super().__init__()

        if learnable_alpha:
            self._alpha_logit = nn.Parameter(torch.tensor(0.85))
        else:
            self.register_buffer('_alpha_fixed', torch.tensor(alpha))

        self.learnable_alpha = learnable_alpha
        self._h_prev: Optional[torch.Tensor] = None
        self._active = True

    @property
    def alpha(self) -> torch.Tensor:
        if self.learnable_alpha:
            return torch.sigmoid(self._alpha_logit)
        return self._alpha_fixed

    def blend(self, h_new: torch.Tensor) -> torch.Tensor:
        """Blend fresh embedding with stored state."""
        if not self._active or self._h_prev is None:
            return h_new

        if self._h_prev.shape[0] != h_new.shape[0] or self._h_prev.shape[1] != h_new.shape[1]:
            self._h_prev = None
            return h_new

        # Normalize h_prev to match h_new's global scale before blending.
        # ODE output norm (~4000) would otherwise drown fresh embedding (~50-100).
        h_prev_scaled = self._h_prev * (h_new.norm() / (self._h_prev.norm() + 1e-8))

        alpha = self.alpha
        return alpha * h_new + (1.0 - alpha) * h_prev_scaled

    def store(self, h_final: torch.Tensor) -> None:
        """Store h_final for next forward pass. Always detached."""
        if self._active:
            self._h_prev = h_final.detach()

    def reset(self) -> None:
        """Clear stored state (e.g., at episode boundary)."""
        self._h_prev = None

    def set_active(self, active: bool) -> None:
        """Enable/disable persistence."""
        self._active = active
        if not active:
            self._h_prev = None

    def get_diagnostics(self) -> dict:
        diag = {
            'persist_alpha': self.alpha.item(),
            'persist_active': self._active,
            'persist_has_state': self._h_prev is not None,
        }
        if self._h_prev is not None:
            diag['persist_h_norm'] = self._h_prev.norm().item()
        return diag
