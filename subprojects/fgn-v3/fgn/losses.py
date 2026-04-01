"""Curvature regularization losses with learnable correlation length.

Phase 1a.1: Flipped regularization to REWARD non-trivial curvature.

Loss = -mu * mean(|kappa|)        [curvature reward — encourages geometry]
     + eta * ||grad(kappa)||^2     [smoothness — keeps geometry structured]

The curvature reward (-mu * mean(|kappa|)) gives the metric direct gradient
signal to develop non-trivial geometry. Without this, Q/K/V projections claim
the solution space before the slow metric can contribute.

The smoothness penalty (eta * ||grad(kappa)||^2) prevents the metric from
developing noisy, unstructured curvature to cheaply satisfy the reward.

Per-layer learnable parameters control the balance.
"""

import math
from typing import List

import torch
import torch.nn as nn


class CurvatureRegularization(nn.Module):
    def __init__(self, n_layers: int, curvature_lambda: float = 0.01,
                 correlation_length_init: float = 5.0,
                 curvature_reward_mu: float = 0.0):
        """Initialize curvature regularization.

        Args:
            n_layers: number of transformer layers
            curvature_lambda: variance penalty weight (legacy, kept for compat)
            correlation_length_init: initial correlation length
            curvature_reward_mu: weight for -mu*mean(|kappa|) reward term.
                If > 0, enables curvature reward mode (Phase 1a.1).
                If 0, falls back to legacy variance penalty (Phase 1a).
        """
        super().__init__()
        self.curvature_reward_mu = curvature_reward_mu
        self.disabled = (curvature_lambda == 0.0 and curvature_reward_mu == 0.0)

        if curvature_lambda > 0:
            log_lambda = math.log(curvature_lambda)
            log_eta = math.log(curvature_lambda * correlation_length_init ** 2)
        else:
            # lambda=0: no smoothness or variance penalty
            log_lambda = 0.0
            log_eta = 0.0

        self.log_lambda = nn.ParameterList([
            nn.Parameter(torch.tensor(log_lambda)) for _ in range(n_layers)
        ])
        self.log_eta = nn.ParameterList([
            nn.Parameter(torch.tensor(log_eta)) for _ in range(n_layers)
        ])

    def forward(self, curvatures: List[torch.Tensor]) -> torch.Tensor:
        """Compute regularization loss from per-layer curvatures.

        Args:
            curvatures: list of [B, N] tensors, one per layer

        Returns:
            Scalar regularization loss
        """
        total = torch.tensor(0.0, device=curvatures[0].device)

        if self.disabled:
            return total

        for i, kappa in enumerate(curvatures):
            eta = self.log_eta[i].exp()

            # Smoothness penalty: keeps curvature structured
            dk = kappa[:, 1:] - kappa[:, :-1]
            grad_sq = (dk * dk).mean()
            total = total + eta * grad_sq

            if self.curvature_reward_mu > 0:
                # Phase 1a.1: REWARD non-trivial curvature (SATURATING)
                # tanh prevents runaway: reward plateaus once |kappa| >> 1
                # At |kappa|=1: tanh(1)=0.76 (strong reward)
                # At |kappa|=3: tanh(3)=0.995 (near max, diminishing returns)
                total = total - self.curvature_reward_mu * torch.tanh(kappa.abs()).mean()
            else:
                # Legacy Phase 1a: penalize curvature variance
                lam = self.log_lambda[i].exp()
                total = total + lam * kappa.var()

        return total

    def correlation_lengths(self) -> List[float]:
        """Return current correlation length per layer."""
        lengths = []
        for i in range(len(self.log_lambda)):
            lam = self.log_lambda[i].exp().item()
            eta = self.log_eta[i].exp().item()
            lengths.append(math.sqrt(eta / lam))
        return lengths


if __name__ == "__main__":
    # Test legacy mode
    reg = CurvatureRegularization(n_layers=6)
    lengths = reg.correlation_lengths()
    print(f"Initial correlation lengths: {[f'{l:.2f}' for l in lengths]}")
    assert all(abs(l - 5.0) < 0.1 for l in lengths), "Should init to ~5.0"

    curvatures = [torch.randn(4, 32) for _ in range(6)]
    loss = reg(curvatures)
    assert loss.shape == ()
    loss.backward()
    for p in reg.parameters():
        assert p.grad is not None
    print("CurvatureRegularization (legacy) OK")

    # Test reward mode
    reg2 = CurvatureRegularization(n_layers=6, curvature_reward_mu=0.01)
    curvatures2 = [torch.randn(4, 32) for _ in range(6)]
    loss2 = reg2(curvatures2)
    assert loss2.shape == ()

    # Reward mode should produce NEGATIVE loss component when curvature exists
    curvatures_large = [torch.randn(4, 32) * 5.0 for _ in range(6)]
    curvatures_small = [torch.randn(4, 32) * 0.01 for _ in range(6)]
    loss_large = reg2(curvatures_large)
    loss_small = reg2(curvatures_small)
    assert loss_large < loss_small, "Larger curvature should give lower loss in reward mode"
    print("CurvatureRegularization (reward mode) OK")
