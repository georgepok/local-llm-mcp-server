"""Action output head for robotics control.

Reads final hidden states from actuated entity tokens and projects
each to its action dimension. Zero-init for small initial actions.
"""

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    """Project entity hidden states to continuous action space."""

    def __init__(self, d_model: int = 768, action_dim: int = 1, n_actuated: int = 1):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.n_actuated = n_actuated

        self.action_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, action_dim // n_actuated),
        )

        # Default: zero-init for stable startup (ARC, cartpole)
        # Call init_exploratory() for locomotion discovery
        nn.init.zeros_(self.action_proj[-1].weight)
        nn.init.zeros_(self.action_proj[-1].bias)

    def init_exploratory(self, scale: float = 0.5):
        """Initialize action head with larger weights for locomotion discovery.

        Zero-init produces near-zero actions — safe for ARC but prevents
        locomotion discovery. This initializes with Xavier-scale weights so
        the model starts with diverse action outputs, like an MLP baseline.
        """
        nn.init.xavier_normal_(self.action_proj[-1].weight, gain=scale)
        nn.init.zeros_(self.action_proj[-1].bias)

    def forward(
        self,
        h_final: torch.Tensor,
        actuated_indices: torch.Tensor,
    ) -> torch.Tensor:
        B = h_final.shape[0]

        if actuated_indices.dim() == 1:
            actuated_indices = actuated_indices.unsqueeze(0).expand(B, -1)

        h_actuated = torch.gather(
            h_final, 1,
            actuated_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        )

        actions_per_entity = self.action_proj(h_actuated)
        actions = actions_per_entity.reshape(B, -1)

        return actions
