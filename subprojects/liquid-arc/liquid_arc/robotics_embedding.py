"""Robotics embedding — continuous state entity-as-token representation.

Replaces ARCEmbedding for robotics tasks. Each rigid body / joint / sensor
in the robot becomes a token with continuous state features and spatial coords.

h = StateProjection(state_features) + SpatialEmbed(x, y, z)
    + TypeEmbed(entity_type) + IdEmbed(entity_id)
    → LayerNorm → Dropout
"""

import torch
import torch.nn as nn


class RoboticsEmbedding(nn.Module):
    """Additive embedding for robotics entity-as-token representation."""

    def __init__(
        self,
        d_model: int = 768,
        max_state_dim: int = 16,
        n_entity_types: int = 8,
        max_entities: int = 32,
        n_spatial_bins: int = 64,
        spatial_range: tuple = (-5.0, 5.0),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_spatial_bins = n_spatial_bins
        self.spatial_range = spatial_range

        # State projection: continuous features → d_model (MLP, not lookup)
        self.state_proj = nn.Sequential(
            nn.Linear(max_state_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Spatial embeddings (discretized world coordinates)
        self.spatial_x_embed = nn.Embedding(n_spatial_bins, d_model)
        self.spatial_y_embed = nn.Embedding(n_spatial_bins, d_model)
        self.spatial_z_embed = nn.Embedding(n_spatial_bins, d_model)

        # Entity type and ID embeddings
        self.type_embed = nn.Embedding(n_entity_types, d_model)
        self.id_embed = nn.Embedding(max_entities, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _discretize_spatial(self, coords: torch.Tensor) -> torch.Tensor:
        """Bin continuous spatial coordinates into discrete indices."""
        low, high = self.spatial_range
        normalized = (coords - low) / (high - low)
        binned = (normalized * (self.n_spatial_bins - 1)).clamp(0, self.n_spatial_bins - 1).long()
        return binned

    def forward(
        self,
        state_features: torch.Tensor,
        spatial_x: torch.Tensor,
        spatial_y: torch.Tensor,
        spatial_z: torch.Tensor,
        entity_types: torch.Tensor,
        entity_ids: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, N, D = state_features.shape
        max_dim = self.state_proj[0].in_features
        if D < max_dim:
            pad = torch.zeros(B, N, max_dim - D, device=state_features.device)
            state_features = torch.cat([state_features, pad], dim=-1)

        h_state = self.state_proj(state_features)

        x_bins = self._discretize_spatial(spatial_x)
        y_bins = self._discretize_spatial(spatial_y)
        z_bins = self._discretize_spatial(spatial_z)

        h = (h_state
             + self.spatial_x_embed(x_bins)
             + self.spatial_y_embed(y_bins)
             + self.spatial_z_embed(z_bins)
             + self.type_embed(entity_types)
             + self.id_embed(entity_ids))

        h = self.norm(h)
        h = self.dropout(h)

        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return h
