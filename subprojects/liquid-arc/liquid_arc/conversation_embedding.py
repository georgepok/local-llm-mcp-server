"""Dual-mode conversation embedding for two-phase ODE processing.

Mode A (token_level): text -> tokens -> embeddings [B, T, d_model]
  Used by Phase 1 to give the ODE token-level input.

Mode B (event_level): encoded_event + metadata -> [B, 1, d_model]
  Used by Phase 2 to combine Phase 1's output with event metadata.

Mode B also supports legacy 384-dim sentence-transformer input for
bootstrap/backward compatibility.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .tokenizer import MindTokenizer


class ConversationEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        content_embed_dim: int = 384,
        n_metadata_features: int = 8,
        n_event_types: int = 10,
        max_events: int = 128,
        max_tokens: int = 64,
        dropout: float = 0.1,
        tokenizer_path: str = None,
    ):
        super().__init__()
        self.d_model = d_model

        # Mode A: Token-level (Mind's own tokenizer)
        self.tokenizer = MindTokenizer(
            d_model=d_model,
            max_tokens=max_tokens,
            tokenizer_path=tokenizer_path,
        )

        # Mode B: Event-level
        # Legacy content projection (384 -> 768) for sentence-transformer compatibility
        self.content_proj = nn.Sequential(
            nn.Linear(content_embed_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Event representation projection (768 -> 768) for Phase 1 output
        self.event_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.metadata_proj = nn.Sequential(
            nn.Linear(n_metadata_features, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

        self.type_embed = nn.Embedding(n_event_types, d_model)
        self.pos_embed = nn.Embedding(max_events, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def embed_tokens(self, text: str, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
        """Mode A: Tokenize and embed text for Phase 1 ODE processing.

        Returns:
            token_embeddings: [1, T, d_model]
            token_mask: [1, T] boolean mask
        """
        token_ids = self.tokenizer.tokenize(text).unsqueeze(0).to(device)
        return self.tokenizer(token_ids)

    def embed_event(
        self,
        encoded_event: torch.Tensor,
        metadata_features: torch.Tensor,
        event_type: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        """Mode B (Phase 1 output): Combine ODE-encoded event with metadata.

        Args:
            encoded_event: [1, d_model] from Phase 1 mean-pooling
            metadata_features: [1, n_metadata]
            event_type: [1] long
            position: [1] long

        Returns: [1, 1, d_model]
        """
        h = (self.event_proj(encoded_event).unsqueeze(1)
             + self.metadata_proj(metadata_features).unsqueeze(1)
             + self.type_embed(event_type).unsqueeze(1)
             + self.pos_embed(position).unsqueeze(1))
        return self.dropout_layer(self.norm(h))

    def forward(
        self,
        content_embeddings: torch.Tensor,
        metadata_features: torch.Tensor,
        event_types: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Mode B (legacy): sentence-transformer 384-dim input.

        Kept for backward compatibility and bootstrap blending.
        """
        h = (self.content_proj(content_embeddings)
             + self.metadata_proj(metadata_features)
             + self.type_embed(event_types)
             + self.pos_embed(positions))
        return self.dropout_layer(self.norm(h))
