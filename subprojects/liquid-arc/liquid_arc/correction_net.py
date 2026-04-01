"""Output correction network for LiquidARC.

Small trainable head that refines frozen base model predictions.
Operates on detached hidden states + base prediction embeddings.
Correction output initialized to zeros — starts as identity (no change).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputCorrectionNet(nn.Module):
    """Additive correction on top of frozen base model logits.

    final_logits = base_logits.detach() + correction_logits

    Args:
        d_model: hidden dimension of base model
        n_colors: number of output classes (10 for ARC)
    """

    def __init__(self, d_model=256, n_colors=10):
        super().__init__()
        self.pred_embed = nn.Embedding(n_colors + 1, d_model)  # +1 for pad
        self.fc1 = nn.Linear(d_model * 2, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc3 = nn.Linear(d_model, n_colors)
        # Zero-init output — starts with no correction
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, hidden_states, base_predictions):
        """Compute correction logits.

        Args:
            hidden_states: [B, N, d] detached ODE output from base model
            base_predictions: [B, N] argmax of base logits

        Returns:
            [B, N, n_colors] correction logits (add to base_logits)
        """
        pred_emb = self.pred_embed(base_predictions.clamp(0, self.pred_embed.num_embeddings - 1))
        x = torch.cat([hidden_states, pred_emb], dim=-1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)
