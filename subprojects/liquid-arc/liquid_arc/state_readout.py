"""State readout — project ODE state into outputs for Claude.

Produces:
  1. Relevance scores: per-event relevance to current focus [N]
  2. State summary: compressed current understanding [d_summary]
  3. Goal status: per-goal completion/priority [n_goals, 2]
  4. Focus indices: top-K events Claude should attend to [K]

These are PROJECTIONS — they don't modify h.
"""

import torch
import torch.nn as nn


class StateReadout(nn.Module):
    def __init__(self, d_model: int = 768, d_summary: int = 256, max_events: int = 128):
        super().__init__()

        self.relevance_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        self.summary_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_summary),
        )

        self.goal_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2),
            nn.Sigmoid(),
        )

        self.focus_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, h: torch.Tensor, event_types: torch.Tensor):
        B, N, d = h.shape

        relevance = self.relevance_head(h).squeeze(-1)
        relevance_scores = torch.sigmoid(relevance)

        query = self.summary_query.expand(B, -1, -1)
        attn_weights = torch.bmm(query, h.transpose(1, 2)) / (d ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        summary_state = torch.bmm(attn_weights, h).squeeze(1)
        summary = self.summary_proj(summary_state)

        goal_status = self.goal_head(h)

        focus_scores = self.focus_head(h).squeeze(-1)
        top_k = min(5, N)
        focus_values, focus_indices = torch.topk(focus_scores, top_k, dim=-1)

        return {
            'relevance_scores': relevance_scores,
            'summary': summary,
            'goal_status': goal_status,
            'focus_indices': focus_indices,
            'focus_scores': torch.sigmoid(focus_values),
        }
