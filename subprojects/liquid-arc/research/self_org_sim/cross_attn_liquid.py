"""Cross-Attention Hidden Liquid: adds content-comparison primitive between
LLM hidden state and goal tokens, then evolves via Liquid dynamics.

The missing piece for cross-category generalization: a goal-content-agnostic
primitive that compares 'what is the model doing' against 'what was asked.'
Cross-attention provides this — it works for any (hidden_state, goal_tokens)
regardless of goal type.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
from liquid_arc.config import LiquidARCConfig  # type: ignore
from liquid_arc.dynamics import ContinuousDynamics  # type: ignore
from liquid_arc.context_pool import ContextPool  # type: ignore
from liquid_arc.solver import euler_solve_halting  # type: ignore


def make_ca_config(d=128):
    return LiquidARCConfig(
        d_model=d, d_metric=16, d_ffn=192, max_seq_len=8,
        n_ode_steps=3, ode_steps_min=2, ode_steps_max=4,
        integration_time=0.5,
        tau_min=0.3, tau_max=2.0, t_diffusion_init=0.5,
        routing_mode="metric",
        tau_freeze_steps=500,
        halting_enabled=True, halting_min_steps=1,
        halting_ponder_lambda=0.0001,
        rezero_enabled=True, rezero_gate_init=-3.0,
        metric_bias_init_std=0.1,
        deep_supervision_enabled=False, ponder_kl_lambda=0.0,
        criticality_loss_enabled=False,
        curvature_diversity_loss_enabled=True,
        curvature_diversity_lambda=0.0001,
    )


class CrossAttnHiddenLiquid(nn.Module):
    """Liquid substrate with cross-attention goal-content-comparison primitive.

    Per chunk:
      1. hidden_state (Qwen activation) attends over goal_tokens
      2. Cross-attention output = goal-aware view of current activation
      3. Concat (hidden_state, cross_attn_output) → Liquid input
      4. Liquid evolves h_fast
      5. Value head V(h_fast, z_goal_pooled) → P(committed)
    """

    def __init__(self, d_hidden=1536, d_goal=384, d=128, K=4,
                 d_attn=128, n_heads=4, value_hidden=256):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_goal = d_goal
        self.d = d
        self.K = K
        self.d_attn = d_attn
        self.n_heads = n_heads
        self.config = make_ca_config(d=d)

        # Cross-attention: query=hidden_state, keys/values=goal_tokens
        assert d_attn % n_heads == 0, f"d_attn={d_attn} not divisible by n_heads={n_heads}"
        self.head_dim = d_attn // n_heads
        self.hidden_layernorm = nn.LayerNorm(d_hidden, elementwise_affine=True)
        self.goal_layernorm = nn.LayerNorm(d_goal, elementwise_affine=True)
        self.W_q = nn.Linear(d_hidden, d_attn)
        self.W_k = nn.Linear(d_goal, d_attn)
        self.W_v = nn.Linear(d_goal, d_attn)
        self.out_proj = nn.Linear(d_attn, d_attn)
        # Initialization
        for m in (self.W_q, self.W_k, self.W_v, self.out_proj):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

        # Liquid input projection: hidden_state (pooled) + cross_attn_output
        # We project hidden_state through a small linear too
        self.hidden_proj = nn.Linear(d_hidden, d_attn)
        nn.init.normal_(self.hidden_proj.weight, std=0.02)
        nn.init.zeros_(self.hidden_proj.bias)
        # Combined: d_attn (cross_attn) + d_attn (hidden_proj) = 2*d_attn → d
        self.in_combined = nn.Linear(2 * d_attn, d)
        nn.init.normal_(self.in_combined.weight, std=0.02)
        nn.init.zeros_(self.in_combined.bias)

        # Initial belief
        self.init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.init_belief, std=0.05)

        # Position weighting for evidence
        self.evidence_mix = nn.Parameter(torch.ones(K, 1))
        with torch.no_grad():
            self.evidence_mix[0] = 2.0
            self.evidence_mix[1] = 1.5
            self.evidence_mix[2] = 1.5
            self.evidence_mix[3] = 1.0

        # Liquid core
        self.context_pool = ContextPool(self.config)
        self.dynamics = ContinuousDynamics(self.config)

        # Stability
        self.evidence_layernorm = nn.LayerNorm(d, elementwise_affine=True)
        self.h_input_clamp = 50.0

        # SLOW CHANNEL: commitment tracker, same as before
        self.K_slow = K
        self.d_slow = d
        self.slow_in_goal = nn.Linear(d_goal, d)
        nn.init.normal_(self.slow_in_goal.weight, std=0.02)
        nn.init.zeros_(self.slow_in_goal.bias)
        self.slow_layernorm = nn.LayerNorm(d, elementwise_affine=True)
        self.slow_init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.slow_init_belief, std=0.05)
        self.slow_alpha = nn.Parameter(torch.tensor(0.05))
        self.trigger_boost = 8.0
        self.head_trigger = nn.Sequential(
            nn.Linear(d_goal, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
        with torch.no_grad():
            self.head_trigger[-1].weight.mul_(0.01)
            self.head_trigger[-1].bias.fill_(-2.0)

        # VALUE HEAD reads h_fast + cross_attn_output (goal-aware) for goal-tracking signal
        # Include cross_attn_output gives V direct access to "is this aligned"
        self.head_value = nn.Sequential(
            nn.Linear(K * d + d_attn, value_hidden),
            nn.SiLU(), nn.LayerNorm(value_hidden),
            nn.Dropout(0.2),
            nn.Linear(value_hidden, value_hidden),
            nn.SiLU(), nn.LayerNorm(value_hidden),
            nn.Dropout(0.2),
            nn.Linear(value_hidden, 1),
        )

        # JEPA self-prediction (regularizer)
        self.head_jepa = nn.Sequential(
            nn.Linear(K * d, value_hidden),
            nn.SiLU(), nn.LayerNorm(value_hidden),
            nn.Linear(value_hidden, K * d),
        )

    def cross_attention(self, hidden_state, goal_tokens):
        """Compute cross-attention from hidden_state (query) to goal_tokens (K, V).

        hidden_state: [B, d_hidden]
        goal_tokens:  [B, G, d_goal]  multi-token goal embedding

        Returns: [B, d_attn]  goal-aware context vector
        """
        B = hidden_state.shape[0]
        G = goal_tokens.shape[1]
        hs = self.hidden_layernorm(hidden_state)
        gt = self.goal_layernorm(goal_tokens)
        q = self.W_q(hs).view(B, self.n_heads, self.head_dim)        # [B, H, D]
        k = self.W_k(gt).view(B, G, self.n_heads, self.head_dim)      # [B, G, H, D]
        v = self.W_v(gt).view(B, G, self.n_heads, self.head_dim)      # [B, G, H, D]
        # Attention scores: q · k → [B, H, G]
        scores = torch.einsum("bhd,bghd->bhg", q, k) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        # Output: attn @ v → [B, H, D]
        out = torch.einsum("bhg,bghd->bhd", attn, v)
        # Combine heads
        out = out.reshape(B, self.n_heads * self.head_dim)
        return self.out_proj(out)  # [B, d_attn]

    def init_state(self, batch_size, device):
        return self.init_belief.unsqueeze(0).expand(batch_size, -1, -1).contiguous().to(device)

    def init_slow_state(self, batch_size, device):
        return self.slow_init_belief.unsqueeze(0).expand(batch_size, -1, -1).contiguous().to(device)

    def _soft_clamp(self, h):
        c = self.h_input_clamp
        if c > 0:
            return c * torch.tanh(h / c)
        return h

    def fast_step(self, h_fast_prev, hidden_state, goal_tokens):
        """Forward fast channel one step.
        h_fast_prev:   [B, K, d]
        hidden_state:  [B, d_hidden]
        goal_tokens:   [B, G, d_goal]
        Returns:
            h_fast_new [B, K, d], cross_attn_out [B, d_attn] (returned for value head)
        """
        cross_ctx = self.cross_attention(hidden_state, goal_tokens)   # [B, d_attn]
        hs_proj = self.hidden_proj(self.hidden_layernorm(hidden_state)) # [B, d_attn]
        combined = torch.cat([cross_ctx, hs_proj], dim=-1)              # [B, 2*d_attn]
        e = self.in_combined(combined)                                   # [B, d]
        e = self.evidence_layernorm(e)
        evidence = e.unsqueeze(1) * self.evidence_mix.unsqueeze(0)      # [B, K, d]
        h_input = h_fast_prev + evidence
        h_input = self._soft_clamp(h_input)
        context = self.context_pool(h_input, None)
        self.dynamics.set_context(context, mask=None)
        n_steps = int(self.config.n_ode_steps)
        self.dynamics.set_n_steps(n_steps)
        T = float(self.config.integration_time)
        out = euler_solve_halting(
            self.dynamics, h_input, (0.0, T), n_steps,
            min_steps=self.config.halting_min_steps,
        )
        if isinstance(out, tuple):
            h_out = out[0]
        else:
            h_out = out
        return h_out, cross_ctx

    def slow_step(self, h_slow_prev, z_goal, z_goal_prev=None):
        if z_goal_prev is None:
            z_goal_delta = z_goal
        else:
            z_goal_delta = z_goal - z_goal_prev
        e = self.slow_in_goal(z_goal)
        evidence = self.slow_layernorm(e)
        trigger_logit = self.head_trigger(z_goal_delta).squeeze(-1)
        trigger_prob = torch.sigmoid(trigger_logit)
        alpha = torch.clamp(self.slow_alpha + trigger_prob.unsqueeze(-1).unsqueeze(-1)
                              * self.slow_alpha * self.trigger_boost, max=0.5)
        injection = torch.tanh(evidence).unsqueeze(1)
        h_slow_new = (1.0 - alpha) * h_slow_prev + alpha * injection
        return h_slow_new, trigger_prob

    def value(self, h_fast, cross_attn_out):
        """V(h_fast, cross_attn_out) → P(committed). The cross_attn_out is the
        goal-aware view from the current chunk's cross-attention."""
        h_flat = h_fast.flatten(1)
        x = torch.cat([h_flat, cross_attn_out], dim=-1)
        return self.head_value(x).squeeze(-1)

    def jepa_predict(self, h_fast):
        h_flat = h_fast.flatten(1)
        out = self.head_jepa(h_flat)
        return out.view(h_fast.shape)


def forward_trajectory(model, hidden_state_traj, goal_tokens_per_chunk,
                          z_goal_traj, device, target_t=None, training=True):
    """Forward CrossAttnHiddenLiquid through a trajectory.

    hidden_state_traj:        [T, d_hidden]
    goal_tokens_per_chunk:    [T, G, d_goal]  — goal tokens per chunk (jumps at turn boundary)
    z_goal_traj:              [T, d_goal]     — pooled goal (for slow channel + value head fallback)

    Returns: (h_fast_traj [T, K, d], h_slow_traj [T, K, d], cross_attn_traj [T, d_attn])
    """
    T = hidden_state_traj.shape[0]
    h_fast = model.init_state(1, device)
    h_slow = model.init_slow_state(1, device)
    z_goal_prev = None
    h_fast_traj = []
    h_slow_traj = []
    cross_attn_traj = []
    end_t = target_t + 1 if target_t is not None else T
    for t in range(end_t):
        hs = hidden_state_traj[t].unsqueeze(0)       # [1, d_hidden]
        gt = goal_tokens_per_chunk[t].unsqueeze(0)    # [1, G, d_goal]
        z_g = z_goal_traj[t].unsqueeze(0)             # [1, d_goal]
        grad_here = training and (target_t is None or t == target_t)
        if grad_here:
            h_fast, cross_ctx = model.fast_step(h_fast, hs, gt)
            h_slow, _ = model.slow_step(h_slow, z_g, z_goal_prev)
        else:
            with torch.no_grad():
                h_fast, cross_ctx = model.fast_step(h_fast, hs, gt)
                h_slow, _ = model.slow_step(h_slow, z_g, z_goal_prev)
            h_fast = h_fast.detach()
            h_slow = h_slow.detach()
            cross_ctx = cross_ctx.detach()
        z_goal_prev = z_g
        h_fast_traj.append(h_fast[0])
        h_slow_traj.append(h_slow[0])
        cross_attn_traj.append(cross_ctx[0])
    return (torch.stack(h_fast_traj, dim=0),
            torch.stack(h_slow_traj, dim=0),
            torch.stack(cross_attn_traj, dim=0))
