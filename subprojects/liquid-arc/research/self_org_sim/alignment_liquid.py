"""Alignment-Liquid: Liquid substrate on alignment features (not raw embeddings).

For multi-turn linguistic goal tracking:
  - Compute per-chunk alignment features f[t] = (cos, L2, dot, momentum, ...)
  - Feed alignment trajectory to Liquid substrate (small d, K positions)
  - Slow channel tracks commitment (jumps at goal transitions)
  - Value head predicts goal-following: V(h_fast, z_goal) → P(committed)

Task-agnostic: alignment-feature computation is universal across linguistic goals.
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


def make_alignment_config(d=32):
    return LiquidARCConfig(
        d_model=d, d_metric=16, d_ffn=64, max_seq_len=8,
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


ALIGNMENT_FEATURE_DIM = 8


def compute_alignment_features(z_t, z_goal, z_t_prev=None, z_goal_prev=None):
    """Compute 8-d alignment features from raw embeddings at a single timestep.

    Args:
      z_t: [B, dim] current generation chunk embedding
      z_goal: [B, dim] current goal embedding
      z_t_prev: [B, dim] previous chunk embedding (for momentum) or None
      z_goal_prev: [B, dim] previous goal embedding (for goal-change detection) or None

    Returns: [B, 8] feature vector
    """
    B = z_t.shape[0]
    device = z_t.device
    cos_g = F.cosine_similarity(z_t, z_goal, dim=-1)             # alignment to goal
    l2_g = (z_t - z_goal).norm(dim=-1)                           # L2 distance
    dot_g = (z_t * z_goal).sum(dim=-1)                           # dot product
    g_norm = z_goal.norm(dim=-1)                                 # goal magnitude
    if z_t_prev is None:
        cos_prev = torch.ones(B, device=device)
        delta_norm = torch.zeros(B, device=device)
        delta_to_goal = torch.zeros(B, device=device)
    else:
        cos_prev = F.cosine_similarity(z_t, z_t_prev, dim=-1)    # momentum cosine
        delta = z_t - z_t_prev
        delta_norm = delta.norm(dim=-1)                          # step magnitude
        delta_to_goal = F.cosine_similarity(delta, z_goal - z_t_prev + 1e-8, dim=-1)  # motion toward goal
    if z_goal_prev is None:
        goal_jump = torch.zeros(B, device=device)
    else:
        goal_jump = (z_goal - z_goal_prev).norm(dim=-1)          # goal change magnitude
    return torch.stack([cos_g, l2_g, dot_g, g_norm, cos_prev,
                          delta_norm, delta_to_goal, goal_jump], dim=-1)


class AlignmentLiquid(nn.Module):
    """Liquid substrate operating on alignment-feature trajectory.

    Forward per-chunk: compute alignment features, project to belief, run dynamics.
    Slow channel: integrates z_goal to track commitment (jumps at goal transitions).
    Value head: V(h_fast, z_goal) → P(committed-to-goal) for drift detection.
    """

    def __init__(self, z_goal_dim=384, d=32, K=4, value_hidden=128):
        super().__init__()
        self.z_goal_dim = z_goal_dim
        self.d = d
        self.K = K
        self.config = make_alignment_config(d=d)

        # Initial belief
        self.init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.init_belief, std=0.05)

        # Alignment feature → belief space
        self.in_align = nn.Linear(ALIGNMENT_FEATURE_DIM, d)
        nn.init.normal_(self.in_align.weight, std=0.05)
        nn.init.zeros_(self.in_align.bias)

        # Direct goal projection (gives substrate access to which goal it's tracking)
        self.in_goal = nn.Linear(z_goal_dim, d)
        nn.init.normal_(self.in_goal.weight, std=0.02)
        nn.init.zeros_(self.in_goal.bias)

        # Gates
        self.align_gate = nn.Parameter(torch.tensor(1.5))
        self.goal_gate = nn.Parameter(torch.tensor(0.5))
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

        # SLOW CHANNEL: tracks commitment to current goal
        self.K_slow = K
        self.d_slow = d
        self.slow_in_goal = nn.Linear(z_goal_dim, d)
        nn.init.normal_(self.slow_in_goal.weight, std=0.02)
        nn.init.zeros_(self.slow_in_goal.bias)
        self.slow_layernorm = nn.LayerNorm(d, elementwise_affine=True)
        self.slow_init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.slow_init_belief, std=0.05)
        self.slow_alpha = nn.Parameter(torch.tensor(0.05))  # higher than 0.01 — text goal-jumps need responsiveness
        self.trigger_boost = 8.0
        # Trigger detector reads goal embedding delta
        self.head_trigger = nn.Sequential(
            nn.Linear(z_goal_dim, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
        with torch.no_grad():
            self.head_trigger[-1].weight.mul_(0.01)
            self.head_trigger[-1].bias.fill_(-2.0)

        # VALUE HEAD: predicts goal-following from (h_fast, z_goal)
        # Includes dropout for regularization vs the 250-conversation overfit
        self.head_value = nn.Sequential(
            nn.Linear(K * d + z_goal_dim, value_hidden),
            nn.SiLU(), nn.LayerNorm(value_hidden),
            nn.Dropout(0.2),
            nn.Linear(value_hidden, value_hidden),
            nn.SiLU(), nn.LayerNorm(value_hidden),
            nn.Dropout(0.2),
            nn.Linear(value_hidden, 1),
        )

        # JEPA self-prediction head (regularizer for smooth h_fast trajectory)
        self.head_jepa = nn.Sequential(
            nn.Linear(K * d, value_hidden),
            nn.SiLU(), nn.LayerNorm(value_hidden),
            nn.Linear(value_hidden, K * d),
        )

    def init_state(self, batch_size, device):
        return self.init_belief.unsqueeze(0).expand(batch_size, -1, -1).contiguous().to(device)

    def init_slow_state(self, batch_size, device):
        return self.slow_init_belief.unsqueeze(0).expand(batch_size, -1, -1).contiguous().to(device)

    def _soft_clamp(self, h):
        c = self.h_input_clamp
        if c > 0:
            return c * torch.tanh(h / c)
        return h

    def fast_step(self, h_fast_prev, align_features, z_goal):
        """Forward fast channel one step.
        h_fast_prev: [B, K, d]
        align_features: [B, ALIGNMENT_FEATURE_DIM]
        z_goal: [B, z_goal_dim]
        Returns: h_fast_new [B, K, d]
        """
        e_align = self.in_align(align_features) * self.align_gate         # [B, d]
        e_goal = self.in_goal(z_goal) * self.goal_gate                    # [B, d]
        e = self.evidence_layernorm(e_align + e_goal)                     # [B, d]
        # Per-position evidence with K-position weighting
        evidence = e.unsqueeze(1) * self.evidence_mix.unsqueeze(0)        # [B, K, d]
        h_input = h_fast_prev + evidence
        h_input = self._soft_clamp(h_input)
        # ContextPool returns [B, d]; ContinuousDynamics integrates via euler_solve_halting
        context = self.context_pool(h_input, None)                         # [B, d]
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
        return h_out

    def slow_step(self, h_slow_prev, z_goal, z_goal_prev=None):
        """Slow channel: tracks commitment to goal. Jumps on goal transitions."""
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

    def value(self, h_fast, z_goal):
        """Predict P(committed-to-goal) from current state + current goal."""
        h_flat = h_fast.flatten(1)
        x = torch.cat([h_flat, z_goal], dim=-1)
        return self.head_value(x).squeeze(-1)

    def jepa_predict(self, h_fast):
        """Self-prediction of next h_fast for trajectory smoothness regularizer."""
        h_flat = h_fast.flatten(1)
        out = self.head_jepa(h_flat)
        return out.view(h_fast.shape)


def forward_trajectory(model, z_t_traj, z_goal_traj, device, target_t=None,
                         training=True):
    """Forward AlignmentLiquid through a trajectory.

    z_t_traj: [T, dim] generation chunk embeddings
    z_goal_traj: [T, dim] goal embeddings (per-chunk, jumps at turn boundaries)
    target_t: if set, gradients only flow at this timestep

    Returns: (h_fast_traj [T, K, d], h_slow_traj [T, K, d], align_features [T, 8])
    """
    T = z_t_traj.shape[0]
    h_fast = model.init_state(1, device)
    h_slow = model.init_slow_state(1, device)
    z_t_prev = None
    z_goal_prev = None
    h_fast_traj = []
    h_slow_traj = []
    align_traj = []
    end_t = target_t + 1 if target_t is not None else T
    for t in range(end_t):
        z_t = z_t_traj[t].unsqueeze(0)
        z_g = z_goal_traj[t].unsqueeze(0)
        f = compute_alignment_features(z_t, z_g, z_t_prev, z_goal_prev)  # [1, 8]
        grad_here = training and (target_t is None or t == target_t)
        if grad_here:
            h_fast = model.fast_step(h_fast, f, z_g)
            h_slow, _ = model.slow_step(h_slow, z_g, z_goal_prev)
        else:
            with torch.no_grad():
                h_fast = model.fast_step(h_fast, f, z_g)
                h_slow, _ = model.slow_step(h_slow, z_g, z_goal_prev)
            h_fast = h_fast.detach()
            h_slow = h_slow.detach()
        z_t_prev = z_t
        z_goal_prev = z_g
        h_fast_traj.append(h_fast[0])
        h_slow_traj.append(h_slow[0])
        align_traj.append(f[0])
    return (torch.stack(h_fast_traj, dim=0),
            torch.stack(h_slow_traj, dim=0),
            torch.stack(align_traj, dim=0))
