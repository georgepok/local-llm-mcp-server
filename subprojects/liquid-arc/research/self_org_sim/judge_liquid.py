"""Judge-Liquid: Liquid wrapper that consumes LLM hidden state + LLM judgment signal.

The LLM provides task-agnostic goal-following judgment (judge_score per chunk).
Liquid provides temporal smoothing and commitment tracking on top.

Architecture:
  PER CHUNK INPUT:
    hidden_state[t] (Qwen activation, 1536-d)
    judge_score[t]  (Qwen self-judgment of partial response, scalar)
       ↓
  in_hidden: LayerNorm(1536) → d
  in_judge:  scalar → d via projection
  Combined evidence → Liquid dynamics
       ↓
  Slow channel tracks commitment (jumps at goal transitions)
       ↓
  V(h_fast, z_goal) → P(committed)
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
from liquid_arc.config import LiquidARCConfig  # type: ignore
from liquid_arc.dynamics import ContinuousDynamics  # type: ignore
from liquid_arc.context_pool import ContextPool  # type: ignore
from liquid_arc.solver import euler_solve_halting  # type: ignore


def make_jl_config(d=64, d_metric=16, d_ffn=128, n_ode_steps=3,
                    tau_depth=False):
    """tau_depth=True enables the prior framework's self-organizing ODE-depth
    machinery: tau-convergence coupling (surprise→deep), tau-CV coupling,
    wider tau range for effective-depth variation."""
    extra = {}
    if tau_depth:
        extra = dict(
            tau_convergence_coupling_enabled=True,
            tau_convergence_beta=1.0,
            tau_convergence_floor=0.3,   # allow aggressive depth on surprise
            tau_cv_coupling_enabled=True,
            tau_quality_loss_enabled=True,
            tau_quality_lambda=0.05,
        )
    return LiquidARCConfig(
        d_model=d, d_metric=d_metric, d_ffn=d_ffn, max_seq_len=8,
        n_ode_steps=n_ode_steps,
        ode_steps_min=max(2, n_ode_steps - 1),
        ode_steps_max=min(n_ode_steps + 1, n_ode_steps + 2),
        integration_time=0.5,
        tau_min=0.2 if tau_depth else 0.3,
        tau_max=3.0 if tau_depth else 2.0,
        t_diffusion_init=0.5,
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
        **extra,
    )


class JudgeLiquid(nn.Module):
    """Liquid wrapper consuming Qwen hidden state + per-chunk LLM judgment signal."""

    def __init__(self, d_hidden=1536, z_goal_dim=384, d=64, K=4, value_hidden=192,
                 d_metric=16, d_ffn=128, n_ode_steps=3, tau_depth=False):
        super().__init__()
        self.d_hidden = d_hidden
        self.z_goal_dim = z_goal_dim
        self.d = d
        self.K = K
        self.config = make_jl_config(d=d, d_metric=d_metric, d_ffn=d_ffn,
                                        n_ode_steps=n_ode_steps, tau_depth=tau_depth)

        self.init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.init_belief, std=0.05)

        # Hidden state projection
        self.hidden_layernorm = nn.LayerNorm(d_hidden, elementwise_affine=True)
        self.in_hidden = nn.Linear(d_hidden, d)
        nn.init.normal_(self.in_hidden.weight, std=0.02)
        nn.init.zeros_(self.in_hidden.bias)

        # Judgment projection: scalar → d-dim signal
        # Embed judge_score richly: use sinusoidal-like features
        # Or simple linear: judge_score [B,1] → d
        self.in_judge = nn.Linear(1, d)
        nn.init.normal_(self.in_judge.weight, std=0.05)
        nn.init.zeros_(self.in_judge.bias)

        # Goal projection
        self.in_goal = nn.Linear(z_goal_dim, d)
        nn.init.normal_(self.in_goal.weight, std=0.02)
        nn.init.zeros_(self.in_goal.bias)

        self.hidden_gate = nn.Parameter(torch.tensor(0.7))
        self.judge_gate = nn.Parameter(torch.tensor(1.5))  # judge signal is critical
        self.goal_gate = nn.Parameter(torch.tensor(0.5))

        self.evidence_mix = nn.Parameter(torch.ones(K, 1))
        with torch.no_grad():
            self.evidence_mix[0] = 2.0
            self.evidence_mix[1] = 1.5
            self.evidence_mix[2] = 1.5
            self.evidence_mix[3] = 1.0

        self.context_pool = ContextPool(self.config)
        self.dynamics = ContinuousDynamics(self.config)

        self.evidence_layernorm = nn.LayerNorm(d, elementwise_affine=True)
        self.h_input_clamp = 50.0

        # Slow channel for commitment
        self.K_slow = K
        self.d_slow = d
        self.slow_in_goal = nn.Linear(z_goal_dim, d)
        nn.init.normal_(self.slow_in_goal.weight, std=0.02)
        nn.init.zeros_(self.slow_in_goal.bias)
        self.slow_layernorm = nn.LayerNorm(d, elementwise_affine=True)
        self.slow_init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.slow_init_belief, std=0.05)
        self.slow_alpha = nn.Parameter(torch.tensor(0.05))
        self.trigger_boost = 8.0
        self.head_trigger = nn.Sequential(
            nn.Linear(z_goal_dim, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
        with torch.no_grad():
            self.head_trigger[-1].weight.mul_(0.01)
            self.head_trigger[-1].bias.fill_(-2.0)

        # REFINEMENT head: outputs signed correction to add to judge_score
        # Initialized small so model starts close to "use judge_score directly"
        self.head_refinement = nn.Sequential(
            nn.Linear(K * d + z_goal_dim, value_hidden),
            nn.SiLU(), nn.LayerNorm(value_hidden),
            nn.Dropout(0.2),
            nn.Linear(value_hidden, value_hidden),
            nn.SiLU(), nn.LayerNorm(value_hidden),
            nn.Dropout(0.2),
            nn.Linear(value_hidden, 1),
        )
        # Tiny init on last layer so refinement starts near zero
        with torch.no_grad():
            self.head_refinement[-1].weight.mul_(0.01)
            self.head_refinement[-1].bias.zero_()
        # Trainable scale on judge signal (init=1.0 = trust judge)
        self.judge_scale = nn.Parameter(torch.tensor(1.0))

        # JEPA self-prediction (regularizer)
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

    def fast_step(self, h_fast_prev, hidden_state, judge_score, z_goal):
        """
        h_fast_prev: [B, K, d]
        hidden_state: [B, d_hidden]
        judge_score: [B] (scalar per sample)
        z_goal: [B, z_goal_dim]
        """
        hs_normed = self.hidden_layernorm(hidden_state)
        e_h = self.in_hidden(hs_normed) * self.hidden_gate
        e_j = self.in_judge(judge_score.unsqueeze(-1)) * self.judge_gate
        e_g = self.in_goal(z_goal) * self.goal_gate
        e = self.evidence_layernorm(e_h + e_j + e_g)
        evidence = e.unsqueeze(1) * self.evidence_mix.unsqueeze(0)
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
        return h_out

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

    def value(self, h_fast, z_goal, judge_score):
        """V output = judge_scale * judge_score + Liquid refinement.

        Forces model to USE the judge signal as primary; Liquid's h_fast provides
        correction. Initialization keeps refinement near zero so untrained model
        behaves like raw judge baseline.
        """
        h_flat = h_fast.flatten(1)
        x = torch.cat([h_flat, z_goal], dim=-1)
        refinement = self.head_refinement(x).squeeze(-1)
        return self.judge_scale * judge_score + refinement

    def jepa_predict(self, h_fast):
        h_flat = h_fast.flatten(1)
        out = self.head_jepa(h_flat)
        return out.view(h_fast.shape)


def forward_trajectory(model, hidden_state_traj, judge_traj, z_goal_traj, device,
                          target_t=None, training=True):
    """Forward JudgeLiquid through a trajectory.

    hidden_state_traj: [T, d_hidden]
    judge_traj:        [T]            scalar per chunk
    z_goal_traj:       [T, z_goal_dim]
    """
    T = hidden_state_traj.shape[0]
    h_fast = model.init_state(1, device)
    h_slow = model.init_slow_state(1, device)
    z_goal_prev = None
    h_fast_traj = []
    h_slow_traj = []
    end_t = target_t + 1 if target_t is not None else T
    for t in range(end_t):
        hs = hidden_state_traj[t].unsqueeze(0)
        js = judge_traj[t].unsqueeze(0)  # [1]
        z_g = z_goal_traj[t].unsqueeze(0)
        grad_here = training and (target_t is None or t == target_t)
        if grad_here:
            h_fast = model.fast_step(h_fast, hs, js, z_g)
            h_slow, _ = model.slow_step(h_slow, z_g, z_goal_prev)
        else:
            with torch.no_grad():
                h_fast = model.fast_step(h_fast, hs, js, z_g)
                h_slow, _ = model.slow_step(h_slow, z_g, z_goal_prev)
            h_fast = h_fast.detach()
            h_slow = h_slow.detach()
        z_goal_prev = z_g
        h_fast_traj.append(h_fast[0])
        h_slow_traj.append(h_slow[0])
    return torch.stack(h_fast_traj, dim=0), torch.stack(h_slow_traj, dim=0)
