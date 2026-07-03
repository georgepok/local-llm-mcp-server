"""JudgeDistillLiquid: substrate learns to PREDICT judge_score without using it as input.

The LLM judge is the TEACHER. Substrate must learn goal-following abstraction
to predict judge_score from (hidden_state, z_goal) alone.

At inference on unseen categories:
  - If substrate predicts judge_score with AUC similar to direct judge → ABSTRACTION LEARNED
  - If much lower → just memorized seen patterns
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


def make_jdl_config(d=64, d_metric=16, d_ffn=128, n_ode_steps=3):
    return LiquidARCConfig(
        d_model=d, d_metric=d_metric, d_ffn=d_ffn, max_seq_len=8,
        n_ode_steps=n_ode_steps,
        ode_steps_min=max(2, n_ode_steps - 1),
        ode_steps_max=min(n_ode_steps + 1, n_ode_steps + 2),
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


class JudgeDistillLiquid(nn.Module):
    """Substrate distilling LLM judge — NO judge as input, predicts it as output."""

    def __init__(self, d_hidden=1536, z_goal_dim=384, d=128, K=4, value_hidden=256,
                 d_metric=16, d_ffn=128, n_ode_steps=3):
        super().__init__()
        self.d_hidden = d_hidden
        self.z_goal_dim = z_goal_dim
        self.d = d
        self.K = K
        self.config = make_jdl_config(d=d, d_metric=d_metric, d_ffn=d_ffn,
                                          n_ode_steps=n_ode_steps)

        self.init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.init_belief, std=0.05)

        # Input projections — NO judge input
        self.hidden_layernorm = nn.LayerNorm(d_hidden, elementwise_affine=True)
        self.in_hidden = nn.Linear(d_hidden, d)
        nn.init.normal_(self.in_hidden.weight, std=0.02)
        nn.init.zeros_(self.in_hidden.bias)

        self.in_goal = nn.Linear(z_goal_dim, d)
        nn.init.normal_(self.in_goal.weight, std=0.02)
        nn.init.zeros_(self.in_goal.bias)

        self.hidden_gate = nn.Parameter(torch.tensor(1.0))
        self.goal_gate = nn.Parameter(torch.tensor(0.7))

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

        # JUDGE PREDICTOR HEAD: predicts judge_score from (h_fast, z_goal)
        # This is the abstraction test — substrate must learn judge's function
        self.head_judge_pred = nn.Sequential(
            nn.Linear(K * d + z_goal_dim, value_hidden),
            nn.SiLU(), nn.LayerNorm(value_hidden),
            nn.Dropout(0.2),
            nn.Linear(value_hidden, value_hidden),
            nn.SiLU(), nn.LayerNorm(value_hidden),
            nn.Dropout(0.2),
            nn.Linear(value_hidden, 1),  # scalar judge_score prediction
        )

        # VALUE HEAD: predicts binary follow from h_fast (no judge needed)
        # Trained jointly so substrate learns BOTH the judge function AND outcome prediction
        self.head_value = nn.Sequential(
            nn.Linear(K * d + z_goal_dim, value_hidden),
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

        # PROTOTYPE: success attractor in belief space — geometric anchor for task dynamics
        # Successful trajectories' end states should converge toward this; drift away.
        # This shapes phase-transition direction to be task-aligned.
        self.success_anchor = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.success_anchor, std=0.1)
        self.proto_margin = 1.0  # margin for drift trajectories

    def init_state(self, batch_size, device):
        return self.init_belief.unsqueeze(0).expand(batch_size, -1, -1).contiguous().to(device)

    def init_slow_state(self, batch_size, device):
        return self.slow_init_belief.unsqueeze(0).expand(batch_size, -1, -1).contiguous().to(device)

    def _soft_clamp(self, h):
        c = self.h_input_clamp
        if c > 0:
            return c * torch.tanh(h / c)
        return h

    def fast_step(self, h_fast_prev, hidden_state, z_goal):
        hs_normed = self.hidden_layernorm(hidden_state)
        e_h = self.in_hidden(hs_normed) * self.hidden_gate
        e_g = self.in_goal(z_goal) * self.goal_gate
        e = self.evidence_layernorm(e_h + e_g)
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

    def predict_judge(self, h_fast, z_goal):
        """Substrate's prediction of judge_score (abstraction test)."""
        h_flat = h_fast.flatten(1)
        x = torch.cat([h_flat, z_goal], dim=-1)
        return self.head_judge_pred(x).squeeze(-1)

    def value(self, h_fast, z_goal):
        """Substrate's prediction of P(committed) — no judge input."""
        h_flat = h_fast.flatten(1)
        x = torch.cat([h_flat, z_goal], dim=-1)
        return self.head_value(x).squeeze(-1)

    def jepa_predict(self, h_fast):
        h_flat = h_fast.flatten(1)
        out = self.head_jepa(h_flat)
        return out.view(h_fast.shape)


def forward_trajectory(model, hidden_state_traj, z_goal_traj, device,
                          target_t=None, training=True):
    T = hidden_state_traj.shape[0]
    h_fast = model.init_state(1, device)
    h_slow = model.init_slow_state(1, device)
    z_goal_prev = None
    h_fast_traj = []
    h_slow_traj = []
    end_t = target_t + 1 if target_t is not None else T
    for t in range(end_t):
        hs = hidden_state_traj[t].unsqueeze(0)
        z_g = z_goal_traj[t].unsqueeze(0)
        grad_here = training and (target_t is None or t == target_t)
        if grad_here:
            h_fast = model.fast_step(h_fast, hs, z_g)
            h_slow, _ = model.slow_step(h_slow, z_g, z_goal_prev)
        else:
            with torch.no_grad():
                h_fast = model.fast_step(h_fast, hs, z_g)
                h_slow, _ = model.slow_step(h_slow, z_g, z_goal_prev)
            h_fast = h_fast.detach()
            h_slow = h_slow.detach()
        z_goal_prev = z_g
        h_fast_traj.append(h_fast[0])
        h_slow_traj.append(h_slow[0])
    return torch.stack(h_fast_traj, dim=0), torch.stack(h_slow_traj, dim=0)
