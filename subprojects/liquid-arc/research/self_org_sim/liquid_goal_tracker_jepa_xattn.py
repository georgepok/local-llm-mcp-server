"""Cross-attention JEPA-LGT: no mean-pool destruction anywhere.

Bottlenecks fixed:
  - Encoder side: substrate sees FULL [seq, 2048] backbone_features, not pooled.
    K learned latent positions attend over the full sequence (preserves spatial+lang info).
  - Decoder side: each backbone token queries h to get its own [2048] tangent
    (per-token modulation; no uniform broadcast).

JEPA loss is now per-token:
  L = smooth_L1(ẑ_{t+1} [B,seq,2048], stop_grad(z_{t+1} [B,seq,2048]))

Per-turn signature:
  step(h_prev[B,K,d], bb_t[B,seq,2048], z_goal[B,2048], action_chunk_t[B,H,A])
    → h_new[B,K,d], z_pred_next[B,seq,2048], tangent[B,seq,2048], diag
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
from liquid_arc.config import LiquidARCConfig  # type: ignore
from liquid_arc.dynamics import ContinuousDynamics  # type: ignore
from liquid_arc.solver import euler_solve_halting  # type: ignore
from liquid_arc.context_pool import ContextPool  # type: ignore


def make_lgt_config(d=128):
    return LiquidARCConfig(
        d_model=d, d_metric=32, d_ffn=256, max_seq_len=16,
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
        curvature_cv_floor=1.0, curvature_cv_ceiling=8.0,
        tau_quality_loss_enabled=False,
        step_embed_enabled=False,
        step_conditional_operator=False,
        structural_tau_enabled=True, structural_tau_min=0.3, structural_tau_max=3.0,
        norm_ref=10.0, norm_lambda=0.1,
        base_lr=3e-4, structural_lr_ratio=0.1,
        warmup_steps=200, weight_decay=0.01,
        use_torch_compile=False,
    )


class JEPA_LGT_XAttn(nn.Module):
    """Cross-attention JEPA world model with per-token tangent output.

    Args:
        z_vl_dim:    2048
        action_dim:  7
        horizon:     16
        d:           128   (larger than goal variants — substrate is bigger)
        K:           8     (more latent positions for attention to spread over)
        n_heads:     4
        tangent_scale: 0.1 (smaller since per-token; aggregate magnitude still ~ √seq · 0.1)
    """

    def __init__(self, z_vl_dim=2048, action_dim=7, horizon=16,
                 d=128, K=8, n_heads=4, tangent_scale=0.1):
        super().__init__()
        self.z_vl_dim = z_vl_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.d = d
        self.K = K
        self.n_heads = n_heads
        self.tangent_scale = tangent_scale
        self.config = make_lgt_config(d=d)

        self.init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.init_belief, std=0.05)

        # ENCODER side: K queries attend over bb_features [B,seq,2048]
        self.encoder_k = nn.Linear(z_vl_dim, d)
        self.encoder_v = nn.Linear(z_vl_dim, d)
        self.encoder_q_proj = nn.Linear(d, d)  # project h_prev → query
        self.encoder_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.0)
        self.encoder_ln_pre_q = nn.LayerNorm(d)
        self.encoder_ln_post = nn.LayerNorm(d)

        # Goal + action conditioning (added to encoded h before ODE)
        self.goal_proj = nn.Linear(z_vl_dim, d)
        self.action_proj = nn.Linear(horizon * action_dim, d)
        self.goal_gate = nn.Parameter(torch.tensor(1.0))
        self.action_gate = nn.Parameter(torch.tensor(0.1))

        # Liquid ODE evolution
        self.context_pool = ContextPool(self.config)
        self.dynamics = ContinuousDynamics(self.config)

        # DECODER side: each bb token queries h_new → per-token output
        self.decoder_q = nn.Linear(z_vl_dim, d)
        self.decoder_ln_pre_q = nn.LayerNorm(d)
        self.decoder_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.0)
        self.decoder_ln_post = nn.LayerNorm(d)
        self.tangent_out = nn.Linear(d, z_vl_dim)
        with torch.no_grad():
            self.tangent_out.weight.mul_(0.01)
            self.tangent_out.bias.zero_()

    def init_state(self, batch_size: int, device, dtype=torch.float32):
        return self.init_belief.unsqueeze(0).expand(batch_size, -1, -1).to(
            device=device, dtype=dtype).contiguous()

    def step(
        self,
        h_goal_prev: torch.Tensor,           # [B, K, d]
        bb_features: torch.Tensor,            # [B, seq, z_vl_dim]
        z_goal: torch.Tensor,                 # [B, z_vl_dim]
        action_chunk_t: torch.Tensor,         # [B, horizon, action_dim]
        n_steps_override: Optional[int] = None,
    ):
        B, seq, _ = bb_features.shape
        device = h_goal_prev.device

        # ENCODER: h_prev queries cross-attend over bb_features
        q = self.encoder_q_proj(self.encoder_ln_pre_q(h_goal_prev))     # [B, K, d]
        k = self.encoder_k(bb_features)                                  # [B, seq, d]
        v = self.encoder_v(bb_features)
        h_enc, _ = self.encoder_attn(q, k, v, need_weights=False)        # [B, K, d]
        h_enc = self.encoder_ln_post(h_enc + h_goal_prev)                # residual

        # Inject goal + action evidence (broadcast across K positions)
        e_g = (self.goal_proj(z_goal) * self.goal_gate).unsqueeze(1)     # [B, 1, d]
        e_a = (self.action_proj(action_chunk_t.reshape(B, -1))
               * self.action_gate).unsqueeze(1)
        h_input = h_enc + e_g + e_a                                       # [B, K, d]

        # Liquid ODE on K-position state
        context = self.context_pool(h_input, None)
        self.dynamics.set_context(context, mask=None)
        if n_steps_override is not None:
            n_steps = int(n_steps_override)
        elif self.training:
            lo = int(self.config.ode_steps_min)
            hi = int(self.config.ode_steps_max)
            n_steps = int(torch.randint(lo, hi + 1, (1,)).item())
        else:
            n_steps = int(self.config.n_ode_steps)
        self.dynamics.set_n_steps(n_steps)
        T = float(self.config.integration_time)
        out = euler_solve_halting(
            self.dynamics, h_input, (0.0, T), n_steps,
            min_steps=self.config.halting_min_steps,
        )
        if isinstance(out, tuple):
            h_goal_new = out[0]
            ponder = out[1]
        else:
            h_goal_new = out
            ponder = torch.zeros(B, device=device)

        # DECODER: each bb token queries h_new for its own [d] response
        # then projects to z_vl_dim — per-token tangent
        q_dec = self.decoder_ln_pre_q(self.decoder_q(bb_features))       # [B, seq, d]
        t_dec, _ = self.decoder_attn(q_dec, h_goal_new, h_goal_new,
                                       need_weights=False)               # [B, seq, d]
        t_dec = self.decoder_ln_post(t_dec)
        tangent = torch.tanh(self.tangent_out(t_dec)) * self.tangent_scale  # [B, seq, 2048]

        z_pred_next = bb_features + tangent                              # [B, seq, 2048]

        g = self.dynamics.compute_metric_diag(h_input)
        metric_cv = g.std() / (g.mean() + 1e-8)

        return h_goal_new, z_pred_next, tangent, {
            "metric_cv": metric_cv,
            "ponder": ponder.mean(),
            "n_steps": n_steps,
            "tangent_norm_per_token": tangent.norm(dim=-1).mean().detach(),  # avg over [B,seq]
            "tangent_total_norm": tangent.reshape(B, -1).norm(dim=-1).mean().detach(),
            "goal_gate": self.goal_gate.detach(),
            "action_gate": self.action_gate.detach(),
        }
