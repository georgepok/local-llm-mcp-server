"""Proprio JEPA-LGT — adds state8 (xyz, rpy, gripper_qpos×2) as 5th evidence input.

Diagnostic findings A1+A2: substrate's pooled z_vl input lacks success-discriminating
information (uniform tangent across success/failure). Add proprioception so substrate
can directly observe robot state (mechanically-grounded failure signatures: gripper
qpos stuck, end-effector hovering, position deltas).

Multi-target heads:
  - pred_goaldist (regression — recover the only learnable scalar from A1)
  - pred_gripper_moving (binary — "did gripper qpos change > eps in last K env steps")
    Directly observable from state8_t vs state8_next; sharp gradient.
  - pred_state_delta (regression — predict ||state8_next - state8_t||,
    a proxy for "is anything happening?")

Per-turn signature:
  step(h_prev[B,K,d], z_t[B,2048], z_goal[B,2048], action_chunk[B,H,A], state8[B,8])
       → h_new[B,K,d], pred_goaldist[B], aux{moving, state_delta}, diag
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


def make_lgt_config(d=64):
    return LiquidARCConfig(
        d_model=d, d_metric=16, d_ffn=128, max_seq_len=8,
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


class JEPA_LGT_Proprio(nn.Module):
    """5-channel substrate with proprioception. Multi-target scalar heads."""

    def __init__(self, z_vl_dim=2048, action_dim=7, horizon=16, state_dim=8,
                 d=64, K=4, n_tok_per_k=1):
        super().__init__()
        self.z_vl_dim = z_vl_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.state_dim = state_dim
        self.d = d
        self.K = K
        self.n_tok_per_k = n_tok_per_k
        self.config = make_lgt_config(d=d)

        self.init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.init_belief, std=0.05)

        # 6 input channels: current z, goal z, progress delta, last action, current state8,
        # JEPA-VL: pooled LANGUAGE tokens (from GR00T's image_mask==0 slice).
        self.in_z     = nn.Linear(z_vl_dim, d)
        self.in_goal  = nn.Linear(z_vl_dim, d)
        self.in_delta = nn.Linear(z_vl_dim, d)
        self.in_action = nn.Linear(horizon * action_dim, d)
        self.in_state  = nn.Linear(state_dim, d)
        self.in_lang   = nn.Linear(z_vl_dim, d)
        for layer in (self.in_z, self.in_goal, self.in_delta, self.in_action,
                      self.in_state, self.in_lang):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

        # Gates — state init HIGH (directly observable, important signal)
        self.action_gate = nn.Parameter(torch.tensor(0.1))
        self.goal_gate   = nn.Parameter(torch.tensor(1.0))
        self.delta_gate  = nn.Parameter(torch.tensor(1.0))
        self.state_gate  = nn.Parameter(torch.tensor(1.0))
        # Language gate — init high so substrate is encouraged to use compositional
        # task semantics rather than collapse to z_t (which loses left/right structure).
        self.lang_gate   = nn.Parameter(torch.tensor(1.0))

        self.evidence_mix = nn.Parameter(torch.ones(K, 1))
        with torch.no_grad():
            self.evidence_mix[0] = 2.0
            self.evidence_mix[1] = 1.5
            self.evidence_mix[2] = 1.5
            self.evidence_mix[3] = 1.0

        self.context_pool = ContextPool(self.config)
        self.dynamics = ContinuousDynamics(self.config)

        # GRADIENT SAFETY: soft-clamp h_input via scaled tanh before context_pool /
        # dynamics. In normal forward (h_input per-dim ~ 1-10), tanh(x/CLAMP) ≈ x/CLAMP
        # so CLAMP * tanh(x/CLAMP) ≈ x (forward identity within 2% for |x| < CLAMP/3).
        # Outside that range, output saturates to ±CLAMP, preventing attention/ODE
        # saturation that would produce NaN gradients.
        # Set h_input_clamp=0 to disable (legacy behavior).
        self.h_input_clamp = 50.0
        # LayerNorm on evidence aggregation to bound magnitude flowing into context_pool.
        # Enabled when use_evidence_layernorm=True (default off for backward compat;
        # JEPA-body training flips it on).
        self.evidence_layernorm = nn.LayerNorm(d, elementwise_affine=True)
        self.use_evidence_layernorm = False

        # Scalar heads
        self.head_goaldist = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(), nn.LayerNorm(d * 2),
            nn.Linear(d * 2, 1),
        )
        # Binary: will the gripper actually move (qpos delta > threshold)?
        self.head_gripper_moving = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(), nn.LayerNorm(d * 2),
            nn.Linear(d * 2, 1),
        )
        # Regression: magnitude of state8 change next turn
        self.head_state_delta = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(), nn.LayerNorm(d * 2),
            nn.Linear(d * 2, 1),
        )
        # VIRTUAL-TOKEN PROJECTION: per-position h_goal[k, d=64] → n_tok_per_k tokens of dim 2048.
        # Output: K * n_tok_per_k tokens to append to GR00T's vl_embeds sequence.
        # Trained end-to-end via BC through GR00T action head.
        self.head_virtual_tokens = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(), nn.LayerNorm(d * 2),
            nn.Linear(d * 2, d * 4), nn.SiLU(), nn.LayerNorm(d * 4),
            nn.Linear(d * 4, z_vl_dim * n_tok_per_k),
        )
        with torch.no_grad():
            self.head_virtual_tokens[-1].weight.mul_(0.01)
            self.head_virtual_tokens[-1].bias.zero_()
        # BROADCAST RESIDUAL head: pooled h_goal → 2048-d residual added to every bb_features token.
        # This is the FORCED-contribution mechanism (variant #9 BC) vs append (optional).
        self.head_residual = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(), nn.LayerNorm(d * 2),
            nn.Linear(d * 2, d * 4), nn.SiLU(), nn.LayerNorm(d * 4),
            nn.Linear(d * 4, z_vl_dim),
        )
        with torch.no_grad():
            self.head_residual[-1].weight.mul_(0.01)
            self.head_residual[-1].bias.zero_()

        # CROSS-ATTENTION head: bb_features tokens query substrate's K belief positions
        # to get a PER-TOKEN residual. Breaks the "pool → single residual → broadcast"
        # bottleneck that collapsed all variants to noise. Each token gets its own
        # substrate-derived modification based on which belief position it best matches.
        # Architecture:
        #   K_sub [B, K, d_attn] from h_goal via head_k_sub
        #   V_sub [B, K, z_vl_dim] from h_goal via head_v_sub (carries content)
        #   Q_bb  [B, seq, d_attn] from bb_features via head_q_bb
        #   attn  [B, seq, K] = softmax(Q_bb @ K_sub^T / sqrt(d_attn))
        #   residual_per_tok [B, seq, z_vl_dim] = attn @ V_sub
        self.d_attn = 64
        self.head_q_bb  = nn.Linear(z_vl_dim, self.d_attn)
        self.head_k_sub = nn.Linear(d, self.d_attn)
        self.head_v_sub = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(), nn.LayerNorm(d * 2),
            nn.Linear(d * 2, z_vl_dim),
        )
        # Init: small Q/K so attention starts near-uniform; V scaled small so residual
        # starts near-zero (substrate doesn't disrupt action head initially).
        nn.init.normal_(self.head_q_bb.weight, std=0.02)
        nn.init.zeros_(self.head_q_bb.bias)
        nn.init.normal_(self.head_k_sub.weight, std=0.02)
        nn.init.zeros_(self.head_k_sub.bias)
        with torch.no_grad():
            self.head_v_sub[-1].weight.mul_(0.01)
            self.head_v_sub[-1].bias.zero_()
        # Bound output via 0.1× (matches broadcast head); per-token scale applied at output
        self.xattn_out_scale = 0.1

        # GAP-STEP CLOSED-LOOP HEADS:
        # 1) head_dynamics: (state8_t, action7_t, h_intent_pool) → state8_{t+1} residual prediction.
        #    Used at every env step inside the gap to predict expected next state.
        # 2) head_correction: (deviation8, h_intent_pool) → action7 correction (bounded).
        #    Triggered when actual proprio deviates from predicted; adjusts next chunk action.
        d_dyn = 64
        self.dyn_in_state  = nn.Linear(state_dim, d_dyn)
        self.dyn_in_action = nn.Linear(action_dim, d_dyn)
        self.dyn_in_intent = nn.Linear(d, d_dyn)  # uses pooled h_goal as intent
        self.head_dynamics = nn.Sequential(
            nn.Linear(3 * d_dyn, d_dyn * 2), nn.SiLU(), nn.LayerNorm(d_dyn * 2),
            nn.Linear(d_dyn * 2, d_dyn), nn.SiLU(),
            nn.Linear(d_dyn, state_dim),  # predicts DELTA state8
        )
        self.head_correction = nn.Sequential(
            nn.Linear(state_dim + d_dyn, d_dyn), nn.SiLU(),
            nn.Linear(d_dyn, action_dim),  # predicts action7 delta
        )
        # Init last layers to near-zero so dynamics predicts identity at start
        with torch.no_grad():
            self.head_dynamics[-1].weight.mul_(0.01)
            self.head_dynamics[-1].bias.zero_()
            self.head_correction[-1].weight.mul_(0.01)
            self.head_correction[-1].bias.zero_()
        # Bound correction output: max ±0.05 per dim
        self.correction_bound = 0.05

        # GOAL-PROJECTION HEADS (forward-looking):
        # 1) head_projection: h_goal[B,K,d] -> state8 prediction 16 env steps ahead.
        #    Substrate's job is to project where state will be (goal-direction projection),
        #    trained on (state_t, ..., state_{t+H}) pairs from expert demos.
        # 2) head_zvl_residual: state8_projection -> z_vl residual [B, 2048].
        #    Encodes future-state projection into GR00T's vision-language embedding
        #    space; injected via get_action_with_zvl_override so GR00T sees a "goal hint"
        #    in its conditioning. Bounded ±projection_zvl_bound per dim.
        self.projection_horizon = 16  # env steps ahead
        # head_projection takes (state8_now, pooled_h_goal[d]) -> state8 at t+H.
        # state8_now provides anchor (always real); h_goal provides goal-aware
        # modulation. Trained with real state, lightly-perturbed h_goal so model
        # learns state-driven trajectory with h_goal as a corrective signal.
        self.head_projection = nn.Sequential(
            nn.Linear(state_dim + d, d * 2), nn.SiLU(), nn.LayerNorm(d * 2),
            nn.Linear(d * 2, d * 2), nn.SiLU(),
            nn.Linear(d * 2, state_dim),  # state8 at t+H
        )
        self.head_zvl_residual = nn.Sequential(
            nn.Linear(state_dim, d * 2), nn.SiLU(), nn.LayerNorm(d * 2),
            nn.Linear(d * 2, d * 4), nn.SiLU(), nn.LayerNorm(d * 4),
            nn.Linear(d * 4, z_vl_dim),
        )
        with torch.no_grad():
            self.head_projection[-1].weight.mul_(0.01)
            self.head_projection[-1].bias.zero_()
            self.head_zvl_residual[-1].weight.mul_(0.01)
            self.head_zvl_residual[-1].bias.zero_()
        self.projection_zvl_bound = 0.5  # max |residual| per dim

        # JEPA PREDICTOR HEAD: (h_goal[t] [K,d] + action_chunks[t..t+W-1] flattened)
        # → predicted h_goal[t+W] [K, d]. Trained with stop-grad on target side.
        # Forces substrate's belief to encode trajectory geometry such that future
        # belief is predictable from current belief + intended actions.
        # action_dim*horizon per chunk × W chunks of context = jepa_action_ctx_dim
        # (configured at runtime via project_window kwarg in forward pass).
        # JEPA SUCCESS PREDICTOR HEAD: pooled h_goal[t] → P(trajectory succeeds)
        # Auxiliary head for outcome-aware JEPA training. Forces substrate body to
        # retain outcome-discriminative features (which pure JEPA discards in favor
        # of pure trajectory dynamics).
        self.head_success_predictor = nn.Sequential(
            nn.Linear(K * d, 64), nn.SiLU(), nn.LayerNorm(64),
            nn.Linear(64, 1),
        )
        with torch.no_grad():
            self.head_success_predictor[-1].weight.mul_(0.01)
            self.head_success_predictor[-1].bias.zero_()

        self.jepa_pred_hidden = 256
        self.jepa_pred_in_h = K * d
        # head_jepa_predictor expects (h_pool + action_ctx) where action_ctx is
        # variable size; rebuild on first call if needed. For simplicity here, use
        # a context of action_dim*horizon (one chunk worth of action context = 112).
        # The training script accepts W and builds the predictor accordingly.
        # We define a DEFAULT predictor sized for one-chunk context.
        jepa_action_ctx_dim = horizon * action_dim
        # ============================================================
        # SLOW SUBSTRATE BRANCH (goal-manifold flow).
        # Parallel to the fast substrate body. Reads ONLY z_lang + z_goal
        # (no observation, no chunk, no proprio). Slow effective time-constant
        # via large per-step decay. Carries "what task am I tracking" representation.
        # Coupling readout (head_coupling_predictor) reads (h_slow, h_fast) and
        # predicts future h_fast — high prediction quality = high coupling =
        # legitimate trajectory evolution; low = decoupled = drift signal.
        # ============================================================
        self.K_slow = K  # same K as fast for symmetric coupling readout
        self.d_slow = d
        self.slow_in_lang = nn.Linear(z_vl_dim, d)
        self.slow_in_goal = nn.Linear(z_vl_dim, d)
        nn.init.normal_(self.slow_in_lang.weight, std=0.02)
        nn.init.zeros_(self.slow_in_lang.bias)
        nn.init.normal_(self.slow_in_goal.weight, std=0.02)
        nn.init.zeros_(self.slow_in_goal.bias)
        self.slow_layernorm = nn.LayerNorm(d, elementwise_affine=True)
        # init_belief for slow channel — separate parameter
        self.slow_init_belief = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.slow_init_belief, std=0.05)
        # Slow update: h_slow ← (1 - alpha) * h_slow_prev + alpha * tanh(injection)
        # alpha very small (default 0.01) so slow truly drifts on a different timescale
        # than fast (which updates fully each chunk). Trainable; bounded ≤0.5 via clamp.
        self.slow_alpha = nn.Parameter(torch.tensor(0.01))
        # Trigger detector: from |Δz_lang| over chunks, learn P(intentional goal update)
        # When trigger fires, allow alpha to spike (alpha → alpha * trigger_boost).
        self.head_trigger = nn.Sequential(
            nn.Linear(z_vl_dim, 64), nn.SiLU(),
            nn.Linear(64, 1),  # logit; sigmoid at use
        )
        with torch.no_grad():
            self.head_trigger[-1].weight.mul_(0.01)
            self.head_trigger[-1].bias.fill_(-2.0)  # bias toward "no trigger"
        self.trigger_boost = 10.0  # alpha multiplier when trigger fires

        # COUPLING PREDICTOR: from (h_slow [K,d] + chunks) predict h_fast[t+W].
        # DOES NOT take h_fast — forces slow channel to carry the predictive info.
        # If slow has degenerated to a constant or noise, coupling loss won't beat
        # fast-self-prediction baseline → diagnostic for slow channel health.
        self.head_coupling_predictor = nn.Sequential(
            nn.Linear(K * d + horizon * action_dim, self.jepa_pred_hidden),
            nn.SiLU(), nn.LayerNorm(self.jepa_pred_hidden),
            nn.Linear(self.jepa_pred_hidden, self.jepa_pred_hidden),
            nn.SiLU(), nn.LayerNorm(self.jepa_pred_hidden),
            nn.Linear(self.jepa_pred_hidden, K * d),
        )
        with torch.no_grad():
            self.head_coupling_predictor[-1].weight.mul_(0.01)
            self.head_coupling_predictor[-1].bias.zero_()

        self.head_jepa_predictor = nn.Sequential(
            nn.Linear(self.jepa_pred_in_h + jepa_action_ctx_dim, self.jepa_pred_hidden),
            nn.SiLU(), nn.LayerNorm(self.jepa_pred_hidden),
            nn.Linear(self.jepa_pred_hidden, self.jepa_pred_hidden),
            nn.SiLU(), nn.LayerNorm(self.jepa_pred_hidden),
            nn.Linear(self.jepa_pred_hidden, K * d),  # output predicted h_goal flat
        )
        with torch.no_grad():
            self.head_jepa_predictor[-1].weight.mul_(0.01)
            self.head_jepa_predictor[-1].bias.zero_()
        with torch.no_grad():
            for h in (self.head_goaldist, self.head_gripper_moving,
                      self.head_state_delta):
                h[-1].weight.mul_(0.01)
                h[-1].bias.zero_()

    def init_state(self, batch_size: int, device, dtype=torch.float32):
        return self.init_belief.unsqueeze(0).expand(batch_size, -1, -1).to(
            device=device, dtype=dtype).contiguous()

    def forward_dynamics(self, state8_t: torch.Tensor, action7_t: torch.Tensor,
                         h_intent: torch.Tensor) -> torch.Tensor:
        """Predict state8_{t+1} given current state, action, and substrate's intent
        representation (h_goal). Used at every env step in the gap between GR00T
        chunk emissions to monitor expected trajectory.

        Inputs:
          state8_t   [B, 8]
          action7_t  [B, 7]
          h_intent   [B, K, d]  — substrate's h_goal at chunk emission
        Returns:
          state8_next [B, 8]  — predicted next state (residual prediction)
        """
        e_s = self.dyn_in_state(state8_t)                          # [B, d_dyn]
        e_a = self.dyn_in_action(action7_t)                        # [B, d_dyn]
        e_i = self.dyn_in_intent(h_intent.mean(dim=1))             # [B, d_dyn]
        combined = torch.cat([e_s, e_a, e_i], dim=-1)              # [B, 3*d_dyn]
        delta_state = self.head_dynamics(combined)                  # [B, 8]
        return state8_t + delta_state                               # residual prediction

    def jepa_predict_success(self, h_goal: torch.Tensor) -> torch.Tensor:
        """Aux head: from h_goal[B,K,d] predict P(trajectory succeeds) logit [B]."""
        h_flat = h_goal.flatten(1)
        return self.head_success_predictor(h_flat).squeeze(-1)

    def slow_step(self, h_slow_prev: torch.Tensor, z_lang_t: torch.Tensor,
                    z_goal: torch.Tensor, z_lang_prev=None):
        """One step of slow substrate. Updates h_slow via leaky integrator.

        Inputs to slow channel:
            z_lang_DELTA = z_lang_t - z_lang_prev  (task-change signal; zero
                if no z_lang_prev passed → degrades gracefully to z_lang_t).
            z_goal raw   (target embedding; stays constant within a sub-task,
                identifies which task).
        Using delta (not raw z_lang) prevents slow integrator from absorbing
        constant-language content; it only updates when language actually changes.

        Trigger detector fires on ||z_lang_delta|| → alpha boost.
        """
        if z_lang_prev is None:
            z_lang_delta = z_lang_t  # fallback: first chunk has no prev
        else:
            z_lang_delta = z_lang_t - z_lang_prev
        e_lang = self.slow_in_lang(z_lang_delta)          # [B, d]
        e_goal = self.slow_in_goal(z_goal)                # [B, d]
        evidence = self.slow_layernorm(e_lang + e_goal)   # bounded
        # Trigger from z_lang DELTA magnitude — fires on task changes only
        trigger_logit = self.head_trigger(z_lang_delta).squeeze(-1)
        trigger_prob = torch.sigmoid(trigger_logit)
        # Per-batch alpha: base_alpha + trigger_prob * boost (capped at 0.5)
        alpha = torch.clamp(self.slow_alpha + trigger_prob.unsqueeze(-1).unsqueeze(-1)
                              * self.slow_alpha * self.trigger_boost, max=0.5)
        # Inject: broadcast evidence to all K positions
        injection = torch.tanh(evidence).unsqueeze(1)  # [B, 1, d]
        h_slow_new = (1.0 - alpha) * h_slow_prev + alpha * injection
        return h_slow_new, trigger_prob

    def init_slow_state(self, batch_size: int, device, dtype=torch.float32):
        return self.slow_init_belief.unsqueeze(0).expand(
            batch_size, -1, -1).to(device=device, dtype=dtype).contiguous()

    def coupling_predict(self, h_slow: torch.Tensor,
                          action_ctx: torch.Tensor,
                          h_fast_shape_ref: torch.Tensor) -> torch.Tensor:
        """Predict h_fast[t+W] from (h_slow[t], chunks at t) — NO h_fast input.
        Forces slow channel to carry the predictive information; if slow has
        collapsed to constant or noise, coupling loss won't differ from
        random-output baseline.

        Inputs:
            h_slow [B, K_slow, d_slow]
            action_ctx [B, horizon * action_dim]
            h_fast_shape_ref [B, K, d] — only used for output shape
        Returns: predicted h_fast_future [B, K, d]
        """
        h_slow_flat = h_slow.flatten(1)
        x = torch.cat([h_slow_flat, action_ctx], dim=-1)
        out = self.head_coupling_predictor(x)
        return out.view(h_fast_shape_ref.shape)

    def jepa_predict_future_h_goal(self, h_goal: torch.Tensor,
                                     action_ctx: torch.Tensor) -> torch.Tensor:
        """Predict h_goal[t+W] from current h_goal[t] + intended action context.
        h_goal: [B, K, d]
        action_ctx: [B, horizon * action_dim]  (one chunk OR pooled multi-chunk)
        Returns predicted h_goal_future [B, K, d].
        """
        h_flat = h_goal.flatten(1)  # [B, K*d]
        x = torch.cat([h_flat, action_ctx], dim=-1)
        out = self.head_jepa_predictor(x)  # [B, K*d]
        return out.view(h_goal.shape)

    def project_future_state(self, state8_now: torch.Tensor,
                              h_goal: torch.Tensor) -> torch.Tensor:
        """Predict state8 at t+projection_horizon from current state + goal belief.
        state8_now: [B, state_dim]
        h_goal: [B, K, d]
        Returns state8_future [B, state_dim].

        Substrate's forward prediction of where the goal-direction trajectory
        leads in the next H env steps. state8 anchors the prediction (always
        observable, real at both train and inference); h_goal modulates with
        goal-aware context.
        """
        h_pool = h_goal.mean(dim=1)  # [B, d]
        x = torch.cat([state8_now, h_pool], dim=-1)  # [B, state_dim + d]
        return self.head_projection(x)

    def encode_projection_residual(self, state8_future: torch.Tensor) -> torch.Tensor:
        """Encode predicted future state as a z_vl-space residual to feed back to GR00T.
        state8_future: [B, state_dim] -> z_vl_residual [B, z_vl_dim], bounded.
        """
        raw = self.head_zvl_residual(state8_future)
        return torch.tanh(raw) * self.projection_zvl_bound

    def compute_correction(self, deviation8: torch.Tensor,
                           h_intent: torch.Tensor) -> torch.Tensor:
        """Given state deviation from prediction, output bounded action correction.
        Applied to the next chunk action to make execution adaptive.

        Inputs:
          deviation8  [B, 8]  — actual_state - predicted_state
          h_intent    [B, K, d]
        Returns:
          delta_action7 [B, 7]  — bounded to ±correction_bound per dim
        """
        e_i = self.dyn_in_intent(h_intent.mean(dim=1))             # [B, d_dyn]
        combined = torch.cat([deviation8, e_i], dim=-1)            # [B, 8 + d_dyn]
        raw = self.head_correction(combined)                       # [B, 7]
        return torch.tanh(raw) * self.correction_bound             # bounded

    def step(
        self,
        h_goal_prev: torch.Tensor,
        z_t: torch.Tensor,
        z_goal: torch.Tensor,
        action_chunk_t: torch.Tensor,
        state8_t: torch.Tensor,
        z_lang_t: Optional[torch.Tensor] = None,
        bb_features: Optional[torch.Tensor] = None,
        n_steps_override: Optional[int] = None,
    ):
        B = h_goal_prev.shape[0]
        device = h_goal_prev.device

        # NaN diagnostic: set _STEP_DEBUG class attribute to True to log first NaN
        def _check(name, t):
            if not getattr(self, "_STEP_DEBUG", False):
                return
            if not torch.isfinite(t).all():
                bad = (~torch.isfinite(t)).sum().item()
                print(f"  [substrate.step NaN] {name}: shape={tuple(t.shape)} "
                      f"min={t.min().item():.4f} max={t.max().item():.4f} "
                      f"n_bad={bad}", flush=True)
            else:
                tmin, tmax = t.min().item(), t.max().item()
                if abs(tmin) > 1e3 or abs(tmax) > 1e3:
                    print(f"  [substrate.step LARGE] {name}: min={tmin:.4f} max={tmax:.4f}",
                          flush=True)

        delta = z_goal - z_t
        _check("z_t", z_t); _check("z_goal", z_goal); _check("delta", delta)

        e_z = self.in_z(z_t); _check("e_z", e_z)
        e_g = self.in_goal(z_goal) * self.goal_gate; _check("e_g", e_g)
        e_d = self.in_delta(delta) * self.delta_gate; _check("e_d", e_d)
        chunk_flat = action_chunk_t.reshape(B, -1)
        e_a = self.in_action(chunk_flat) * self.action_gate; _check("e_a", e_a)
        e_s = self.in_state(state8_t) * self.state_gate; _check("e_s", e_s)
        if z_lang_t is None:
            e_l = torch.zeros_like(e_z)
        else:
            e_l = self.in_lang(z_lang_t) * self.lang_gate
        _check("e_l", e_l)
        e_evidence = e_z + e_g + e_d + e_a + e_s + e_l
        # Optional LayerNorm to bound evidence magnitude (gradient-safe).
        if self.use_evidence_layernorm:
            e_evidence = self.evidence_layernorm(e_evidence)
        _check("e_evidence", e_evidence)

        injection = self.evidence_mix.unsqueeze(0) * e_evidence.unsqueeze(1)
        _check("injection", injection); _check("evidence_mix", self.evidence_mix)
        h_input = h_goal_prev + injection
        # SOFT-CLAMP for gradient safety: scaled tanh bounds extreme drift while
        # preserving forward in normal range. Set h_input_clamp=0 to disable.
        if getattr(self, "h_input_clamp", 0.0) > 0.0:
            c = self.h_input_clamp
            h_input = c * torch.tanh(h_input / c)
        _check("h_input", h_input); _check("h_goal_prev", h_goal_prev)

        context = self.context_pool(h_input, None)
        _check("context_pool", context)
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
        _check("euler_solve_out", out[0] if isinstance(out, tuple) else out)
        if isinstance(out, tuple):
            h_goal_new = out[0]
            ponder = out[1]
        else:
            h_goal_new = out
            ponder = torch.zeros(B, device=device)

        pooled = h_goal_new.mean(dim=1)
        pred_goaldist = self.head_goaldist(pooled).squeeze(-1)              # [B]
        pred_gripper_logit = self.head_gripper_moving(pooled).squeeze(-1)    # [B]
        pred_state_delta = self.head_state_delta(pooled).squeeze(-1)         # [B]
        # Per-position virtual tokens [B, K*n_tok_per_k, z_vl_dim] — append to vl_embeds
        vt_raw = self.head_virtual_tokens(h_goal_new)                       # [B, K, z_vl_dim*n_tok]
        virtual_tokens = vt_raw.view(B, self.K * self.n_tok_per_k, self.z_vl_dim)
        # Broadcast residual [B, z_vl_dim] — added to every bb_features token at inference.
        # Bound output via 0.1× scale (matches variant #9's out_scale=0.2 regime; keeps
        # residual norm ~5-10 vs bb_features norm ~30/token, so action head still sees
        # original content with substrate as additive bias rather than overwrite).
        residual = self.head_residual(pooled) * 0.1

        # CROSS-ATTENTION per-token residual (preserves substrate K-structure AND
        # bb_features sequence-structure; breaks the broadcast bottleneck).
        per_token_residual = None
        if bb_features is not None:
            # bb_features: [B, seq, z_vl_dim]
            K_sub = self.head_k_sub(h_goal_new)                        # [B, K, d_attn]
            V_sub = self.head_v_sub(h_goal_new)                        # [B, K, z_vl_dim]
            Q_bb = self.head_q_bb(bb_features.float())                 # [B, seq, d_attn]
            scale = self.d_attn ** 0.5
            attn_logits = torch.matmul(Q_bb, K_sub.transpose(-2, -1)) / scale  # [B, seq, K]
            attn_weights = torch.softmax(attn_logits, dim=-1)          # [B, seq, K]
            per_token_residual = torch.matmul(attn_weights, V_sub)     # [B, seq, z_vl_dim]
            per_token_residual = per_token_residual * self.xattn_out_scale

        g = self.dynamics.compute_metric_diag(h_input)
        metric_cv = g.std() / (g.mean() + 1e-8)

        return h_goal_new, pred_goaldist, {
            "pred_gripper_moving_logit": pred_gripper_logit,
            "pred_state_delta": pred_state_delta,
            "virtual_tokens": virtual_tokens,
            "residual": residual,
            "per_token_residual": per_token_residual,
        }, {
            "metric_cv": metric_cv,
            "ponder": ponder.mean(),
            "n_steps": n_steps,
            "action_gate": self.action_gate.detach(),
            "goal_gate": self.goal_gate.detach(),
            "delta_gate": self.delta_gate.detach(),
            "state_gate": self.state_gate.detach(),
            "lang_gate": self.lang_gate.detach(),
        }
