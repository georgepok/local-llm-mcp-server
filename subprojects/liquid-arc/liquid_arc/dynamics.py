"""ContinuousDynamics — the single shared ODE dynamics module.

This is the core innovation: one weight-tied dynamics module applied 16 times
via Euler integration. ALL computation lives inside the ODE:
  - Metric computation from current h
  - SDPA-based heat kernel diffusion (FlashAttention compatible)
  - LTC contraction: dh/dt = -(1/tau) * (h - target)
  - FFN residual (amortized by n_ode_steps)

The heat kernel K = softmax(-D^2/(4t)) is factored as SDPA:
  K = softmax(q.k/(2t) - ||k||^2/(4t))  where q = k = h * sqrt(g)
This avoids materializing the N*N distance matrix, enabling FlashAttention.
The N*N matrix stays in SRAM — never hits HBM.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LiquidARCConfig


class ContinuousDynamics(nn.Module):
    """Shared dynamics for ODE integration. Weight-tied across all steps.

    dh/dt = -(1/tau) * (h - target) + FFN(h) / n_ode_steps
    where target = h + W_o(alpha*V + (1-alpha)*SDPA(q,k,V))
    """

    def __init__(self, config: LiquidARCConfig):
        super().__init__()
        d = config.d_model
        d_met = config.d_metric
        # Use a buffer tensor so torch.compile treats it as dynamic (not static int)
        self.register_buffer('_n_ode_steps', torch.tensor(config.n_ode_steps, dtype=torch.float32),
                             persistent=False)
        self.tau_min = config.tau_min
        self.tau_max = getattr(config, 'tau_max', 3.0)

        # Pre-norms
        self.norm_geo = nn.LayerNorm(d)
        self.norm_val = nn.LayerNorm(d)
        self.norm_ff = nn.LayerNorm(d)

        # MetricNet: [LN(h) || ctx] -> bottleneck -> g via Softplus
        # Fluid metric: optionally wider bottleneck + low-rank off-diagonal
        d_metric_bn = getattr(config, 'd_metric_bottleneck', 0) or d_met
        self.metric_rank = getattr(config, 'metric_rank', 0)
        self._d_metric_bn = d_metric_bn

        self.metric_net_linear1 = nn.Linear(2 * d, d_metric_bn)
        self.metric_net_linear2_diag = nn.Linear(d_metric_bn, d)

        # Init for identity metric: Softplus(1.3133) ~ 1.0
        # Optional bias noise (metric_bias_init_std) breaks the flat-metric
        # fixed point — with std>0 the metric has per-dim variance at init,
        # so gradient can escape from flat-soup equilibrium. No-op when 0.
        with torch.no_grad():
            self.metric_net_linear2_diag.bias.fill_(math.log(math.e - 1))
            bias_noise_std = getattr(config, 'metric_bias_init_std', 0.0)
            if bias_noise_std > 0:
                self.metric_net_linear2_diag.bias.add_(
                    torch.randn_like(self.metric_net_linear2_diag.bias)
                    * bias_noise_std)
            nn.init.normal_(self.metric_net_linear2_diag.weight, std=0.05)

        # Low-rank factors: g = diag(D) + L·L^T
        # Small random init (not zero!) — bilinear form has zero gradient at zero
        if self.metric_rank > 0:
            self.metric_net_linear2_lr = nn.Linear(d_metric_bn, d * self.metric_rank)
            nn.init.normal_(self.metric_net_linear2_lr.weight, std=0.001)
            nn.init.zeros_(self.metric_net_linear2_lr.bias)

        # Step-evolving metric: learnable step embeddings modulate MetricNet
        self.step_embed_enabled = getattr(config, 'step_embed_enabled', False)
        if self.step_embed_enabled:
            n_embeds = getattr(config, 'n_step_embeds', 20)
            self.step_embeds = nn.Parameter(torch.zeros(n_embeds, d_metric_bn))
            self.register_buffer('_current_step_embed', torch.zeros(d_metric_bn),
                                 persistent=False)

        # Channel-wise gate (working memory) OR scalar TauNet
        self.channel_gate_enabled = getattr(config, 'channel_gate_enabled', False)
        if self.channel_gate_enabled:
            # Gate replaces tau: d-dimensional per-position gate
            self.gate_net_linear1 = nn.Linear(d, d_met)
            self.gate_net_linear2 = nn.Linear(d_met, d)
            # Init: sigmoid(2.0) ≈ 0.88 ≈ current 1/tau_init ≈ 1.0
            with torch.no_grad():
                self.gate_net_linear2.bias.fill_(2.0)
        else:
            # TauNet: per-position adaptive time constant
            self.tau_net_linear1 = nn.Linear(d, d_met)
            self.tau_net_linear2 = nn.Linear(d_met, 1)
            # Init tau to softplus_inv(1.0) so initial tau ~ 1.0 + tau_min
            with torch.no_grad():
                self.tau_net_linear2.bias.fill_(math.log(math.e - 1))

        # Step-aware tau: lets TauNet know WHERE in the ODE it is
        # Zero-init → no-op at start, checkpoint-compatible (strict=False)
        self.tau_step_embed = nn.Embedding(20, d_met)  # 20 max steps
        nn.init.zeros_(self.tau_step_embed.weight)

        # === Step-conditional operator (flexible geometric reasoning Tier 1) ===
        # FiLM modulation on MetricNet + TauNet bottlenecks, per-step t_diffusion.
        # Init as no-op: γ=1, β=0, t_diff[s]=softplus⁻¹(t_diffusion_init).
        # Preserves backward compat with pre-Tier1 checkpoints (strict=False load).
        self.step_conditional_enabled = getattr(
            config, 'step_conditional_operator', False)
        if self.step_conditional_enabled:
            self._step_cond_n_max = getattr(config, 'step_conditional_n_max', 32)
            n_max = self._step_cond_n_max
            # MetricNet FiLM (on bottleneck features, dim d_metric_bn)
            self.metric_film_gamma = nn.Embedding(n_max, d_metric_bn)
            self.metric_film_beta  = nn.Embedding(n_max, d_metric_bn)
            nn.init.ones_(self.metric_film_gamma.weight)
            nn.init.zeros_(self.metric_film_beta.weight)
            # TauNet FiLM (on bottleneck features, dim d_met) — skip if channel_gate path
            if not self.channel_gate_enabled:
                self.tau_film_gamma = nn.Embedding(n_max, d_met)
                self.tau_film_beta  = nn.Embedding(n_max, d_met)
                nn.init.ones_(self.tau_film_gamma.weight)
                nn.init.zeros_(self.tau_film_beta.weight)
            # Per-step diffusion radius. Init to softplus⁻¹(t_diffusion_init)
            # so per-step t_diff equals baseline scalar at every step initially.
            self.t_diff_per_step = nn.Embedding(n_max, 1)
            inv_softplus_init = math.log(
                math.exp(float(config.t_diffusion_init)) - 1.0)
            with torch.no_grad():
                self.t_diff_per_step.weight.fill_(inv_softplus_init)

        # === Tier 3: Per-position ACT-style halting ===
        # Each ODE step computes a halt probability per position. A "still_active"
        # fraction (product of (1 - p_halt) across steps) multiplies dh — positions
        # with high cumulative halt stop contributing. Ponder cost (mean of
        # still_active across steps) rewards earlier halting.
        # Init: halt_head zeroed + large negative bias → p_halt ≈ 0 → still_active
        # ≈ 1 → baseline behavior preserved. Training drives halt_head away from
        # zero where fewer steps suffice.
        # With halting, n_ode_steps becomes a MAX, not fixed: hard positions get
        # the full budget, easy ones exit early.
        self.halting_enabled = getattr(config, 'halting_enabled', False)
        if self.halting_enabled:
            self.halt_head = nn.Linear(d, 1, bias=True)
            with torch.no_grad():
                nn.init.zeros_(self.halt_head.weight)
                self.halt_head.bias.fill_(-10.0)  # sigmoid(-10) ≈ 4.5e-5 ≈ 0

        # === Tier 2: FiLM on attention routing (Q/K projections) ===
        # Only relevant when routing_mode uses attention (attention or coupled).
        # Each step gets its own γ/β modulation of h_normed before W_q/W_k.
        # This lets attention ROUTING (not just geometric) specialize per step.
        # Controlled by separate flag for granular A/B testing.
        self.step_conditional_qk_enabled = getattr(
            config, 'step_conditional_qk', False)
        if (self.step_conditional_qk_enabled
                and getattr(config, 'routing_mode', 'metric') in
                ('attention', 'coupled')):
            n_max_qk = getattr(config, 'step_conditional_n_max', 32)
            self._step_cond_qk_n_max = n_max_qk
            self.qk_film_gamma = nn.Embedding(n_max_qk, d)
            self.qk_film_beta  = nn.Embedding(n_max_qk, d)
            nn.init.ones_(self.qk_film_gamma.weight)
            nn.init.zeros_(self.qk_film_beta.weight)
        else:
            self.step_conditional_qk_enabled = False

        # τ-CV coupling: tau responds to local metric complexity
        self.tau_cv_coupling_enabled = getattr(config, 'tau_cv_coupling_enabled', False)
        self.cv_coupling_target = getattr(config, 'cv_coupling_target', 3.5)
        self.cv_coupling_strength = getattr(config, 'cv_coupling_strength', 0.5)

        # Tau-convergence coupling: positions struggling to converge get faster integration
        self.tau_convergence_coupling_enabled = getattr(config, 'tau_convergence_coupling_enabled', False)
        self.tau_convergence_beta = getattr(config, 'tau_convergence_beta', 1.0)
        self.tau_convergence_floor = getattr(config, 'tau_convergence_floor', 0.5)

        # Structural tau — input-INDEPENDENT per-position timescale.
        # Modulates dynamic tau at BOTH inference time (tau_dynamic * s_tau)
        # and learning time (gradient scaling via apply_structural_gradient_coupling).
        # Ones-init → sigmoid(1)≈0.73, s_tau≈2.3 — starts slow, differentiates via training.
        self.structural_tau_enabled = getattr(config, 'structural_tau_enabled', False)
        if self.structural_tau_enabled:
            self.structural_tau_min = getattr(config, 'structural_tau_min', 0.3)
            self.structural_tau_max = getattr(config, 'structural_tau_max', 3.0)
            self.structural_tau = nn.Parameter(torch.ones(config.max_seq_len))

        # Hierarchical tau schedule: initialized by init_hierarchical_tau()
        # When set, biases early steps toward low tau (fast) and late steps toward high tau (slow)
        self._hierarchical_tau_init = False

        # Value and output projections
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)

        # Multi-timescale local learning: per-batch low-rank fast-weight overlay
        # on W_o. F = U·V^T accumulates outer(post, pre) by Hebbian rule across
        # ODE steps, reset to 0 at the start of each forward pass. No gradient
        # flows through the Hebbian update — it is a genuine local rule running
        # at a faster timescale than gradient learning.
        self.fast_weights_enabled = getattr(config, 'fast_weights_enabled', False)
        if self.fast_weights_enabled:
            self.fast_weights_rank = int(getattr(config, 'fast_weights_rank', 4))
            self.fast_weights_eta = float(getattr(config, 'fast_weights_eta', 0.01))
            self.fast_weights_decay = float(
                getattr(config, 'fast_weights_decay', 0.05))
            r = self.fast_weights_rank
            # Per-batch state buffers (resized via reset_fast_weights).
            self.register_buffer('_fast_U', torch.zeros(1, d, r), persistent=False)
            self.register_buffer('_fast_V', torch.zeros(1, d, r), persistent=False)
            # FIXED random projections (sketching) for compile-safe rank-r
            # Hebbian update. Persistent so every run uses the same projection
            # (deterministic, reproducible). Initialized once.
            with torch.no_grad():
                proj_pre = torch.randn(d, r) / math.sqrt(d)
                proj_post = torch.randn(d, r) / math.sqrt(d)
            self.register_buffer('_fw_proj_pre', proj_pre, persistent=True)
            self.register_buffer('_fw_proj_post', proj_post, persistent=True)

        # Routing mode: metric / attention / coupled.
        self.routing_mode = getattr(config, 'routing_mode', 'metric')
        if self.routing_mode in ('attention', 'coupled'):
            self.W_q = nn.Linear(d, d, bias=False)
            self.W_k = nn.Linear(d, d, bias=False)
            with torch.no_grad():
                nn.init.xavier_uniform_(self.W_q.weight)
                nn.init.xavier_uniform_(self.W_k.weight)
        if self.routing_mode == 'coupled':
            self.W_gate = nn.Linear(d, 1, bias=True)
            with torch.no_grad():
                nn.init.normal_(self.W_gate.weight, std=0.01)
                self.W_gate.bias.fill_(0.0)

        # Sparse activation (Probe 4). When sparse_fraction > 0, only the
        # top-k positions (by learned activity score) receive dh/dt updates
        # this step; the rest hold. Tests biological sparsity hypothesis.
        self.sparse_fraction = getattr(config, 'sparse_fraction', 0.0)
        self.sparse_random = getattr(config, 'sparse_random', False)
        if self.sparse_fraction > 0:
            self.W_activity = nn.Linear(d, 1, bias=True)
            with torch.no_grad():
                nn.init.normal_(self.W_activity.weight, std=0.01)
                self.W_activity.bias.fill_(2.0)  # sigmoid(2)≈0.88 → most active early

        # Learned diffusion timescale
        self.t_diffusion = nn.Parameter(torch.tensor(config.t_diffusion_init))

        # Identity residual — sigmoid(2.2) ~ 0.90 self-attention at init
        self.alpha_logit = nn.Parameter(torch.tensor(config.alpha_logit_init))

        # ReZero-style gate on dh/dt. When enabled, the entire ODE update is
        # multiplied by sigmoid(rezero_gate_logit). Init at a large negative
        # value → near-zero gate → identity at step 0. The gate receives
        # gradient directly (dL/dγ = ⟨dh_dt, dL/dy⟩), so it grows iff the
        # dynamics produces loss-reducing updates. Lets the geometry wake
        # from a dead-init state without scheduling.
        self.rezero_enabled = getattr(config, 'rezero_enabled', False)
        if self.rezero_enabled:
            init_val = float(getattr(config, 'rezero_gate_init', -5.0))
            self.rezero_gate_logit = nn.Parameter(torch.tensor(init_val))

        # Self-organizing identity routing — adds a parallel SDPA branch using
        # raw h-similarity (no learned g scaling), plus a per-batch EMA
        # accumulator across ODE steps. Validated on toy substrate (see
        # research/self_org_sim/).
        self.identity_routing_enabled = getattr(
            config, 'identity_routing_enabled', False)
        if self.identity_routing_enabled:
            init_val = float(getattr(
                config, 'identity_routing_alpha_init', 0.0))
            self.identity_routing_alpha = nn.Parameter(torch.tensor(init_val))
            self.identity_routing_decay = float(getattr(
                config, 'identity_routing_decay', 0.1))
            # EMA accumulator buffer — sized lazily per-batch.
            self.register_buffer('_id_history', torch.zeros(1, 1, d),
                                 persistent=False)

        # FFN inside dynamics (applied at every step, amortized)
        self.ffn = nn.Sequential(
            nn.Linear(d, config.d_ffn),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ffn, d),
        )

        # Normal init for W_o and FFN — the residual pattern (target = h + update)
        # already prevents signal destruction. Non-zero W_o breaks copy symmetry
        # so positions get different perturbations from SDPA routing.

        # Fast metric overlay from working memory (set by euler_solve_with_memory)
        self._metric_overlay = None

        # Norm homeostasis: soft decay that activates above a reference norm.
        # norm_ref = typical embedding scale the ODE should operate at.
        # norm_lambda = strength of restoring force (higher = stronger pull back).
        self._norm_ref = getattr(config, 'norm_ref', 50.0)  # per-position L2 reference
        self._norm_lambda = getattr(config, 'norm_lambda', 0.1)  # decay rate

        # Progressive damping: later ODE steps make smaller updates
        self._progressive_damping = getattr(config, 'progressive_damping', False)
        self._damping_strength = getattr(config, 'damping_strength', 0.5)

        # Metric freeze: cache g at a specific ODE step, reuse for later steps
        self.metric_freeze_step = getattr(config, 'metric_freeze_step', -1)
        self.metric_freeze_after_training_step = getattr(config, 'metric_freeze_after_training_step', 0)
        self._cached_g = None
        # Boolean flag set by training loop — plain bool causes one recompile when it flips
        self.metric_freeze_active: bool = False

        # System 2: EMA of metric weights for multi-timescale routing
        self.system2_enabled = getattr(config, 'system2_enabled', False)
        self._ema_momentum = getattr(config, 'system2_ema_momentum', 0.995)
        self._ema_max_alpha = getattr(config, 'system2_max_alpha', 0.4)
        self._ema_w1 = None
        self._ema_b1 = None
        self._ema_w2 = None
        self._ema_b2 = None
        # Use buffer tensors so torch.compile treats them as dynamic, not static ints
        self.register_buffer('_current_step_index_buf',
                             torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer('_current_n_steps_buf',
                             torch.tensor(16, dtype=torch.long), persistent=False)

        # HyperNet delta for W_o — zero buffer = original behavior (W_o.weight + 0)
        self.register_buffer('_delta_W_o', torch.zeros(d, d), persistent=False)

        # Stored context (set once before ODE, read at every step)
        self._context: Optional[torch.Tensor] = None
        self._mask: Optional[torch.Tensor] = None

        # External τ bias (set by Mind for per-event τ floor)
        self._tau_external_bias: Optional[torch.Tensor] = None  # [N] or None

        # Tau freeze flag — plain bool, one recompile at unfreeze is acceptable
        self.freeze_tau: bool = True

    def init_hierarchical_tau(self, tau_fast: float = 0.2, tau_slow: float = 0.9,
                               n_steps: int = 16):
        """Initialize tau_step_embed with a hierarchical schedule.

        Biases early ODE steps toward low tau (fast reactive dynamics) and
        late steps toward high tau (slow consolidation). Like a nervous system:
          - Steps 0-3:  spinal reflex (fast, tau ~ tau_fast)
          - Steps 4-10: motor coordination (medium)
          - Steps 11-15: cortical consolidation (slow, tau ~ tau_slow)

        The tau_step_embed adds to TauNet hidden state BEFORE the final linear
        projection → sigmoid. We compute the bias needed to shift sigmoid output
        toward the desired tau value.

        Args:
            tau_fast: target tau for step 0 (e.g., 0.2 = fast dynamics)
            tau_slow: target tau for the final step (e.g., 0.9 = slow)
            n_steps: number of ODE steps to schedule
        """
        import math
        with torch.no_grad():
            # tau = sigmoid(logit) * (tau_max - tau_min) + tau_min
            # We want to bias the logit toward specific tau values at each step.
            # The tau_step_embed is added to the hidden state before linear2,
            # so we scale our bias by the approximate gain of linear2.
            # Use a uniform direction in d_met space scaled by desired magnitude.
            d_met = self.tau_step_embed.weight.shape[1]

            for step in range(min(n_steps, 20)):
                # Linear interpolation from tau_fast to tau_slow
                frac = step / max(n_steps - 1, 1)
                target_tau = tau_fast + frac * (tau_slow - tau_fast)

                # Convert to sigmoid logit: sigmoid(x) = (target - tau_min) / (tau_max - tau_min)
                sig_target = (target_tau - self.tau_min) / (self.tau_max - self.tau_min)
                sig_target = max(0.01, min(0.99, sig_target))  # clamp for numerical safety
                target_logit = math.log(sig_target / (1.0 - sig_target))

                # The default TauNet bias produces sigmoid ≈ 1.0 (tau ≈ tau_max)
                # We want to shift it to target_logit. The step_embed is added
                # to tau_hidden before linear2, so the effect depends on linear2 weights.
                # Use a scaled uniform vector that linear2 maps to approximately target_logit shift.
                # Approximate: embed magnitude ≈ desired_shift / sqrt(d_met) * scale
                # This is rough — the model will learn the exact values
                desired_shift = target_logit - math.log(math.e - 1)  # shift from default init
                scale = desired_shift / math.sqrt(d_met) * 2.0  # factor for linear layer gain
                self.tau_step_embed.weight[step] = scale

        self._hierarchical_tau_init = True
        print(f"Hierarchical tau initialized: step 0 → tau≈{tau_fast:.2f}, "
              f"step {n_steps-1} → tau≈{tau_slow:.2f}")

    def set_context(self, context: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Set episode context before ODE integration."""
        self._context = context
        self._mask = mask
        self._cached_g = None  # reset metric cache for new input
        self._cached_L = None

    def reset_fast_weights(self, batch_size: int, device, dtype):
        """Zero the per-batch Hebbian fast weights at the start of an ODE pass.

        Called by the solver before iteration. Buffers are resized if batch
        size changed. No-op when fast_weights_enabled is False.
        """
        if not self.fast_weights_enabled:
            return
        d = self.W_o.weight.shape[0]
        r = self.fast_weights_rank
        self._fast_U = torch.zeros(batch_size, d, r, device=device, dtype=dtype)
        self._fast_V = torch.zeros(batch_size, d, r, device=device, dtype=dtype)

    def reset_id_history(self, batch_size: int, seq_len: int, device, dtype):
        """Zero the identity-routing EMA accumulator at the start of an ODE
        pass.  Called by the solver. No-op when identity_routing disabled.
        """
        if not self.identity_routing_enabled:
            return
        d = self.W_o.weight.shape[0]
        self._id_history = torch.zeros(batch_size, seq_len, d,
                                         device=device, dtype=dtype)

    def set_n_steps(self, n_steps: int):
        """Override step count for FFN amortization — no inplace to avoid autograd conflicts."""
        self._n_ode_steps = torch.tensor(float(n_steps), device=self._n_ode_steps.device)

    def set_delta_W_o(self, delta: Optional[torch.Tensor] = None):
        """Set HyperNet W_o delta. Must be called before every forward pass.

        This buffer is NOT persisted in state_dict (persistent=False). The caller
        is responsible for setting it before each ODE integration. Not setting it
        (or passing None) restores original behavior (no W_o modification).

        Both branches use direct assignment (not in-place ops) because after
        set_delta_W_o(hypernet_output), _delta_W_o may be a grad-tracking tensor
        where in-place zero_() is illegal. Direct assignment also keeps the
        hypernet output in the autograd graph for gradient flow. Same pattern as
        set_step_embed.

        Called BEFORE the compiled forward, not during it — torch.compile sees
        the tensor as a dynamic input on each call.
        """
        if delta is None:
            self._delta_W_o = torch.zeros_like(self.W_o.weight)
        else:
            self._delta_W_o = delta

    def forward(self, t_ode, h: torch.Tensor) -> torch.Tensor:
        """Compute dh/dt. Called n_ode_steps times with same weights.

        The heat kernel is computed via SDPA (FlashAttention):
          softmax(-D^2/(4t)) = softmax(q.k/(2t) - ||k_j||^2/(4t))
        where q = k = h_normed * sqrt(g). The ||h_i||^2_g term drops
        out by softmax row-invariance. The N*N matrix never hits HBM.

        Args:
            t_ode: current ODE time (unused — autonomous system)
            h: [B, N, d] current hidden states

        Returns:
            dh_dt: [B, N, d] time derivative
        """
        B, N, d = h.shape
        context = self._context
        mask = self._mask

        # Asymmetric-attention branch (either full-attention or half of coupled).
        routed_v_attn = None
        gate_val = None
        if self.routing_mode in ('attention', 'coupled'):
            h_normed = self.norm_geo(h)
            # Tier 2: step-conditional FiLM on inputs to Q/K. No-op at init.
            if self.step_conditional_qk_enabled:
                _s_qk = self._current_step_index_buf.long().clamp(
                    0, self._step_cond_qk_n_max - 1)
                _gqk = self.qk_film_gamma(_s_qk).view(1, 1, -1)
                _bqk = self.qk_film_beta(_s_qk).view(1, 1, -1)
                h_qk = _gqk * h_normed + _bqk
            else:
                h_qk = h_normed
            Q = self.W_q(h_qk)
            K = self.W_k(h_qk)
            V = self.W_v(self.norm_val(h))
            scale_factor = 1.0 / math.sqrt(d)
            attn_mask_sdpa = None
            if mask is not None:
                mask_bc = mask if mask.dim() == 3 else mask.unsqueeze(0)
                attn_mask_sdpa = torch.zeros_like(
                    mask_bc, dtype=Q.dtype, device=Q.device)
                attn_mask_sdpa = attn_mask_sdpa.masked_fill(
                    mask_bc, float('-inf'))
            routed_v_attn = F.scaled_dot_product_attention(
                query=Q * scale_factor, key=K, value=V,
                attn_mask=attn_mask_sdpa,
            )
            if self.routing_mode == 'attention':
                # Pure attention: no metric path, no gate.
                routed_v = routed_v_attn
                g = None
                L = None
            else:
                # 'coupled' — gate computed now, metric path runs below
                gate_val = torch.sigmoid(self.W_gate(h_normed))  # [B, N, 1]

        # 1. Metric-routing path (wraps the whole heat-kernel block).
        # Runs in 'metric' mode and also in 'coupled' mode (where its output
        # is blended with the attention path via the sigmoid gate).
        if self.routing_mode in ('metric', 'coupled'):
            h_normed = self.norm_geo(h)
            ctx_exp = context.unsqueeze(1).expand(B, N, d)

            step_idx = self._current_step_index_buf

            cat_input = torch.cat([h_normed, ctx_exp], dim=-1)  # [B, N, 2d]

            freeze_active = self.metric_freeze_active
            if (freeze_active
                    and step_idx > self.metric_freeze_step
                    and self._cached_g is not None):
                g = self._cached_g
                L = self._cached_L if hasattr(self, '_cached_L') else None
            else:
                met_hidden = F.gelu(self.metric_net_linear1(cat_input))
                if self._metric_overlay is not None:
                    met_hidden = met_hidden + self._metric_overlay
                if self.step_embed_enabled:
                    met_hidden = met_hidden + self._current_step_embed
                # Step-conditional FiLM on MetricNet bottleneck (no-op at init)
                if self.step_conditional_enabled:
                    _s = self._current_step_index_buf.long().clamp(
                        0, self._step_cond_n_max - 1)
                    _gamma = self.metric_film_gamma(_s).view(1, 1, -1)
                    _beta = self.metric_film_beta(_s).view(1, 1, -1)
                    met_hidden = _gamma * met_hidden + _beta
                g = F.softplus(self.metric_net_linear2_diag(met_hidden))
                L = None
                if self.metric_rank > 0:
                    L_flat = self.metric_net_linear2_lr(met_hidden)
                    L = L_flat.view(B, N, d, self.metric_rank)
                if freeze_active and step_idx == self.metric_freeze_step:
                    self._cached_g = g.detach()
                    self._cached_L = L.detach() if self.metric_rank > 0 else None

            if self.system2_enabled and self._ema_w1 is not None and self._cached_g is None:
                step_idx = self._current_step_index_buf
                n_total = self._current_n_steps_buf
                alpha = (step_idx / max(n_total - 1, 1)) * self._ema_max_alpha
                if alpha > 0.01:
                    with torch.no_grad():
                        slow_hidden = F.gelu(F.linear(cat_input, self._ema_w1, self._ema_b1))
                        if self.step_embed_enabled:
                            slow_hidden = slow_hidden + self._current_step_embed
                        slow_g = F.softplus(F.linear(slow_hidden, self._ema_w2, self._ema_b2))
                    g = (1.0 - alpha) * g + alpha * slow_g

            if self.system2_enabled and self._current_step_index_buf == 0 and self.training:
                with torch.no_grad():
                    m = self._ema_momentum
                    w1 = self.metric_net_linear1.weight.data
                    b1 = self.metric_net_linear1.bias.data
                    w2 = self.metric_net_linear2_diag.weight.data
                    b2 = self.metric_net_linear2_diag.bias.data
                    if self._ema_w1 is None:
                        self._ema_w1 = w1.clone()
                        self._ema_b1 = b1.clone()
                        self._ema_w2 = w2.clone()
                        self._ema_b2 = b2.clone()
                    else:
                        self._ema_w1 = m * self._ema_w1 + (1.0 - m) * w1
                        self._ema_b1 = m * self._ema_b1 + (1.0 - m) * b1
                        self._ema_w2 = m * self._ema_w2 + (1.0 - m) * w2
                        self._ema_b2 = m * self._ema_b2 + (1.0 - m) * b2

            # Step-conditional t_diffusion: per-step diffusion radius
            if self.step_conditional_enabled:
                _s_td = self._current_step_index_buf.long().clamp(
                    0, self._step_cond_n_max - 1)
                t_diff = F.softplus(self.t_diff_per_step(_s_td).squeeze())
            else:
                t_diff = F.softplus(self.t_diffusion)
            sqrt_g = torch.sqrt(g)
            q_diag = h_normed * sqrt_g
            k_diag = h_normed * sqrt_g
            if self.metric_rank > 0:
                h_proj = torch.einsum('bnd,bndr->bnr', h_normed, L)
                Q = torch.cat([q_diag, h_proj], dim=-1)
                K = torch.cat([k_diag, h_proj], dim=-1)
                d_qk = d + self.metric_rank
            else:
                Q = q_diag
                K = k_diag
                d_qk = d

            k_norm_sq = (K * K).sum(dim=-1, keepdim=True)
            attn_bias = -k_norm_sq.transpose(1, 2) / (4.0 * t_diff)

            V = self.W_v(self.norm_val(h))

            if mask is not None:
                full_bias = attn_bias.expand(B, N, N).clone()
                mask_bc = mask if mask.dim() == 3 else mask.unsqueeze(0)
                full_bias.masked_fill_(mask_bc, float('-inf'))
                attn_bias = full_bias

            if self.metric_rank > 0:
                logits = torch.bmm(Q, K.transpose(1, 2)) / (2.0 * t_diff)
                logits = logits + attn_bias
                attn_weights = F.softmax(logits, dim=-1)
                routed_v_metric = torch.bmm(attn_weights, V)
            else:
                scale_factor = math.sqrt(d_qk) / (2.0 * t_diff)
                Q_scaled = Q * scale_factor
                routed_v_metric = F.scaled_dot_product_attention(
                    query=Q_scaled, key=K, value=V,
                    attn_mask=attn_bias,
                )
            # In 'metric' mode this IS the final routed_v; in 'coupled'
            # we blend it with routed_v_attn via the sigmoid gate below.
            if self.routing_mode == 'metric':
                routed_v = routed_v_metric

            # Self-organizing identity routing — parallel SDPA branch using
            # raw h-similarity (no g scaling), accumulated across ODE steps
            # via per-batch EMA. Toy substrate validation: this kind of
            # similarity-based update self-organizes connection structure
            # to match input topology.
            if self.identity_routing_enabled:
                # For mask handling, reuse attn_bias (it already includes -inf
                # for blocked positions). The metric-bias term on it (k_norm/4t)
                # is small relative to identity-similarity logits and acts as a
                # weak regularizer on the identity branch — acceptable.
                routed_v_id = F.scaled_dot_product_attention(
                    query=h_normed, key=h_normed, value=V,
                    attn_mask=attn_bias,
                )
                # Update EMA accumulator (no gradient — local Hebbian rule)
                with torch.no_grad():
                    decay = self.identity_routing_decay
                    self._id_history = (
                        (1.0 - decay) * self._id_history
                        + decay * routed_v_id)
                # Mix current identity routing with accumulated history,
                # then mix with the metric routing via learned alpha.
                alpha_id = torch.sigmoid(self.identity_routing_alpha)
                routed_v_id_eff = 0.5 * routed_v_id + 0.5 * self._id_history
                routed_v = (1.0 - alpha_id) * routed_v + alpha_id * routed_v_id_eff

        # Coupled mode: blend metric and attention paths via learned gate.
        # gate_val is [B, N, 1] sigmoid; gate=1 → pure attention, gate=0 → pure metric.
        if self.routing_mode == 'coupled':
            routed_v = gate_val * routed_v_attn + (1.0 - gate_val) * routed_v_metric
            # Store gate value for diagnostic logging (detached).
            self._last_gate_mean = gate_val.mean().detach()
            self._last_gate_std = gate_val.std().detach()

        # 4. Zero-init residual target (W_o starts at zeros -> dh/dt ~ 0)
        # F.linear with delta allows HyperNet to modulate W_o per-task.
        # When _delta_W_o is zero (default), this is equivalent to self.W_o(routed_v).
        update = F.linear(routed_v, self.W_o.weight + self._delta_W_o, None)
        # Multi-timescale fast-weight overlay: per-batch low-rank Hebbian
        # correction.  fw_update = routed_v @ U @ V^T   (batched bmm).
        # Hebbian update sketches outer(routed_v, update) via fixed random
        # projections — deterministic, compile-safe, no random calls inside
        # the trace.
        if self.fast_weights_enabled:
            fw_proj = torch.bmm(routed_v, self._fast_U)              # [B,N,r]
            fw_update = torch.bmm(fw_proj,
                                   self._fast_V.transpose(1, 2))      # [B,N,d]
            update = update + fw_update
            with torch.no_grad():
                N_ = update.shape[1]
                # Sketched outer-product: project pre/post through fixed
                # random matrices, then average across positions.
                pre_sketch = torch.einsum(
                    'bnd,dr->bnr', routed_v, self._fw_proj_pre)       # [B,N,r]
                post_sketch = torch.einsum(
                    'bnd,dr->bnr', update, self._fw_proj_post)        # [B,N,r]
                upd_U = torch.einsum('bnd,bnr->bdr',
                                       routed_v, post_sketch) / N_     # [B,d,r]
                upd_V = torch.einsum('bnd,bnr->bdr',
                                       update, pre_sketch) / N_        # [B,d,r]
                eta = self.fast_weights_eta
                lam = self.fast_weights_decay
                # Decay + Hebbian + clamp (numerical stability under bf16)
                self._fast_U = ((1.0 - lam) * self._fast_U
                                + eta * upd_U).clamp(-1.0, 1.0)
                self._fast_V = ((1.0 - lam) * self._fast_V
                                + eta * upd_V).clamp(-1.0, 1.0)
        target = h + update

        # 5. Gate/Tau: per-position dynamics control
        if self.channel_gate_enabled:
            # Channel-wise gate: [B, N, d] — each dimension independently holds or flows
            if self.freeze_tau:
                gate = torch.full((B, N, d), 0.88, device=h.device, dtype=h.dtype)
            else:
                gate = torch.sigmoid(self.gate_net_linear2(
                    F.gelu(self.gate_net_linear1(h_normed))
                ))  # [B, N, d]
            # 6. LTC: decay toward diffusion target (channel-wise)
            dh_dt = -gate * (h - target)
        else:
            # Scalar tau: per-position time constant (bounded sigmoid)
            # Step-aware: TauNet knows WHERE in the ODE it is via step embedding
            if self.freeze_tau:
                tau = torch.ones(B, N, 1, device=h.device, dtype=h.dtype)
            else:
                tau_hidden = F.gelu(self.tau_net_linear1(h_normed))
                # Add step embedding — model learns to produce different tau at different steps
                # Early steps: low tau (aggressive processing), Late steps: high tau (coast)
                step_idx = self._current_step_index_buf.long().clamp(0, 19)
                tau_hidden = tau_hidden + self.tau_step_embed(step_idx)
                # Step-conditional FiLM on TauNet bottleneck (no-op at init)
                if self.step_conditional_enabled:
                    _s_t = self._current_step_index_buf.long().clamp(
                        0, self._step_cond_n_max - 1)
                    _gt = self.tau_film_gamma(_s_t).view(1, 1, -1)
                    _bt = self.tau_film_beta(_s_t).view(1, 1, -1)
                    tau_hidden = _gt * tau_hidden + _bt
                tau_logits = self.tau_net_linear2(tau_hidden)
                # External τ bias (from Mind's per-event τ floor)
                if self._tau_external_bias is not None:
                    bias = self._tau_external_bias[:N].unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
                    tau_logits = tau_logits + bias
                tau = torch.sigmoid(tau_logits) * (self.tau_max - self.tau_min) + self.tau_min
                # Structural tau: input-independent per-position multiplier
                if self.structural_tau_enabled:
                    s_raw = self.structural_tau[:N]  # [N]
                    s_tau = self.structural_tau_min + (
                        self.structural_tau_max - self.structural_tau_min
                    ) * torch.sigmoid(s_raw)
                    s_tau = s_tau.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
                    tau = (tau * s_tau).clamp(
                        min=self.tau_min,
                        max=self.tau_max * self.structural_tau_max,
                    )
                # τ-CV coupling: tau responds to local metric complexity.
                # Requires a metric (g); skipped in 'attention' routing mode.
                if self.tau_cv_coupling_enabled and self.routing_mode == 'metric':
                    with torch.no_grad():
                        g_mean_local = g.mean(dim=-1, keepdim=True)   # [B, N, 1]
                        g_std_local = g.std(dim=-1, keepdim=True)      # [B, N, 1]
                        local_cv = g_std_local / (g_mean_local + 1e-8) # [B, N, 1]
                    cv_target = self.cv_coupling_target
                    alpha = self.cv_coupling_strength
                    coupling_factor = 1.0 + alpha * (local_cv - cv_target)
                    coupling_factor = coupling_factor.clamp(0.3, 3.0)
                    tau = (tau * coupling_factor).clamp(
                        min=self.tau_min, max=self.tau_max * 3.0)
                # Tau-convergence coupling: positions struggling to converge get faster integration
                if self.tau_convergence_coupling_enabled:
                    with torch.no_grad():
                        residual = (h - target).norm(dim=-1, keepdim=True)  # [B, N, 1]
                        residual_norm = residual / (residual.mean() + 1e-8)
                        conv_factor = 1.0 / (1.0 + self.tau_convergence_beta * residual_norm)
                        _floor = self.tau_convergence_floor
                        tau_scale = _floor + (1.0 - _floor) * conv_factor  # modulate within [floor*τ, τ]
                        # Store diagnostics as detached tensors (no .item() → no compile break)
                        self._last_convergence_residual_mean = residual.mean().detach()
                        self._last_convergence_residual_std = residual.std().detach()
                    tau = (tau * tau_scale).clamp(min=self.tau_min, max=self.tau_max)
            # Structural tau anchor: rescale tau to target range.
            # TauNet was trained on ARC and produces low tau on text deltas.
            # Instead of fighting the weights, rescale the output: map
            # [current_min, current_max] → [target*0.5, target*1.5]
            # preserving relative per-position differentiation.
            tau_anchor_target = getattr(self, '_tau_anchor_target', 0.0)
            if tau_anchor_target > 0:
                with torch.no_grad():
                    tau_min_val = tau.min()
                    tau_max_val = tau.max()
                    tau_range = tau_max_val - tau_min_val + 1e-8
                    # Normalize to [0, 1], then rescale to [target*0.3, target*1.5]
                    tau_norm = (tau - tau_min_val) / tau_range
                    target_lo = tau_anchor_target * 0.3
                    target_hi = min(tau_anchor_target * 1.5, self.tau_max)
                    tau = target_lo + tau_norm * (target_hi - target_lo)
                tau = tau.clamp(min=self.tau_min, max=self.tau_max)

            # 6. LTC: decay toward diffusion target
            dh_dt = -(1.0 / tau) * (h - target)

        # Store LTC convergence residual — the model's own internal surprise.
        # ||h - target|| is large when the model hasn't figured out where h should go.
        # Bounded by state norms, doesn't drive dynamics magnitude → NaN-safe.
        self._last_residual = (h - target).detach().norm(dim=-1).mean(dim=-1)  # [B]

        # 7. FFN residual (amortized across steps)
        dh_dt = dh_dt + self.ffn(self.norm_ff(h)) / self._n_ode_steps

        # 8. Adaptive stability damping: restoring force that grows with ||dh/dt||
        dh_norm = dh_dt.detach().norm(dim=-1, keepdim=True)  # [B, N, 1]
        stability_threshold = 50.0
        damping_factor = stability_threshold / (dh_norm + stability_threshold)
        dh_dt = dh_dt * damping_factor

        # Norm homeostasis applied in solver (euler_solve) after each step,
        # not here — dynamics-level decay fights stability damping.

        # Progressive damping: later ODE steps make smaller updates
        if self._progressive_damping and self._current_step_index_buf > 0:
            damping = 1.0 - self._damping_strength * (
                self._current_step_index_buf / max(self._current_n_steps_buf - 1, 1))
            dh_dt = dh_dt * damping

        # Sparse activation (Probe 4). Probe 7 adds a random-selection mode
        # for ablating whether the learned activity score is load-bearing.
        if self.sparse_fraction > 0 and hasattr(self, 'W_activity'):
            B_, N_, _ = dh_dt.shape
            k = max(1, int((1.0 - self.sparse_fraction) * N_))
            if getattr(self, 'sparse_random', False):
                # Probe 7: random top-k — uniformly random k positions per (batch, step)
                rand = torch.rand(B_, N_, device=dh_dt.device, dtype=dh_dt.dtype)
                _, topk_idx = rand.topk(k, dim=-1)
                mask = torch.zeros(B_, N_, device=dh_dt.device, dtype=dh_dt.dtype)
                mask.scatter_(-1, topk_idx, 1.0)
                dh_dt = dh_dt * mask.unsqueeze(-1)
                self._last_activity_mean = torch.tensor(0.5, device=dh_dt.device)
                self._last_activity_std = torch.tensor(0.0, device=dh_dt.device)
            else:
                activity = torch.sigmoid(self.W_activity(h.detach())).squeeze(-1)
                topk_vals, topk_idx = activity.topk(k, dim=-1)
                mask = torch.zeros_like(activity)
                mask.scatter_(-1, topk_idx, 1.0)
                # Straight-through: gradient flows through activity score
                mask_ste = activity + (mask - activity).detach()
                dh_dt = dh_dt * mask_ste.unsqueeze(-1)
                self._last_activity_mean = activity.mean().detach()
                self._last_activity_std = activity.std().detach()
            self._last_sparse_k = k

        # ReZero gate: multiplies the full update. Near-0 at init → identity
        # dynamics; grows as the gate gets reward-carrying gradient through
        # deep supervision.
        if self.rezero_enabled:
            dh_dt = dh_dt * torch.sigmoid(self.rezero_gate_logit)

        # Tier 3: halting — compute per-position halt probability, return as tuple.
        # Solver reads p_halt to compute still_active and scale dh.
        # Input uses norm_geo(h) for stable stats (same as metric/attention routing).
        if self.halting_enabled:
            halt_logit = self.halt_head(self.norm_geo(h))  # [B, N, 1]
            p_halt = torch.sigmoid(halt_logit)
            return dh_dt, p_halt

        return dh_dt

    def set_step_index(self, step_idx: int, n_steps: int):
        """Set current ODE step index — uses new tensors to avoid inplace conflicts."""
        self._current_step_index_buf = torch.tensor(step_idx, device=self._current_step_index_buf.device, dtype=self._current_step_index_buf.dtype)
        self._current_n_steps_buf = torch.tensor(n_steps, device=self._current_n_steps_buf.device, dtype=self._current_n_steps_buf.dtype)

    def set_step_embed(self, step_idx: int, n_steps: int):
        """Set step embedding for current ODE step (called from solver).

        Linearly interpolates between learnable step embeddings.
        No-op when step_embed_enabled=False.

        Uses direct assignment (not .copy_()) so each ODE step gets its own
        tensor in the autograd graph — avoids in-place version conflicts.

        Args:
            step_idx: current step index (0-based)
            n_steps: total number of ODE steps
        """
        if not self.step_embed_enabled:
            return
        n_embeds = self.step_embeds.shape[0]
        # Map step_idx to [0, n_embeds-1] range for interpolation
        t = step_idx / max(n_steps - 1, 1) * (n_embeds - 1)
        idx_lo = int(t)
        idx_hi = min(idx_lo + 1, n_embeds - 1)
        frac = t - idx_lo
        embed = (1 - frac) * self.step_embeds[idx_lo] + frac * self.step_embeds[idx_hi]
        # Direct assignment: each step gets a fresh tensor in the autograd graph.
        # Gradients flow to step_embeds via this path.
        self._current_step_embed = embed

    def compute_gate(self, h: torch.Tensor) -> torch.Tensor:
        """Compute channel-wise gate from h (for diagnostics).

        Args:
            h: [B, N, d]

        Returns:
            gate: [B, N, d] values in (0, 1)
        """
        h_normed = self.norm_geo(h)
        return torch.sigmoid(self.gate_net_linear2(
            F.gelu(self.gate_net_linear1(h_normed))
        ))

    def compute_metric_diag(self, h: torch.Tensor) -> torch.Tensor:
        """Compute diagonal metric only (for diagnostics that expect [B,N,d]).
        Standalone implementation — does NOT call compute_metric() to avoid
        isinstance branching issues inside torch.compile."""
        B, N, d = h.shape
        context = self._context
        h_normed = self.norm_geo(h)
        ctx_exp = context.unsqueeze(1).expand(B, N, d)
        cat_input = torch.cat([h_normed, ctx_exp], dim=-1)
        met_hidden = F.gelu(self.metric_net_linear1(cat_input))
        return F.softplus(self.metric_net_linear2_diag(met_hidden))

    def compute_metric(self, h: torch.Tensor):
        """Compute metric field from h.

        Args:
            h: [B, N, d] hidden states

        Returns:
            If metric_rank > 0: (D, L) where D=[B,N,d], L=[B,N,d,rank]
            If metric_rank == 0: D [B,N,d] (backward compatible)
        """
        B, N, d = h.shape
        context = self._context
        h_normed = self.norm_geo(h)
        ctx_exp = context.unsqueeze(1).expand(B, N, d)
        cat_input = torch.cat([h_normed, ctx_exp], dim=-1)
        met_hidden = F.gelu(self.metric_net_linear1(cat_input))
        D = F.softplus(self.metric_net_linear2_diag(met_hidden))
        if self.metric_rank > 0:
            L_flat = self.metric_net_linear2_lr(met_hidden)
            L = L_flat.view(B, N, d, self.metric_rank)
            return D, L
        return D

    def compute_tau(self, h: torch.Tensor) -> torch.Tensor:
        """Compute tau field from h (for diagnostics AND loss supervision).

        When channel_gate_enabled, returns gate mean across dims as [B, N, 1] proxy.

        This path mirrors the τ composition used inside forward()/ODE,
        including the structural_tau multiplicative branch. Mirroring is
        load-bearing: `tau_init = compute_tau(h0)` is consumed by
        `compute_tau_quality_loss` (sustained_criticality.py), so the
        backward graph from that loss must reach `structural_tau` for the
        parameter to receive gradient. Before this alignment it did not.

        Args:
            h: [B, N, d]

        Returns:
            tau: [B, N, 1]
        """
        if self.channel_gate_enabled:
            # Return gate mean as tau proxy for backward-compat diagnostics
            gate = self.compute_gate(h)
            return gate.mean(dim=-1, keepdim=True)
        N = h.shape[1]
        h_normed = self.norm_geo(h)
        tau_logits = self.tau_net_linear2(
            F.gelu(self.tau_net_linear1(h_normed))
        )
        tau = torch.sigmoid(tau_logits) * (self.tau_max - self.tau_min) + self.tau_min

        # Structural tau: input-independent per-position multiplier.
        # Same branch as forward() (dynamics.py:457-466); placed here so
        # tau_quality_loss gradients reach the structural_tau parameter.
        if self.structural_tau_enabled:
            s_raw = self.structural_tau[:N]
            s_tau = self.structural_tau_min + (
                self.structural_tau_max - self.structural_tau_min
            ) * torch.sigmoid(s_raw)
            s_tau = s_tau.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
            tau = (tau * s_tau).clamp(
                min=self.tau_min,
                max=self.tau_max * self.structural_tau_max,
            )

        # Apply same rescaling as forward() for consistent diagnostics
        tau_anchor_target = getattr(self, '_tau_anchor_target', 0.0)
        if tau_anchor_target > 0:
            tau_min_val = tau.min()
            tau_max_val = tau.max()
            tau_range = tau_max_val - tau_min_val + 1e-8
            tau_norm = (tau - tau_min_val) / tau_range
            target_lo = tau_anchor_target * 0.3
            target_hi = min(tau_anchor_target * 1.5, self.tau_max)
            tau = target_lo + tau_norm * (target_hi - target_lo)
            tau = tau.clamp(min=self.tau_min, max=self.tau_max)

        return tau


if __name__ == "__main__":
    print("Testing ContinuousDynamics (SDPA path)...")
    config = LiquidARCConfig(d_model=64, d_metric=16, d_ffn=128, n_ode_steps=4)
    dyn = ContinuousDynamics(config)

    B, N = 2, 16
    h = torch.randn(B, N, 64)
    ctx = torch.randn(B, 64)

    dyn.set_context(ctx)
    dh = dyn(0.0, h)
    assert dh.shape == (B, N, 64), f"dh shape: {dh.shape}"

    # Gradient flow
    h_g = torch.randn(B, N, 64, requires_grad=True)
    dyn.set_context(ctx)
    dh_g = dyn(0.0, h_g)
    dh_g.sum().backward()
    assert h_g.grad is not None

    # Diagnostic methods
    g_result = dyn.compute_metric(h_g.detach())
    g = g_result[0] if isinstance(g_result, tuple) else g_result
    assert g.shape == (B, N, 64)
    tau = dyn.compute_tau(h_g.detach())
    assert tau.shape == (B, N, 1)

    # Check zero-init: dh should be near zero at initialization
    h_test = torch.randn(B, N, 64)
    dyn_fresh = ContinuousDynamics(config)
    dyn_fresh.set_context(ctx)
    dh_test = dyn_fresh(0.0, h_test)
    print(f"  dh_norm at init: {dh_test.norm().item():.6f} (should be ~0)")

    print(f"  dh shape: {dh.shape}")
    print(f"  Params: {sum(p.numel() for p in dyn.parameters()):,}")
    print("ContinuousDynamics OK")
