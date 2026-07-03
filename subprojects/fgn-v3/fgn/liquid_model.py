"""LiquidSequenceModel — wrap LiquidARC's ContinuousDynamics as a sequence model
for token-stream tasks (parity / affine / TR / TSP / text NTP).

Makes LiquidARC directly comparable to flat and FGN within fgn-v3's training
framework: same I/O (input_ids → logits), same optimizer plumbing, same
training script (train_multitask.py with model_type="liquid").

Architecture:
    input_ids (token IDs)
      → token_embed + pos_embed  [B, N, d]
      → ContextPool (mean over tokens)  [B, d]
      → ContinuousDynamics applied n_ode_steps times via Euler  [B, N, d]
      → LayerNorm
      → LM head (d → vocab)

The ContinuousDynamics module is the substrate under test: weight-tied
continuous-time ODE with SDPA-factored heat-kernel attention, learned
Riemannian metric, adaptive tau. One shared module applied N times.
"""

import sys
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from the liquid-arc package
# (this container has PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3)
from liquid_arc.config import LiquidARCConfig
from liquid_arc.dynamics import ContinuousDynamics
from liquid_arc.multi_substrate import MultiSubstrateDynamics
from liquid_arc.solver import euler_solve, euler_solve_halting
from liquid_arc.sustained_criticality import (
    compute_criticality_loss, compute_tau_quality_loss,
)
import math


def _compile_safe_criticality_loss(
    h: torch.Tensor, g: torch.Tensor, tau: torch.Tensor,
    t_diffusion_param: torch.Tensor,
    target_ratio: float = 18.0, n_pairs: int = 256,
    d_sq_target: float = 60.0,
) -> torch.Tensor:
    """Same math as LiquidARC's compute_criticality_loss but returns ONLY
    the scalar loss tensor (no diagnostics dict with .item() calls) so
    torch.compile doesn't recompile on each step."""
    B, N, d = h.shape
    t_diff = F.softplus(t_diffusion_param)
    g_b = g[0]
    tau_b = tau[0].squeeze(-1)
    pair_count = min(n_pairs, N * N)
    i_idx = torch.randint(0, N, (pair_count,), device=h.device)
    j_idx = torch.randint(0, N, (pair_count,), device=h.device)
    h_b = h[0]
    diff = h_b[i_idx] - h_b[j_idx]
    g_mean = (g_b[i_idx] + g_b[j_idx]) * 0.5
    d_sq = (diff * diff * g_mean).sum(dim=-1)
    tau_pairs = (tau_b[i_idx] + tau_b[j_idx]) * 0.5
    d_sq_median = d_sq.median()
    tau_median = tau_pairs.median()
    ratio = d_sq_median / (4.0 * tau_median + 1e-8)
    log_ratio = torch.log(ratio / target_ratio + 1e-8)
    ratio_loss = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), beta=0.5)
    if d_sq_target > 0:
        D_sq_log = torch.log(d_sq_median + 1e-8)
        D_sq_target_log = math.log(d_sq_target)
        D_sq_anchor = 0.1 * (D_sq_log - D_sq_target_log) ** 2
        return ratio_loss + D_sq_anchor
    return ratio_loss

from .config import FGNConfig


class LiquidSequenceModel(nn.Module):
    """LiquidARC dynamics as a next-token-prediction sequence model."""

    def __init__(self, config: FGNConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        # Build a LiquidARCConfig from the fgn-v3 FGNConfig.
        # Fields that fgn-v3 doesn't expose keep their LiquidARC defaults.
        la_cfg = LiquidARCConfig(
            d_model=d,
            # d_metric: bigger reasoning module; default d/4 → bumped to d·0.75
            # so MetricNet has 3× more params without touching d_model
            # (which hits Triton shared-memory limit at d>256).
            # 0 means "use default" (d_liquid_metric=0 is the sentinel)
            d_metric=(getattr(config, 'd_liquid_metric', 0) or int(d * 0.75)),
            # d_ffn: 2d → 4d for richer per-position transform
            d_ffn=(getattr(config, 'd_liquid_ffn', 0) or 4 * d),
            max_seq_len=config.max_seq_len,
            n_ode_steps=getattr(config, 'n_ode_steps', 16),
            ode_steps_min=getattr(config, 'n_ode_steps', 16),
            ode_steps_max=getattr(config, 'n_ode_steps', 16),
            integration_time=2.0,
            tau_min=getattr(config, 'liquid_tau_min', 0.5),
            tau_max=getattr(config, 'liquid_tau_max', 1.0),
            t_diffusion_init=1.0,
            routing_mode=getattr(config, 'liquid_routing', 'metric'),
            dropout=config.dropout,
            alpha_logit_init=2.2,
            tau_freeze_steps=0,     # no tau freeze for this framework
            use_torch_compile=False,  # compile handled by fgn-v3 wrapper
            chunk_size=256,
            metric_rank=getattr(config, 'metric_rank', 0) if getattr(
                config, 'metric_type', 'diagonal') == 'diagonal' else 0,
            structural_tau_enabled=getattr(
                config, 'liquid_structural_tau', False),
            tau_quality_loss_enabled=False,
            criticality_loss_enabled=False,  # would require aux-loss plumbing
            # Tier 1 flexible geometric reasoning: step-conditional operator
            step_conditional_operator=getattr(
                config, 'step_conditional_operator', False),
            step_conditional_n_max=getattr(
                config, 'step_conditional_n_max', 32),
            # Tier 2: FiLM on Q/K projections
            step_conditional_qk=getattr(
                config, 'step_conditional_qk', False),
            # Tier 3: per-position halting (adaptive per-token compute depth)
            halting_enabled=getattr(config, 'halting_enabled', False),
            halting_min_steps=getattr(config, 'halting_min_steps', 4),
            halting_ponder_lambda=getattr(
                config, 'halting_ponder_lambda', 0.01),
            # Bootstrap pack
            rezero_enabled=getattr(config, 'rezero_enabled', False),
            rezero_gate_init=getattr(config, 'rezero_gate_init', -5.0),
            metric_bias_init_std=getattr(config, 'metric_bias_init_std', 0.0),
            deep_supervision_enabled=getattr(
                config, 'deep_supervision_enabled', False),
            ponder_kl_lambda=getattr(config, 'ponder_kl_lambda', 0.0),
            ponder_kl_prior_rate=getattr(
                config, 'ponder_kl_prior_rate', 0.0625),
            fast_weights_enabled=getattr(config, 'fast_weights_enabled', False),
            fast_weights_rank=getattr(config, 'fast_weights_rank', 4),
            fast_weights_eta=getattr(config, 'fast_weights_eta', 0.01),
            fast_weights_decay=getattr(config, 'fast_weights_decay', 0.05),
            identity_routing_enabled=getattr(
                config, 'identity_routing_enabled', False),
            identity_routing_alpha_init=getattr(
                config, 'identity_routing_alpha_init', 0.0),
            identity_routing_decay=getattr(
                config, 'identity_routing_decay', 0.1),
        )
        self.la_cfg = la_cfg
        self.n_ode_steps = la_cfg.n_ode_steps

        # Token + position embeddings (match flat/fgn for fair comparison)
        self.embed = nn.Embedding(config.vocab_size, d)
        self.pos_embed = nn.Embedding(config.max_seq_len, d)

        # The substrate under test: weight-tied continuous dynamics.
        # When K_substrates > 1, switch to MultiSubstrateDynamics — K parallel
        # ContinuousDynamics with lateral coupling at each ODE step. State
        # carried through ODE is K*d_per wide; we tile h0 by K and reduce at
        # the end.
        self.K_substrates = int(getattr(config, 'k_substrates', 1))
        self.lateral_weight = float(getattr(config, 'lateral_weight', 0.5))
        if self.K_substrates > 1:
            self.dynamics = MultiSubstrateDynamics(
                la_cfg, K=self.K_substrates,
                lateral_weight=self.lateral_weight)
            self._fuse_proj = nn.Linear(self.K_substrates * d, d, bias=False)
            for sub in self.dynamics.substrates:
                if hasattr(sub, 'freeze_tau'):
                    sub.freeze_tau = False
        else:
            self.dynamics = ContinuousDynamics(la_cfg)
            if hasattr(self.dynamics, 'freeze_tau'):
                self.dynamics.freeze_tau = False
            self._fuse_proj = None
        # Compile JUST the dynamics module (one Euler step's worth of ops),
        # NOT the whole model. LiquidARC's canonical training does this —
        # otherwise torch.compile unrolls 16 Euler iterations into one huge
        # kernel that exceeds Triton's 101KB shared-memory limit with coupled
        # routing or rank>0. We signal to fgn-v3 to skip top-level compile
        # via the `_skip_top_compile` flag (read by train_multitask).
        self._skip_top_compile = True
        if torch.cuda.is_available() and getattr(config, 'use_torch_compile', True):
            self.dynamics = torch.compile(self.dynamics, mode="default")

        # Context pool: mean over sequence (matches fgn-v3 simplicity)
        # LiquidARC has a fancier context_pool with mask handling; we use
        # simple mean here because our tasks are unmasked within the window.

        # Output head
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)

    def _pool_context(self, h: torch.Tensor) -> torch.Tensor:
        """Simple mean-pool over sequence → [B, d]."""
        return h.mean(dim=1)

    def compute_aux_losses(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute SOC aux losses from outside the compiled forward.

        Mirrors LiquidARC's canonical pattern: aux losses are computed in
        the training script (not fused into the compiled model forward).
        This keeps the main forward compile-friendly while still shaping
        metric and tau via criticality + tau-quality gradients.

        Called AFTER model.forward() returns. Gradient from these losses
        flows back through dynamics params.
        """
        device = input_ids.device
        B, N = input_ids.shape
        pos = torch.arange(N, device=device).unsqueeze(0)
        h0 = self.embed(input_ids) + self.pos_embed(pos)
        context = self._pool_context(h0)
        mask = torch.triu(
            torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        self.dynamics.set_context(context, mask=mask)

        g_init = self.dynamics.compute_metric_diag(h0)
        tau_init = self.dynamics.compute_tau(h0)
        crit_loss = _compile_safe_criticality_loss(
            h0, g_init, tau_init, self.dynamics.t_diffusion,
            target_ratio=18.0, n_pairs=256, d_sq_target=60.0,
        )
        tq_loss = compute_tau_quality_loss(
            tau_init, mean_target=1.0, log_spread_target=0.6)
        with torch.no_grad():
            cv = g_init.std() / (g_init.mean() + 1e-8)
            tau_mean = tau_init.mean()
            t_diff_eff = F.softplus(self.dynamics.t_diffusion)
            h_c = h0[0] - h0[0].mean(dim=0, keepdim=True)
            d_sq_approx = 2.0 * (g_init[0] * h_c ** 2).sum(-1).mean()
            d_over_4t = d_sq_approx / (4.0 * t_diff_eff * tau_mean + 1e-8)
        return {
            "crit_loss": crit_loss,
            "tq_loss": tq_loss,
            "cv": cv.detach(),
            "d_over_4t": d_over_4t.detach(),
        }

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        B, N = input_ids.shape
        device = input_ids.device

        pos = torch.arange(N, device=device).unsqueeze(0)
        h0 = self.embed(input_ids) + self.pos_embed(pos)

        # Causal mask (True=blocked)
        mask = torch.triu(
            torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)

        context = self._pool_context(h0)
        self.dynamics.set_context(context, mask=mask)
        self.dynamics.set_n_steps(self.n_ode_steps)

        # Main dynamics forward — the ONLY tensor work in the compiled
        # path. Aux losses (criticality, tau_quality) are now computed
        # outside the compiled forward, in the training script via
        # model.compute_aux_losses(), matching LiquidARC's canonical
        # pattern.
        ponder_cost = None
        steps_used = None  # [B, N] effective steps per position
        sup = None          # deep-supervision dict (h_stack, p_halt_stack, ...)
        do_deep_sup = (
            self.la_cfg.halting_enabled
            and getattr(self.la_cfg, 'deep_supervision_enabled', False)
            and labels is not None
        )
        label_mask = None
        if do_deep_sup:
            label_mask = (labels != -100)
            # If no label positions in this batch, fall back to vanilla halting.
            if not bool(label_mask.any()):
                do_deep_sup = False
                label_mask = None

        # Multi-substrate: tile h0 by K so each substrate gets its own initial
        # state copy. K=1 path is unchanged.
        if self.K_substrates > 1:
            h0_evolved = h0.repeat(1, 1, self.K_substrates)  # [B, N, K*d]
        else:
            h0_evolved = h0

        if self.la_cfg.halting_enabled:
            if do_deep_sup:
                h, ponder_cost, steps_used, sup = euler_solve_halting(
                    self.dynamics, h0_evolved,
                    t_span=(0.0, self.la_cfg.integration_time),
                    n_steps=self.n_ode_steps,
                    min_steps=self.la_cfg.halting_min_steps,
                    label_mask=label_mask,
                )
            else:
                h, ponder_cost, steps_used = euler_solve_halting(
                    self.dynamics, h0_evolved,
                    t_span=(0.0, self.la_cfg.integration_time),
                    n_steps=self.n_ode_steps,
                    min_steps=self.la_cfg.halting_min_steps,
                )
        else:
            h = euler_solve(
                self.dynamics, h0_evolved,
                t_span=(0.0, self.la_cfg.integration_time),
                n_steps=self.n_ode_steps)
            if isinstance(h, tuple):
                h = h[0]

        # Multi-substrate: fuse K substrate states back to d via linear
        if self.K_substrates > 1:
            h = self._fuse_proj(h)
            if sup is not None:
                # h_stack came back as [K_steps, total_labels, K_substrates*d];
                # fuse the substrate dim so per-step CE uses the same readout.
                sup['h_stack'] = self._fuse_proj(sup['h_stack'])

        h = self.norm(h)
        logits = self.lm_head(h)
        result: Dict[str, torch.Tensor] = {"logits": logits}

        # Compatibility placeholders (real values set by compute_aux_losses)
        result["metric_cv"] = torch.tensor(0.0, device=device)
        result["avg_kappa"] = torch.tensor(0.0, device=device)

        # Tier 3 halting stats: expose per-position step usage for logging
        if steps_used is not None:
            with torch.no_grad():
                result["steps_mean"] = steps_used.mean().detach()
                result["steps_min"] = steps_used.min().detach()
                result["steps_max"] = steps_used.max().detach()
                result["steps_std"] = steps_used.std().detach()
                # Fraction of positions using <80% of full budget
                halt_threshold = 0.8 * self.n_ode_steps
                result["halt_frac"] = (
                    (steps_used < halt_threshold).float().mean().detach())

        if labels is not None:
            ce = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1), ignore_index=-100,
            )
            result["ce_loss"] = ce
            result["curv_loss"] = torch.tensor(0.0, device=device)
            result["scale_loss"] = torch.tensor(0.0, device=device)

            if sup is not None:
                # PonderNet-style deep supervision:
                #   L_task = Σ_k p(halt=k) · CE_k  +  p_never_halted · CE_{final}
                # Each step's CE is computed from h at the label positions at
                # that step. Gradient reaches MetricNet/ReZero/halt_head at every
                # iteration — not just the final state.
                h_stack = sup['h_stack']            # [K, L, d]
                p_halt_stack = sup['p_halt_stack']  # [K, L, 1]
                p_active_stack = sup['p_active_stack']  # [K, L, 1]
                K = h_stack.shape[0]
                # labels at label positions [L]
                labels_at_mask = labels[label_mask]
                # Logits per step [K, L, V] — we project through the same norm+head.
                logits_stack = self.lm_head(self.norm(h_stack))
                # Per-step CE at label positions → [K, L]
                logp = F.log_softmax(logits_stack, dim=-1)
                step_ce = -logp.gather(-1, labels_at_mask.view(1, -1, 1)
                                       .expand(K, -1, 1)).squeeze(-1)
                # Halt distribution per label position: [K, L]
                p_halt_dist = (p_active_stack * p_halt_stack).squeeze(-1)  # [K, L]
                # Mass remaining after all K steps — forced to halt at last step.
                p_remainder = 1.0 - p_halt_dist.sum(dim=0)  # [L]
                # Weighted CE: expected CE under halt distribution.
                task_loss_per_label = (p_halt_dist * step_ce).sum(dim=0) \
                    + p_remainder * step_ce[-1]  # [L]
                task_loss = task_loss_per_label.mean()

                result["ce_loss"] = task_loss  # log the deep-sup loss as ce
                total_loss = task_loss

                # Geometric KL prior on halt distribution
                kl_lam = float(getattr(self.la_cfg, 'ponder_kl_lambda', 0.0))
                if kl_lam > 0:
                    rate = float(getattr(
                        self.la_cfg, 'ponder_kl_prior_rate', 0.0625))
                    # Prior Geom(rate): p_G(k) = rate·(1-rate)^k for k=0..K-1.
                    # Renormalize over {0..K-1, remainder} to match support.
                    k_idx = torch.arange(K, device=h_stack.device,
                                          dtype=h_stack.dtype)
                    log1m = math.log(max(1.0 - rate, 1e-8))
                    log_r = math.log(max(rate, 1e-8))
                    log_p_G = log_r + k_idx * log1m  # [K]
                    p_G = torch.exp(log_p_G)         # [K]
                    p_G_rem = (1.0 - p_G.sum()).clamp(min=1e-8)
                    # Append remainder bucket to both model + prior
                    p_model_full = torch.cat([
                        p_halt_dist.clamp(min=1e-8),           # [K, L]
                        p_remainder.clamp(min=1e-8).unsqueeze(0)  # [1, L]
                    ], dim=0)
                    p_G_full = torch.cat([p_G, p_G_rem.unsqueeze(0)])  # [K+1]
                    # Renormalize model (safety for numerical drift)
                    p_model_full = p_model_full / p_model_full.sum(dim=0,
                                                                    keepdim=True)
                    kl = (p_model_full
                          * (torch.log(p_model_full)
                             - torch.log(p_G_full.unsqueeze(1)))
                          ).sum(dim=0).mean()
                    kl_loss = kl_lam * kl
                    result["kl_loss"] = kl_loss.detach()
                    total_loss = total_loss + kl_loss

                # Also log vanilla CE at final state for comparability.
                result["ce_final"] = ce.detach()
                if ponder_cost is not None:
                    # Explicit compute-cost penalty (PonderNet ρ term).
                    # Deep-sup task loss can reward using full budget when CE
                    # decreases monotonically with k. Without this term, halt
                    # head has only KL pressure to halt early — usually too
                    # weak. ponder_cost ∈ [0,1] = mean(still_active) over steps.
                    ponder_lam = float(getattr(
                        self.la_cfg, 'halting_ponder_lambda', 0.0))
                    if ponder_lam > 0:
                        ponder_loss = ponder_lam * ponder_cost
                        result["ponder_loss"] = ponder_loss.detach()
                        total_loss = total_loss + ponder_loss
                    result["ponder_cost"] = ponder_cost.detach()
            else:
                total_loss = ce
                if ponder_cost is not None:
                    # Legacy gated-ponder path (no deep supervision).
                    ponder_gate = torch.exp(-ce.detach()).clamp(0.0, 1.0)
                    ponder_loss = (self.la_cfg.halting_ponder_lambda
                                   * ponder_gate * ponder_cost)
                    result["ponder_loss"] = ponder_loss.detach()
                    result["ponder_cost"] = ponder_cost.detach()
                    result["ponder_gate"] = ponder_gate.detach()
                    total_loss = total_loss + ponder_loss
            result["loss"] = total_loss
        return result

    def slow_parameters(self):
        """Geometric params that get the slower LR (metric_lr_mult × base_lr).
        Matches LiquidARC's canonical training: MetricNet, TauNet, t_diffusion,
        alpha_logit, and attention-routing W_q/W_k get 10-100× lower LR than
        content params. MEMORY flags this as 'the single most impactful
        discovery' for stable training.
        """
        slow = []
        dyn = self.dynamics
        for name in [
            'metric_net_linear1', 'metric_net_linear2_diag',
            'metric_net_linear2_lr',
            'tau_net_linear1', 'tau_net_linear2',
            'tau_step_embed', 'step_embeds',
            'gate_net_linear1', 'gate_net_linear2',
            'W_q', 'W_k', 'W_gate', 'W_activity',
        ]:
            mod = getattr(dyn, name, None)
            if mod is not None:
                if hasattr(mod, 'parameters'):
                    slow.extend(list(mod.parameters()))
                elif isinstance(mod, torch.nn.Parameter):
                    slow.append(mod)
        # Scalars / standalone params
        for name in ['t_diffusion', 'alpha_logit', 'structural_tau']:
            p = getattr(dyn, name, None)
            if isinstance(p, torch.nn.Parameter):
                slow.append(p)
        return slow

    def fast_parameters(self):
        slow_ids = {id(p) for p in self.slow_parameters()}
        return [p for p in self.parameters() if id(p) not in slow_ids]


if __name__ == "__main__":
    from .config import FGNConfig
    cfg = FGNConfig(d_model=128, n_heads=4, n_layers=6,
                    d_ff=256, vocab_size=1024, max_seq_len=128)
    m = LiquidSequenceModel(cfg)
    print(f"LiquidSequenceModel params: {sum(p.numel() for p in m.parameters()):,}")
    ids = torch.randint(0, 1024, (2, 32))
    labels = torch.full((2, 32), -100, dtype=torch.long)
    labels[:, -1] = 5
    out = m(ids, labels)
    print(f"logits shape: {out['logits'].shape}, loss={out['loss'].item():.3f}")
    out['loss'].backward()
    n_with_grad = sum(1 for p in m.parameters() if p.grad is not None)
    n_total = sum(1 for _ in m.parameters())
    print(f"backward OK: {n_with_grad}/{n_total} params have grad")
