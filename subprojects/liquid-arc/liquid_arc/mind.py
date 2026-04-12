"""LiquidARC Mind — persistent state controller for conversation.

Wraps ContinuousDynamics with:
  - ConversationEmbedding (replaces RoboticsEmbedding)
  - StateReadout (replaces ActionHead)
  - Event buffer management
  - Thread-safe autonomous processing
  - Online learning from feedback

CRITICAL STABILITY NOTES (from lifecycle/adaptive autonomy experiments):
  1. NEVER use ||dh/dt|| as a curiosity reward — causes NaN every time.
     Use prediction error ||embed(obs) - h|| instead — bounded by embedding scale.
  2. Include efficiency regularizer (λ_eff=0.001-0.005) on autonomous dynamics
     to prevent degenerate behavior and runaway dynamics.
  3. All GPU operations must go through a single thread or use a threading lock —
     concurrent CUDA calls from main thread + autonomous thread cause races.
  4. Adaptive damping in dynamics: dh_dt *= threshold/(||dh_dt|| + threshold)
     must be active — this was the critical fix preventing NaN in lifecycle runs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import time
from typing import Any, Dict, List, Optional

from .config import LiquidARCConfig
from .lifecycle import SensoryForcing
from .conversation_embedding import ConversationEmbedding
from .state_readout import StateReadout
from .context_pool import ContextPool
from .dynamics import ContinuousDynamics


class PlasticityController:
    """Adaptive LR controller for embedding online learning.

    Maintains the system at the productive edge of stability-plasticity:
    - Raises embed_lr when suffocating (xform=0%, NTP stable)
    - Lowers embed_lr when running loose (NTP rising, xform climbing fast)
    - Holds when productive (xform 3-15%, NTP stable)
    """

    def __init__(self, embed_lr_min=1e-5, embed_lr_max=1e-3, embed_lr_init=1e-4,
                 ntp_ema_alpha=0.05, ntp_rise_threshold=1.15, ntp_spike_threshold=1.5,
                 xform_suffocate=0.0, xform_productive_low=0.03,
                 xform_productive_high=0.15, xform_danger=0.30,
                 lr_up_factor=1.3, lr_down_factor=0.5, lr_emergency_factor=0.1,
                 suffocate_patience=50):
        self.embed_lr_min = embed_lr_min
        self.embed_lr_max = embed_lr_max
        self.current_embed_lr = embed_lr_init
        self.ntp_ema_alpha = ntp_ema_alpha
        self.ntp_rise_threshold = ntp_rise_threshold
        self.ntp_spike_threshold = ntp_spike_threshold
        self.xform_suffocate = xform_suffocate
        self.xform_productive_low = xform_productive_low
        self.xform_productive_high = xform_productive_high
        self.xform_danger = xform_danger
        self.lr_up_factor = lr_up_factor
        self.lr_down_factor = lr_down_factor
        self.lr_emergency_factor = lr_emergency_factor
        self.suffocate_patience = suffocate_patience
        self.ntp_loss_ema = None
        self.zero_xform_streak = 0
        self.step_count = 0
        self.last_action = 'init'
        self.history = []

    def update(self, ntp_loss: float, xform: float) -> dict:
        self.step_count += 1
        if self.ntp_loss_ema is None:
            self.ntp_loss_ema = ntp_loss
            return self._result('init', 'First step — establishing baseline')

        self.ntp_loss_ema = (self.ntp_ema_alpha * ntp_loss +
                             (1 - self.ntp_ema_alpha) * self.ntp_loss_ema)

        if xform <= self.xform_suffocate:
            self.zero_xform_streak += 1
        else:
            self.zero_xform_streak = 0

        # Emergency brake: NTP spiking
        if ntp_loss > self.ntp_spike_threshold * self.ntp_loss_ema:
            self.current_embed_lr = max(
                self.embed_lr_min, self.current_embed_lr * self.lr_emergency_factor)
            return self._result('emergency_brake',
                f'NTP spike: {ntp_loss:.2f} vs EMA {self.ntp_loss_ema:.2f}')

        # Pull back: NTP rising or xform dangerously high
        ntp_rising = ntp_loss > self.ntp_rise_threshold * self.ntp_loss_ema
        xform_high = xform > self.xform_danger
        if ntp_rising or xform_high:
            self.current_embed_lr = max(
                self.embed_lr_min, self.current_embed_lr * self.lr_down_factor)
            reasons = []
            if ntp_rising:
                reasons.append(f'NTP rising: {ntp_loss:.2f} vs EMA {self.ntp_loss_ema:.2f}')
            if xform_high:
                reasons.append(f'xform high: {xform:.1%}')
            return self._result('pull_back', '; '.join(reasons))

        # Productive zone: hold
        if (self.xform_productive_low <= xform <= self.xform_productive_high
                and not ntp_rising):
            return self._result('hold',
                f'Productive: xform={xform:.1%}, NTP={ntp_loss:.2f}')

        # Suffocating: push harder
        if self.zero_xform_streak >= self.suffocate_patience and not ntp_rising:
            self.current_embed_lr = min(
                self.embed_lr_max, self.current_embed_lr * self.lr_up_factor)
            self.zero_xform_streak = 0
            return self._result('push',
                f'Suffocating for {self.suffocate_patience} cycles, NTP={ntp_loss:.2f}')

        # Wait
        return self._result('wait',
            f'xform={xform:.1%}, streak={self.zero_xform_streak}/{self.suffocate_patience}')

    def _result(self, action, reason):
        self.last_action = action
        entry = {'step': self.step_count, 'embed_lr': self.current_embed_lr,
                 'ntp_ema': self.ntp_loss_ema, 'action': action, 'reason': reason}
        self.history.append(entry)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return entry

    def get_status(self):
        return {
            'current_embed_lr': self.current_embed_lr,
            'ntp_loss_ema': self.ntp_loss_ema,
            'zero_xform_streak': self.zero_xform_streak,
            'step_count': self.step_count,
            'last_action': self.last_action,
            'lr_bounds': [self.embed_lr_min, self.embed_lr_max],
            'recent_history': self.history[-10:],
        }


class ReflectionTrigger:
    """Monitors ODE state and decides when LLM interpretation is warranted."""

    def __init__(self, cv_shift_threshold=1.5, h_norm_ceiling=5000.0,
                 tau_std_floor=0.02, self_pe_ceiling=500.0):
        self.prev_cv = None
        self.prev_h_norm = None
        self.prev_tau_std = None

        self.cv_shift_threshold = cv_shift_threshold
        self.h_norm_ceiling = h_norm_ceiling
        self.tau_std_floor = tau_std_floor
        self.self_pe_ceiling = self_pe_ceiling

        self.trigger_sensitivity = {
            'cv_shift': 1.0,
            'h_norm_drift': 1.0,
            'tau_stagnation': 1.0,
            'self_divergence': 1.0,
        }
        self.trigger_history = []

    def check(self, diagnostics: Dict, last_reflection_pe: float = 0) -> Optional[str]:
        """Check whether any condition warrants LLM reflection.
        Returns trigger reason string, or None.
        """
        triggers = []

        cv = diagnostics.get('metric_cv', 0)
        h_norm = diagnostics.get('h_norm', 0)
        tau_std = diagnostics.get('tau_std', 0)

        # Condition 1: Geometric reorganization
        if self.prev_cv is not None:
            delta_cv = abs(cv - self.prev_cv)
            eff_thresh = self.cv_shift_threshold / self.trigger_sensitivity['cv_shift']
            if delta_cv > eff_thresh:
                triggers.append(('cv_shift', delta_cv,
                    f"cv_shift: CV moved {delta_cv:.1f} ({self.prev_cv:.1f}->{cv:.1f})"))

        # Condition 2: Grounding needed
        eff_ceiling = self.h_norm_ceiling / self.trigger_sensitivity['h_norm_drift']
        if h_norm > eff_ceiling:
            triggers.append(('h_norm_drift', h_norm,
                f"h_norm_drift: h_norm={h_norm:.0f} (ceiling {eff_ceiling:.0f})"))

        # Condition 3: Tau stagnation
        eff_floor = self.tau_std_floor * self.trigger_sensitivity['tau_stagnation']
        if tau_std < eff_floor and self.prev_tau_std is not None:
            triggers.append(('tau_stagnation', 1.0 / (tau_std + 1e-8),
                f"tau_stagnation: tau_std={tau_std:.4f}"))

        # Condition 4: Self-description divergence
        eff_pe = self.self_pe_ceiling * self.trigger_sensitivity['self_divergence']
        if last_reflection_pe > eff_pe:
            triggers.append(('self_divergence', last_reflection_pe,
                f"self_divergence: PE={last_reflection_pe:.0f}"))

        self.prev_cv = cv
        self.prev_h_norm = h_norm
        self.prev_tau_std = tau_std

        if not triggers:
            return None

        triggers.sort(key=lambda t: t[1], reverse=True)
        return triggers[0][2]

    def record_outcome(self, trigger_type: str, reflection_pe: float):
        """Learn from reflection outcome — adjust trigger sensitivity."""
        self.trigger_history.append((trigger_type, reflection_pe))
        type_pes = [pe for t, pe in self.trigger_history if t == trigger_type]
        if len(type_pes) >= 3:
            avg_pe = sum(type_pes[-5:]) / len(type_pes[-5:])
            if avg_pe > 300:
                self.trigger_sensitivity[trigger_type] = min(
                    2.0, self.trigger_sensitivity[trigger_type] * 1.1)
            elif avg_pe < 100:
                self.trigger_sensitivity[trigger_type] = max(
                    0.3, self.trigger_sensitivity[trigger_type] * 0.9)


class CuriosityController:
    """Intrinsic motivation: reads PE trajectory and drives curriculum selection.

    Replaces fixed-interval feed/digest with self-regulated exploration:
    - Boredom (low stable PE) → inject from most-novel domain
    - Satiation (high PE) → digest, let ODE consolidate
    - Reflection stagnation → break fixation with novel domain injection
    """

    def __init__(self,
                 pe_history_size=50,
                 boredom_threshold=0.3,
                 satiation_threshold=0.8,
                 min_digest_cycles=100,
                 max_feed_streak=5,
                 domain_temperature=0.3):
        import collections
        self.pe_history = collections.deque(maxlen=pe_history_size)
        self.pe_baseline = None
        self.boredom_threshold = boredom_threshold
        self.satiation_threshold = satiation_threshold
        self.min_digest_cycles = min_digest_cycles
        self.max_feed_streak = max_feed_streak
        self.domain_temperature = domain_temperature

        self.cycles_since_stimulus = 0
        self.consecutive_stimuli = 0
        self.current_phase = 'calibrating'
        self.calibration_count = 0
        self.total_injections = 0
        self.last_reason = ''

    def select_domain(self, curriculum_stats):
        """Choose domain where the Mind has most to learn (highest PE)."""
        import math
        import random
        domain_pe = curriculum_stats.get('domain_avg_pe', {})
        if not domain_pe:
            domains = curriculum_stats.get('domains', [])
            return random.choice(domains) if domains else None

        domains = list(domain_pe.keys())
        pes = [domain_pe[d] for d in domains]
        max_pe, min_pe = max(pes), min(pes)
        if max_pe == min_pe:
            return random.choice(domains)

        weights = [(pe - min_pe) / (max_pe - min_pe) for pe in pes]
        exp_weights = [math.exp(w / self.domain_temperature) for w in weights]
        total = sum(exp_weights)
        probs = [w / total for w in exp_weights]
        return random.choices(domains, weights=probs, k=1)[0]

    def detect_reflection_stagnation(self, recent_reflections, n=5):
        """Check if recent reflections repeat the same content."""
        from collections import Counter
        if len(recent_reflections) < n:
            return False, None

        texts = [r.get('text', '') for r in recent_reflections[-n:]]
        phrases = Counter()
        for text in texts:
            words = text.split()
            for i in range(len(words) - 1):
                if words[i][0:1].isupper() and words[i + 1][0:1].isupper():
                    phrase = f"{words[i]} {words[i + 1]}"
                    phrases[phrase] += 1

        if not phrases:
            return False, None

        most_common, count = phrases.most_common(1)[0]
        if count >= n * 0.6:
            return True, most_common
        return False, None

    def should_inject(self, current_pe, curriculum_stats,
                      recent_reflections=None):
        """Main decision: should the Mind receive a stimulus now?

        Returns: (should_inject: bool, domain: str|None, reason: str)
        """
        self.cycles_since_stimulus += 1
        self.pe_history.append(current_pe)

        # Phase: CALIBRATING — inject at fixed interval to build PE baseline
        if self.current_phase == 'calibrating':
            self.calibration_count += 1
            non_zero = [p for p in self.pe_history if p > 0]
            if len(non_zero) >= 10:
                self.pe_baseline = sum(non_zero) / len(non_zero)
                self.current_phase = 'exploring'
                self.consecutive_stimuli = 0
                print(f"  [curiosity] Calibrated: PE baseline={self.pe_baseline:.0f}")
                return False, None, 'calibrated'
            # Inject every 50 cycles during calibration to gather PE data
            if self.cycles_since_stimulus >= 50:
                import random
                domains = curriculum_stats.get('domains', [])
                domain = random.choice(domains) if domains else None
                self._record_injection('calibrating')
                return True, domain, 'calibrating'
            return False, None, 'calibrating'

        # Minimum digest time
        if self.cycles_since_stimulus < self.min_digest_cycles:
            return False, None, 'digesting'

        # Forced digest after streak
        if self.consecutive_stimuli >= self.max_feed_streak:
            if self.cycles_since_stimulus < self.min_digest_cycles * 3:
                return False, None, 'forced_digest'
            else:
                self.consecutive_stimuli = 0

        # Reflection stagnation override
        if recent_reflections:
            stagnant, stuck_topic = self.detect_reflection_stagnation(recent_reflections)
            if stagnant:
                domain = self.select_domain(curriculum_stats)
                self._record_injection(f'breaking_fixation_on_{stuck_topic}')
                return True, domain, self.last_reason

        # Need enough PE history
        if len(self.pe_history) < 10:
            return False, None, 'insufficient_history'

        recent_pe = list(self.pe_history)[-10:]
        pe_mean = sum(recent_pe) / len(recent_pe)
        pe_std = (sum((p - pe_mean) ** 2 for p in recent_pe) / len(recent_pe)) ** 0.5

        pe_relative = pe_mean / max(self.pe_baseline, 100.0)
        pe_cv = pe_std / max(pe_mean, 1.0)

        # Boredom: PE is low AND stable → inject immediately
        if pe_relative < self.boredom_threshold and pe_cv < 0.3:
            domain = self.select_domain(curriculum_stats)
            self._record_injection('bored')
            return True, domain, self.last_reason

        # Satiation: PE still high → wait longer (but not forever)
        if pe_relative > self.satiation_threshold:
            # Even when satiated, inject at reduced rate (3× minimum) to keep PE fresh
            if self.cycles_since_stimulus >= self.min_digest_cycles * 3:
                domain = self.select_domain(curriculum_stats)
                self._record_injection('satiated_refresh')
                return True, domain, self.last_reason
            return False, None, 'satiated'

        # Moderate: allow injection at standard rate (2× minimum)
        if self.cycles_since_stimulus >= self.min_digest_cycles * 2:
            domain = self.select_domain(curriculum_stats)
            self._record_injection('moderate_curiosity')
            return True, domain, self.last_reason

        return False, None, 'waiting'

    def _record_injection(self, reason):
        self.cycles_since_stimulus = 0
        self.consecutive_stimuli += 1
        self.total_injections += 1
        self.last_reason = reason

    def get_status(self):
        recent_pe = list(self.pe_history)[-10:] if len(self.pe_history) >= 10 else list(self.pe_history)
        pe_mean = sum(recent_pe) / len(recent_pe) if recent_pe else 0
        pe_std = (sum((p - pe_mean) ** 2 for p in recent_pe) / len(recent_pe)) ** 0.5 if recent_pe else 0
        return {
            'phase': self.current_phase,
            'pe_mean': round(pe_mean, 1),
            'pe_std': round(pe_std, 1),
            'pe_baseline': round(self.pe_baseline, 1) if self.pe_baseline else None,
            'cycles_since_stimulus': self.cycles_since_stimulus,
            'consecutive_stimuli': self.consecutive_stimuli,
            'max_feed_streak': self.max_feed_streak,
            'total_injections': self.total_injections,
            'last_reason': self.last_reason,
            'boredom_threshold': self.boredom_threshold,
            'satiation_threshold': self.satiation_threshold,
        }

    def get_params(self):
        return {
            'boredom_threshold': self.boredom_threshold,
            'satiation_threshold': self.satiation_threshold,
            'min_digest_cycles': self.min_digest_cycles,
            'max_feed_streak': self.max_feed_streak,
            'domain_temperature': self.domain_temperature,
        }


class ReflectionLimiter:
    """Rate-limit reflections to prevent buffer crowding.

    Ensures curriculum content dominates the event buffer (~67%)
    instead of being crowded out by reflections (~33%).
    """

    def __init__(self, max_ratio=0.33):
        self.max_ratio = max_ratio
        self.curriculum_count = 0
        self.reflection_count = 0

    def on_curriculum(self):
        self.curriculum_count += 1

    def can_reflect(self):
        if self.curriculum_count == 0:
            return self.reflection_count < 1
        return (self.reflection_count / max(1, self.curriculum_count)) < self.max_ratio

    def on_reflection(self):
        self.reflection_count += 1


class LiquidARCMind:
    """Persistent conversational mind.

    The ODE state h lives as long as the server runs.
    Events arrive as sensory forcing. Context is read from h.
    Between events, autonomous dynamics consolidate.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: LiquidARCConfig,
        text_embedder: Any = None,
        device: str = 'cuda',
        max_context_events: int = 64,
        lambda_eff: float = 0.001,
        freeze_dynamics: bool = False,  # unfrozen by default — self-adapting system
        online_lr: float = 1e-5,
        enable_online_learning: bool = True,
        use_ode_encoder: bool = False,
        tokenizer_path: str = None,
        bootstrap_mode: bool = True,
        bootstrap_events: int = 5000,
        use_trained_text_embed: bool = False,
        qwen_model: Any = None,
        qwen_tokenizer: Any = None,
        coupling: Any = None,
        delta_extractor: Any = None,
        qwen_bridge: Any = None,
        layer_wise_bridge: Any = None,
    ):
        self.device = device
        self.text_embedder = text_embedder  # legacy sentence-transformer (None if ODE encoder)
        self.max_events = max_context_events
        self.lambda_eff = lambda_eff
        self._delta_extractor = delta_extractor  # DeltaExtractor for LLM trajectory deltas
        self._qwen_bridge = qwen_bridge  # QwenBridge for bias-injected generation
        self._layer_wise_bridge = layer_wise_bridge  # LayerWiseBridge for co-processing

        # Two-phase ODE encoding
        self.use_ode_encoder = use_ode_encoder
        self._bootstrap_mode = bootstrap_mode and (text_embedder is not None)
        self._bootstrap_events = bootstrap_events

        d = config.d_model

        # Core dynamics (from checkpoint)
        self.dynamics = ContinuousDynamics(config).to(device)
        self.context_pool = ContextPool(config).to(device)

        # Load checkpoint weights for dynamics + context_pool
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        # Rename old checkpoint key for backward compat
        cleaned = {k.replace('metric_net_linear2.', 'metric_net_linear2_diag.'): v
                   for k, v in cleaned.items()}
        dynamics_keys = {k: v for k, v in cleaned.items()
                         if k.startswith('dynamics.') or k.startswith('context_pool.')}
        holder = nn.ModuleDict({'dynamics': self.dynamics, 'context_pool': self.context_pool})
        holder.load_state_dict(dynamics_keys, strict=False)
        print(f"Loaded {len(dynamics_keys)} dynamics/context_pool parameters")

        # Unfreeze tau if past tau_freeze_steps (dynamics inits with freeze_tau=True)
        if getattr(config, 'tau_freeze_steps', 5000) == 0:
            self.dynamics.freeze_tau = False
            print(f"  TauNet: unfrozen (tau_freeze_steps=0)")

        # Set structural tau anchor for deployment (prevents tau collapse)
        tau_target = getattr(config, 'tau_mean_target', 0.0)
        if tau_target <= 0:
            T = getattr(config, 'integration_time', 2.0)
            tau_target = T / config.n_ode_steps * 16
        self.dynamics._tau_anchor_target = tau_target

        # Tau rescaling handled in dynamics forward — maps TauNet output to [target*0.3, target*1.5]
        print(f"  Tau anchor: target={tau_target:.2f} "
              f"(rescale: [{tau_target*0.3:.1f}, {min(tau_target*1.5, config.tau_max):.1f}])")

        if freeze_dynamics:
            for param in self.dynamics.parameters():
                param.requires_grad = False
            for param in self.context_pool.parameters():
                param.requires_grad = False

        # Path C hybrid: trained TextEmbedding for Phase 1 + Mind metadata for Phase 2
        self.use_trained_text_embed = use_trained_text_embed
        self._text_embed = None
        self._text_tokenizer = None
        # Determine direct prefix mode early (needed for TextEmbedding tokenizer choice)
        _is_direct = (coupling is None and qwen_model is not None)
        if use_trained_text_embed and use_ode_encoder:
            try:
                from .tasks.text_task import TextEmbedding
                from transformers import AutoTokenizer
                # Use LLM tokenizer if direct prefix, otherwise GPT-2
                if _is_direct and qwen_tokenizer is not None:
                    self._text_tokenizer = qwen_tokenizer
                    te_vocab = len(qwen_tokenizer)
                    print(f"  TextEmbedding: using LLM tokenizer (vocab={te_vocab})")
                else:
                    self._text_tokenizer = AutoTokenizer.from_pretrained("gpt2")
                    te_vocab = self._text_tokenizer.vocab_size
                # Detect max_seq_len from checkpoint if available
                te_max_seq = 2048
                if isinstance(ckpt, dict) and 'text_embed_state' in ckpt:
                    pos_shape = ckpt['text_embed_state'].get('pos_embed.weight')
                    if pos_shape is not None:
                        te_max_seq = pos_shape.shape[0]
                self._text_embed = TextEmbedding(
                    vocab_size=te_vocab,
                    d_model=d,
                    max_seq_len=te_max_seq,
                ).to(device)
                # Load trained TextEmbedding weights
                # First check main checkpoint, then look for separate file alongside it
                import os
                loaded_te = False
                ckpt_full = ckpt if isinstance(ckpt, dict) else {}
                if 'text_embed_state' in ckpt_full:
                    self._text_embed.load_state_dict(ckpt_full['text_embed_state'])
                    print(f"  Loaded trained TextEmbedding from checkpoint")
                    loaded_te = True
                if not loaded_te:
                    # Look for text_embed_trained.pt alongside the checkpoint
                    ckpt_dir = os.path.dirname(checkpoint_path)
                    for search_dir in [ckpt_dir, os.path.dirname(ckpt_dir),
                                       os.path.join(os.path.dirname(ckpt_dir), 'output_fluid')]:
                        te_path = os.path.join(search_dir, 'text_embed_trained.pt')
                        if os.path.exists(te_path):
                            te_ckpt = torch.load(te_path, map_location=device, weights_only=False)
                            self._text_embed.load_state_dict(te_ckpt['text_embed_state'])
                            print(f"  Loaded trained TextEmbedding from {te_path}")
                            loaded_te = True
                            break
                if not loaded_te:
                    print("  TextEmbedding: no trained weights found, using random init")
                # Record trained embedding norm scale for norm-clamping
                with torch.no_grad():
                    self._embed_target_norm = self._text_embed.token_embed.weight.norm(dim=-1).mean().item()
                print(f"  Path C hybrid: GPT-2 tokenizer + trained TextEmbedding "
                      f"(norm={self._embed_target_norm:.1f})")
            except Exception as e:
                print(f"  TextEmbedding init failed ({e}), falling back to MindTokenizer")
                self.use_trained_text_embed = False
                self._text_embed = None

        # Conversation-specific modules (Phase 2 metadata preserved)
        self.embedding = ConversationEmbedding(
            d_model=d,
            content_embed_dim=384,
            n_metadata_features=8,
            max_events=max_context_events,
            max_tokens=64,
            tokenizer_path=tokenizer_path,
        ).to(device)

        self.forcing = SensoryForcing(d_model=d, n_entities=max_context_events).to(device)

        self.readout = StateReadout(
            d_model=d, d_summary=256, max_events=max_context_events,
        ).to(device)

        # Persistent ODE state
        self._h: Optional[torch.Tensor] = None

        self._gpu_lock = threading.RLock()  # reentrant — Phase 1 encoding + Phase 2 integration

        # Event buffer
        self.events: List[Dict] = []
        self.event_count = 0
        self._last_event_time = time.time()

        # Integration config
        self.T = getattr(config, 'integration_time', 2.0)
        self.internal_steps = config.n_ode_steps
        self.ntp_loss_weight = getattr(config, 'ntp_loss_weight', 1.0)
        self.ntp_mode = getattr(config, 'ntp_mode', 'raw')  # 'raw' or 'ode'

        # Online learning
        if enable_online_learning:
            param_groups = []
            if self.use_trained_text_embed and self._text_embed is not None:
                # Path C: TextEmbedding at conservative LR (NTP-only gradients)
                text_embed_lr = 1e-4  # fixed conservative — NTP loss maintains structure
                param_groups.append(
                    {'params': list(self._text_embed.parameters()), 'lr': text_embed_lr})
                print(f"  Online learning [Path C]: text_embed_lr={text_embed_lr:.1e}, "
                      f"other_lr={online_lr:.1e}")
            else:
                # Original: MindTokenizer at high LR (learning from scratch)
                embed_lr = online_lr * 100  # 1e-3
                param_groups.append(
                    {'params': list(self.embedding.tokenizer.parameters()), 'lr': embed_lr})
                print(f"  Online learning: embed_lr={embed_lr:.1e}, other_lr={online_lr:.1e}")
            # Dynamics at 100x slower LR (geometric params — preserve trained structure)
            geo_lr = online_lr * 0.1  # 1e-6 — very gentle adaptation
            param_groups.extend([
                {'params': list(self.dynamics.parameters()) +
                           list(self.context_pool.parameters()),
                 'lr': geo_lr},
                {'params': list(self.embedding.metadata_proj.parameters()) +
                           list(self.embedding.event_proj.parameters()) +
                           list(self.embedding.type_embed.parameters()) +
                           list(self.embedding.pos_embed.parameters()) +
                           list(self.embedding.norm.parameters()),
                 'lr': online_lr},
                {'params': list(self.forcing.parameters()), 'lr': online_lr},
                {'params': list(self.readout.parameters()), 'lr': online_lr},
            ])
            self.optimizer = torch.optim.Adam(param_groups)
            print(f"  Dynamics in optimizer: geo_lr={geo_lr:.1e} "
                  f"({sum(p.numel() for p in self.dynamics.parameters()):,} params)")
        else:
            self.optimizer = None

        # Adaptive plasticity controller
        self._plasticity_ctrl = PlasticityController(
            embed_lr_init=5e-3,       # start where productive regime was
            embed_lr_max=5e-3,        # sphere constraint prevents collapse at any LR
            embed_lr_min=1e-4,
            suffocate_patience=50,
            ntp_rise_threshold=1.15,
            ntp_spike_threshold=1.5,
            xform_productive_low=0.03,
            xform_productive_high=0.15,
            xform_danger=0.30,        # higher threshold — sphere prevents collapse
            lr_up_factor=1.2,
            lr_down_factor=0.5,
            lr_emergency_factor=0.1,
        ) if self.optimizer is not None else None

        # ──── LLM INTERFACE ────
        # Direct architecture: ODE state lives in LLM's embedding space (d=2688)
        # h(t) [N, d] → N prefix tokens, no coupling projection needed
        # Inbound: LLM embed_tokens → ODE forcing in same space
        self._qwen_model = qwen_model
        self._qwen_tokenizer = qwen_tokenizer
        self._coupling = coupling  # legacy, may be None for direct architecture
        self._qwen_available = qwen_model is not None or layer_wise_bridge is not None
        self._qwen_is_vllm = hasattr(qwen_model, 'generate') and hasattr(qwen_model, 'is_available')
        self._direct_prefix = (coupling is None and (qwen_model is not None or delta_extractor is not None))
        self._layerwise_mode = layer_wise_bridge is not None
        if self._layerwise_mode:
            print(f"  LLM layer-wise ODE: co-processing at every layer "
                  f"(no delta extraction, no token buffer)")
        elif self._direct_prefix:
            print(f"  LLM direct prefix: ODE state [{max_context_events}, {d}] "
                  f"→ {max_context_events} prefix tokens (no coupling)")
        elif self._qwen_available:
            mode = "vLLM API" if self._qwen_is_vllm else "in-process HF"
            print(f"  LLM coupling active ({mode}): {coupling.n_virtual_tokens} virtual tokens, "
                  f"d_arc={coupling.d_arc} → d_qwen={coupling.d_qwen}")

        # Autonomous processing thread
        self._running = False
        self._auto_thread = None

        # Voice and reflection cycle
        self.voice = None  # set by mcp_serve.py
        self._reflection_interval = 30  # fallback for non-adaptive mode
        self._last_reflection_time = time.time()
        self._last_reflection_text: Optional[str] = None
        self._reflection_count = 0

        # Adaptive routing
        self.trigger: Optional['ReflectionTrigger'] = None
        self._external_event_pending = False
        self._last_user_event_time = 0.0  # timestamp of last user interaction
        self._conversation_quiet_period = 120  # seconds before resuming curriculum
        self._cycles_since_reflection = 0
        self._last_reflection_pe = 0.0
        self.maintenance_interval = 500  # reduced frequency to prevent CUDA OOM from reflection+generation

        # Curriculum
        self.curriculum = None  # set by mcp_serve.py
        self._stimulus_interval = 14  # ~30% ratio with reflection frequency
        self._cycles_since_stimulus = 0
        self._trigger_stats = {
            'total_ode_cycles': 0,
            'triggered_reflections': 0,
            'maintenance_reflections': 0,
            'external_reflections': 0,
            'triggers_by_type': {},
        }

        # ──── THREE WRITE MECHANISMS ────

        # 1. Salience feedback: sustained high-τ attention → relevance vote
        self._salience = torch.zeros(max_context_events, device=device)
        self._high_tau_streak = torch.zeros(max_context_events, dtype=torch.long, device=device)
        self._salience_tau_threshold = 0.85  # τ above this counts as "sustained attention"
        self._salience_streak_threshold = 5  # consecutive cycles before salience increments
        self._salience_increment = 0.05
        self._salience_decay = 0.995  # slow exponential decay per cycle

        # 2. τ floor per event-type: unclustered events get slower integration
        # Computed per-cycle based on event types and cluster membership
        self._tau_bias = torch.zeros(max_context_events, device=device)
        self._tau_bias_unclustered = 0.15  # additive bias for events without a cluster
        self._tau_bias_by_type = {6: 0.10, 7: 0.05}  # reflection, expression get slight bias

        # 3. Hebbian cluster crystallization: co-reflection → metric convergence
        self._consolidation_emb = torch.zeros(max_context_events, d, device=device)
        self._hebbian_lr = 0.001  # very slow — much smaller than gradient LR
        self._last_cluster_assignment: Optional[List] = None  # tracks cluster membership

        # Geometric prediction error (distance between new signal and current state)
        self._last_geometric_pe = 0.0

        # Event embedding cache for text similarity scoring (Channel 1 of hybrid interface)
        # Stores normalized embeddings [max_events, d] — updated on every observe_event
        self._event_embed_cache = torch.zeros(max_context_events, d, device=device)

        # Autonomous ODE norm ceiling — prevents unbounded growth during consolidation.
        # Updated from running norm during active event processing (observe_event).
        # 2x headroom allows geometry to breathe while preventing runaway.
        self._h_norm_ceiling = 50000.0  # initial conservative ceiling
        self._h_norm_ema = 0.0  # running EMA of h_norm during active operation

        # Initialize state
        self._h = torch.zeros(1, max_context_events, d, device=device)

        # ──── TOKEN BUFFER (per-token ODE positions when delta_extractor is active) ────
        # Each entry: {delta_h: [d], source: str, text: str, timestamp: float}
        # When _delta_extractor is set, ODE state is rebuilt from this buffer
        # instead of from the legacy per-event mean-pooled embeddings.
        self._token_buffer: List[Dict] = []
        self._max_tokens_idle: int = 512   # during autonomous cycling
        self._max_tokens_convo: int = 1024  # during active conversation (more retention)
        self._max_tokens: int = 512  # current limit (switches based on activity)

        # Adaptive criticality: EMA of actual D² for automatic target calibration
        self._D_sq_ema: float = 0.0
        self._D_sq_ema_alpha: float = 0.05  # update rate

    # ──────────────────── TEXT ENCODING ────────────────────

    def _embed_text_legacy(self, text: str) -> torch.Tensor:
        """Legacy: embed text using sentence-transformers (384-dim)."""
        if self.text_embedder is None:
            return torch.zeros(384, device=self.device)
        with torch.no_grad():
            emb = self.text_embedder.encode(text, convert_to_tensor=True)
        return emb.to(self.device)

    def _encode_text_ode(self, text: str) -> torch.Tensor:
        """Phase 1: Use the ODE to encode text into an event representation.

        Path C hybrid: if trained TextEmbedding available (from fluid metric
        Stage B), uses GPT-2 tokenizer → trained TextEmbedding → ODE.
        Otherwise falls back to MindTokenizer → ODE.

        Uses a standalone Euler loop — NOT _run_ode_segment which applies
        Phase 2 infrastructure (consolidation_emb, forcing, tau_bias) that
        assumes N ≤ max_events=64. Phase 1 token sequences can be much longer.

        Returns: [768] encoded event representation (mean-pooled)
        """
        with self._gpu_lock:
            d = self.dynamics.norm_geo.normalized_shape[0]

            if self.use_trained_text_embed and self._text_embed is not None:
                # Path C: GPT-2 tokenizer → trained TextEmbedding
                tokens = self._text_tokenizer.encode(
                    text, add_special_tokens=False,
                    truncation=True, max_length=2048,
                )
                if not tokens:
                    return torch.zeros(d, device=self.device)
                input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
                token_h = self._text_embed(input_ids)  # [1, T, d]
                token_mask = torch.ones(1, len(tokens), dtype=torch.bool, device=self.device)
            else:
                # Original: MindTokenizer → embed
                token_h, token_mask = self.embedding.embed_tokens(text, self.device)

            T_actual = token_mask.sum().item()
            if T_actual == 0:
                return torch.zeros(d, device=self.device)

            # Compute context for Phase 1 ODE
            context = self.context_pool(token_h, token_mask)
            self.dynamics.set_context(context, mask=None)
            self.dynamics.set_n_steps(self.internal_steps)

            # Phase 1 Euler loop — clean, no Phase 2 buffers
            with torch.no_grad():
                dt = self.T / self.internal_steps
                h = token_h
                t = 0.0
                for step_i in range(self.internal_steps):
                    if hasattr(self.dynamics, 'set_step_index'):
                        self.dynamics.set_step_index(step_i, self.internal_steps)
                    dh = self.dynamics(t, h)
                    h = h + dt * dh
                    t += dt

            # Mean-pool non-padding positions -> single event representation
            mask_expanded = token_mask.unsqueeze(-1).float()
            h_pooled = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            return h_pooled.squeeze(0)  # [768]

    def _embed_text(self, text: str) -> torch.Tensor:
        """Encode text into ODE-compatible representation.

        Delta mode: LLM hidden state deltas → mean pool → [d] (trajectory signal)
        Direct mode: LLM embed_tokens → mean pool → [d] in LLM's native space
        ODE encoder: GPT-2 TextEmbedding → ODE → [d]
        Legacy: sentence-transformer → [384]
        """
        # Delta extraction: feed LiquidARC the LLM's velocity through meaning-space
        if self._delta_extractor is not None:
            result = self._delta_extractor.extract(text, max_tokens=128)
            return result['delta_h'].float().mean(dim=1).squeeze(0)  # [d_arc], float32

        # Direct architecture: use trained TextEmbedding co-adapted with ODE
        if self._direct_prefix and self.use_trained_text_embed and self._text_embed is not None:
            with torch.no_grad():
                token_ids = self._text_tokenizer(
                    text, return_tensors='pt', truncation=True,
                    max_length=128).input_ids.to(self.device)
                token_embeds = self._text_embed(token_ids)  # [1, T, d]
                return token_embeds.mean(dim=1).squeeze(0)  # [d]

        if not self.use_ode_encoder:
            return self._embed_text_legacy(text)

        ode_repr = self._encode_text_ode(text)  # [768]

        if self._bootstrap_mode and self.text_embedder is not None:
            legacy_repr = self._embed_text_legacy(text)  # [384]
            # Project legacy to 768 via content_proj
            with torch.no_grad():
                legacy_768 = self.embedding.content_proj(legacy_repr.unsqueeze(0)).squeeze(0)
            alpha = min(1.0, self.event_count / max(1, self._bootstrap_events))
            return (1 - alpha) * legacy_768 + alpha * ode_repr

        return ode_repr

    def _process_text_tokens(self, text: str, source: str = 'unknown') -> Dict:
        """Process text into per-token deltas and append to ODE token buffer.

        When a DeltaExtractor is active, every token becomes an individual
        ODE position rather than mean-pooling to a single event vector.
        The _h tensor is rebuilt from the token buffer after each call.

        Args:
            text: input text to process
            source: event type label for token metadata (e.g. 'user_message')

        Returns:
            dict with n_new (tokens added) and n_total (total buffer size)
        """
        if self._delta_extractor is None:
            return {}

        result = self._delta_extractor.extract(text, max_tokens=128)
        n_new = result['n_tokens']
        delta_h_all = result['delta_h'][0]  # [N, d_arc]
        token_texts = result.get('token_texts', [''] * n_new)

        # Append each token as a separate buffer entry
        now = time.time()
        event_id = self.event_count  # unique per observe_event call
        for i in range(n_new):
            self._token_buffer.append({
                'delta_h': delta_h_all[i].detach().float(),  # [d]
                'source': source,
                'event_id': event_id,  # unique per event for cross-event D² diagnostic
                'text': token_texts[i] if i < len(token_texts) else '',
                'timestamp': now,
            })

        # Drop tokens if over limit — priority: keep user/assistant, drop bootstrap/generated
        if len(self._token_buffer) > self._max_tokens:
            n_drop = len(self._token_buffer) - self._max_tokens
            # Score each token: 0=drop first (bootstrap/generated), 1=keep (user/assistant)
            priorities = []
            for t in self._token_buffer:
                src = t.get('source', '')
                if src in ('temporal', 'generated', 'internal_reflection'):
                    priorities.append(0)  # low priority
                elif src in ('user_message', 'assistant_message'):
                    priorities.append(1)  # high priority
                else:
                    priorities.append(0)  # default: low
            # Drop lowest-priority tokens first (by index, preserving temporal order within priority)
            indexed = sorted(enumerate(priorities), key=lambda x: (x[1], x[0]))
            drop_indices = set(idx for idx, _ in indexed[:n_drop])
            self._token_buffer = [t for i, t in enumerate(self._token_buffer) if i not in drop_indices]

        # Rebuild ODE state from token buffer
        N = len(self._token_buffer)
        if N > 0:
            stacked = torch.stack(
                [t['delta_h'] for t in self._token_buffer]
            )  # [N, d]
            self._h = stacked.unsqueeze(0).to(self.device)  # [1, N, d]

        print(f"  [tokens] buffer: +{n_new} new, total={N} tokens (source={source})")
        return {'n_new': n_new, 'n_total': N}

    # ──────────────────── EVENT MANAGEMENT ────────────────────

    def _build_metadata(self, event_type: str, content: str,
                        metadata: Optional[Dict]) -> tuple:
        now = time.time()
        time_delta = now - self._last_event_time
        self._last_event_time = now

        type_map = {
            'user_message': 0, 'assistant_message': 1, 'tool_result': 2,
            'goal': 3, 'context': 4, 'temporal': 5, 'reflection': 6,
            'expression': 7, 'voice_response': 8,
        }
        type_id = type_map.get(event_type, 0)

        meta = torch.zeros(8, device=self.device)
        meta[0] = min(time_delta, 600.0) / 600.0
        meta[1] = min(len(content), 5000) / 5000.0
        meta[2] = float(metadata.get('sentiment', 0.0)) if metadata else 0.0
        meta[3] = float(metadata.get('confidence', 0.5)) if metadata else 0.5
        meta[4] = float(metadata.get('tool_count', 0)) if metadata else 0.0
        meta[5] = float(metadata.get('success', 1.0)) if metadata else 1.0
        meta[6] = min(self.event_count, 500) / 500.0
        meta[7] = type_id / 5.0
        return meta, type_id

    def _get_embedded_events(self) -> torch.Tensor:
        """Get embedded event representations [1, N, d_model].
        Handles both ODE encoder (768-dim) and legacy (384-dim) paths.
        Must be called inside _gpu_lock.
        """
        N = min(len(self.events), self.max_events)
        recent = self.events[-N:]

        if self.use_ode_encoder:
            event_embeds = []
            for i, ev in enumerate(recent):
                emb = ev['embedding']
                if ev.get('geometric', False) or self._direct_prefix:
                    # Direct prefix / geometric: use embedding directly, no metadata wrapper
                    e_emb = emb.unsqueeze(0).unsqueeze(0) if emb.dim() == 1 else emb.unsqueeze(0)
                    e_emb = e_emb.to(self.device)
                else:
                    # Legacy path — embed through ConversationEmbedding
                    e_emb = self.embedding.embed_event(
                        emb.unsqueeze(0) if emb.dim() == 1 else emb,
                        ev['metadata'].unsqueeze(0),
                        torch.tensor([ev['type']], device=self.device),
                        torch.tensor([i], device=self.device),
                    )
                event_embeds.append(e_emb)
            return torch.cat(event_embeds, dim=1)  # [1, N, 768]
        else:
            tokens = self._tokenize_current_events()
            return self.embedding(
                tokens['content_embeddings'],
                tokens['metadata_features'],
                tokens['event_types'],
                tokens['positions'],
            )

    def _tokenize_current_events(self) -> Dict[str, torch.Tensor]:
        N = min(len(self.events), self.max_events)
        recent = self.events[-N:]

        content_embs = torch.stack([e['embedding'] for e in recent]).unsqueeze(0)
        metadata = torch.stack([e['metadata'] for e in recent]).unsqueeze(0)
        types = torch.tensor([e['type'] for e in recent], device=self.device).unsqueeze(0)
        positions = torch.arange(N, device=self.device).unsqueeze(0)

        return {
            'content_embeddings': content_embs,
            'metadata_features': metadata,
            'event_types': types,
            'positions': positions,
            'n_events': N,
        }

    # ──────────────────── WRITE MECHANISMS ────────────────────

    def _update_salience(self, h: torch.Tensor):
        """Update per-event salience from sustained high-τ attention.
        Must be called inside _gpu_lock. Modifies _salience in-place.
        Skipped in token-level mode (salience is event-level, not token-level).
        """
        if self._delta_extractor is not None:
            return  # token-level mode — salience doesn't map to token positions
        N = min(len(self.events), self.max_events)
        if N == 0:
            return
        with torch.no_grad():
            tau = self.dynamics.compute_tau(h[:, :N, :])  # [1, N, 1]
            tau_flat = tau[0, :, 0]  # [N]

            # Update high-τ streak
            high = (tau_flat > self._salience_tau_threshold).long()
            self._high_tau_streak[:N] = self._high_tau_streak[:N] * high + high
            self._high_tau_streak[N:] = 0

            # Increment salience where streak exceeds threshold
            earned = (self._high_tau_streak[:N] >= self._salience_streak_threshold).float()
            self._salience[:N] = self._salience[:N] + earned * self._salience_increment

            # Slow decay
            self._salience[:N] = self._salience[:N] * self._salience_decay

    def _compute_tau_bias(self):
        """Compute per-event τ bias from event type and cluster membership.
        Sets self._tau_bias and self.dynamics._tau_external_bias.
        """
        N = min(len(self.events), self.max_events)
        if N == 0:
            return
        recent = self.events[-N:]
        bias = torch.zeros(N, device=self.device)

        # Per-type bias
        for i, ev in enumerate(recent):
            etype = ev['type']
            if etype in self._tau_bias_by_type:
                bias[i] = self._tau_bias_by_type[etype]

        # Unclustered events get extra bias
        if self._last_cluster_assignment is not None:
            clustered = set()
            for cluster in self._last_cluster_assignment:
                for idx in cluster:
                    clustered.add(idx)
            for i in range(N):
                if i not in clustered:
                    bias[i] += self._tau_bias_unclustered

        self._tau_bias[:N] = bias
        self._tau_bias[N:] = 0

        # Push to dynamics — token-level mode has more positions than events,
        # so tau_bias (per-event) doesn't map 1:1. Disable per-event bias in token mode.
        if self._delta_extractor is not None:
            self.dynamics._tau_external_bias = None
        else:
            self.dynamics._tau_external_bias = self._tau_bias

    def _hebbian_nudge(self, h: torch.Tensor, focus_indices: List[int]):
        """Co-reflection Hebbian update on consolidation embeddings.
        Events that co-appear in reflection focus pull closer in h-space.
        Must be called inside _gpu_lock.
        """
        if len(focus_indices) < 2:
            return
        N = min(len(self.events), self.max_events)

        with torch.no_grad():
            # For all pairs of co-focused events, nudge toward each other
            for a in range(len(focus_indices)):
                for b in range(a + 1, len(focus_indices)):
                    i, j = focus_indices[a], focus_indices[b]
                    if i >= N or j >= N:
                        continue
                    diff = h[0, j] - h[0, i]
                    self._consolidation_emb[i] += self._hebbian_lr * diff
                    self._consolidation_emb[j] -= self._hebbian_lr * diff

    # ──────────────────── ODE INTEGRATION ────────────────────

    def _run_ode_segment(self, h: torch.Tensor, n_steps: int,
                         forcing: Optional[torch.Tensor] = None,
                         return_efficiency: bool = False,
                         accumulate_bias: bool = False):
        """Run n_steps of ODE. Caller must hold _gpu_lock.
        Applies consolidation embeddings to h before dynamics computation.

        If accumulate_bias=True, computes and sums the SDPA bias logits
        B_ij at each step across the full ODE trajectory. The accumulated
        bias captures which tokens the ODE consistently routed together —
        the trajectory of routing decisions, not a snapshot. Stored in
        self._accumulated_bias [N, N].
        """
        dt = self.T / n_steps
        t = 0.0
        eff_accum = torch.tensor(0.0, device=h.device)
        N = h.shape[1]

        # Apply Hebbian consolidation embedding (additive bias on h)
        # Only for Phase 2 (persistent event state), not Phase 1 (token encoding)
        if N <= self._consolidation_emb.shape[0]:
            consol = self._consolidation_emb[:N].unsqueeze(0)  # [1, N, d]
            if consol.any():
                h = h + consol

        for i in range(n_steps):
            if hasattr(self.dynamics, 'set_step_embed'):
                self.dynamics.set_step_embed(i, n_steps)
            if hasattr(self.dynamics, 'set_step_index'):
                self.dynamics.set_step_index(i, n_steps)

            dy = self.dynamics(t, h)

            if forcing is not None:
                decay = 1.0 - (i / n_steps)
                dy = dy + decay * forcing

            if return_efficiency:
                eff_accum = eff_accum + (dy.detach() ** 2).mean()

            h = h + dt * dy

            # Accumulate bias logits across ODE trajectory
            if accumulate_bias and N <= 512:
                with torch.no_grad():
                    h_n = self.dynamics.norm_geo(h)
                    g_step = torch.nn.functional.softplus(
                        self.dynamics.metric_net_linear2_diag(
                            torch.nn.functional.gelu(
                                self.dynamics.metric_net_linear1(
                                    torch.cat([h_n, self.dynamics._context.unsqueeze(1).expand(-1, N, -1)], dim=-1)
                                )
                            )
                        )
                    )
                    qk = h_n * g_step.sqrt()
                    t_d = torch.nn.functional.softplus(self.dynamics.t_diffusion)
                    B_step = (torch.bmm(qk, qk.transpose(1, 2)) / (2.0 * t_d)
                              - (qk * qk).sum(dim=-1, keepdim=True).transpose(1, 2) / (4.0 * t_d))
                    self._accumulated_bias = self._accumulated_bias + B_step[0]

            # Per-position norm homeostasis — same as euler_solve
            norm_ref = getattr(self.dynamics, '_norm_ref', 0.0)
            norm_lambda = getattr(self.dynamics, '_norm_lambda', 0.0)
            if norm_ref > 0 and norm_lambda > 0:
                pos_norm = h.detach().norm(dim=-1, keepdim=True).clamp(min=1e-8)
                scale = torch.where(
                    pos_norm > norm_ref,
                    1.0 - norm_lambda * (1.0 - norm_ref / pos_norm),
                    torch.ones_like(pos_norm),
                )
                h = h * scale

            t = t + dt

        if return_efficiency:
            return h, eff_accum / n_steps
        return h

    # ──────────────────── CORE METHODS (MCP tools call these) ────────────────────

    def observe_event(self, event_type: str, content: str,
                      metadata: Optional[Dict] = None) -> Dict:
        """Inject a conversation event as sensory forcing.

        When _delta_extractor is active: processes text into per-token ODE
        positions via _process_text_tokens, then runs ODE integration over the
        full token buffer. Events list still stores metadata for context building.

        When _delta_extractor is None: falls back to legacy mean-pooled embedding.

        SAFE CURIOSITY: Uses prediction error ||h_before - h_after||,
        NOT dynamics magnitude ||dh/dt|| (which causes NaN).
        """
        # Flag for adaptive routing — only HUMAN events trigger immediate reflection
        # Internal events (reflection, expression, curriculum) should not cascade
        is_internal = (
            event_type in ('reflection', 'expression', 'voice_response') or
            (metadata and metadata.get('source', '').startswith(
                ('curriculum', 'geometric_curriculum', 'internal', 'express_state',
                 'self_', 'adaptive_', 'geometric_', 'manual_', 'qwen3_')))
        )
        if not is_internal:
            self._external_event_pending = True
            self._last_user_event_time = time.time()
            self._max_tokens = self._max_tokens_convo  # expand buffer for conversation

        # ── Token-level path (DeltaExtractor active) ──
        if self._delta_extractor is not None:
            # Extract tokens OUTSIDE lock (Qwen3 forward pass is slow)
            result = self._delta_extractor.extract(content, max_tokens=128)

            with self._gpu_lock:
                # Rebuild token buffer and self._h INSIDE lock to prevent race with autonomous loop
                n_new = result['n_tokens']
                delta_h_all = result['delta_h'][0]  # [N, d_arc]
                token_texts = result.get('token_texts', [''] * n_new)
                now = time.time()
                event_id = self.event_count  # unique per observe_event call
                for i in range(n_new):
                    self._token_buffer.append({
                        'delta_h': delta_h_all[i].detach().float(),
                        'source': event_type,
                        'event_id': event_id,
                        'text': token_texts[i] if i < len(token_texts) else '',
                        'timestamp': now,
                    })
                if len(self._token_buffer) > self._max_tokens:
                    self._token_buffer = self._token_buffer[-self._max_tokens:]
                N = len(self._token_buffer)
                if N > 0:
                    self._h = torch.stack(
                        [t['delta_h'] for t in self._token_buffer]
                    ).unsqueeze(0).to(self.device)
                print(f"  [tokens] buffer: +{n_new} new, total={N} tokens (source={event_type})")

                # Store event metadata
                meta, type_id = self._build_metadata(event_type, content, metadata)
                self.events.append({
                    'embedding': self._h[0, -1, :].detach().float() if N > 0
                                 else torch.zeros(self.dynamics.norm_geo.normalized_shape[0],
                                                  device=self.device),
                    'metadata': meta,
                    'type': type_id,
                    'content_preview': content[:200],
                    'timestamp': now,
                    'n_tokens': n_new,
                })
                self.event_count += 1
                if len(self.events) > self.max_events:
                    self.events = self.events[-self.max_events:]

                # Run ODE integration over full token buffer
                N = self._h.shape[1]
                if N == 0:
                    return {'prediction_error': 0.0, 'cv': 0.0,
                            'events_in_context': len(self.events), 'h_norm': 0.0,
                            'n_tokens': 0}

                context = self.context_pool(
                    self._h, torch.ones(1, N, dtype=torch.bool, device=self.device))
                self.dynamics.set_context(context, mask=None)
                self.dynamics.set_n_steps(self.internal_steps)

                h_before = self._h.detach().clone()
                h_new = self._run_ode_segment(
                    self._h, self.internal_steps, forcing=None)

                with torch.no_grad():
                    h_b_norm = h_before.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    h_a_norm = h_new.detach().norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    cos_disp = (h_before / h_b_norm * h_new.detach() / h_a_norm).sum(dim=-1)
                    pe = (1.0 - cos_disp).mean().item() * 500.0

                    # Displacement correlation bias: tokens that the ODE moved in the
                    # same direction were geometrically connected by routing.
                    # This IS the ODE's computation — extracted from dynamics, not metric.
                    delta_h = h_new.detach() - h_before  # [1, N, d] — how each position changed
                    delta_norm = delta_h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    delta_unit = delta_h / delta_norm  # [1, N, d] unit displacement vectors
                    # Correlation: cosine similarity of displacement vectors
                    # corr_ij = Δh_i · Δh_j / (||Δh_i|| · ||Δh_j||)
                    self._displacement_bias = torch.bmm(
                        delta_unit, delta_unit.transpose(1, 2)
                    )[0]  # [N, N] in [-1, +1]

                self._h = h_new.detach()

                # ── Self-supervised MetricNet adaptation ──
                # Run a SEPARATE gradient-enabled ODE step on a small subsample
                # to compute routing quality loss and update MetricNet.
                # This is cheap (subsample of 32 tokens, 1 ODE step) and runs
                # every observe_event, allowing MetricNet to adapt to text.
                if self.optimizer is not None and N >= 8:
                    try:
                        n_sub = min(32, N)
                        # Sample tokens spread across events
                        step_s = max(1, N // n_sub)
                        sub_idx = list(range(0, N, step_s))[:n_sub]
                        h_sub = self._h[:, sub_idx, :].clone().requires_grad_(True)

                        # One ODE step with gradients
                        context_sub = self.context_pool(h_sub)
                        self.dynamics.set_context(context_sub, mask=None)
                        self.dynamics.set_n_steps(1)
                        dy = self.dynamics(0.0, h_sub)
                        h_sub_post = h_sub + (self.T / self.internal_steps) * dy

                        # Self-supervised loss: CV should stay in productive range
                        g_sub = self.dynamics.compute_metric_diag(h_sub_post)
                        cv_sub = g_sub.std() / (g_sub.mean() + 1e-8)
                        cv_loss = torch.nn.functional.relu(3.0 - cv_sub) ** 2  # floor at 3.0

                        # Tau quality: mean near target
                        tau_sub = self.dynamics.compute_tau(h_sub_post)
                        tau_anchor_target = getattr(self.dynamics, '_tau_anchor_target', 2.0)
                        tau_loss = (tau_sub.mean() - tau_anchor_target) ** 2

                        adapt_loss = 0.1 * cv_loss + 0.05 * tau_loss
                        if adapt_loss.item() > 1e-6 and not torch.isnan(adapt_loss):
                            self.optimizer.zero_grad()
                            adapt_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 0.1)
                            self.optimizer.step()
                    except Exception:
                        pass  # don't crash on adaptation failure

                g = self.dynamics.compute_metric_diag(self._h.detach())
                cv = (g.std() / (g.mean() + 1e-8)).item()
                h_norm_val = self._h.detach().norm().item()
                tau_val = self.dynamics.compute_tau(self._h.detach()).mean().item()

                alpha = 0.05
                self._h_norm_ema = (alpha * h_norm_val +
                                    (1 - alpha) * self._h_norm_ema
                                    ) if self._h_norm_ema > 0 else h_norm_val

            import math
            h_norm_per_pos = h_norm_val / math.sqrt(N) if N > 0 else 0.0
            tau_std_val = self.dynamics.compute_tau(h_new.detach()).std().item()

            print(f"  [observe] #{self.event_count} type={event_type} "
                  f"PE={pe:.1f} CV={cv:.2f} tau={tau_val:.2f}±{tau_std_val:.4f} "
                  f"h={h_norm_val:.0f} h/√N={h_norm_per_pos:.1f} "
                  f"tokens={N} events={len(self.events)} "
                  f"\"{content[:60]}\"")

            return {
                'prediction_error': pe,
                'prediction_error_per_event': [pe],
                'cv': cv,
                'metric_cv': cv,
                'tau_mean': tau_val,
                'tau_std': tau_std_val,
                'events_in_context': len(self.events),
                'h_norm': h_norm_val,
                'h_norm_per_position': h_norm_per_pos,
                'n_tokens': n_new,
                'token_buffer_size': len(self._token_buffer),
            }

        # ── Legacy mean-pooled embedding path ──
        embedding = self._embed_text(content)
        meta, type_id = self._build_metadata(event_type, content, metadata)

        self.events.append({
            'embedding': embedding,
            'metadata': meta,
            'type': type_id,
            'content_preview': content[:200],
            'timestamp': time.time(),
        })
        self.event_count += 1

        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        # Update event embedding cache — rebuild from current buffer
        self._rebuild_event_embed_cache()

        with self._gpu_lock:
            N = min(len(self.events), self.max_events)
            recent = self.events[-N:]

            if self.use_ode_encoder:
                event_embeds = []
                for i, ev in enumerate(recent):
                    emb = ev['embedding']
                    if ev.get('geometric', False) or self._direct_prefix:
                        # Direct prefix / geometric: use embedding directly
                        e_emb = emb.unsqueeze(0).unsqueeze(0) if emb.dim() == 1 else emb.unsqueeze(0)
                        e_emb = e_emb.to(self.device)
                    else:
                        # Legacy: embed through ConversationEmbedding
                        e_emb = self.embedding.embed_event(
                            emb.unsqueeze(0) if emb.dim() == 1 else emb,
                            ev['metadata'].unsqueeze(0),
                            torch.tensor([ev['type']], device=self.device),
                            torch.tensor([i], device=self.device),
                        )
                    event_embeds.append(e_emb)
                obs_embed = torch.cat(event_embeds, dim=1)  # [1, N, d]
            else:
                # Legacy path: 384-dim sentence-transformer embeddings
                tokens = self._tokenize_current_events()
                N = tokens['n_events']
                obs_embed = self.embedding(
                    tokens['content_embeddings'],
                    tokens['metadata_features'],
                    tokens['event_types'],
                    tokens['positions'],
                )

            h_slice = self._h[:, :N, :]

            forcing = self.forcing.compute_forcing(h_slice, obs_embed)

            context_mask = torch.ones(1, N, dtype=torch.bool, device=self.device)
            context = self.context_pool(h_slice, context_mask)
            self.dynamics.set_context(context, mask=None)
            self.dynamics.set_n_steps(self.internal_steps)

            h_before = h_slice.detach().clone()
            h_new = self._run_ode_segment(h_slice, self.internal_steps, forcing=forcing)

            # PE = state displacement from forcing injection.
            # Measures how much the observation actually perturbed the ODE state.
            # Novel content → large displacement (system had to reorganize).
            # Familiar content → small displacement (system already aligned).
            # Per-position cosine distance between h_before and h_after, scaled to [0, 1000].
            with torch.no_grad():
                h_b_norm = h_before.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                h_a_norm = h_new.detach().norm(dim=-1, keepdim=True).clamp(min=1e-8)
                cos_displacement = (h_before / h_b_norm * h_new.detach() / h_a_norm).sum(dim=-1)
                prediction_error = (1.0 - cos_displacement) * 500.0  # [B, N]

            self._h = self._h.clone()
            self._h[:, :N, :] = h_new.detach()

            g = self.dynamics.compute_metric_diag(h_new.detach())
            cv = (g.std() / (g.mean() + 1e-8)).item()

            # Track h_norm for diagnostics (homeostasis handled in ODE dynamics)
            current_norm = h_new.detach().norm().item()
            alpha = 0.05
            self._h_norm_ema = (alpha * current_norm +
                                (1 - alpha) * self._h_norm_ema) if self._h_norm_ema > 0 else current_norm

        pe_val = prediction_error.mean().item()
        h_norm_val = h_new.detach().norm().item()
        tau_val = self.dynamics.compute_tau(h_new.detach()).mean().item()
        print(f"  [observe] #{self.event_count} type={event_type} "
              f"PE={pe_val:.1f} CV={cv:.2f} tau={tau_val:.2f} "
              f"h={h_norm_val:.0f} events={len(self.events)} "
              f"\"{content[:60]}\"")

        return {
            'prediction_error': pe_val,
            'prediction_error_per_event': prediction_error[0].cpu().tolist()[:10],
            'cv': cv,
            'events_in_context': len(self.events),
            'h_norm': h_norm_val,
        }

    def get_context(self, query: Optional[str] = None) -> Dict:
        """Read current relevance scores, summary, and focus directives."""
        if self._h is None or len(self.events) == 0:
            return {'status': 'no_state', 'events': []}

        with self._gpu_lock:
            N = min(len(self.events), self.max_events)
            h_current = self._h[:, :N, :]
            event_types = torch.tensor(
                [e['type'] for e in self.events[-N:]], device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                readout_result = self.readout(h_current, event_types)

        relevance = readout_result['relevance_scores'][0].cpu().tolist()
        focus_idx = readout_result['focus_indices'][0].cpu().tolist()
        focus_scores = readout_result['focus_scores'][0].cpu().tolist()

        recent = self.events[-N:]
        context_items = []
        type_names = ['user_msg', 'assistant_msg', 'tool_result',
                     'goal', 'context', 'temporal', 'reflection', 'expression', 'voice_response']
        for i, (event, rel) in enumerate(zip(recent, relevance)):
            # Blend salience into relevance (mechanism 1 readout)
            sal = self._salience[i].item() if i < self._salience.shape[0] else 0
            blended_rel = rel + sal * 0.3  # salience adds up to 30% relevance boost
            context_items.append({
                'index': i,
                'type': type_names[event['type']] if event['type'] < len(type_names) else 'unknown',
                'preview': event['content_preview'],
                'relevance': round(blended_rel, 3),
                'salience': round(sal, 3),
                'age_seconds': round(time.time() - event['timestamp'], 1),
            })

        context_items.sort(key=lambda x: x['relevance'], reverse=True)

        return {
            'status': 'active',
            'n_events': N,
            'context': context_items,
            'focus_indices': focus_idx,
            'focus_scores': focus_scores,
            'summary_norm': readout_result['summary'].norm().item(),
        }

    def get_diagnostics(self) -> Dict:
        """Read model internals for monitoring."""
        if self._h is None:
            return {'status': 'inactive'}

        with self._gpu_lock:
            h = self._h
            N_tokens = h.shape[1]
            N_events = min(len(self.events), self.max_events)

            if N_tokens == 0:
                return {'status': 'no_events', 'h_norm': 0.0}

            # Use full token buffer for geometry diagnostics
            g = self.dynamics.compute_metric_diag(h.detach())
            tau = self.dynamics.compute_tau(h.detach())

            import math
            h_norm_val = h.norm().item()
            h_norm_per_pos = h_norm_val / math.sqrt(N_tokens)

            # Beta only available for event-level N
            beta_mean = 0.0
            if N_events > 0 and not self._delta_extractor:
                beta = self.forcing.beta[:N_events]
                beta_mean = beta.mean().item()

        return {
            'status': 'active',
            'h_norm': h_norm_val,
            'h_norm_per_position': h_norm_per_pos,
            'metric_cv': (g.std() / (g.mean() + 1e-8)).item(),
            'tau_mean': tau.mean().item(),
            'tau_std': tau.std().item(),
            'events_in_context': N_events,
            'token_buffer_size': N_tokens,
            'event_count_total': self.event_count,
        }

    # ──────────────────── HYBRID INTERFACE HELPERS ────────────────────

    def _rebuild_event_embed_cache(self):
        """Rebuild the normalized event embedding cache from current buffer.

        Called after every observe_event and _force_geometric_signal.
        Stores F.normalize'd embeddings for fast cosine similarity at query time.
        """
        N = min(len(self.events), self.max_events)
        self._event_embed_cache.zero_()
        with torch.no_grad():
            for i, ev in enumerate(self.events[-N:]):
                emb = ev.get('embedding')
                if emb is not None:
                    if emb.dim() == 1:
                        self._event_embed_cache[i] = F.normalize(emb.float(), dim=0)
                    else:
                        self._event_embed_cache[i] = F.normalize(
                            emb.float().mean(dim=0), dim=0)

    def _compute_text_relevance(self, query: str, n_events: int) -> List[float]:
        """Score events by text similarity to the query.

        Layer 1 of three-layer relevance scoring.
        Uses cached normalized embeddings — only one embed call for the query.
        """
        if n_events == 0:
            return []

        # Encode query using the same path as events
        with torch.no_grad():
            q_emb = self._embed_text(query)
            if q_emb.dim() > 1:
                q_emb = q_emb.mean(dim=0)
            q_emb = F.normalize(q_emb.float(), dim=0)

            # Batch dot product against cache
            scores = (self._event_embed_cache[:n_events] @ q_emb).cpu().tolist()

        return scores

    def _compute_structural_relevance(self, events: List[Dict],
                                       query_type: str = 'conversation') -> List[float]:
        """Score events by type and recency.

        Layer 2 of three-layer relevance scoring.
        Conversation queries boost user/assistant events, deprioritize curriculum.
        """
        import math
        type_names = ['user_msg', 'assistant_msg', 'tool_result',
                      'goal', 'context', 'temporal', 'reflection',
                      'expression', 'voice_response']
        scores = []
        for ev in events:
            score = 1.0
            type_id = ev.get('type', 0)
            etype = type_names[type_id] if type_id < len(type_names) else 'unknown'
            age = time.time() - ev.get('timestamp', time.time())

            if query_type == 'conversation':
                if etype in ('user_msg', 'assistant_msg'):
                    score *= 2.0
                elif etype == 'goal':
                    score *= 1.5
                elif etype == 'expression':
                    score *= 1.0
                elif etype == 'context':
                    score *= 0.5  # curriculum deprioritized for conversation
            # Domain queries could boost context events, but default is fine

            # Recency: exponential decay with ~5 minute half-life, floor at 0.3
            recency = math.exp(-age / 300.0)
            score *= (0.3 + 0.7 * recency)

            scores.append(score)
        return scores

    def _compute_combined_relevance(self, query: str,
                                     w_text: float = 0.5,
                                     w_structural: float = 0.3,
                                     w_geometric: float = 0.2) -> List[Dict]:
        """Three-layer relevance scoring for hybrid interface Channel 1.

        Combines:
          Layer 1: Text similarity (cosine in embedding space)
          Layer 2: Structural (event type + recency)
          Layer 3: Geometric (StateReadout scores, trained via feedback)

        Returns events sorted by combined score (descending).
        """
        N = min(len(self.events), self.max_events)
        if N == 0:
            return []

        recent = self.events[-N:]

        # Layer 1: Text similarity
        text_scores = self._compute_text_relevance(query, N)

        # Layer 2: Structural
        struct_scores = self._compute_structural_relevance(recent)

        # Layer 3: Geometric (existing readout)
        geo_scores = [0.0] * N
        ctx = self.get_context(query=query)
        if ctx.get('status') == 'active':
            for item in ctx.get('context', []):
                idx = item.get('index', -1)
                if 0 <= idx < N:
                    geo_scores[idx] = item.get('relevance', 0)

        # Normalize each layer to [0, 1]
        def normalize(scores):
            if not scores:
                return scores
            mn, mx = min(scores), max(scores)
            rng = mx - mn if mx > mn else 1.0
            return [(s - mn) / rng for s in scores]

        text_norm = normalize(text_scores)
        struct_norm = normalize(struct_scores)
        geo_norm = normalize(geo_scores)

        # Build scored event list
        type_names = ['user_msg', 'assistant_msg', 'tool_result',
                      'goal', 'context', 'temporal', 'reflection',
                      'expression', 'voice_response']
        result = []
        for i, ev in enumerate(recent):
            combined = (w_text * text_norm[i] +
                        w_structural * struct_norm[i] +
                        w_geometric * geo_norm[i])
            type_id = ev.get('type', 0)
            result.append({
                'index': i,
                'type': type_names[type_id] if type_id < len(type_names) else 'unknown',
                'preview': ev.get('content_preview', ''),
                'relevance': round(combined, 3),
                'text_sim': round(text_norm[i], 3),
                'structural': round(struct_norm[i], 3),
                'geometric': round(geo_norm[i], 3),
                'age_seconds': round(time.time() - ev.get('timestamp', time.time()), 1),
            })

        result.sort(key=lambda x: x['relevance'], reverse=True)
        return result

    def get_relevant_events(self, query: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """Get top-K relevance-scored events using three-layer combined scoring.

        Channel 1 of hybrid interface: selects the most relevant events
        for inclusion as text context in Qwen3 generation.
        """
        if not query or len(self.events) == 0:
            # No query — fall back to recency
            N = min(len(self.events), self.max_events)
            type_names = ['user_msg', 'assistant_msg', 'tool_result',
                          'goal', 'context', 'temporal', 'reflection',
                          'expression', 'voice_response']
            recent = self.events[-min(top_k, N):]
            return [{
                'index': i,
                'type': type_names[ev.get('type', 0)] if ev.get('type', 0) < len(type_names) else 'unknown',
                'preview': ev.get('content_preview', ''),
                'relevance': 1.0,
                'age_seconds': round(time.time() - ev.get('timestamp', time.time()), 1),
            } for i, ev in enumerate(recent)]

        scored = self._compute_combined_relevance(query)
        return scored[:top_k]

    def get_recent_reflections(self, n: int = 5) -> List[Dict]:
        """Get last N reflection/expression events for stagnation detection."""
        reflections = []
        for ev in reversed(self.events):
            if ev.get('type') in (6, 7):  # reflection, expression
                reflections.append({
                    'text': ev.get('content_preview', ''),
                    'timestamp': ev.get('timestamp', 0),
                })
                if len(reflections) >= n:
                    break
        reflections.reverse()
        return reflections

    def get_all_events(self) -> List[Dict]:
        """Return all events in the buffer with metadata and text."""
        N = min(len(self.events), self.max_events)
        type_names = ['user_msg', 'assistant_msg', 'tool_result',
                      'goal', 'context', 'temporal', 'reflection',
                      'expression', 'voice_response']
        result = []
        for i, ev in enumerate(self.events[-N:]):
            type_id = ev.get('type', 0)
            result.append({
                'index': i,
                'type': type_names[type_id] if type_id < len(type_names) else 'unknown',
                'preview': ev.get('content_preview', ''),
                'age_seconds': round(time.time() - ev.get('timestamp', time.time()), 1),
                'salience': self._salience[i].item() if i < self._salience.shape[0] else 0,
            })
        return result

    def get_curriculum_stats_dict(self) -> Dict:
        """Get curriculum statistics for metadata formatting."""
        stats = getattr(self, '_curriculum_stats', None)
        if not stats or not stats.get('domain_avg_pe'):
            return {'most_familiar_domain': 'unknown', 'most_novel_domain': 'unknown'}

        avg_pe = stats['domain_avg_pe']
        if not avg_pe:
            return {'most_familiar_domain': 'unknown', 'most_novel_domain': 'unknown'}

        familiar = min(avg_pe, key=avg_pe.get)
        novel = max(avg_pe, key=avg_pe.get)
        return {
            'most_familiar_domain': self.DOMAIN_NAMES.get(familiar, familiar),
            'most_novel_domain': self.DOMAIN_NAMES.get(novel, novel),
        }

    # ──────────────────── QWEN3 KNOWLEDGE INTERFACE ────────────────────

    def _get_pooled_state(self) -> Optional[torch.Tensor]:
        """Get mean-pooled ODE state [d] for coupling path (legacy)."""
        if self._h is None:
            return None
        N = min(len(self.events), self.max_events)
        if N == 0:
            return None
        with torch.no_grad():
            return self._h[:, :N, :].mean(dim=1).squeeze(0).to(torch.bfloat16)  # [d]

    def _get_prefix_embeds(self) -> Optional[torch.Tensor]:
        """Get full ODE state as prefix tokens for direct architecture.

        Returns h(t) [1, N, d] — each event position becomes a prefix token.
        The ODE state lives in the LLM's embedding space, no projection needed.
        """
        if self._h is None:
            return None
        N = min(len(self.events), self.max_events)
        if N == 0:
            return None
        with torch.no_grad():
            return self._h[:, :N, :].detach().to(torch.bfloat16)  # [1, N, d]

    def _encode_through_qwen(self, text: str) -> Optional[torch.Tensor]:
        """Encode text through Qwen3 → W_read → geometric signal for LiquidARC.

        The inbound path: text enters LiquidARC not as tokens but as geometry.
        Qwen3 processes the text (with current state prefix), and the hidden
        states at prefix positions are projected back to LiquidARC's space
        via W_read. This geometric signal becomes sensory forcing.

        Args:
            text: Input text to encode geometrically

        Returns:
            arc_signal: [d_arc] geometric representation, or None if unavailable
        """
        if not self._qwen_available or self._coupling is None:
            return None

        h_state = self._get_pooled_state()
        if h_state is None:
            d = self.dynamics.norm_geo.normalized_shape[0]
            h_state = torch.zeros(d, device=self.device,
                                  dtype=torch.bfloat16)

        with self._gpu_lock, torch.no_grad():
            # Project current state → prefix
            prefix_embeds = self._coupling.inject(h_state)  # [1, n_vt, d_qwen]

            # Tokenize text through Qwen3
            tokens = self._qwen_tokenizer(
                text, return_tensors='pt', truncation=True,
                max_length=512).to(self.device)
            input_ids = tokens['input_ids']

            # Forward through Qwen3 with prefix (no generation, just encoding)
            input_embeds = self._qwen_model.get_input_embeddings()(input_ids)
            combined = torch.cat([prefix_embeds, input_embeds], dim=1)
            attn_mask = torch.ones(1, combined.shape[1],
                                   dtype=torch.long, device=self.device)

            outputs = self._qwen_model(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            # Read prefix positions from last hidden state → geometric signal
            last_hidden = outputs.hidden_states[-1]
            n_vt = self._coupling.n_virtual_tokens
            prefix_output = last_hidden[:, :n_vt, :]  # [1, n_vt, d_qwen]
            arc_signal = self._coupling.read(prefix_output)  # [d_arc]

        return arc_signal

    def _force_geometric_signal(self, signal: torch.Tensor, source_text: str,
                                event_type: str = 'user_message',
                                metadata: Optional[Dict] = None):
        """Inject a geometric signal into LiquidARC's ODE state as forcing.

        The signal enters directly as a d_arc vector — no tokenization,
        no embedding table lookup. Pure geometric forcing.

        Args:
            signal: [d_arc] geometric representation from Qwen3
            source_text: Original text (stored in event buffer for context)
            event_type: Event type for the buffer
            metadata: Optional metadata dict
        """
        # Store in event buffer (for get_context and readout)
        meta_tensor, type_id = self._build_metadata(event_type, source_text, metadata)
        self.events.append({
            'embedding': signal.detach().float(),  # store as float32
            'metadata': meta_tensor,
            'type': type_id,
            'content_preview': source_text[:200],
            'timestamp': time.time(),
            'geometric': True,  # flag: skip embed_event, use directly
        })
        self.event_count += 1

        # Trim buffer
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        # Update event embedding cache
        self._rebuild_event_embed_cache()

        # Inject signal directly into ODE state — no re-embedding
        with self._gpu_lock:
            N = min(len(self.events), self.max_events)
            if self._h is not None and N <= self._h.shape[1]:
                # Compute geometric PE: cosine distance (magnitude-independent)
                with torch.no_grad():
                    h_prev = self._h[:, N - 1, :].squeeze(0).float()
                    sig = signal.detach().float()
                    cos_sim = F.cosine_similarity(sig.unsqueeze(0), h_prev.unsqueeze(0)).item()
                    self._last_geometric_pe = (1.0 - cos_sim) * 500.0

                # Place geometric signal at the latest event position
                with torch.no_grad():
                    self._h[:, N - 1, :] = signal.detach().float()

                    # Run one ODE integration to let dynamics route the new signal
                    context = self.context_pool(self._h[:, :N, :])
                    self.dynamics.set_context(context, mask=None)
                    self.dynamics.set_n_steps(self.internal_steps)

                    from .solver import euler_solve
                    h_updated = euler_solve(
                        self.dynamics, self._h[:, :N, :],
                        t_span=(0.0, self.T),
                        n_steps=self.internal_steps,
                    )
                    # Norm floor: prevent more than 50% collapse from single event
                    h_norm_before = self._h[:, :N, :].norm()
                    self._h = self._h.detach()
                    self._h[:, :N, :] = h_updated
                    h_norm_after = self._h[:, :N, :].norm()
                    if h_norm_after < 0.5 * h_norm_before and h_norm_after > 1e-8:
                        scale = (0.5 * h_norm_before) / h_norm_after
                        self._h[:, :N, :] *= scale

    def hybrid_generate(self, prompt: str, max_new_tokens: int = 200,
                        temperature: float = 0.7,
                        include_text_context: bool = True,
                        include_geometric_prefix: bool = True,
                        include_metadata: bool = True,
                        max_context_events: int = 5) -> Dict:
        """Generate response using three channels.

        Channel 1: Text context — relevance-scored events as text in prompt
        Channel 2: Geometric prefix — W_inject virtual tokens (HF path only)
        Channel 3: Structured metadata — PE, CV, tau, domain awareness

        Falls back gracefully: without coupling, Channels 1+3 still work.
        """
        if not self._qwen_available:
            return {'error': 'LLM not connected'}

        # ═══ Channel 1: Text Context ═══
        text_context = ""
        if include_text_context:
            events = self.get_relevant_events(query=prompt, top_k=max_context_events)
            if events:
                text_context = "Recent context:\n"
                for event in events:
                    age = event.get('age_seconds', 0)
                    age_str = f"{age:.0f}s ago" if age < 60 else f"{age/60:.0f}m ago"
                    text_context += f"- [{age_str}] {event['preview'][:200]}\n"
                text_context += "\n"

        # ═══ Channel 3: Structured Metadata ═══
        metadata_context = ""
        if include_metadata:
            diag = self.get_diagnostics()
            pe = self._pe_history[-1] if hasattr(self, '_pe_history') and self._pe_history else 0

            novelty = ("very high" if pe > 500 else "high" if pe > 300
                       else "moderate" if pe > 100 else "low")

            curriculum = self.get_curriculum_stats_dict()
            familiar = curriculum.get('most_familiar_domain', 'unknown')
            novel = curriculum.get('most_novel_domain', 'unknown')

            tau_mean = diag.get('tau_mean', 1.0)
            depth = ('deep' if tau_mean < 0.8 else
                     'moderate' if tau_mean < 1.0 else 'surface')

            metadata_context = (
                f"[System: Query novelty is {novelty}. "
                f"Familiar domains: {familiar}. Novel domains: {novel}. "
                f"Processing depth: {depth}. "
                f"Geometric complexity: {diag.get('metric_cv', 0):.1f}]\n\n"
            )

        # ═══ Build composed prompt ═══
        full_prompt = metadata_context + text_context + prompt

        # ═══ Get geometric state for prefix ═══
        if self._direct_prefix:
            # Direct: full ODE state as prefix tokens (no coupling)
            prefix_embeds = self._get_prefix_embeds()
        elif self._coupling is not None:
            # Legacy coupling: pooled state → W_inject → virtual tokens
            h_state = self._get_pooled_state()
            if h_state is None:
                return {'error': 'No ODE state — observe events first'}
            with self._gpu_lock, torch.no_grad():
                prefix_embeds = self._coupling.inject(h_state)
        else:
            prefix_embeds = None

        if prefix_embeds is None and include_geometric_prefix:
            return {'error': 'No ODE state — observe events first'}

        if self._qwen_is_vllm:
            # ═══ vLLM path ═══
            response = self._qwen_model.generate(
                prefix_embeds if include_geometric_prefix else None,
                full_prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                use_prefix=include_geometric_prefix,
            )
        else:
            # ═══ HF path: All three channels ═══
            with self._gpu_lock, torch.no_grad():
                tokenizer = self._qwen_tokenizer
                messages = [
                    {"role": "system",
                     "content": "You are a scientific assistant. Always respond in English."},
                    {"role": "user", "content": full_prompt},
                ]
                if hasattr(tokenizer, 'apply_chat_template'):
                    chat_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False)
                else:
                    chat_text = full_prompt

                tokens = tokenizer(chat_text, return_tensors='pt', truncation=True,
                                   max_length=512).to(self.device)
                input_embeds = self._qwen_model.get_input_embeddings()(tokens['input_ids'])

                if include_geometric_prefix:
                    combined = torch.cat([input_embeds, prefix_embeds], dim=1)
                else:
                    combined = input_embeds

                attn_mask = torch.ones(1, combined.shape[1],
                                       dtype=torch.long, device=self.device)

                outputs = self._qwen_model.generate(
                    inputs_embeds=combined,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    repetition_penalty=1.3,
                    pad_token_id=tokenizer.pad_token_id,
                )
                input_len = combined.shape[1]
                generated_ids = outputs[0][input_len:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        diag = self.get_diagnostics()
        return {
            'response': response,
            'prompt': prompt,
            'channels_used': {
                'text_context': include_text_context and bool(text_context),
                'geometric_prefix': include_geometric_prefix and prefix_embeds is not None,
                'direct_prefix': self._direct_prefix,
                'n_prefix_tokens': prefix_embeds.shape[1] if prefix_embeds is not None and prefix_embeds.dim() >= 2 else 0,
                'metadata': include_metadata,
            },
            'h_norm': self._h.norm().item() if self._h is not None else 0,
            'metric_cv': diag.get('metric_cv', 0),
            'tau_mean': diag.get('tau_mean', 0),
            'events_in_context': diag.get('events_in_context', 0),
        }

    def query_knowledge(self, prompt: str, max_new_tokens: int = 200,
                        temperature: float = 0.7,
                        use_prefix: bool = False,
                        h_override: Optional[torch.Tensor] = None) -> Dict:
        """Query Qwen3 conditioned on LiquidARC's geometric state.

        Projects h(t) → virtual prefix tokens → Qwen3 generates response.
        The response is shaped by LiquidARC's accumulated context.

        Args:
            prompt: Text to send to Qwen3
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            use_prefix: Whether to inject geometric prefix
            h_override: Optional pre-computed pooled state [d_arc] to use
                        instead of current h(t). Used by converse() to capture
                        state before inbound processing.

        Returns:
            dict with: response, diagnostics (CV, tau), coupling_info
        """
        if not self._qwen_available:
            return {'error': 'LLM not connected'}

        # Get prefix embeddings
        if self._direct_prefix:
            prefix_embeds = self._get_prefix_embeds() if use_prefix else None
        elif self._coupling is not None:
            h_state = h_override if h_override is not None else self._get_pooled_state()
            if h_state is None:
                return {'error': 'No ODE state — observe events first'}
            with self._gpu_lock, torch.no_grad():
                prefix_embeds = self._coupling.inject(h_state)
        else:
            prefix_embeds = None

        if self._qwen_is_vllm:
            response = self._qwen_model.generate(
                prefix_embeds, prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                use_prefix=use_prefix and prefix_embeds is not None,
            )
        else:
            # ═══ In-process HuggingFace path — fallback ═══
            with self._gpu_lock, torch.no_grad():
                tokenizer = self._qwen_tokenizer
                if hasattr(tokenizer, 'apply_chat_template'):
                    messages = [
                        {"role": "system", "content": "You are a scientific assistant. Always respond in English."},
                        {"role": "user", "content": prompt},
                    ]
                    chat_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False)
                else:
                    chat_text = prompt
                tokens = tokenizer(chat_text, return_tensors='pt', truncation=True,
                                   max_length=512).to(self.device)
                input_ids = tokens['input_ids']

                input_embeds = self._qwen_model.get_input_embeddings()(input_ids)
                combined = torch.cat([input_embeds, prefix_embeds], dim=1)
                attn_mask = torch.ones(1, combined.shape[1],
                                       dtype=torch.long, device=self.device)

                outputs = self._qwen_model.generate(
                    inputs_embeds=combined,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    repetition_penalty=1.3,
                    pad_token_id=tokenizer.pad_token_id,
                )
                input_len = combined.shape[1]
                generated_ids = outputs[0][input_len:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        diag = self.get_diagnostics()

        return {
            'response': response,
            'prompt': prompt,
            'n_prefix_tokens': prefix_embeds.shape[1] if prefix_embeds is not None and prefix_embeds.dim() >= 2 else 0,
            'h_norm': self._h.norm().item() if self._h is not None else 0,
            'metric_cv': diag.get('metric_cv', 0),
            'tau_mean': diag.get('tau_mean', 0),
            'events_in_context': diag.get('events_in_context', 0),
        }

    def _build_context_prompt(self, n_events: int = 5) -> str:
        """Assemble recent events into a text prompt fragment for QwenBridge.

        Returns a string with the last n_events content previews, one per line,
        oldest first. Used as prefix context when generating with attention bias.
        """
        N = min(len(self.events), self.max_events)
        recent = self.events[-N:]
        selected = recent[-n_events:] if len(recent) > n_events else recent

        type_labels = {
            0: 'User', 1: 'Assistant', 2: 'Tool', 3: 'Goal',
            4: 'Context', 5: 'Temporal', 6: 'Reflection', 7: 'Expression',
        }
        # Only include conversational events (user + assistant), not system/internal
        conversational_types = {0, 1}  # user_message, assistant_message
        lines = []
        for ev in selected:
            ev_type = ev.get('type', 0)
            if ev_type not in conversational_types:
                continue
            label = type_labels.get(ev_type, 'Event')
            preview = ev.get('content_preview', '')[:200]
            if preview:
                lines.append(f"{label}: {preview}")

        return '\n'.join(lines)

    def generate_with_bias(self, prompt: str, max_tokens: int = 128) -> Dict:
        """Generate text using ODE geometric bias on Qwen3 attention.

        Computes attention bias B_ij from current ODE state and injects
        into Qwen3's attention layers during generation.

        Args:
            prompt: Input prompt for Qwen3
            max_tokens: Maximum new tokens to generate

        Returns:
            dict with: response, cv, D_sq_4tau, tau_mean, criticality_flag,
                       bias_applied, error (on failure)
        """
        if self._qwen_bridge is None:
            return {'response': '', 'error': 'no qwen_bridge configured'}

        from .attention_bias import compute_attention_bias

        with self._gpu_lock:
            if self._h is None or self._h.shape[1] == 0:
                return {'response': '', 'error': 'no ODE state — call observe_event first'}

            N = self._h.shape[1]

            event_ids = [t.get('event_id', 0) for t in self._token_buffer] if self._token_buffer else None

            # State cosine bias with recency compensation.
            # The ODE state cosine captures routing alignment but has recency bias —
            # older events drift from the current trajectory.
            # Compensation: boost cosine for token pairs involving older tokens.
            # This counters the drift and gives distal causes more attention.

            with torch.no_grad():
                h_norm_vec = self._h / self._h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                cos_sim = torch.bmm(h_norm_vec, h_norm_vec.transpose(1, 2))[0]  # [N, N]

                # Recency compensation: tokens at index 0 are oldest, index N-1 newest.
                # Boost pairs involving old tokens by a factor of up to 1.5×.
                # This counteracts the ODE convergence that weakens old alignments.
                positions = torch.arange(N, device=self.device, dtype=torch.float)
                age = 1.0 - positions / max(N - 1, 1)  # 1.0 for oldest, 0.0 for newest
                # Pairwise age boost: max(age_i, age_j) — boost if EITHER token is old
                age_boost = age.unsqueeze(1).expand(N, N).max(
                    age.unsqueeze(0).expand(N, N))
                # Scale: 1.0 at newest pair, up to 2.0 at oldest pair.
                # 2× compensates for ~50% cosine decay over 5 conversation turns.
                recency_weight = 1.0 + 1.0 * age_boost

                bias = cos_sim * recency_weight

                # Add displacement if available
                disp_bias = getattr(self, '_displacement_bias', None)
                if disp_bias is not None and disp_bias.shape[0] == N:
                    bias = 0.7 * bias + 0.3 * disp_bias
                    bias_source = 'state_compensated+disp'
                else:
                    bias_source = 'state_compensated'

            # Get diagnostics
            _, diag = compute_attention_bias(self.dynamics, self._h, token_sources=event_ids)
            diag['bias_source'] = bias_source

            # Update adaptive D² EMA
            if diag.get('D_sq_median', 0) > 0:
                if self._D_sq_ema == 0:
                    self._D_sq_ema = diag['D_sq_median']
                else:
                    self._D_sq_ema = ((1 - self._D_sq_ema_alpha) * self._D_sq_ema
                                     + self._D_sq_ema_alpha * diag['D_sq_median'])

            # Compute entropy on the ACTUAL bias being injected (after per-row normalization)
            import math as _math
            sample_n = min(64, N)
            step = max(1, N // sample_n)
            sample_idx = list(range(0, N, step))[:sample_n]
            B_sample = bias[sample_idx, :]
            target_range = 2.0 * _math.log(max(N, 2))
            row_mean = B_sample.mean(dim=-1, keepdim=True)
            row_centered = B_sample - row_mean
            row_range = (row_centered.max(dim=-1, keepdim=True).values
                         - row_centered.min(dim=-1, keepdim=True).values).clamp(min=1e-8)
            B_scaled = row_centered / row_range * target_range
            K_sample = torch.softmax(B_scaled, dim=-1)
            actual_entropy = -(K_sample * (K_sample + 1e-10).log()).sum(dim=-1).mean().item()
            max_ent = _math.log(N) if N > 1 else 1.0
            actual_entropy_ratio = actual_entropy / max_ent
            diag['entropy_ratio'] = actual_entropy_ratio
            diag['attn_entropy'] = actual_entropy

            # B statistics on the actual bias
            if event_ids and len(event_ids) == N:
                import random as _rng
                bw, bx = [], []
                for _ in range(min(500, N*N)):
                    i, j = _rng.randint(0, N-1), _rng.randint(0, N-1)
                    if i == j: continue
                    v = bias[i, j].item()
                    if event_ids[i] == event_ids[j]:
                        bw.append(v)
                    else:
                        bx.append(v)
                diag['B_within_mean'] = sum(bw) / max(len(bw), 1)
                diag['B_across_mean'] = sum(bx) / max(len(bx), 1)
                diag['B_across_max'] = max(bx) if bx else 0.0

            n_unique_eids = len(set(event_ids)) if event_ids else 0
            n_cross = len(bx) if 'bx' in dir() else 0
            print(f"  [generate] bias [{N}x{N}] src={diag.get('bias_source', '?')} "
                  f"H={actual_entropy_ratio:.2f} "
                  f"Bw={diag.get('B_within_mean', 0):.3f} Bx={diag.get('B_across_mean', 0):.3f} "
                  f"eids={n_unique_eids} xpairs={n_cross} "
                  f"tau={diag['tau_mean']:.2f}")

            # Generate with iterative ODE feedback
            # Generate without post-hoc feedback to preserve buffer space.
            # The response enters the ODE via observe_event in converse(),
            # which adds it as a proper event with its own event_id.
            # Post-hoc feedback floods the buffer with generated tokens,
            # crowding out user content and reducing cross-event pairs.
            response = self._qwen_bridge.generate(
                prompt, bias=bias, max_new_tokens=max_tokens)

        # Pass through all diagnostics from bias computation
        result = {
            'response': response,
            'bias_applied': True,
        }
        result.update(diag)
        return result

    def converse(self, user_message: str, max_new_tokens: int = 300,
                 temperature: float = 0.7) -> Dict:
        """Complete Phase 6 conversation loop — symmetric geometric interface.

        Both directions go through Qwen3 × LiquidARC coupling:

        INBOUND (user → Mind):
          1. User text → Qwen3 processes with current state prefix
          2. W_read projects Qwen3's prefix hidden states → geometric signal
          3. Signal enters LiquidARC as sensory forcing (ODE integrates)

        OUTBOUND (Mind → user):
          4. Updated h(t) → W_inject → new prefix tokens
          5. Qwen3 generates response conditioned on geometric prefix
          6. W_read projects response hidden states → forcing back into LiquidARC

        No tokenizer touches LiquidARC. Text enters and exits as geometry.

        Args:
            user_message: User's text input
            max_new_tokens: Max response length
            temperature: Sampling temperature

        Returns:
            dict with: response, diagnostics, geometric signal norms
        """
        if not self._qwen_available and self._qwen_bridge is None and self._layer_wise_bridge is None:
            return {'error': 'LLM not connected'}

        print(f"\n  [converse] \"{user_message[:80]}\"")
        pre_diag = self.get_diagnostics()

        # ═══ INBOUND: User message → observe into ODE state ═══
        obs_result = self.observe_event(
            event_type='user_message',
            content=user_message,
            metadata={'source': 'conversation'},
        )

        # ═══ OUTBOUND: Layer-Wise ODE co-processing (preferred) ═══
        if self._layer_wise_bridge is not None:
            context_text = self._build_context_prompt(n_events=5)
            prompt = f"{context_text}\nUser: {user_message}\nAssistant:" if context_text else user_message

            result = self._layer_wise_bridge.generate(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature)

            response = result.get('response', '')
            if response and len(response) > 5:
                self.observe_event(
                    event_type='assistant_message',
                    content=response[:1000],
                    metadata={'source': 'layerwise_response'},
                )

            post_diag = self.get_diagnostics()
            summary = result.get('diagnostics', {})
            return {
                'response': response,
                'prediction_error': obs_result.get('prediction_error', 0),
                'cv_before': pre_diag.get('metric_cv', 0),
                'cv_after': post_diag.get('metric_cv', 0),
                'cv_early': summary.get('cv_early', 0),
                'cv_mid': summary.get('cv_mid', 0),
                'cv_late': summary.get('cv_late', 0),
                'tau_mean': summary.get('tau_mean', 0),
                'events_in_context': post_diag.get('events_in_context', 0),
                'h_norm': post_diag.get('h_norm', 0),
                'n_layers_processed': summary.get('n_layers_processed', 0),
                'B_range_early': summary.get('B_range_early', 0),
                'B_range_late': summary.get('B_range_late', 0),
                'architecture': 'layer_wise_ode',
            }

        # ═══ OUTBOUND: QwenBridge path (attention bias injection) ═══
        if self._qwen_bridge is not None:
            context_text = self._build_context_prompt(n_events=5)
            prompt = f"{context_text}\nUser: {user_message}\nAssistant:" if context_text else f"User: {user_message}\nAssistant:"
            bias_result = self.generate_with_bias(prompt, max_tokens=max_new_tokens)

            response = bias_result.get('response', '')
            if response and len(response) > 5:
                self.observe_event(
                    event_type='assistant_message',
                    content=response[:1000],
                    metadata={'source': 'qwen_bridge_response'},
                )

            post_diag = self.get_diagnostics()
            return {
                'response': response,
                'prediction_error': obs_result.get('prediction_error', 0),
                'cv_before': pre_diag.get('metric_cv', 0),
                'cv_after': post_diag.get('metric_cv', 0),
                'tau_mean': bias_result.get('tau_mean', 0),
                'tau_std': bias_result.get('tau_std', 0),
                'events_in_context': post_diag.get('events_in_context', 0),
                'token_buffer_size': post_diag.get('token_buffer_size', 0),
                'h_norm': self._h.norm().item() if self._h is not None else 0,
                'bias_applied': bias_result.get('bias_applied', False),
                'criticality_flag': bias_result.get('criticality_flag', False),
                'attn_entropy': bias_result.get('attn_entropy', 0),
                'entropy_ratio': bias_result.get('entropy_ratio', 0),
                'D_sq_across': bias_result.get('D_sq_across', 0),
                'B_within_mean': bias_result.get('B_within_mean', 0),
                'B_across_mean': bias_result.get('B_across_mean', 0),
                'B_across_max': bias_result.get('B_across_max', 0),
                'B_range': bias_result.get('B_range', 0),
                'bias_source': bias_result.get('bias_source', 'unknown'),
            }

        # ═══ OUTBOUND: Hybrid generation with all three channels ═══
        qwen_result = self.hybrid_generate(
            user_message,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            include_text_context=True,
            include_geometric_prefix=True,
            include_metadata=True,
            max_context_events=5,
        )

        if 'error' in qwen_result:
            return qwen_result

        response = qwen_result.get('response', '')

        # ═══ FEEDBACK: Response → observe into ODE state ═══
        if response and len(response) > 5:
            self.observe_event(
                event_type='assistant_message',
                content=response[:1000],
                metadata={'source': 'qwen3_response'},
            )

        # ═══ AUTO-FEEDBACK: Train geometric readout heads ═══
        # Conversation events get positive signal, curriculum in top-K gets negative
        try:
            N = min(len(self.events), self.max_events)
            scored = self._compute_combined_relevance(user_message)
            for item in scored[:5]:
                idx = item.get('index', -1)
                if 0 <= idx < N:
                    etype = item.get('type', '')
                    if etype in ('user_msg', 'assistant_msg'):
                        self.provide_feedback(idx, 'correct', signal=0.3)
                    elif etype == 'context':
                        self.provide_feedback(idx, 'irrelevant', signal=0.2)
        except Exception:
            pass  # auto-feedback is best-effort

        post_diag = self.get_diagnostics()

        return {
            'response': response,
            'prediction_error': obs_result.get('prediction_error', 0),
            'cv_before': pre_diag.get('metric_cv', 0),
            'cv_after': post_diag.get('metric_cv', 0),
            'tau_mean': post_diag.get('tau_mean', 0),
            'events_in_context': post_diag.get('events_in_context', 0),
            'h_norm': qwen_result.get('h_norm', 0),
        }

    def express_through_qwen(self, focus_query: Optional[str] = None) -> Dict:
        """Express the Mind's state through Qwen3's language.

        Uses hybrid generation with all three channels — text context provides
        factual grounding, geometric prefix provides implicit bias, metadata
        provides processing signals. More context events (8) for richer self-reflection.

        Args:
            focus_query: Optional focus to direct the expression

        Returns:
            dict with: expression text, diagnostics, channels used
        """
        if not self._qwen_available and self._qwen_bridge is None and self._layer_wise_bridge is None:
            return {'error': 'LLM not connected'}

        prompt = focus_query or "What themes and patterns dominate your current processing?"
        print(f"\n  [express] \"{prompt[:80]}\"")

        # ═══ Layer-Wise ODE path (preferred) ═══
        if self._layer_wise_bridge is not None:
            context_text = self._build_context_prompt(n_events=8)
            full_prompt = (f"{context_text}\n{prompt}" if context_text else prompt)
            result = self._layer_wise_bridge.generate(
                full_prompt, max_new_tokens=300, temperature=0.7)

            expression = result.get('response', '')
            if expression and len(expression) > 10:
                self.observe_event(
                    event_type='expression',
                    content=expression[:500],
                    metadata={'source': 'layerwise_expression', 'focus': focus_query},
                )

            summary = result.get('diagnostics', {})
            return {
                'response': expression,
                'source': 'layer_wise_ode',
                'cv_mean': summary.get('cv_mean', 0),
                'tau_mean': summary.get('tau_mean', 0),
                'n_layers_processed': summary.get('n_layers_processed', 0),
            }

        # ═══ QwenBridge path (attention bias injection) ═══
        if self._qwen_bridge is not None:
            context_text = self._build_context_prompt(n_events=8)
            full_prompt = (f"{context_text}\n{prompt}" if context_text else prompt)
            bias_result = self.generate_with_bias(full_prompt, max_tokens=300)

            if 'error' in bias_result:
                return bias_result

            expression = bias_result.get('response', '')
            if expression and len(expression) > 10:
                self.observe_event(
                    event_type='expression',
                    content=expression[:500],
                    metadata={'source': 'qwen_bridge_expression', 'focus': focus_query},
                )

            bias_result['source'] = 'qwen_bridge'
            return bias_result

        result = self.hybrid_generate(
            prompt,
            max_new_tokens=300,
            temperature=0.7,
            include_text_context=True,
            include_geometric_prefix=True,
            include_metadata=True,
            max_context_events=8,
        )

        if 'error' in result:
            return result

        # Feed expression back as an event (self-reference loop)
        expression = result.get('response', '')
        if expression and len(expression) > 10:
            self.observe_event(
                event_type='expression',
                content=expression[:500],
                metadata={'source': 'qwen3_expression', 'focus': focus_query},
            )

        result['source'] = 'qwen3_coupled'
        return result

    def probe_encoding(self, text: str) -> Dict:
        """Project Phase 1 ODE output back to token space.

        Returns the Mind's own linguistic transformation of input text —
        what each token moved TOWARD through 16 ODE integration steps.
        No LLM involved — pure ODE dynamics projected through the embedding table.

        Path C: uses GPT-2 TextEmbedding for both encoding and projection.
        """
        with self._gpu_lock:
            if self.use_trained_text_embed and self._text_embed is not None:
                # Path C: GPT-2 tokenizer + trained TextEmbedding
                tokens = self._text_tokenizer.encode(
                    text, add_special_tokens=False, truncation=True, max_length=2048)
                if not tokens:
                    return {'input_text': text, 'n_tokens': 0, 'positions': [],
                            'mind_sentence': '', 'state_vocabulary': [],
                            'transform_ratio': 0, 'mean_displacement': 0}
                token_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
                token_h = self._text_embed(token_ids)
                T = len(tokens)
                embed_weight = self._text_embed.token_embed.weight
                tokenizer = self._text_tokenizer
            else:
                # Original: MindTokenizer
                token_ids = self.embedding.tokenizer.tokenize(text).unsqueeze(0).to(self.device)
                token_h, mask = self.embedding.tokenizer(token_ids)
                T = mask[0].sum().item()
                if T == 0:
                    return {'input_text': text, 'n_tokens': 0, 'positions': [],
                            'mind_sentence': '', 'state_vocabulary': [],
                            'transform_ratio': 0, 'mean_displacement': 0}
                embed_weight = self.embedding.tokenizer.token_embed.weight
                tokenizer = self.embedding.tokenizer._tokenizer

            context = self.context_pool(token_h)
            self.dynamics.set_context(context, mask=None)
            self.dynamics.set_n_steps(self.internal_steps)

            # Standalone Euler loop (no Phase 2 infrastructure)
            with torch.no_grad():
                dt = self.T / self.internal_steps
                h_processed = token_h
                t = 0.0
                for step_i in range(self.internal_steps):
                    if hasattr(self.dynamics, 'set_step_index'):
                        self.dynamics.set_step_index(step_i, self.internal_steps)
                    dh = self.dynamics(t, h_processed)
                    h_processed = h_processed + dt * dh
                    t += dt

            logits = h_processed[0, :T, :] @ embed_weight.T
            topk = logits.topk(5, dim=-1)
            displacement = (h_processed[0, :T, :] - token_h[0, :T, :]).norm(dim=-1)

            # NTP loss for monitoring (how interpretable is the ODE output?)
            ntp_loss_val = None
            if T > 1:
                ntp_logits = logits[:-1]  # [T-1, vocab]
                ntp_targets = token_ids[0, 1:T]  # [T-1]
                ntp_loss_val = F.cross_entropy(ntp_logits, ntp_targets).item()

            positions = []
            state_vocabulary = []

            for pos in range(T):
                input_id = token_ids[0, pos].item()
                input_tok = tokenizer.decode([input_id])

                output_ids = topk.indices[pos].tolist()
                output_toks = [tokenizer.decode([tid]) for tid in output_ids]
                output_scores = topk.values[pos].tolist()

                transformed = (output_ids[0] != input_id)

                positions.append({
                    'pos': pos,
                    'input': input_tok,
                    'output_top5': output_toks,
                    'scores': [round(s, 2) for s in output_scores],
                    'displacement': round(displacement[pos].item(), 2),
                    'transformed': transformed,
                })

                for tok in output_toks[:3]:
                    if tok.strip() and tok not in state_vocabulary:
                        state_vocabulary.append(tok)

            mind_sentence_tokens = [p['output_top5'][0] for p in positions]
            n_transformed = sum(1 for p in positions if p['transformed'])

            transformations = []
            mean_disp = displacement[:T].mean().item()
            for p in positions:
                if p['transformed'] or p['displacement'] > mean_disp:
                    transformations.append(
                        f"{p['input']} -> {p['output_top5'][0]} "
                        f"(also: {', '.join(p['output_top5'][1:3])}) "
                        f"[d={p['displacement']}]"
                    )

        return {
            'input_text': text,
            'n_tokens': T,
            'positions': positions,
            'mind_sentence': ' '.join(mind_sentence_tokens),
            'state_vocabulary': state_vocabulary[:20],
            'transform_ratio': round(n_transformed / T if T > 0 else 0, 2),
            'mean_displacement': round(mean_disp, 2),
            'transformations': transformations,
            'ntp_loss': ntp_loss_val,
        }

    def get_geometric_profile(self) -> Dict:
        """Extract rich geometric features for the Voice.

        Returns per-event metrics, inter-event similarity, clusters,
        ODE trajectory, and prediction error landscape.
        """
        if self._h is None or len(self.events) == 0:
            return {'status': 'no_state'}

        with self._gpu_lock:
            N = min(len(self.events), self.max_events)
            h = self._h[:, :N, :]  # [1, N, d]

            # 1. Per-event metric and tau
            g = self.dynamics.compute_metric_diag(h)
            tau = self.dynamics.compute_tau(h)
            g_per_event = g[0].mean(dim=-1).cpu().tolist()
            tau_per_event = tau[0, :, 0].cpu().tolist()

            # 2. Inter-event geometry (cosine similarity)
            h_normed = torch.nn.functional.normalize(h[0], dim=-1)
            sim_matrix = (h_normed @ h_normed.T).cpu()
            sim_matrix.fill_diagonal_(-1)
            nearest_idx = sim_matrix.argmax(dim=-1).tolist()
            nearest_sim = sim_matrix.max(dim=-1).values.tolist()
            farthest_idx = sim_matrix.argmin(dim=-1).tolist()
            farthest_sim = sim_matrix.min(dim=-1).values.tolist()

            # 3. Geometric clusters (threshold-based)
            threshold = 0.7
            clusters = []
            assigned = set()
            for i in range(N):
                if i in assigned:
                    continue
                cluster = [i]
                assigned.add(i)
                for j in range(i + 1, N):
                    if j not in assigned and sim_matrix[i, j] > threshold:
                        cluster.append(j)
                        assigned.add(j)
                if len(cluster) > 1:
                    clusters.append(cluster)

            # 4. ODE trajectory snapshot (first 8 steps)
            step_profile = []
            if N <= 32:
                h_trace = h.clone()
                dt = self.T / self.internal_steps
                context_mask = torch.ones(1, N, dtype=torch.bool, device=self.device)
                context = self.context_pool(h_trace, context_mask)
                self.dynamics.set_context(context, mask=None)

                for step in range(min(self.internal_steps, 8)):
                    if hasattr(self.dynamics, 'set_step_index'):
                        self.dynamics.set_step_index(step, self.internal_steps)
                    g_step = self.dynamics.compute_metric_diag(h_trace)
                    cv_step = (g_step.std() / (g_step.mean() + 1e-8)).item()
                    dy = self.dynamics(step * dt, h_trace)
                    step_profile.append({
                        'step': step,
                        'cv': round(cv_step, 2),
                        'h_norm': round(h_trace.norm().item(), 1),
                        'dynamics_magnitude': round(dy.norm(dim=-1).mean().item(), 3),
                    })
                    h_trace = h_trace + dt * dy

            # 5. Prediction error landscape
            obs_embed = self._get_embedded_events()
            pe_per_event = (obs_embed[0] - h[0]).norm(dim=-1).cpu().tolist()

            # 6. Attention/relevance from readout
            event_types_t = torch.tensor(
                [e['type'] for e in self.events[-N:]], device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                readout = self.readout(h, event_types_t)
            relevance = readout['relevance_scores'][0].cpu().tolist()

        # Build per-event profiles
        type_names = ['user_msg', 'assistant_msg', 'tool_result',
                      'goal', 'context', 'temporal', 'reflection', 'expression', 'voice_response']
        recent = self.events[-N:]
        event_profiles = []
        for i in range(N):
            etype = recent[i]['type']
            event_profiles.append({
                'index': i,
                'type': type_names[etype] if etype < len(type_names) else 'unknown',
                'preview': recent[i]['content_preview'],
                'metric_intensity': round(g_per_event[i], 3),
                'tau': round(tau_per_event[i], 3),
                'prediction_error': round(pe_per_event[i], 1),
                'relevance': round(relevance[i], 3),
                'nearest_event': nearest_idx[i],
                'nearest_similarity': round(nearest_sim[i], 3),
                'farthest_event': farthest_idx[i],
                'farthest_similarity': round(farthest_sim[i], 3),
            })

        return {
            'status': 'active',
            'n_events': N,
            'events': event_profiles,
            'clusters': clusters,
            'step_profile': step_profile,
            'global': {
                'h_norm': round(h.norm().item(), 1),
                'cv': round((g.std() / (g.mean() + 1e-8)).item(), 2),
                'tau_mean': round(tau.mean().item(), 3),
                'tau_std': round(tau.std().item(), 3),
            },
        }

    def provide_feedback(self, event_index: int, feedback_type: str,
                         signal: float = 1.0) -> Dict:
        """Online learning from human feedback.

        Phase 1: Gradients flow through readout only (h detached).
        This updates relevance scoring without touching dynamics.
        """
        if self.optimizer is None:
            return {'status': 'online_learning_disabled'}
        if event_index >= len(self.events):
            return {'status': 'invalid_index'}

        with self._gpu_lock:
            N = min(len(self.events), self.max_events)
            h = self._h[:, :N, :].detach().requires_grad_(True)
            event_types = torch.tensor(
                [e['type'] for e in self.events[-N:]], device=self.device
            ).unsqueeze(0)

            readout_result = self.readout(h, event_types)
            relevance = readout_result['relevance_scores'][0, event_index]

            if feedback_type == 'correct':
                loss = -torch.log(relevance + 1e-8) * signal
            elif feedback_type == 'wrong':
                loss = -torch.log(1 - relevance + 1e-8) * signal
            elif feedback_type == 'irrelevant':
                loss = relevance * signal
            else:
                return {'status': 'unknown_feedback_type'}

            if torch.isnan(loss):
                return {'status': 'nan_loss', 'relevance': relevance.item()}

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.embedding.parameters()) +
                list(self.readout.parameters()) +
                list(self.forcing.parameters()),
                max_norm=0.1,
            )
            self.optimizer.step()

        return {
            'status': 'updated',
            'loss': loss.item(),
            'relevance_at_feedback': relevance.item(),
        }

    def signal_goal(self, goal_text: str, priority: float = 1.0) -> Dict:
        """Inject a goal as a persistent entity token."""
        return self.observe_event('goal', goal_text, metadata={'confidence': priority})

    def reset(self):
        """Reset state for new conversation/topic."""
        with self._gpu_lock:
            self.events = []
            self.event_count = 0
            self._h = torch.zeros_like(self._h)
            self._last_event_time = time.time()

    # ──────────────────── AUTONOMOUS PROCESSING ────────────────────

    # ──── Curriculum system ────

    DOMAIN_NAMES = {
        'topology': 'algebraic topology',
        'mathematics': 'pure mathematics',
        'physics': 'theoretical physics',
        'biology': 'developmental biology',
        'ecology': 'ecology',
        'music_theory': 'music theory',
        'philosophy': 'philosophy',
        'poetry': 'poetry and poetics',
    }

    COMPLEXITY_TIERS = [
        "basic concept explained simply",
        "intermediate concept with connections to other ideas",
        "advanced concept requiring prior knowledge",
        "cutting-edge research question or open problem",
        "cross-domain connection between this field and another",
    ]

    CURATED_TOPICS = {
        'topology': [
            "Explain the Euler characteristic and how it classifies surfaces",
            "What is a fiber bundle and why does it matter in physics?",
            "How does persistent homology extract shape from data?",
            "What makes the Poincare conjecture so important?",
            "Explain covering spaces and their relationship to fundamental groups",
            "What is a CW complex and how does it simplify topology?",
            "How does Morse theory connect topology to calculus?",
            "What are knot invariants and why are knots hard to classify?",
            "Explain the hairy ball theorem and its consequences",
            "What is homological algebra and how does it generalize topology?",
        ],
        'mathematics': [
            "Explain Galois theory and why quintics have no general formula",
            "What is a sheaf and why does algebraic geometry need them?",
            "How does the Langlands program connect number theory to geometry?",
            "What is a modular form and why did they help prove Fermat's Last Theorem?",
            "Explain the Yoneda lemma and why category theorists love it",
            "What is p-adic analysis and how does it differ from real analysis?",
            "How does spectral theory connect linear algebra to quantum mechanics?",
            "What is an ergodic system and why does ergodicity matter?",
            "Explain the Riemann hypothesis in terms of prime distribution",
            "What is a Lie group and how does it encode symmetry?",
        ],
        'physics': [
            "Explain spontaneous symmetry breaking in the Higgs mechanism",
            "What is the renormalization group and why does it matter?",
            "How does the Aharonov-Bohm effect challenge classical intuition?",
            "What is topological order in condensed matter?",
            "Explain the connection between entropy and information",
            "What are anyons and why do they matter for quantum computing?",
            "How does AdS/CFT connect gravity to quantum field theory?",
            "What is Berry phase and how does geometry enter quantum mechanics?",
            "Explain decoherence and why Schrodinger's cat doesn't work at large scales",
            "What is a quasicrystal and how does it challenge crystallography?",
        ],
        'biology': [
            "How do Hox genes control body plan organization?",
            "What is epigenetic inheritance and how does it work without DNA changes?",
            "Explain the Waddington landscape model of cell differentiation",
            "How do reaction-diffusion systems create biological patterns?",
            "What is horizontal gene transfer and why does it complicate phylogenetics?",
            "How do prions propagate without nucleic acid?",
            "What is the RNA world hypothesis?",
            "Explain quorum sensing in bacterial communities",
            "How does the immune system distinguish self from non-self?",
            "What are transposable elements and how do they shape genomes?",
        ],
        'ecology': [
            "What is the intermediate disturbance hypothesis?",
            "How do keystone species maintain ecosystem diversity?",
            "Explain metacommunity theory and landscape-scale ecology",
            "What is the paradox of the plankton?",
            "How do mycorrhizal networks create forest communication?",
            "What is ecological niche construction?",
            "Explain island biogeography and species-area relationships",
            "How do tipping points work in ecosystem collapse?",
            "What is the competitive exclusion principle and its exceptions?",
            "How does nutrient spiraling work in stream ecosystems?",
        ],
        'music_theory': [
            "Explain the circle of fifths and why it organizes tonality",
            "What is serialism and how did Schoenberg use tone rows?",
            "How does counterpoint create independent melodic lines?",
            "What is a Schenkerian analysis and what does it reveal?",
            "Explain microtonal music and alternative tuning systems",
            "What are modes and how do they differ from major/minor scales?",
            "How does polyrhythm create complex temporal patterns?",
            "What is spectral music and how does it use overtones?",
            "Explain the difference between diatonic and chromatic harmony",
            "How does gamelan music organize pitch and rhythm differently?",
        ],
        'philosophy': [
            "What is the hard problem of consciousness?",
            "Explain Wittgenstein's private language argument",
            "What is emergence and how does it relate to reductionism?",
            "How does Heidegger's concept of Dasein differ from Cartesian subjectivity?",
            "What is the Chinese Room argument and what does it prove?",
            "Explain pragmatism and how it defines truth differently",
            "What is the frame problem in AI and philosophy of mind?",
            "How does Merleau-Ponty's phenomenology of perception work?",
            "What is the is-ought problem and why can't we derive values from facts?",
            "Explain panpsychism and why some philosophers take it seriously",
        ],
        'poetry': [
            "How does enjambment create tension between line and sentence?",
            "What is the objective correlative and how did Eliot use it?",
            "Explain sprung rhythm and how Hopkins broke metrical convention",
            "How does the villanelle's repetitive structure create meaning?",
            "What is language poetry and how does it challenge representation?",
            "Explain the ghazal form and its tradition of autonomous couplets",
            "How does concrete poetry merge visual and linguistic meaning?",
            "What is negative capability and how does Keats relate to poetry?",
            "Explain the difference between lyric, narrative, and dramatic poetry",
            "How does haiku's constraint force precision of image?",
        ],
    }

    def start_autonomous(self, voice=None):
        """Background thread: ODE consolidation + geometric reflection through Qwen3.

        All reflection and curriculum flows through the Qwen3 geometric coupling:
        h(t) → W_inject → prefix → Qwen3 → W_read → geometric signal → ODE forcing.

        No Voice/Nemotron dependency. Language enters and exits as geometry.

        Three reflection modes:
        A. Triggered: geometric conditions warrant reflection (CV shift, h_norm drift)
        B. Maintenance: periodic pathway exercise (~100 ODE cycles)
        C. External: triggered by incoming conversation events
        """
        import re as _re
        import collections as _collections

        self._running = True
        self.voice = voice
        # Disable h_norm_drift trigger entirely — h_norm grows naturally with events
        # and is not indicative of problems. CV shift is the meaningful geometric signal.
        self.trigger = ReflectionTrigger(cv_shift_threshold=3.0, h_norm_ceiling=1e9)
        self._reflection_limiter = ReflectionLimiter(max_ratio=0.33)
        self._curiosity = CuriosityController()
        self._curriculum_domains = list(self.DOMAIN_NAMES.keys())
        self._curriculum_idx = 0
        self._curriculum_count = 0

        # Topic tracking — anti-repetition
        if not hasattr(self, '_curriculum_history'):
            self._curriculum_history = {d: [] for d in self._curriculum_domains}
        if not hasattr(self, '_curriculum_tier'):
            self._curriculum_tier = {d: 0 for d in self._curriculum_domains}

        # PE-based reflection trigger
        self._pe_history = _collections.deque(maxlen=50)

        # Feed/digest scheduler
        if not hasattr(self, '_scheduler_phase'):
            self._scheduler_phase = 'feed'
            self._scheduler_stimuli_this_phase = 0
            self._scheduler_cycles_this_phase = 0
        self._scheduler_feed_count = 20
        self._scheduler_digest_cycles = 200

        # Curriculum instrumentation
        if not hasattr(self, '_curriculum_stats') or self._curriculum_stats is None:
            self._curriculum_stats = {
                'domain_counts': {d: 0 for d in self._curriculum_domains},
                'domain_pe_history': {d: [] for d in self._curriculum_domains},
                'domain_cv_history': {d: [] for d in self._curriculum_domains},
                'domain_tau_history': {d: [] for d in self._curriculum_domains},
                'domain_avg_pe': {},
                'domain_avg_cv': {},
                'domain_avg_tau': {},
                'domain_effectiveness': {},
                'growth_zone_domains': [],
            }

        def _extract_topic(text):
            """Extract topic from curriculum text for anti-repetition."""
            bold = _re.search(r'\*\*(.+?)\*\*', text)
            if bold:
                return bold.group(1).lower().strip()[:50]
            first_line = text.strip().split('\n')[0]
            return first_line[:50].lower().strip()

        def _generate_curriculum_prompt(domain):
            """Build diverse prompt with anti-repetition + tiered complexity."""
            already = self._curriculum_history.get(domain, [])
            already_str = ', '.join(already[-20:]) if already else 'none'
            tier_idx = self._curriculum_tier.get(domain, 0) % len(self.COMPLEXITY_TIERS)
            tier = self.COMPLEXITY_TIERS[tier_idx]
            self._curriculum_tier[domain] = self._curriculum_tier.get(domain, 0) + 1
            domain_name = self.DOMAIN_NAMES.get(domain, domain)

            return (
                f"Explain a concept from {domain_name}. "
                f"Difficulty level: {tier}. "
                f"AVOID these already-covered topics: {already_str}. "
                f"Choose something genuinely different. "
                f"Respond only in English. Keep it under 200 words."
            )

        # Load external curriculum bank if available (from Nemotron batch generation)
        self._curriculum_bank = {}
        import os as _os
        for bank_path in [
            '/workspace/liquid-arc/curriculum_bank.json',
            _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), 'curriculum_bank.json'),
        ]:
            if _os.path.exists(bank_path):
                try:
                    import json as _json
                    with open(bank_path) as f:
                        self._curriculum_bank = _json.load(f)
                    n_total = sum(len(v) for v in self._curriculum_bank.values())
                    print(f"  Curriculum bank loaded: {n_total} topics from {bank_path}")
                except Exception as e:
                    print(f"  Curriculum bank load failed: {e}")
                break

        def _get_curriculum_prompt(domain):
            """Get stimulus — prefer bank, then curated fallback, then generated."""
            history = self._curriculum_history.get(domain, [])

            # Priority 1: External bank (Nemotron-generated, diverse)
            bank_topics = self._curriculum_bank.get(domain, [])
            if bank_topics:
                idx = len(history) % len(bank_topics)
                return bank_topics[idx]  # Full text, not a prompt — use directly

            # Priority 2: Curated fallback if generating repeats
            recent = history[-10:] if len(history) >= 10 else history
            if len(recent) >= 5 and len(set(recent)) < len(recent) // 2:
                curated = self.CURATED_TOPICS.get(domain, [])
                if curated:
                    idx = len(history) % len(curated)
                    return curated[idx]

            # Priority 3: Generated prompt
            return _generate_curriculum_prompt(domain)

        self._pe_trigger_cooldown = 0

        def _check_pe_trigger(current_pe):
            """Trigger reflection when PE is unusual relative to running average."""
            if self._pe_trigger_cooldown > 0:
                self._pe_trigger_cooldown -= 1
                return False
            if len(self._pe_history) < 10:
                self._pe_history.append(current_pe)
                return False
            pe_mean = sum(self._pe_history) / len(self._pe_history)
            pe_std = (sum((p - pe_mean)**2 for p in self._pe_history) / len(self._pe_history)) ** 0.5
            self._pe_history.append(current_pe)
            if abs(current_pe - pe_mean) > 1.5 * max(pe_std, 10.0):
                self._pe_trigger_cooldown = 20  # 20 cycles cooldown
                return True
            return False

        def _loop():
            while self._running:
                if self._h is not None and self._h.shape[1] > 0:

                    # ═══ PHASE 1: Pure ODE processing + write mechanisms ═══
                    N = self._h.shape[1]
                    with self._gpu_lock:
                        try:
                            self._compute_tau_bias()

                            h_slice = self._h[:, :N, :]
                            context_mask = torch.ones(1, N, dtype=torch.bool,
                                                      device=self.device)
                            context = self.context_pool(h_slice, context_mask)
                            self.dynamics.set_context(context, mask=None)
                            self.dynamics.set_n_steps(16)

                            with torch.no_grad():
                                h_auto = self._run_ode_segment(
                                    h_slice, 16, forcing=None)

                            self._h = h_auto.detach()
                            # Sync token buffer with ODE-updated positions
                            if self._token_buffer and self._h.shape[1] == len(self._token_buffer):
                                for i in range(len(self._token_buffer)):
                                    self._token_buffer[i]['delta_h'] = self._h[0, i].detach().float()
                            self._update_salience(self._h)
                        except Exception as e:
                            print(f"Autonomous ODE error: {e}")

                    self._trigger_stats['total_ode_cycles'] += 1
                    self._cycles_since_reflection += 1
                    self._cycles_since_stimulus += 1

                    # Curiosity logging suppressed — only log on phase transitions (handled in curiosity controller)

                    # ═══ PHASE 2: Decide whether to reflect/stimulate ═══
                    # Suppress during active conversation — curriculum and reflections
                    # dilute the ODE state and context with unrelated content
                    in_conversation = (time.time() - self._last_user_event_time
                                       < self._conversation_quiet_period)

                    # Shrink buffer back to idle size after conversation ends
                    if not in_conversation and self._max_tokens > self._max_tokens_idle:
                        self._max_tokens = self._max_tokens_idle

                    should_reflect = False
                    should_stimulate = False
                    curiosity_domain = None
                    curiosity_reason = None
                    reflection_mode = None
                    trigger_reason = None

                    # Ask curiosity controller if we should inject
                    last_pe = self._pe_history[-1] if self._pe_history else 0
                    c_stats = getattr(self, '_curriculum_stats', None) or {}
                    c_stats['domains'] = self._curriculum_domains
                    recent_refs = self.get_recent_reflections(n=5)
                    should_stimulate_c, curiosity_domain, curiosity_reason = \
                        self._curiosity.should_inject(last_pe, c_stats, recent_refs)

                    # Consolidation mode: curiosity is digesting OR manual pause
                    is_consolidating = (self._curiosity.current_phase in ('calibrating', 'digesting')
                                        or curiosity_reason in ('digesting', 'forced_digest', 'satiated')
                                        or self._stimulus_interval >= 999)

                    # Minimum cooldown + ratio limiter (max 1 reflection per 2 curriculum)
                    # During consolidation, only maintenance reflections (diagnostic-only)
                    # During active conversation, suppress ALL curriculum and non-essential reflections
                    reflection_allowed = (self._cycles_since_reflection >= 10
                                          and not is_consolidating
                                          and not in_conversation
                                          and self._reflection_limiter.can_reflect())

                    if self._qwen_available or self._qwen_bridge is not None:
                        # Check A: External event pending (only if cooldown passed)
                        if reflection_allowed and self._external_event_pending:
                            should_reflect = True
                            reflection_mode = 'external'
                            trigger_reason = 'External event — geometric integration'
                            self._external_event_pending = False

                        # Check B: Triggered conditions (every 10 cycles)
                        if (not should_reflect and reflection_allowed
                                and self._cycles_since_reflection % 10 == 0):
                            diag = self.get_diagnostics()
                            trigger_reason = self.trigger.check(
                                diag, self._last_reflection_pe)
                            if trigger_reason:
                                should_reflect = True
                                reflection_mode = 'triggered'

                        # Check B2: PE-based trigger (with cooldown)
                        if (not should_reflect and reflection_allowed
                                and len(self._pe_history) >= 10):
                            last_pe = self._pe_history[-1] if self._pe_history else 0
                            if _check_pe_trigger(last_pe):
                                should_reflect = True
                                reflection_mode = 'triggered'
                                trigger_reason = f'PE anomaly: {last_pe:.0f}'

                        # Check C: Curiosity-driven curriculum stimulus
                        # SUPPRESSED during active conversation to protect ODE state alignment
                        if not should_reflect and should_stimulate_c and not in_conversation:
                            should_stimulate = True

                        # Check D: Maintenance (only during idle)
                        if not should_reflect and not should_stimulate and not in_conversation:
                            if self._cycles_since_reflection >= self.maintenance_interval:
                                should_reflect = True
                                reflection_mode = 'maintenance'
                                trigger_reason = f'Maintenance ({self._cycles_since_reflection} cycles)'

                    # ═══ PHASE 3a: Hybrid reflection through Qwen3 ═══
                    if should_reflect:
                        try:
                            if reflection_mode == 'maintenance':
                                prompt = ("Briefly reflect on the current state of your processing. "
                                          "Respond in English only. One paragraph.")
                            else:
                                prompt = ("What patterns, connections, or shifts do you notice "
                                          "in your current state? Respond in English only. "
                                          "Keep your response concise — one paragraph.")

                            # Generate reflection via QwenBridge (bias-guided) or legacy hybrid
                            # Use generate() directly — NOT generate_with_bias which does
                            # post-hoc feedback (double Qwen3 forward → OOM on autonomous loop).
                            # The reflection text feeds back via observe_event instead.
                            if self._qwen_bridge is not None:
                                context_text = self._build_context_prompt(n_events=3)
                                full_prompt = f"{context_text}\n{prompt}" if context_text else prompt
                                from .attention_bias import compute_attention_bias
                                with self._gpu_lock:
                                    if self._h is not None and self._h.shape[1] > 0:
                                        sources = [t.get('event_id', 0) for t in self._token_buffer] if self._token_buffer else None
                                        bias, _ = compute_attention_bias(self.dynamics, self._h, token_sources=sources)
                                    else:
                                        bias = None
                                reflection_text = self._qwen_bridge.generate(
                                    full_prompt, bias=bias, max_new_tokens=100)
                                result = {'response': reflection_text}
                            else:
                                result = self.hybrid_generate(
                                    prompt,
                                    max_new_tokens=200,
                                    temperature=0.7,
                                    include_text_context=True,
                                    include_geometric_prefix=False,
                                    include_metadata=False,
                                    max_context_events=5,
                                )
                            reflection_text = result.get('response', '')

                            if reflection_text and len(reflection_text) > 5:
                                # During consolidation: diagnostic-only (EEG mode)
                                # Read the reflection like an EEG — observe without disturbing
                                if is_consolidating:
                                    # DO NOT feed back as event — ODE processes in silence
                                    self._last_reflection_text = reflection_text
                                    self._reflection_count += 1
                                    self._cycles_since_reflection = 0
                                    diag = self.get_diagnostics()
                                    print(f"  [diagnostic] #{self._reflection_count}: "
                                          f"CV={diag.get('metric_cv', 0):.2f} "
                                          f"tau={diag.get('tau_mean', 0):.2f} "
                                          f"h={diag.get('h_norm', 0):.0f} "
                                          f"\"{reflection_text[:60]}\"")
                                else:
                                    # Active mode: feed reflection back as event
                                    self.observe_event(
                                        event_type='expression',
                                        content=reflection_text[:500],
                                        metadata={'source': 'qwen3_expression',
                                                  'focus': prompt},
                                    )

                                    self._last_reflection_pe = 0
                                    self._last_reflection_text = reflection_text
                                    self._reflection_count += 1
                                    self._reflection_limiter.on_reflection()
                                    self._cycles_since_reflection = 0

                                    # Hebbian nudge on non-maintenance reflections
                                    if reflection_mode != 'maintenance':
                                        try:
                                            ctx = self.get_context()
                                            if ctx.get('focus_indices'):
                                                with self._gpu_lock:
                                                    self._hebbian_nudge(
                                                        self._h, ctx['focus_indices'])
                                        except Exception:
                                            pass

                                    self._trigger_stats[f'{reflection_mode}_reflections'] = \
                                        self._trigger_stats.get(f'{reflection_mode}_reflections', 0) + 1

                                    print(f"  [{reflection_mode}] #{self._reflection_count}: "
                                          f"\"{reflection_text[:60]}\" "
                                          f"(reason: {trigger_reason[:40] if trigger_reason else 'n/a'})")

                        except Exception as e:
                            print(f"Reflection error: {e}")
                            self._cycles_since_reflection = 0

                    # ═══ PHASE 3b: Curiosity-driven curriculum ═══
                    elif should_stimulate:
                        try:
                            # Domain selected by curiosity controller (novelty-weighted)
                            domain = curiosity_domain or self._curriculum_domains[
                                self._curriculum_idx % len(self._curriculum_domains)]
                            self._curriculum_idx += 1
                            self._curriculum_count += 1

                            # Get stimulus — bank text or generated prompt
                            prompt_or_text = _get_curriculum_prompt(domain)
                            bank_topics = self._curriculum_bank.get(domain, [])

                            if bank_topics:
                                # Bank provides full text — use directly, no Qwen3 call
                                stimulus_text = prompt_or_text
                            else:
                                # Generated prompt — query Qwen3
                                result = self.query_knowledge(
                                    prompt_or_text, max_new_tokens=150, temperature=0.7)
                                stimulus_text = result.get('response', '')

                            if stimulus_text and len(stimulus_text) > 10:
                                # Use observe_event for curriculum — stimulus already
                                # came FROM Qwen3 conditioned on state, no need to
                                # encode back through Qwen3 again (saves ~3s per event)
                                obs_result = self.observe_event(
                                    event_type='context',
                                    content=stimulus_text[:500],
                                    metadata={
                                        'source': 'geometric_curriculum',
                                        'domain': domain,
                                    })

                                self._cycles_since_stimulus = 0
                                self._reflection_limiter.on_curriculum()

                                # Topic tracking for anti-repetition
                                topic = _extract_topic(stimulus_text)
                                self._curriculum_history.setdefault(domain, []).append(topic)

                                # PE tracking for trigger (curiosity controller uses this too)
                                pe = obs_result.get('prediction_error', 0)
                                self._pe_history.append(pe)

                                # ── Curriculum stats tracking ──
                                diag = self.get_diagnostics()
                                current_cv = diag.get('metric_cv', 0)
                                current_tau = diag.get('tau_mean', 0)

                                stats = self._curriculum_stats
                                stats['domain_counts'][domain] = stats['domain_counts'].get(domain, 0) + 1
                                stats['domain_pe_history'].setdefault(domain, []).append(pe)
                                stats['domain_cv_history'].setdefault(domain, []).append(current_cv)
                                stats['domain_tau_history'].setdefault(domain, []).append(current_tau)

                                # Running averages (last 50)
                                for d in self._curriculum_domains:
                                    pe_hist = stats['domain_pe_history'].get(d, [])
                                    cv_hist = stats['domain_cv_history'].get(d, [])
                                    tau_hist = stats['domain_tau_history'].get(d, [])
                                    if pe_hist:
                                        stats['domain_avg_pe'][d] = sum(pe_hist[-50:]) / min(50, len(pe_hist))
                                    if cv_hist:
                                        stats['domain_avg_cv'][d] = sum(cv_hist[-50:]) / min(50, len(cv_hist))
                                    if tau_hist:
                                        stats['domain_avg_tau'][d] = sum(tau_hist[-50:]) / min(50, len(tau_hist))

                                # Domain effectiveness (bell curve peaking at moderate PE)
                                all_pe = [v for v in stats['domain_avg_pe'].values() if v > 0]
                                if len(all_pe) >= 2:
                                    import math as _math
                                    pe_min, pe_max = min(all_pe), max(all_pe)
                                    pe_range = pe_max - pe_min if pe_max > pe_min else 1.0
                                    for d in stats['domain_avg_pe']:
                                        norm_pe = (stats['domain_avg_pe'][d] - pe_min) / pe_range
                                        stats['domain_effectiveness'][d] = round(
                                            _math.exp(-((norm_pe - 0.4) ** 2) / 0.1), 3)
                                    stats['growth_zone_domains'] = [
                                        d for d in stats['domain_avg_pe']
                                        if 0.25 < ((stats['domain_avg_pe'][d] - pe_min) / pe_range) < 0.65
                                    ]

                                # Console logging every 10 stimuli
                                total = sum(stats['domain_counts'].values())
                                if total % 10 == 0 and total > 0:
                                    print(f"  [curriculum] stimuli={total}")
                                    for d in sorted(stats['domain_avg_pe'].keys()):
                                        d_pe = stats['domain_avg_pe'].get(d, 0)
                                        d_cv = stats['domain_avg_cv'].get(d, 0)
                                        d_tau = stats['domain_avg_tau'].get(d, 0)
                                        d_n = stats['domain_counts'].get(d, 0)
                                        print(f"    {d:15s}: PE={d_pe:6.1f} CV={d_cv:5.2f} "
                                              f"tau={d_tau:4.2f} n={d_n}")
                                    if all_pe:
                                        spread = 100 * (max(all_pe) - min(all_pe)) / max(all_pe)
                                        print(f"  [curriculum] PE spread: {spread:.1f}%")
                                        if stats['growth_zone_domains']:
                                            print(f"  [curriculum] growth zones: "
                                                  f"{stats['growth_zone_domains']}")

                                print(f"  [curiosity] #{self._curriculum_count} "
                                      f"domain={domain} PE={pe:.1f} CV={current_cv:.2f} "
                                      f"reason={curiosity_reason} "
                                      f"\"{stimulus_text[:60]}\"")

                        except Exception as e:
                            print(f"Curriculum error: {e}")
                            self._cycles_since_stimulus = 0

                time.sleep(0.05)

        self._auto_thread = threading.Thread(target=_loop, daemon=True)
        self._auto_thread.start()

    def stop_autonomous(self):
        self._running = False
        if self._auto_thread:
            self._auto_thread.join(timeout=5.0)

    # ──────────────────── PERSISTENCE ────────────────────

    def save_state(self, path: str) -> None:
        """Save all learned weights + ODE state to disk.

        Persists dynamics (MetricNet, TauNet, W_o, FFN), context_pool,
        embedding, forcing, readout, TextEmbedding, ODE state, events,
        write mechanism state, and curriculum stats.
        """
        with self._gpu_lock:
            state = {
                # Core dynamics (the geometric brain — 4.69M params)
                'dynamics': self.dynamics.state_dict(),
                'context_pool': self.context_pool.state_dict(),
                # Mind infrastructure
                'embedding': self.embedding.state_dict(),
                'forcing': self.forcing.state_dict(),
                'readout': self.readout.state_dict(),
                # TextEmbedding (Path C)
                'text_embed': self._text_embed.state_dict() if self._text_embed is not None else None,
                # ODE state
                'h': self._h.cpu() if self._h is not None else None,
                'events': [
                    {k: v.cpu() if isinstance(v, torch.Tensor) else v
                     for k, v in e.items()}
                    for e in self.events
                ],
                'event_count': self.event_count,
                'last_event_time': self._last_event_time,
                # Write mechanism state
                'salience': self._salience.cpu(),
                'consolidation_emb': self._consolidation_emb.cpu(),
                'high_tau_streak': self._high_tau_streak.cpu(),
                'event_embed_cache': self._event_embed_cache.cpu(),
                'h_norm_ceiling': self._h_norm_ceiling,
                'h_norm_ema': self._h_norm_ema,
                # Curriculum state (full)
                'curriculum_stats': getattr(self, '_curriculum_stats', None),
                'curriculum_count': getattr(self, '_curriculum_count', 0),
                'curriculum_idx': getattr(self, '_curriculum_idx', 0),
                'curriculum_history': getattr(self, '_curriculum_history', {}),
                'curriculum_tier': getattr(self, '_curriculum_tier', {}),
                'scheduler_phase': getattr(self, '_scheduler_phase', 'feed'),
                'scheduler_stimuli_this_phase': getattr(self, '_scheduler_stimuli_this_phase', 0),
                'scheduler_cycles_this_phase': getattr(self, '_scheduler_cycles_this_phase', 0),
                # Reflection state
                'reflection_count': self._reflection_count,
                'last_reflection_text': self._last_reflection_text,
                # Reflection limiter
                'reflection_limiter_curriculum': getattr(self, '_reflection_limiter', None) and self._reflection_limiter.curriculum_count,
                'reflection_limiter_reflection': getattr(self, '_reflection_limiter', None) and self._reflection_limiter.reflection_count,
                # Curiosity controller
                'curiosity_pe_baseline': getattr(self, '_curiosity', None) and self._curiosity.pe_baseline,
                'curiosity_phase': getattr(self, '_curiosity', None) and self._curiosity.current_phase,
                'curiosity_total_injections': getattr(self, '_curiosity', None) and self._curiosity.total_injections,
                'curiosity_consecutive': getattr(self, '_curiosity', None) and self._curiosity.consecutive_stimuli,
            }
        torch.save(state, path)
        print(f"Mind state saved: {path} ({len(self.events)} events, "
              f"dynamics + context_pool + text_embed persisted)")

    def load_state(self, path: str) -> None:
        """Restore all learned weights + ODE state from disk."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        with self._gpu_lock:
            # Core dynamics (the geometric brain)
            if 'dynamics' in state:
                self.dynamics.load_state_dict(state['dynamics'], strict=False)
                print(f"  Restored dynamics weights")
            if 'context_pool' in state:
                self.context_pool.load_state_dict(state['context_pool'], strict=False)
                print(f"  Restored context_pool weights")

            # Mind infrastructure
            self.embedding.load_state_dict(state['embedding'], strict=False)
            self.forcing.load_state_dict(state['forcing'], strict=False)
            self.readout.load_state_dict(state['readout'], strict=False)

            # TextEmbedding (Path C)
            if state.get('text_embed') is not None and self._text_embed is not None:
                self._text_embed.load_state_dict(state['text_embed'], strict=False)
                print(f"  Restored TextEmbedding weights")

            # ODE state
            if state['h'] is not None:
                self._h = state['h'].to(self.device)
            self.events = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in e.items()}
                for e in state['events']
            ]
            self.event_count = state['event_count']
            self._last_event_time = state['last_event_time']

            # Write mechanism state
            if 'salience' in state:
                n = min(state['salience'].shape[0], self._salience.shape[0])
                self._salience[:n] = state['salience'][:n].to(self.device)
            if 'consolidation_emb' in state:
                n = min(state['consolidation_emb'].shape[0], self._consolidation_emb.shape[0])
                self._consolidation_emb[:n] = state['consolidation_emb'][:n].to(self.device)
            if 'high_tau_streak' in state:
                n = min(state['high_tau_streak'].shape[0], self._high_tau_streak.shape[0])
                self._high_tau_streak[:n] = state['high_tau_streak'][:n].to(self.device)
            if 'event_embed_cache' in state:
                n = min(state['event_embed_cache'].shape[0], self._event_embed_cache.shape[0])
                self._event_embed_cache[:n] = state['event_embed_cache'][:n].to(self.device)
            else:
                # Rebuild from loaded events if no cache in state
                self._rebuild_event_embed_cache()
            if 'h_norm_ceiling' in state:
                self._h_norm_ceiling = state['h_norm_ceiling']
                self._h_norm_ema = state.get('h_norm_ema', 0.0)

            # Curriculum state (full)
            if state.get('curriculum_stats') is not None:
                self._curriculum_stats = state['curriculum_stats']
                self._curriculum_count = state.get('curriculum_count', 0)
                self._curriculum_idx = state.get('curriculum_idx', 0)
                self._curriculum_history = state.get('curriculum_history', {})
                self._curriculum_tier = state.get('curriculum_tier', {})
                self._scheduler_phase = state.get('scheduler_phase', 'feed')
                self._scheduler_stimuli_this_phase = state.get('scheduler_stimuli_this_phase', 0)
                self._scheduler_cycles_this_phase = state.get('scheduler_cycles_this_phase', 0)
                n_topics = sum(len(v) for v in self._curriculum_history.values())
                print(f"  Restored curriculum ({self._curriculum_count} stimuli, "
                      f"{n_topics} topics, phase={self._scheduler_phase})")

            # Reflection state
            if 'reflection_count' in state:
                self._reflection_count = state['reflection_count']
                self._last_reflection_text = state.get('last_reflection_text')

            # Reflection limiter
            if hasattr(self, '_reflection_limiter') and state.get('reflection_limiter_curriculum') is not None:
                self._reflection_limiter.curriculum_count = state['reflection_limiter_curriculum']
                self._reflection_limiter.reflection_count = state.get('reflection_limiter_reflection', 0)

            # Curiosity controller
            if hasattr(self, '_curiosity') and state.get('curiosity_pe_baseline') is not None:
                baseline = state['curiosity_pe_baseline']
                self._curiosity.total_injections = state.get('curiosity_total_injections', 0)
                self._curiosity.consecutive_stimuli = state.get('curiosity_consecutive', 0)
                if baseline and baseline > 0:
                    self._curiosity.pe_baseline = baseline
                    self._curiosity.current_phase = state.get('curiosity_phase', 'exploring')
                else:
                    # Force recalibration if baseline was 0
                    self._curiosity.current_phase = 'calibrating'
                print(f"  Restored curiosity (baseline={self._curiosity.pe_baseline}, "
                      f"phase={self._curiosity.current_phase})")

        print(f"Mind state loaded: {path} ({len(self.events)} events, "
              f"event_count={self.event_count})")
