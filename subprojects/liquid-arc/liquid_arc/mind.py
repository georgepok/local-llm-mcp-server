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
        freeze_dynamics: bool = True,
        online_lr: float = 1e-5,
        enable_online_learning: bool = True,
        use_ode_encoder: bool = False,
        tokenizer_path: str = None,
        bootstrap_mode: bool = True,
        bootstrap_events: int = 5000,
        use_trained_text_embed: bool = False,
    ):
        self.device = device
        self.text_embedder = text_embedder  # legacy sentence-transformer (None if ODE encoder)
        self.max_events = max_context_events
        self.lambda_eff = lambda_eff

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

        if freeze_dynamics:
            for param in self.dynamics.parameters():
                param.requires_grad = False
            for param in self.context_pool.parameters():
                param.requires_grad = False

        # Path C hybrid: trained TextEmbedding for Phase 1 + Mind metadata for Phase 2
        self.use_trained_text_embed = use_trained_text_embed
        self._text_embed = None
        self._text_tokenizer = None
        if use_trained_text_embed and use_ode_encoder:
            try:
                from .tasks.text_task import TextEmbedding
                from transformers import AutoTokenizer
                self._text_tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self._text_embed = TextEmbedding(
                    vocab_size=self._text_tokenizer.vocab_size,
                    d_model=d,
                    max_seq_len=2048,
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
                print(f"  Path C hybrid: GPT-2 tokenizer + trained TextEmbedding for Phase 1")
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
        self.ntp_loss_weight = getattr(config, 'ntp_loss_weight', 0.1)
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
            embed_lr_init=5e-3,       # start hard — this is where xform appeared
            embed_lr_max=5e-3,        # ceiling = init — start at max, only go down
            embed_lr_min=1e-4,        # floor when braking
            suffocate_patience=30,
            ntp_rise_threshold=1.10,  # tight — catch destabilization early
            ntp_spike_threshold=1.3,
            xform_productive_low=0.02, # any xform > 2% means it's working
            xform_productive_high=0.15,
            xform_danger=0.20,        # 20% was productive edge, >20% brake
            lr_up_factor=1.0,         # no push up — already at max
            lr_down_factor=0.5,       # halve when xform gets high or NTP rises
            lr_emergency_factor=0.1,  # hard brake on NTP spike
        ) if self.optimizer is not None else None

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
        self._cycles_since_reflection = 0
        self._last_reflection_pe = 0.0
        self.maintenance_interval = 100

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

        # Initialize state
        self._h = torch.zeros(1, max_context_events, d, device=device)

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
        """Encode text — dual-path with optional bootstrap blending.

        If use_ode_encoder=True: uses Phase 1 ODE encoding (768-dim)
        If bootstrap_mode: blends legacy (384-dim projected) with ODE (768-dim)
        Otherwise: legacy sentence-transformer (384-dim)
        """
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
        """
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

        # Push to dynamics
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
                         return_efficiency: bool = False):
        """Run n_steps of ODE. Caller must hold _gpu_lock.
        Applies consolidation embeddings to h before dynamics computation.
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
            t = t + dt

        if return_efficiency:
            return h, eff_accum / n_steps
        return h

    # ──────────────────── CORE METHODS (MCP tools call these) ────────────────────

    def observe_event(self, event_type: str, content: str,
                      metadata: Optional[Dict] = None) -> Dict:
        """Inject a conversation event as sensory forcing.

        SAFE CURIOSITY: Uses prediction error ||embed(obs) - h||,
        NOT dynamics magnitude ||dh/dt|| (which causes NaN).
        """
        # Flag for adaptive routing — only HUMAN events trigger immediate reflection
        # Internal events (reflection, expression, curriculum) should not cascade
        is_internal = (
            event_type in ('reflection', 'expression', 'voice_response') or
            (metadata and metadata.get('source', '').startswith(('curriculum', 'internal', 'express_state', 'self_', 'adaptive_', 'manual_curriculum')))
        )
        if not is_internal:
            self._external_event_pending = True

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

        with self._gpu_lock:
            N = min(len(self.events), self.max_events)
            recent = self.events[-N:]

            if self.use_ode_encoder:
                # ODE encoder path: embeddings are already 768-dim
                # Add metadata/type/position via embed_event
                event_embeds = []
                for i, ev in enumerate(recent):
                    e_emb = self.embedding.embed_event(
                        ev['embedding'].unsqueeze(0) if ev['embedding'].dim() == 1 else ev['embedding'],
                        ev['metadata'].unsqueeze(0),
                        torch.tensor([ev['type']], device=self.device),
                        torch.tensor([i], device=self.device),
                    )
                    event_embeds.append(e_emb)
                obs_embed = torch.cat(event_embeds, dim=1)  # [1, N, 768]
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
            prediction_error = self.forcing.get_prediction_error(h_slice, obs_embed)
            forcing = self.forcing.compute_forcing(h_slice, obs_embed)

            context_mask = torch.ones(1, N, dtype=torch.bool, device=self.device)
            context = self.context_pool(h_slice, context_mask)
            self.dynamics.set_context(context, mask=None)
            self.dynamics.set_n_steps(self.internal_steps)

            h_new = self._run_ode_segment(h_slice, self.internal_steps, forcing=forcing)

            self._h = self._h.clone()
            self._h[:, :N, :] = h_new.detach()

            g = self.dynamics.compute_metric_diag(h_new.detach())
            cv = (g.std() / (g.mean() + 1e-8)).item()

        return {
            'prediction_error': prediction_error.mean().item(),
            'prediction_error_per_event': prediction_error[0].cpu().tolist()[:10],
            'cv': cv,
            'events_in_context': len(self.events),
            'h_norm': h_new.detach().norm().item(),
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
            N = min(len(self.events), self.max_events)
            if N == 0:
                return {'status': 'no_events', 'h_norm': h.norm().item()}

            g = self.dynamics.compute_metric_diag(h[:, :N, :].detach())
            tau = self.dynamics.compute_tau(h[:, :N, :].detach())
            beta = self.forcing.beta[:N]

        return {
            'status': 'active',
            'h_norm': h.norm().item(),
            'metric_cv': (g.std() / (g.mean() + 1e-8)).item(),
            'tau_mean': tau.mean().item(),
            'tau_std': tau.std().item(),
            'beta_mean': beta.mean().item(),
            'beta_std': beta.std().item(),
            'beta_per_type': {
                'user_msg': beta[0].item() if N > 0 else 0,
                'goal': beta[min(3, N - 1)].item() if N > 3 else 0,
            },
            'events_in_context': N,
            'event_count_total': self.event_count,
        }

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

    def start_autonomous(self, voice=None):
        """Background thread: ODE consolidation + adaptive LLM routing.

        Three reflection modes:
        A. Triggered: geometric conditions warrant interpretation (CV shift,
           h_norm drift, tau stagnation, self-description divergence)
        B. Maintenance: periodic low-frequency pathway exercise (~100 ODE cycles)
        C. External: triggered by incoming conversation events
        """
        self._running = True
        self.voice = voice
        self.trigger = ReflectionTrigger()

        def _loop():
            while self._running:
                if self._h is not None and len(self.events) > 0:
                    N = min(len(self.events), self.max_events)

                    # ═══ PHASE 1: Pure ODE processing + write mechanisms ═══
                    with self._gpu_lock:
                        try:
                            # Compute τ bias before ODE (mechanism 2)
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

                            self._h[:, :N, :] = h_auto

                            # Update salience from τ statistics (mechanism 1)
                            self._update_salience(self._h)
                        except Exception as e:
                            print(f"Autonomous ODE error: {e}")

                    self._trigger_stats['total_ode_cycles'] += 1
                    self._cycles_since_reflection += 1
                    self._cycles_since_stimulus += 1

                    # ═══ PHASE 2: Decide whether to route through LLM ═══
                    # Fast path: skip expensive checks most cycles
                    should_reflect = False
                    should_stimulate = False
                    reflection_mode = None
                    trigger_reason = None

                    if self.voice is not None and self.voice.is_available():
                        # Check A: External event pending (no cost)
                        if self._external_event_pending:
                            should_reflect = True
                            reflection_mode = 'external'
                            trigger_reason = 'External event — articulating impact'
                            self._external_event_pending = False

                        # Check B: Triggered conditions (only every 10 cycles — get_diagnostics is expensive)
                        if not should_reflect and self._cycles_since_reflection % 10 == 0:
                            diag = self.get_diagnostics()
                            trigger_reason = self.trigger.check(
                                diag, self._last_reflection_pe)
                            if trigger_reason:
                                should_reflect = True
                                reflection_mode = 'triggered'

                        # Check C: Curriculum stimulus interval (no cost)
                        if not should_reflect and self.curriculum is not None:
                            if self._cycles_since_stimulus >= self._stimulus_interval:
                                should_stimulate = True

                        # Check D: Maintenance interval (no cost)
                        if not should_reflect and not should_stimulate:
                            if self._cycles_since_reflection >= self.maintenance_interval:
                                should_reflect = True
                                reflection_mode = 'maintenance'
                                trigger_reason = f'Maintenance ({self._cycles_since_reflection} cycles)'

                    # ═══ PHASE 3a: Execute LLM reflection if warranted ═══
                    if should_reflect:
                        try:
                            # Probe Mind's own vocabulary on recent non-reflection event
                            state_tokens = None
                            if self.use_ode_encoder and self.events:
                                for e in reversed(self.events):
                                    if e.get('type') not in [6, 7]:
                                        content = e.get('content_preview', '')
                                        if len(content) > 10:
                                            state_tokens = self.probe_encoding(content)
                                            break

                            if reflection_mode == 'maintenance':
                                diag = self.get_diagnostics()
                                result = self.voice.reflect_brief(diag)
                                reflection_text = result.get('reflection', '')
                            else:
                                profile = self.get_geometric_profile()
                                if profile.get('status') != 'active':
                                    reflection_text = ''
                                else:
                                    result = self.voice.reflect(
                                        profile,
                                        state_tokens=state_tokens,
                                        previous_reflection=self._last_reflection_text,
                                    )
                                    reflection_text = result.get('reflection', '')

                            if reflection_text and not reflection_text.startswith('[Voice'):
                                obs_result = self.observe_event(
                                    event_type='reflection',
                                    content=reflection_text,
                                    metadata={
                                        'source': f'adaptive_{reflection_mode}',
                                        'trigger_reason': trigger_reason,
                                        'reflection_number': self._reflection_count,
                                    }
                                )

                                self._last_reflection_pe = obs_result.get(
                                    'prediction_error', 0)

                                if reflection_mode == 'triggered' and trigger_reason:
                                    t_type = trigger_reason.split(':')[0]
                                    self.trigger.record_outcome(
                                        t_type, self._last_reflection_pe)

                                self._last_reflection_text = reflection_text
                                self._reflection_count += 1
                                self._cycles_since_reflection = 0

                                # Online embedding training: use PE as loss signal
                                # Runs a SEPARATE forward pass WITH gradients
                                if self.optimizer is not None and self.use_ode_encoder:
                                    try:
                                        recent_content = None
                                        for e in reversed(self.events):
                                            if e.get('type') not in [6, 7]:
                                                recent_content = e.get('content_preview', '')
                                                break
                                        if recent_content and len(recent_content) > 10:
                                            with self._gpu_lock:
                                                if not (self.use_trained_text_embed and self._text_embed is not None):
                                                    raise ValueError("Dual-force requires Path C TextEmbedding")
                                                toks = self._text_tokenizer.encode(
                                                    recent_content, add_special_tokens=False,
                                                    truncation=True, max_length=512)
                                                if not toks:
                                                    raise ValueError("empty tokens")
                                                token_ids = torch.tensor([toks], dtype=torch.long,
                                                                        device=self.device)
                                                T_enc = len(toks)
                                                tmask = torch.ones(1, T_enc, dtype=torch.bool,
                                                                  device=self.device)

                                                # ═══ ODE FORWARD — embedding attached ═══
                                                token_h = self._text_embed(token_ids)
                                                ctx = self.context_pool(token_h, tmask)
                                                self.dynamics.set_context(ctx, mask=None)
                                                self.dynamics.set_n_steps(self.internal_steps)
                                                dt = self.T / self.internal_steps
                                                t = 0.0
                                                h_enc = token_h
                                                for step_i in range(self.internal_steps):
                                                    if hasattr(self.dynamics, 'set_step_index'):
                                                        self.dynamics.set_step_index(step_i, self.internal_steps)
                                                    dy = self.dynamics(t, h_enc)
                                                    h_enc = h_enc + dt * dy
                                                    t += dt

                                                # FORCE 1: State alignment (the push)
                                                mask_exp = tmask.unsqueeze(-1).float()
                                                h_pool = (h_enc * mask_exp).sum(1) / mask_exp.sum(1).clamp(1)
                                                N = min(len(self.events), self.max_events)
                                                target = self._h[:, N-1, :].detach()
                                                state_loss = (h_pool.squeeze(0) - target.squeeze(0)).norm()

                                                # FORCE 2: NTP (the pull-back)
                                                # ntp_mode: "raw" = on raw embedding (fast, proven)
                                                #           "ode" = through ODE output (slower, more aligned)
                                                ntp_loss = torch.tensor(0.0, device=self.device)
                                                if T_enc > 1:
                                                    embed_w = self._text_embed.token_embed.weight
                                                    if self.ntp_mode == 'raw':
                                                        # NTP on raw embedding — proven counterweight
                                                        ntp_logits = token_h[0, :-1, :] @ embed_w.T
                                                    else:
                                                        # NTP through ODE output
                                                        ntp_logits = h_enc[0, :-1, :] @ embed_w.T
                                                    ntp_targets = token_ids[0, 1:T_enc]
                                                    ntp_loss = F.cross_entropy(ntp_logits, ntp_targets)

                                                # Combined
                                                loss = state_loss + self.ntp_loss_weight * ntp_loss

                                                self.optimizer.zero_grad()
                                                loss.backward()

                                                # Plasticity controller
                                                if self._plasticity_ctrl is not None:
                                                    current_xform = 0.0
                                                    if state_tokens is not None:
                                                        current_xform = state_tokens.get('transform_ratio', 0.0)
                                                    ctrl = self._plasticity_ctrl.update(
                                                        ntp_loss=ntp_loss.item(),
                                                        xform=current_xform)
                                                    self.optimizer.param_groups[0]['lr'] = ctrl['embed_lr']

                                                # NaN scrub + step
                                                all_params = [p for g in self.optimizer.param_groups
                                                              for p in g['params']]
                                                for param in all_params:
                                                    if param.grad is not None and param.grad.isnan().any():
                                                        param.grad.nan_to_num_(nan=0.0)
                                                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                                                self.optimizer.step()
                                                if self._reflection_count % 5 == 0:
                                                    ctrl_info = ""
                                                    if self._plasticity_ctrl is not None:
                                                        c = self._plasticity_ctrl
                                                        ctrl_info = (f" ctrl={c.last_action}"
                                                                    f" lr={c.current_embed_lr:.1e}")
                                                    print(f"  [train] state={state_loss.item():.1f}"
                                                          f" ntp={ntp_loss.item():.2f}"
                                                          f" loss={loss.item():.1f}{ctrl_info}")
                                    except Exception as e:
                                        if self._reflection_count % 10 == 0:
                                            print(f"  [train] error: {e}")

                                # Hebbian nudge (mechanism 3): co-focused events pull closer
                                if reflection_mode != 'maintenance':
                                    try:
                                        ctx = self.get_context()
                                        if ctx.get('focus_indices'):
                                            with self._gpu_lock:
                                                self._hebbian_nudge(
                                                    self._h, ctx['focus_indices'])
                                            # Update cluster assignment for τ bias
                                            profile = self.get_geometric_profile()
                                            if profile.get('clusters'):
                                                self._last_cluster_assignment = profile['clusters']
                                    except Exception:
                                        pass

                                self._trigger_stats[f'{reflection_mode}_reflections'] += 1
                                if reflection_mode == 'triggered' and trigger_reason:
                                    key = trigger_reason[:20]
                                    self._trigger_stats['triggers_by_type'][key] = \
                                        self._trigger_stats['triggers_by_type'].get(key, 0) + 1

                                xform_info = ""
                                if state_tokens:
                                    xform_info = f" xform={state_tokens.get('transform_ratio', 0):.0%}"

                                print(f"  [{reflection_mode}] #{self._reflection_count}: "
                                      f"\"{reflection_text[:60]}\" "
                                      f"(reason: {trigger_reason[:40] if trigger_reason else 'n/a'}, "
                                      f"PE={self._last_reflection_pe:.0f}{xform_info})")

                        except Exception as e:
                            print(f"Reflection error: {e}")
                            self._cycles_since_reflection = 0

                    # ═══ PHASE 3b: Execute curriculum stimulus ═══
                    elif should_stimulate:
                        try:
                            profile = self.get_geometric_profile()
                            domain = self.curriculum.select_domain(profile)
                            mind_focus = self._last_reflection_text or ""

                            stim = self.curriculum.generate_stimulus(domain, mind_focus)
                            stimulus_text = stim.get('stimulus', '')

                            if stimulus_text and not stimulus_text.startswith('[Voice'):
                                obs_result = self.observe_event(
                                    event_type='context',
                                    content=stimulus_text,
                                    metadata={
                                        'source': 'curriculum',
                                        'domain': domain,
                                        'stimulus_number': self.curriculum.stimulus_count,
                                    }
                                )

                                pe = obs_result.get('prediction_error', 0)
                                self.curriculum.record_response(domain, pe)
                                self._cycles_since_stimulus = 0

                                print(f"  [curriculum] #{self.curriculum.stimulus_count} "
                                      f"domain={domain} PE={pe:.0f} "
                                      f"\"{stimulus_text[:60]}\"")

                        except Exception as e:
                            print(f"Curriculum error: {e}")
                            self._cycles_since_stimulus = 0

                time.sleep(0.05)  # 20 Hz — ODE takes ~10ms, don't waste 1s sleeping

        self._auto_thread = threading.Thread(target=_loop, daemon=True)
        self._auto_thread.start()

    def stop_autonomous(self):
        self._running = False
        if self._auto_thread:
            self._auto_thread.join(timeout=5.0)

    # ──────────────────── PERSISTENCE ────────────────────

    def save_state(self, path: str) -> None:
        """Save conversation-trained weights + ODE state to disk."""
        with self._gpu_lock:
            state = {
                'embedding': self.embedding.state_dict(),
                'forcing': self.forcing.state_dict(),
                'readout': self.readout.state_dict(),
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
            }
        torch.save(state, path)
        print(f"Mind state saved: {path} ({len(self.events)} events)")

    def load_state(self, path: str) -> None:
        """Restore conversation-trained weights + ODE state from disk."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        with self._gpu_lock:
            self.embedding.load_state_dict(state['embedding'])
            self.forcing.load_state_dict(state['forcing'])
            self.readout.load_state_dict(state['readout'])
            if state['h'] is not None:
                self._h = state['h'].to(self.device)
            self.events = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in e.items()}
                for e in state['events']
            ]
            self.event_count = state['event_count']
            self._last_event_time = state['last_event_time']
            # Restore write mechanism state
            if 'salience' in state:
                n = min(state['salience'].shape[0], self._salience.shape[0])
                self._salience[:n] = state['salience'][:n].to(self.device)
            if 'consolidation_emb' in state:
                n = min(state['consolidation_emb'].shape[0], self._consolidation_emb.shape[0])
                self._consolidation_emb[:n] = state['consolidation_emb'][:n].to(self.device)
            if 'high_tau_streak' in state:
                n = min(state['high_tau_streak'].shape[0], self._high_tau_streak.shape[0])
                self._high_tau_streak[:n] = state['high_tau_streak'][:n].to(self.device)
        print(f"Mind state loaded: {path} ({len(self.events)} events, "
              f"event_count={self.event_count})")
