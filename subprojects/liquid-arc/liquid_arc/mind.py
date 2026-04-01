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
import threading
import time
from typing import Any, Dict, List, Optional

from .config import LiquidARCConfig
from .lifecycle import SensoryForcing
from .conversation_embedding import ConversationEmbedding
from .state_readout import StateReadout
from .context_pool import ContextPool
from .dynamics import ContinuousDynamics


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
        text_embedder: Any,
        device: str = 'cuda',
        max_context_events: int = 64,
        lambda_eff: float = 0.001,
        freeze_dynamics: bool = True,
        online_lr: float = 1e-5,
        enable_online_learning: bool = True,
    ):
        self.device = device
        self.text_embedder = text_embedder
        self.max_events = max_context_events
        self.lambda_eff = lambda_eff

        d = config.d_model

        # Core dynamics (from checkpoint)
        self.dynamics = ContinuousDynamics(config).to(device)
        self.context_pool = ContextPool(config).to(device)

        # Load checkpoint weights for dynamics + context_pool
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        # Strip _orig_mod. prefix from compiled checkpoints
        cleaned = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
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

        # Conversation-specific modules (new, randomly initialized)
        self.embedding = ConversationEmbedding(
            d_model=d,
            content_embed_dim=384,
            n_metadata_features=8,
            max_events=max_context_events,
        ).to(device)

        self.forcing = SensoryForcing(d_model=d, n_entities=max_context_events).to(device)

        self.readout = StateReadout(
            d_model=d, d_summary=256, max_events=max_context_events,
        ).to(device)

        # Persistent ODE state
        self._h: Optional[torch.Tensor] = None

        self._gpu_lock = threading.Lock()

        # Event buffer
        self.events: List[Dict] = []
        self.event_count = 0
        self._last_event_time = time.time()

        # Integration config
        self.T = getattr(config, 'integration_time', 2.0)
        self.internal_steps = config.n_ode_steps

        # Online learning
        if enable_online_learning:
            self.optimizer = torch.optim.Adam(
                list(self.embedding.parameters()) +
                list(self.forcing.parameters()) +
                list(self.readout.parameters()),
                lr=online_lr,
            )
        else:
            self.optimizer = None

        # Autonomous processing thread
        self._running = False
        self._auto_thread = None

        # Initialize state
        self._h = torch.zeros(1, max_context_events, d, device=device)

    # ──────────────────── TEXT EMBEDDING (LOCAL) ────────────────────

    def _embed_text(self, text: str) -> torch.Tensor:
        """Embed text using local model. NEVER leaves the Spark."""
        with torch.no_grad():
            emb = self.text_embedder.encode(text, convert_to_tensor=True)
        return emb.to(self.device)

    # ──────────────────── EVENT MANAGEMENT ────────────────────

    def _build_metadata(self, event_type: str, content: str,
                        metadata: Optional[Dict]) -> tuple:
        now = time.time()
        time_delta = now - self._last_event_time
        self._last_event_time = now

        type_map = {
            'user_message': 0, 'assistant_message': 1, 'tool_result': 2,
            'goal': 3, 'context': 4, 'temporal': 5,
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

    # ──────────────────── ODE INTEGRATION ────────────────────

    def _run_ode_segment(self, h: torch.Tensor, n_steps: int,
                         forcing: Optional[torch.Tensor] = None,
                         return_efficiency: bool = False):
        """Run n_steps of ODE. Caller must hold _gpu_lock."""
        dt = self.T / n_steps
        t = 0.0
        eff_accum = torch.tensor(0.0, device=h.device)

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

            g = self.dynamics.compute_metric(h_new.detach())
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
        for i, (event, rel) in enumerate(zip(recent, relevance)):
            context_items.append({
                'index': i,
                'type': ['user_msg', 'assistant_msg', 'tool_result',
                         'goal', 'context', 'temporal'][event['type']],
                'preview': event['content_preview'],
                'relevance': round(rel, 3),
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

            g = self.dynamics.compute_metric(h[:, :N, :].detach())
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

    def start_autonomous(self):
        """Background thread: efficiency-gated autonomous dynamics.

        CRITICAL: Uses efficiency regularizer to prevent runaway dynamics.
        Does NOT use ||dh/dt|| as curiosity (causes NaN).
        All GPU operations serialized through _gpu_lock.
        """
        self._running = True

        def _loop():
            while self._running:
                if self._h is not None and len(self.events) > 0:
                    with self._gpu_lock:
                        try:
                            N = min(len(self.events), self.max_events)
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
                        except Exception as e:
                            print(f"Autonomous processing error: {e}")

                time.sleep(1.0)

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
        print(f"Mind state loaded: {path} ({len(self.events)} events, "
              f"event_count={self.event_count})")
