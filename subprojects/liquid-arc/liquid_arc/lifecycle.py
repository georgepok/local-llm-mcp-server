"""Continuous lifecycle runtime for LiquidARC in Isaac Sim.

The ODE runs as a persistent process. Observations inject as sensory
forcing. Actions are read continuously. Between observations, the
dynamics evolve autonomously.

The key equation changes from:
  dh/dt = -(1/τ)(h - f(h))           [standard, autonomous]
to:
  dh/dt = -(1/τ)(h - f(h)) + F(t)    [forced, with sensory input]

where F(t) = β · (embed(obs) - h) is predictive coding:
  - When h matches observation: forcing ≈ 0 (prediction confirmed)
  - When h diverges from observation: forcing is large (prediction error)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .dynamics import ContinuousDynamics
from .robotics_embedding import RoboticsEmbedding
from .action_head import ActionHead
from .context_pool import ContextPool
from .config import LiquidARCConfig


class SensoryForcing(nn.Module):
    """Computes forcing F(t) = β · (obs_embed - h).

    β is per-entity, learned. Controls how strongly each entity token
    is pulled toward the new observation vs maintaining internal state.
    """

    def __init__(self, d_model: int = 768, n_entities: int = 32):
        super().__init__()
        self.beta_logits = nn.Parameter(torch.zeros(n_entities))
        self.beta_max = 2.0

    @property
    def beta(self) -> torch.Tensor:
        return self.beta_max * torch.sigmoid(self.beta_logits)

    def compute_forcing(self, h_current: torch.Tensor, obs_embed: torch.Tensor) -> torch.Tensor:
        N = h_current.shape[1]
        beta = self.beta[:N].unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        return beta * (obs_embed - h_current)

    def get_prediction_error(self, h_current: torch.Tensor, obs_embed: torch.Tensor) -> torch.Tensor:
        return (obs_embed - h_current).detach().norm(dim=-1)  # [B, N]


class ContinuousLifecycleRunner(nn.Module):
    """Always-on dynamical system connected to Isaac Sim.

    The ODE state h persists across physics steps. Observations inject
    via sensory forcing. Actions read from current h. Between observations,
    dynamics evolve autonomously.
    """

    def __init__(
        self,
        config: LiquidARCConfig,
        n_entities: int = 13,
        n_actuated: int = 12,
        action_dim: int = 12,
        state_dim: int = 16,
        internal_steps: int = 16,
        autonomous_steps: int = 0,
        freeze_dynamics: bool = False,
    ):
        super().__init__()
        self.config = config
        self.n_entities = n_entities
        self.n_actuated = n_actuated
        self.internal_steps = internal_steps
        self.autonomous_steps = autonomous_steps
        d = config.d_model

        self.dynamics = ContinuousDynamics(config)
        self.context_pool = ContextPool(config)

        self.embedding = RoboticsEmbedding(
            d_model=d, max_state_dim=state_dim,
            n_entity_types=8, max_entities=max(32, n_entities),
            dropout=config.dropout,
        )
        self.forcing = SensoryForcing(d_model=d, n_entities=n_entities)

        self.action_head = ActionHead(
            d_model=d, action_dim=action_dim, n_actuated=n_actuated,
        )

        self.T_per_obs = getattr(config, 'integration_time', 2.0)

        # Persistent ODE state
        self._h: Optional[torch.Tensor] = None
        self._steps_since_reset = 0

        if freeze_dynamics:
            for param in self.dynamics.parameters():
                param.requires_grad = False
            for param in self.context_pool.parameters():
                param.requires_grad = False

    @property
    def is_alive(self) -> bool:
        return self._h is not None

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: LiquidARCConfig,
                        n_entities: int, n_actuated: int, action_dim: int,
                        state_dim: int, internal_steps: int = 16,
                        autonomous_steps: int = 0, freeze_dynamics: bool = False,
                        device: str = 'cuda'):
        model = cls(
            config=config, n_entities=n_entities, n_actuated=n_actuated,
            action_dim=action_dim, state_dim=state_dim,
            internal_steps=internal_steps, autonomous_steps=autonomous_steps,
            freeze_dynamics=freeze_dynamics,
        ).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model', ckpt)
        cleaned = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
        dynamics_keys = {k: v for k, v in cleaned.items()
                         if k.startswith('dynamics.') or k.startswith('context_pool.')}
        model.load_state_dict(dynamics_keys, strict=False)
        print(f"Loaded {len(dynamics_keys)} dynamics/context parameters")
        return model

    def reset(self, batch_size: int, device: torch.device):
        """Reset ODE state — the model is born."""
        self._h = torch.zeros(
            batch_size, self.n_entities, self.config.d_model, device=device,
        )
        self._steps_since_reset = 0

    def _run_ode_segment(
        self, h: torch.Tensor, n_steps: int,
        forcing: Optional[torch.Tensor] = None,
        return_efficiency: bool = False,
        prediction_error: Optional[torch.Tensor] = None,
    ):
        """Run n_steps of ODE, optionally with decaying sensory forcing.

        If return_efficiency=True, returns (h, efficiency_cost) where
        efficiency_cost is prediction-error-GATED:
          L_eff = mean(||dh/dt||² · gate)
          gate = sigmoid(-prediction_error) → near 0 when surprised (allow dynamics)
                                            → near 1 when routine (penalize waste)

        This lets the model explore surprising states with full dynamics
        while being efficient on routine states.
        """
        T = self.T_per_obs
        dt = T / n_steps
        t = 0.0

        if return_efficiency:
            eff_accum = torch.tensor(0.0, device=h.device)
            # Prediction-error gate: don't penalize dynamics when surprised
            if prediction_error is not None:
                # prediction_error is [B, N], normalize to reasonable scale
                pe_mean = prediction_error.mean(dim=-1, keepdim=True)  # [B, 1]
                gate = torch.sigmoid(-pe_mean / 10.0)  # [B, 1] → near 0 when surprised
            else:
                gate = torch.ones(h.shape[0], 1, device=h.device)

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
                # Per-position dynamics cost, gated by prediction error
                dy_sq = (dy ** 2).mean(dim=-1)  # [B, N]
                gated_cost = (dy_sq * gate).mean()  # scalar
                eff_accum = eff_accum + gated_cost

            h = h + dt * dy
            t = t + dt

        if return_efficiency:
            return h, eff_accum / n_steps
        return h

    def step(
        self, obs_tokens: Dict[str, torch.Tensor],
        actuated_indices: torch.Tensor,
        skip_autonomous: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Process one physics timestep with persistent state.

        Args:
            skip_autonomous: If True, skip autonomous ODE steps (used during PPO
                re-evaluation where h starts fresh and autonomous steps would diverge).
        """
        device = obs_tokens['state_features'].device
        B = obs_tokens['state_features'].shape[0]

        if self._h is None or self._h.shape[0] != B:
            self.reset(B, device)

        # Embed observation
        obs_embed = self.embedding(
            obs_tokens['state_features'],
            obs_tokens['spatial_x'], obs_tokens['spatial_y'], obs_tokens['spatial_z'],
            obs_tokens['entity_types'], obs_tokens['entity_ids'],
        )

        # Prediction error: how surprised by this observation?
        prediction_error = self.forcing.get_prediction_error(self._h, obs_embed)
        # Curiosity: mean prediction error across entities [B]
        curiosity = prediction_error.mean(dim=-1)

        # Sensory forcing
        forcing = self.forcing.compute_forcing(self._h, obs_embed)

        # Context from current state (accumulated understanding)
        context_mask = torch.ones(B, self.n_entities, dtype=torch.bool, device=device)
        context = self.context_pool(self._h, context_mask)
        self.dynamics.set_context(context, mask=None)
        self.dynamics.set_n_steps(self.internal_steps)

        # ODE with sensory forcing + prediction-error-gated efficiency tracking
        ode_result = self._run_ode_segment(
            self._h, self.internal_steps, forcing=forcing,
            return_efficiency=True, prediction_error=prediction_error)
        h_after_obs, efficiency_cost = ode_result

        # Autonomous processing (if configured, and not skipped for PPO eval)
        if self.autonomous_steps > 0 and not skip_autonomous:
            context = self.context_pool(h_after_obs, context_mask)
            self.dynamics.set_context(context, mask=None)
            self.dynamics.set_n_steps(self.autonomous_steps)
            h_after_auto = self._run_ode_segment(h_after_obs, self.autonomous_steps)
        else:
            h_after_auto = h_after_obs

        # Read action
        actions = self.action_head(h_after_auto, actuated_indices)

        # Update persistent state (detached for next step)
        self._h = h_after_auto.detach()
        self._steps_since_reset += 1

        # Diagnostics
        g = self.dynamics.compute_metric_diag(h_after_auto.detach())
        metric_cv = g.std() / (g.mean() + 1e-8)
        tau = self.dynamics.compute_tau(h_after_auto.detach())

        return {
            'actions': actions,
            'curiosity': curiosity,
            'prediction_error': prediction_error,
            'efficiency_cost': efficiency_cost,  # mean(||dh/dt||²) for regularizer
            'h_state': h_after_auto,
            'metric_cv': metric_cv.item(),
            'tau_mean': tau.mean().item(),
            'tau_std': tau.std().item(),
            'beta': self.forcing.beta[:self.n_entities].detach(),
            'steps_alive': self._steps_since_reset,
        }

    def handle_resets(self, reset_mask: torch.Tensor):
        """Reset ODE state for terminated environments."""
        if self._h is not None and reset_mask.any():
            self._h[reset_mask] = 0.0

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def frozen_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
