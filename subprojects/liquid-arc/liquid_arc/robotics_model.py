"""LiquidARC for robotics — wraps the post-transition model for control tasks.

Architecture:
    RoboticsEmbedding → ContextPool → euler_solve(ContinuousDynamics, 16 steps) → ActionHead

ContinuousDynamics is loaded from the post-transition checkpoint.
Only the embedding and action head are new.
"""

import torch
import torch.nn as nn
from typing import Dict

from .config import LiquidARCConfig
from .robotics_embedding import RoboticsEmbedding
from .action_head import ActionHead
from .context_pool import ContextPool
from .dynamics import ContinuousDynamics
from .solver import euler_solve


class LiquidARCRoboticsModel(nn.Module):
    """Robotics controller using the LiquidARC ODE substrate."""

    def __init__(
        self,
        config: LiquidARCConfig,
        action_dim: int = 1,
        n_entities: int = 2,
        n_actuated: int = 1,
        state_dim_per_entity: int = 4,
        freeze_dynamics: bool = True,
    ):
        super().__init__()
        self.config = config
        self.n_entities = n_entities
        self.n_actuated = n_actuated
        d = config.d_model

        self.embedding = RoboticsEmbedding(
            d_model=d,
            max_state_dim=state_dim_per_entity,
            n_entity_types=8,
            max_entities=max(32, n_entities),
            dropout=config.dropout,
        )

        self.context_pool = ContextPool(config)
        self.dynamics = ContinuousDynamics(config)

        self.action_head = ActionHead(
            d_model=d,
            action_dim=action_dim,
            n_actuated=n_actuated,
        )

        if freeze_dynamics:
            for param in self.dynamics.parameters():
                param.requires_grad = False
            for param in self.context_pool.parameters():
                param.requires_grad = False

        self.integration_time = getattr(config, 'integration_time', 2.0)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: LiquidARCConfig,
                        action_dim: int, n_entities: int, n_actuated: int,
                        state_dim_per_entity: int, freeze_dynamics: bool = True,
                        device: str = 'cuda'):
        """Load dynamics from a pre-trained ARC checkpoint, add new heads."""
        model = cls(
            config=config,
            action_dim=action_dim,
            n_entities=n_entities,
            n_actuated=n_actuated,
            state_dim_per_entity=state_dim_per_entity,
            freeze_dynamics=freeze_dynamics,
        ).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model', ckpt)

        # Clean compiled prefixes
        cleaned = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}

        # Load only dynamics and context_pool weights
        dynamics_keys = {k: v for k, v in cleaned.items()
                         if k.startswith('dynamics.') or k.startswith('context_pool.')}

        missing, unexpected = model.load_state_dict(dynamics_keys, strict=False)

        loaded = len(dynamics_keys)
        print(f"Loaded {loaded} dynamics/context parameters from checkpoint")
        print(f"New parameters (randomly initialized): {len(missing)}")

        return model

    def forward(
        self,
        state_features: torch.Tensor,
        spatial_x: torch.Tensor,
        spatial_y: torch.Tensor,
        spatial_z: torch.Tensor,
        entity_types: torch.Tensor,
        entity_ids: torch.Tensor,
        actuated_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h0 = self.embedding(
            state_features, spatial_x, spatial_y, spatial_z,
            entity_types, entity_ids,
        )

        context_mask = torch.ones(h0.shape[:2], dtype=torch.bool, device=h0.device)
        context = self.context_pool(h0, context_mask)
        self.dynamics.set_context(context, mask=None)
        self.dynamics.set_n_steps(self.config.n_ode_steps)

        h_final = euler_solve(
            self.dynamics, h0,
            t_span=(0.0, self.integration_time),
            n_steps=self.config.n_ode_steps,
        )

        actions = self.action_head(h_final, actuated_indices)

        g = self.dynamics.compute_metric(h0)
        metric_cv = g.std() / (g.mean() + 1e-8)
        tau = self.dynamics.compute_tau(h0)

        # Internal curiosity: LTC convergence residual from the last ODE step.
        # ||h - target|| at the final step = model's own surprise about the state.
        # High residual = "I don't know where h should be" = genuine uncertainty.
        # Bounded by state norms. Computed inside the ODE, detached from gradients.
        curiosity = getattr(self.dynamics, '_last_residual',
                            torch.zeros(h0.shape[0], device=h0.device))

        return {
            'actions': actions,
            'curiosity': curiosity,  # [B] LTC residual curiosity
            'h_final': h_final,
            'metric_cv': metric_cv.item(),
            'tau_mean': tau.mean().item(),
            'tau_std': tau.std().item(),
        }

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def frozen_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
