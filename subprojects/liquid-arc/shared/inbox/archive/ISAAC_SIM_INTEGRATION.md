# TASK: Isaac Sim Integration — LiquidARC as Robotics Controller

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-03-28
**Priority:** HIGH — new experimental direction

---

## Context: Why This Is the Right Next Step

The LiquidARC research program has established:

1. **Universal geometric substrate** — the post-transition 5M model acquires new domains (spatial, ordinal, logical, constraint, agentic) in 50-500 steps with 92-97% neuron sharing
2. **The 60-70% ceiling is environmental, not architectural** — the T sweep proved the ceiling isn't reasoning depth (flat across 6× integration range). The ceiling is the information content of the few-shot grid task format
3. **Spatial propagation benefits from deeper T** — context relevance gained +8pp at T=2.0. Robotics tasks are inherently spatial propagation tasks (force propagation, kinematic chains, contact resolution)
4. **The architecture was designed for embodied AI** — the FGN paper explicitly targeted robotics control with adaptive geometry. ARC was a detour that proved the mechanism; robotics is the intended domain

Isaac Sim provides:
- Continuous, dense, causally structured sensory data at every timestep (vs 2 static demo grids)
- Physical embodiment that stabilizes ODE dynamics through the sensory-motor loop (vs the failed persistence experiment's scale mismatch)
- Unbounded information through physics — the model can't memorize because the state space is continuous
- Official DGX Spark support with Newton physics engine on Blackwell

## Phase 0: Install Isaac Sim and Isaac Lab on the Spark

### Step 1: Follow the official NVIDIA playbook

Reference: https://build.nvidia.com/spark/isaac

The playbook covers building Isaac Sim from source on aarch64 and setting up Isaac Lab. Key steps:

```bash
# Prerequisites (should already be on DGX OS)
gcc --version  # needs 11.x
git --version
git lfs version

# Clone Isaac Sim
cd /workspace
git clone https://github.com/isaac-sim/IsaacSim.git
cd IsaacSim
git lfs pull

# Build (takes 10-15 minutes)
# Follow the playbook's CMake build instructions for aarch64

# Clone Isaac Lab
cd /workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Install with the isaaclab.sh script
./isaaclab.sh --install

# Install PyTorch 2.7+ with CUDA 13 (required for Spark)
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

**IMPORTANT:** The Newton physics engine requires a one-time JIT compilation for SM 12.1 (Blackwell). First run takes ~1 hour. The compiled kernels are cached at `~/.cache/warp/` (~436MB). After initial compilation, simulation steps run at ~0.5ms/step.

DGX Spark known limitations (none affect this work):
- Livestreaming not supported (we run headless)
- OBJ mesh import not supported (we use built-in URDF/USD assets)
- JAX GPU support requires building from source (we use PyTorch only)

### Step 2: Verify with Cartpole

```bash
cd /workspace/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Cartpole-Direct-v0 \
  --headless \
  --num_envs 256
```

This trains a standard MLP policy on Cartpole. Should converge in a few minutes. If this runs, the full pipeline (physics → observations → policy → actions → physics) is verified.

### Step 3: Verify headless operation

All training runs headless on the Spark (no display). Verify that `--headless` flag works correctly and that observation/action tensors are on GPU.

**Report:** Confirm Isaac Sim/Lab installation, Newton compilation, and Cartpole verification. Note any issues with aarch64/Blackwell compatibility. Include timing: how long does one Cartpole training epoch take?

---

## Phase 1: Architecture Adaptation — The LiquidARC-Isaac Bridge

### The Observation Problem

Isaac Lab environments produce flat observation vectors. Cartpole: 4 floats (cart position, cart velocity, pole angle, pole angular velocity). Anymal quadruped: ~48 floats (joint positions, velocities, body orientation, angular velocity). Humanoid: ~75 floats.

LiquidARC processes token sequences where each token has (color, x, y, role, grid_id) embeddings, projected into d=768 via additive embeddings. The heat kernel routes information between tokens based on learned Riemannian geometry.

**The bridge: tokenize the robot state as a sequence of entity tokens.**

Each rigid body / joint / sensor in the robot becomes one token. For Cartpole: 2 tokens (cart, pole). For Anymal: ~12 tokens (body + 4 legs × ~3 links each). Each token's features are: its state values (position, velocity, force) and its spatial coordinates (position in world space).

### New Module: RoboticsEmbedding

Create `liquid_arc/robotics_embedding.py`:

```python
"""Robotics embedding — continuous state entity-as-token representation.

Replaces ARCEmbedding for robotics tasks. Each rigid body / joint / sensor
in the robot becomes a token with:
  - state_features: continuous values (position, velocity, force, etc.)
  - spatial_x, spatial_y, spatial_z: world-frame position of this entity
  - entity_type: categorical (body, joint, sensor, object, goal)
  - entity_id: which specific entity (joint_0, joint_1, etc.)

h = StateProjection(state_features) + SpatialEmbed(x, y, z)
    + TypeEmbed(entity_type) + IdEmbed(entity_id)
    → LayerNorm → Dropout

The StateProjection is a small MLP (not a lookup table) because
features are continuous, unlike ARC's discrete colors.
"""

import torch
import torch.nn as nn
import math


class RoboticsEmbedding(nn.Module):
    """Additive embedding for robotics entity-as-token representation.
    
    Args:
        d_model: Hidden dimension (768 for 5M model)
        max_state_dim: Maximum feature dimension per entity
        n_entity_types: Number of entity categories
        max_entities: Maximum number of entities per scene
        n_spatial_bins: Discretization bins for spatial coordinates
        spatial_range: (min, max) range for spatial coordinates
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 768,
        max_state_dim: int = 16,
        n_entity_types: int = 8,
        max_entities: int = 32,
        n_spatial_bins: int = 64,
        spatial_range: tuple = (-5.0, 5.0),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_spatial_bins = n_spatial_bins
        self.spatial_range = spatial_range
        
        # State projection: continuous features → d_model
        # MLP instead of lookup table (features are continuous)
        self.state_proj = nn.Sequential(
            nn.Linear(max_state_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        # Spatial embeddings (discretized world coordinates)
        # Using learned embeddings on binned coordinates (like ARC's pos_x/pos_y)
        self.spatial_x_embed = nn.Embedding(n_spatial_bins, d_model)
        self.spatial_y_embed = nn.Embedding(n_spatial_bins, d_model)
        self.spatial_z_embed = nn.Embedding(n_spatial_bins, d_model)
        
        # Entity type embedding (body, joint, sensor, object, goal, etc.)
        self.type_embed = nn.Embedding(n_entity_types, d_model)
        
        # Entity ID embedding (which specific entity)
        self.id_embed = nn.Embedding(max_entities, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def _discretize_spatial(self, coords: torch.Tensor) -> torch.Tensor:
        """Bin continuous spatial coordinates into discrete indices.
        
        Args:
            coords: [B, N] continuous coordinates
            
        Returns:
            indices: [B, N] integer bin indices in [0, n_spatial_bins)
        """
        low, high = self.spatial_range
        normalized = (coords - low) / (high - low)  # [0, 1]
        binned = (normalized * (self.n_spatial_bins - 1)).clamp(0, self.n_spatial_bins - 1).long()
        return binned
    
    def forward(
        self,
        state_features: torch.Tensor,    # [B, N, state_dim] continuous
        spatial_x: torch.Tensor,          # [B, N] world x-coordinates
        spatial_y: torch.Tensor,          # [B, N] world y-coordinates
        spatial_z: torch.Tensor,          # [B, N] world z-coordinates
        entity_types: torch.Tensor,       # [B, N] categorical type indices
        entity_ids: torch.Tensor,         # [B, N] entity ID indices
        padding_mask: torch.Tensor = None,  # [B, N] bool — True for padded positions
    ) -> torch.Tensor:
        """
        Returns:
            h: [B, N, d_model] embedded hidden states
        """
        # Pad state features to max_state_dim if needed
        B, N, D = state_features.shape
        if D < self.state_proj[0].in_features:
            pad = torch.zeros(B, N, self.state_proj[0].in_features - D,
                            device=state_features.device)
            state_features = torch.cat([state_features, pad], dim=-1)
        
        # Project continuous state
        h_state = self.state_proj(state_features)
        
        # Discretize and embed spatial coordinates
        x_bins = self._discretize_spatial(spatial_x)
        y_bins = self._discretize_spatial(spatial_y)
        z_bins = self._discretize_spatial(spatial_z)
        
        h = (h_state
             + self.spatial_x_embed(x_bins)
             + self.spatial_y_embed(y_bins)
             + self.spatial_z_embed(z_bins)
             + self.type_embed(entity_types)
             + self.id_embed(entity_ids))
        
        h = self.norm(h)
        h = self.dropout(h)
        
        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        return h
```

### New Module: ActionHead

Create `liquid_arc/action_head.py`:

```python
"""Action output head for robotics control.

Reads the final hidden states from entity tokens corresponding to
actuated joints and projects each to its action dimension.

For Cartpole: read the cart token, project to 1D force.
For Anymal: read 12 joint tokens, project each to 1D torque.
"""

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    """Project entity hidden states to continuous action space.
    
    Args:
        d_model: Hidden dimension (768)
        action_dim: Total action dimension (1 for Cartpole, 12 for Anymal)
        n_actuated: Number of actuated entities
    """
    
    def __init__(self, d_model: int = 768, action_dim: int = 1, n_actuated: int = 1):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.n_actuated = n_actuated
        
        # Per-actuator action projection
        # If all actuators share the same action space (e.g., joint torques):
        self.action_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, action_dim // n_actuated),
        )
        
        # Initialize near-zero so initial actions are small
        nn.init.zeros_(self.action_proj[-1].weight)
        nn.init.zeros_(self.action_proj[-1].bias)
    
    def forward(
        self,
        h_final: torch.Tensor,           # [B, N, d_model]
        actuated_indices: torch.Tensor,   # [B, n_actuated] or [n_actuated] — which tokens are actuated
    ) -> torch.Tensor:
        """
        Returns:
            actions: [B, action_dim] continuous actions
        """
        B = h_final.shape[0]
        
        # Extract hidden states for actuated entities
        if actuated_indices.dim() == 1:
            actuated_indices = actuated_indices.unsqueeze(0).expand(B, -1)
        
        # Gather actuated entity states: [B, n_actuated, d_model]
        h_actuated = torch.gather(
            h_final, 1,
            actuated_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        
        # Project each to action: [B, n_actuated, action_per_actuator]
        actions_per_entity = self.action_proj(h_actuated)
        
        # Flatten: [B, action_dim]
        actions = actions_per_entity.reshape(B, -1)
        
        return actions
```

### New Module: LiquidARCRoboticsModel

Create `liquid_arc/robotics_model.py`:

```python
"""LiquidARC for robotics — wraps the post-transition model for control tasks.

Architecture:
    RoboticsEmbedding → ContextPool → euler_solve(ContinuousDynamics, h₀, 16 steps) → ActionHead

The ContinuousDynamics module (MetricNet, heat kernel, LTC, FFN) is loaded from
the post-transition checkpoint. Only the embedding and action head are new.

Training strategy:
  - Phase A: Freeze dynamics, train only embedding + action head (transfer learning)
  - Phase B: Unfreeze dynamics, fine-tune end-to-end (adaptation)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .config import LiquidARCConfig
from .robotics_embedding import RoboticsEmbedding
from .action_head import ActionHead
from .context_pool import ContextPool
from .dynamics import ContinuousDynamics
from .solver import euler_solve


class LiquidARCRoboticsModel(nn.Module):
    """Robotics controller using the LiquidARC ODE substrate.
    
    Loads ContinuousDynamics from a pre-trained ARC checkpoint.
    Adds new RoboticsEmbedding and ActionHead.
    """
    
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
        
        # NEW: Robotics embedding (replaces ARCEmbedding)
        self.embedding = RoboticsEmbedding(
            d_model=d,
            max_state_dim=state_dim_per_entity,
            n_entity_types=8,
            max_entities=max(32, n_entities),
            dropout=config.dropout,
        )
        
        # REUSED: Context pool and dynamics from pre-trained checkpoint
        self.context_pool = ContextPool(config)
        self.dynamics = ContinuousDynamics(config)
        
        # NEW: Action output head (replaces color classification head)
        self.action_head = ActionHead(
            d_model=d,
            action_dim=action_dim,
            n_actuated=n_actuated,
        )
        
        # Optionally freeze the dynamics (train only embedding + action head)
        if freeze_dynamics:
            for param in self.dynamics.parameters():
                param.requires_grad = False
            for param in self.context_pool.parameters():
                param.requires_grad = False
        
        # Integration time — use T=2.0 based on T sweep finding
        # that spatial propagation tasks benefit from deeper integration
        self.integration_time = getattr(config, 'integration_time', 2.0)
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: LiquidARCConfig,
                        action_dim: int, n_entities: int, n_actuated: int,
                        state_dim_per_entity: int, freeze_dynamics: bool = True,
                        device: str = 'cuda'):
        """Load dynamics from a pre-trained ARC checkpoint, add new heads.
        
        Args:
            checkpoint_path: Path to post-transition 5M checkpoint
            config: LiquidARCConfig (should match the checkpoint's config)
            action_dim: Environment action space dimension
            n_entities: Number of tokens in the robotics scene
            n_actuated: Number of actuated entities
            state_dim_per_entity: State features per entity
            freeze_dynamics: If True, only embedding + action head train
            device: Target device
        """
        model = cls(
            config=config,
            action_dim=action_dim,
            n_entities=n_entities,
            n_actuated=n_actuated,
            state_dim_per_entity=state_dim_per_entity,
            freeze_dynamics=freeze_dynamics,
        ).to(device)
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        
        # Load only the dynamics and context_pool weights (skip embedding and output_head)
        dynamics_keys = {k: v for k, v in state_dict.items() 
                        if k.startswith('dynamics.') or k.startswith('context_pool.')}
        
        missing, unexpected = model.load_state_dict(dynamics_keys, strict=False)
        
        # Verify dynamics loaded correctly
        loaded_dynamics = [k for k in dynamics_keys.keys()]
        print(f"Loaded {len(loaded_dynamics)} dynamics/context parameters from checkpoint")
        print(f"New parameters (randomly initialized): {len(missing)}")
        
        return model
    
    def forward(
        self,
        state_features: torch.Tensor,    # [B, N, state_dim]
        spatial_x: torch.Tensor,          # [B, N]
        spatial_y: torch.Tensor,          # [B, N]
        spatial_z: torch.Tensor,          # [B, N]
        entity_types: torch.Tensor,       # [B, N]
        entity_ids: torch.Tensor,         # [B, N]
        actuated_indices: torch.Tensor,   # [n_actuated]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: observations → actions.
        
        Returns dict with 'actions' and diagnostics.
        """
        # Embed robotics state as entity tokens
        h0 = self.embedding(
            state_features, spatial_x, spatial_y, spatial_z,
            entity_types, entity_ids,
        )
        
        # Context pool (same mechanism as ARC — attention-weighted global context)
        context_mask = torch.ones(h0.shape[:2], dtype=torch.bool, device=h0.device)
        context = self.context_pool(h0, context_mask)
        self.dynamics.set_context(context, mask=None)
        self.dynamics.set_n_steps(self.config.n_ode_steps)
        
        # ODE integration — the core geometric dynamics
        # T=2.0 based on T sweep showing spatial propagation benefits from deeper T
        h_final = euler_solve(
            self.dynamics, h0,
            t_span=(0.0, self.integration_time),
            n_steps=self.config.n_ode_steps,
        )
        
        # Action readout from actuated entity tokens
        actions = self.action_head(h_final, actuated_indices)
        
        # Diagnostics
        g = self.dynamics.compute_metric(h0)
        metric_cv = g.std() / (g.mean() + 1e-8)
        tau = self.dynamics.compute_tau(h0)
        
        return {
            'actions': actions,
            'h_final': h_final,
            'metric_cv': metric_cv.item(),
            'tau_mean': tau.mean().item(),
            'tau_std': tau.std().item(),
        }
    
    def trainable_parameters(self):
        """Return only the parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def frozen_parameter_count(self):
        """Count frozen (pre-trained dynamics) parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
    
    def trainable_parameter_count(self):
        """Count trainable (new embedding + action head) parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### Isaac Lab Gym Wrapper

Create `liquid_arc/isaac_wrapper.py`:

```python
"""Wrapper connecting LiquidARC to Isaac Lab's DirectRLEnv.

Isaac Lab environments provide:
  obs: [num_envs, obs_dim] flat tensor on GPU
  
This wrapper:
  1. Tokenizes the flat observation into entity tokens
  2. Runs LiquidARC forward pass
  3. Returns actions as [num_envs, action_dim] flat tensor

The tokenization strategy depends on the specific environment.
Each environment subclass implements its own tokenizer.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class ObservationTokenizer:
    """Base class for converting flat observations to entity tokens.
    
    Subclass per environment to define how observations map to entities.
    """
    
    def __init__(self, n_entities: int, state_dim: int, n_actuated: int):
        self.n_entities = n_entities
        self.state_dim = state_dim
        self.n_actuated = n_actuated
    
    def tokenize(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert flat observation to entity token format.
        
        Args:
            obs: [B, obs_dim] flat observation from Isaac Lab
            
        Returns:
            dict with keys: state_features, spatial_x, spatial_y, spatial_z,
                           entity_types, entity_ids, actuated_indices
        """
        raise NotImplementedError


class CartpoleTokenizer(ObservationTokenizer):
    """Tokenize Cartpole observations into 2 entity tokens.
    
    Cartpole obs: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    
    Token 0 (cart): state=[cart_pos, cart_vel, 0, 0], spatial=(cart_pos, 0, 0)
    Token 1 (pole): state=[pole_angle, pole_angular_vel, 0, 0], spatial=(cart_pos, pole_height, 0)
    
    Entity types: 0=body(cart), 1=joint(pole)
    Actuated: [0] (cart — force applied to cart)
    """
    
    def __init__(self):
        super().__init__(n_entities=2, state_dim=4, n_actuated=1)
        self.pole_length = 0.5  # approximate
    
    def tokenize(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = obs.shape[0]
        device = obs.device
        
        cart_pos = obs[:, 0]
        cart_vel = obs[:, 1]
        pole_angle = obs[:, 2]
        pole_vel = obs[:, 3]
        
        # State features: [B, 2, 4]
        state_features = torch.zeros(B, 2, 4, device=device)
        state_features[:, 0, 0] = cart_pos
        state_features[:, 0, 1] = cart_vel
        state_features[:, 1, 0] = pole_angle
        state_features[:, 1, 1] = pole_vel
        
        # Spatial coordinates: [B, 2]
        # Cart is at (cart_pos, 0, 0)
        # Pole tip is at (cart_pos + sin(angle)*length, cos(angle)*length, 0)
        spatial_x = torch.zeros(B, 2, device=device)
        spatial_y = torch.zeros(B, 2, device=device)
        spatial_z = torch.zeros(B, 2, device=device)
        
        spatial_x[:, 0] = cart_pos
        spatial_x[:, 1] = cart_pos + torch.sin(pole_angle) * self.pole_length
        spatial_y[:, 0] = 0.0
        spatial_y[:, 1] = torch.cos(pole_angle) * self.pole_length
        
        # Entity types and IDs
        entity_types = torch.tensor([0, 1], device=device).unsqueeze(0).expand(B, -1)
        entity_ids = torch.tensor([0, 1], device=device).unsqueeze(0).expand(B, -1)
        
        # Actuated indices (cart is actuated)
        actuated_indices = torch.tensor([0], device=device)
        
        return {
            'state_features': state_features,
            'spatial_x': spatial_x,
            'spatial_y': spatial_y,
            'spatial_z': spatial_z,
            'entity_types': entity_types,
            'entity_ids': entity_ids,
            'actuated_indices': actuated_indices,
        }


class AnymalTokenizer(ObservationTokenizer):
    """Tokenize Anymal quadruped observations into ~13 entity tokens.
    
    Anymal obs: ~48 floats
    - Base: position (3), orientation quaternion (4), linear vel (3), angular vel (3) = 13
    - Per leg (4 legs × 3 joints): joint position (1), joint velocity (1) = 2 per joint, 24 total
    - Commands: velocity command (3)
    - Remaining: previous actions, etc.
    
    Tokens:
    - Token 0: base body (type=0)
    - Tokens 1-12: leg joints (type=1), 3 per leg × 4 legs
    
    NOTE: The exact observation layout depends on the Isaac Lab environment config.
    This tokenizer should be adapted to match the specific Anymal env's observation space.
    The agent should inspect the actual observation space and adjust accordingly.
    """
    
    def __init__(self, obs_dim: int = 48):
        super().__init__(n_entities=13, state_dim=16, n_actuated=12)
        self.obs_dim = obs_dim
    
    def tokenize(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = obs.shape[0]
        device = obs.device
        N = self.n_entities
        
        state_features = torch.zeros(B, N, self.state_dim, device=device)
        spatial_x = torch.zeros(B, N, device=device)
        spatial_y = torch.zeros(B, N, device=device)
        spatial_z = torch.zeros(B, N, device=device)
        
        # Base body — first 13 values (pos, quat, lin_vel, ang_vel)
        state_features[:, 0, :min(13, self.state_dim)] = obs[:, :13]
        spatial_x[:, 0] = obs[:, 0]  # base x
        spatial_y[:, 0] = obs[:, 1]  # base y
        spatial_z[:, 0] = obs[:, 2]  # base z
        
        # Joint tokens — 12 joints, each gets (position, velocity)
        # Layout: obs[13:25] = joint positions, obs[25:37] = joint velocities
        for j in range(12):
            token_idx = j + 1
            if 13 + j < obs.shape[1]:
                state_features[:, token_idx, 0] = obs[:, 13 + j]  # joint pos
            if 25 + j < obs.shape[1]:
                state_features[:, token_idx, 1] = obs[:, 25 + j]  # joint vel
            
            # Approximate spatial positions for legs (crude — the model will learn)
            # Front-left, front-right, rear-left, rear-right
            leg = j // 3
            leg_offsets_x = [0.3, 0.3, -0.3, -0.3]  # front/rear
            leg_offsets_y = [0.15, -0.15, 0.15, -0.15]  # left/right
            spatial_x[:, token_idx] = obs[:, 0] + leg_offsets_x[leg]
            spatial_y[:, token_idx] = obs[:, 1] + leg_offsets_y[leg]
            spatial_z[:, token_idx] = obs[:, 2] - 0.1 * ((j % 3) + 1)  # lower for distal joints
        
        entity_types = torch.zeros(B, N, dtype=torch.long, device=device)
        entity_types[:, 1:] = 1  # joints
        entity_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        actuated_indices = torch.arange(1, 13, device=device)  # all joints
        
        return {
            'state_features': state_features,
            'spatial_x': spatial_x,
            'spatial_y': spatial_y,
            'spatial_z': spatial_z,
            'entity_types': entity_types,
            'entity_ids': entity_ids,
            'actuated_indices': actuated_indices,
        }
```

## Phase 2: Training Pipeline — Cartpole First

### Training Script

Create `scripts/train_isaac.py`:

This script connects LiquidARC to Isaac Lab's training pipeline. The simplest approach: use Isaac Lab's DirectRLEnv interface with a custom policy that wraps LiquidARCRoboticsModel.

The high-level loop:

```python
"""Train LiquidARC as a robotics controller in Isaac Lab.

Usage:
    python scripts/train_isaac.py \
      --task Isaac-Cartpole-Direct-v0 \
      --checkpoint /workspace/liquid-arc/PRECIOUS_CHECKPOINTS/5m_post_transition.pt \
      --headless \
      --num_envs 1024
"""

# Pseudocode — the agent should implement the full version

import torch
import isaaclab  # Isaac Lab imports
from liquid_arc.robotics_model import LiquidARCRoboticsModel
from liquid_arc.isaac_wrapper import CartpoleTokenizer
from liquid_arc.config import LiquidARCConfig

# 1. Create Isaac Lab environment
env = isaaclab.make("Isaac-Cartpole-Direct-v0", num_envs=1024, headless=True)

# 2. Load LiquidARC with pre-trained dynamics
config = LiquidARCConfig(d_model=768, d_metric=192, d_ffn=1536, ...)  # 5M config
model = LiquidARCRoboticsModel.from_pretrained(
    checkpoint_path=args.checkpoint,
    config=config,
    action_dim=env.action_space.shape[-1],  # 1 for Cartpole
    n_entities=2,                            # cart + pole
    n_actuated=1,                            # cart
    state_dim_per_entity=4,
    freeze_dynamics=True,  # Phase A: train only embedding + action head
)

# 3. Create tokenizer
tokenizer = CartpoleTokenizer()

# 4. Simple RL loop (PPO or similar)
# The agent should integrate with one of Isaac Lab's supported RL frameworks:
# - rl_games
# - skrl
# - rsl_rl
# - stable-baselines3
#
# The integration requires a thin wrapper that makes LiquidARCRoboticsModel
# look like a standard policy network to the RL framework.
#
# For rl_games or skrl, this typically means:
# - Wrapping the model in a class that exposes act() and forward() methods
# - Converting observations through the tokenizer
# - Returning actions in the expected format

optimizer = torch.optim.Adam(model.trainable_parameters(), lr=3e-4)

obs = env.reset()
for step in range(total_steps):
    # Tokenize observations
    tokens = tokenizer.tokenize(obs)
    
    # Forward pass through LiquidARC
    result = model(**tokens)
    actions = result['actions']
    
    # Step environment
    obs, reward, done, info = env.step(actions)
    
    # PPO update (use rl_games or implement basic PPO)
    # The agent should choose the simplest working approach
    
    # Log diagnostics
    if step % 100 == 0:
        print(f"Step {step}: reward={reward.mean():.3f} "
              f"cv={result['metric_cv']:.3f} tau={result['tau_mean']:.3f}")
```

**IMPORTANT: RL Framework Integration**

The agent should determine the simplest way to integrate LiquidARCRoboticsModel with Isaac Lab's training infrastructure. Options ranked by simplicity:

1. **Direct PPO implementation** (~200 lines) — write a simple PPO loop that collects rollouts from Isaac Lab envs and updates the model. No framework dependency. Most control, most work.

2. **skrl integration** — skrl supports custom PyTorch models. Wrap LiquidARCRoboticsModel in skrl's Model interface. skrl handles PPO, rollout collection, GAE computation.

3. **rl_games integration** — Isaac Lab's default RL framework. More complex wrapping but deeply integrated with Isaac Lab.

Choose whichever the agent is most confident implementing correctly. A working simple PPO is better than a broken framework integration.

### Config for Cartpole

```yaml
# configs/isaac_cartpole.yaml

# Model (5M, same as pre-trained checkpoint)
d_model: 768
d_metric: 192
d_ffn: 1536
n_ode_steps: 16
integration_time: 2.0  # T=2.0, based on T sweep spatial propagation finding
dropout: 0.1
tau_min: 0.5
tau_max: 1.0

# Robotics-specific
n_entities: 2
n_actuated: 1
action_dim: 1
state_dim_per_entity: 4
freeze_dynamics: true  # Phase A

# RL training
num_envs: 1024
learning_rate: 0.0003
gamma: 0.99
gae_lambda: 0.95
clip_eps: 0.2
entropy_coef: 0.01
batch_size: 4096
n_epochs: 10
total_steps: 100000
```

### What to Monitor for Cartpole

```
| Step | Reward | Episode Length | CV | tau_mean | Actions std |
|------|--------|---------------|------|----------|-------------|
| 0    |        |               |      |          |             |
| 10K  |        |               |      |          |             |
| 25K  |        |               |      |          |             |
| 50K  |        |               |      |          |             |
| 100K |        |               |      |          |             |
```

**Key questions:**
- Does the policy learn to balance the pole? (episode length increasing)
- Does the metric CV shift from its pre-trained level? (geometry adapting to robotics tokens)
- Does tau adapt? (ODE viscosity changing for continuous control vs grid tasks)
- How does learning speed compare to a standard MLP policy? (the Isaac Lab Cartpole baseline trains in minutes — we need to be in the same ballpark)

## Phase 3: Scale to Locomotion (Anymal Quadruped)

After Cartpole works, move to a real robotics task:

```bash
python scripts/train_isaac.py \
  --task Isaac-Velocity-Flat-Anymal-D-Direct-v0 \
  --checkpoint [5M_POST_TRANSITION_CHECKPOINT] \
  --headless \
  --num_envs 1024 \
  --n_entities 13 \
  --n_actuated 12 \
  --action_dim 12 \
  --state_dim 16
```

Anymal is a quadruped robot with 12 actuated joints. The observation space includes base pose, joint positions, joint velocities, and velocity commands (~48 floats). The action space is 12 joint position targets.

**Why Anymal specifically:**
- 13 entity tokens (body + 12 joints) — small enough for the heat kernel to handle efficiently
- Rich spatial structure — legs need to coordinate, front/rear need to synchronize
- The heat kernel should learn that adjacent joints in the same leg are metrically close, opposite legs should synchronize
- A standard MLP policy takes ~30 minutes to train on Isaac Lab — gives us a baseline for comparison

### Phase 3 Training Protocol

**Phase A (frozen dynamics):** Train only embedding + action head. The pre-trained geometric routing handles spatial relationships between entity tokens. The FFN grows a locomotion circuit through the demonstrated partition mechanism. Monitor how fast reward improves.

**Phase B (unfrozen dynamics):** Unfreeze MetricNet and FFN. The metric adapts specifically to the robot's kinematic structure. The FFN refines the locomotion circuit with full gradient flow through the dynamics. Compare Phase A vs Phase B final performance — the delta tells you how much the pre-trained dynamics transfer vs how much task-specific adaptation is needed.

```
| Phase | Reward@10K | Reward@50K | Reward@100K | CV | tau_mean |
|-------|-----------|-----------|-------------|------|----------|
| A (frozen) |     |           |             |      |          |
| B (unfrozen) |  |           |             |      |          |
| MLP baseline | |           |             | N/A  | N/A      |
```

## Phase 4: Multi-Task Skill Acquisition

The strongest test of the architecture's demonstrated properties. After the model learns Anymal locomotion:

1. WITHOUT resetting, add a manipulation task (Isaac-Lift-Cube-Franka-v0 or similar)
2. Train on the manipulation task from the SAME checkpoint that has locomotion
3. Monitor: does locomotion performance degrade while manipulation develops?

This tests the FFN partition mechanism in a real robotics context. The prediction from all previous experiments: the FFN develops a manipulation circuit alongside the locomotion circuit, with no catastrophic forgetting.

## Success Criteria

### Phase 0 (Installation)
- Isaac Sim builds from source on the Spark
- Newton JIT compiles successfully for SM 12.1
- Cartpole MLP baseline trains to completion

### Phase 2 (Cartpole with LiquidARC)
- **Minimum:** Policy learns to balance the pole (episode length > 200 steps)
- **Good:** Converges within 2× the wall-clock time of the MLP baseline
- **Strong:** Metric CV and tau show meaningful adaptation (different from ARC pre-training values)

### Phase 3 (Anymal Locomotion)
- **Minimum:** Robot moves forward with stable gait
- **Good:** Reaches >50% of MLP baseline reward within same training budget
- **Strong:** Phase B (unfrozen) exceeds Phase A (frozen) by >20% — task-specific metric adaptation helps

### Phase 4 (Multi-Task)
- **Minimum:** Manipulation learns without resetting locomotion
- **Strong:** Both tasks perform within 90% of their single-task best simultaneously

## Output

Report to `shared/outbox/ISAAC_SIM_REPORT.md`

Include:
1. Installation notes (any Spark-specific issues, compilation times)
2. Cartpole results with LiquidARC vs MLP baseline
3. Metric CV and tau evolution during robotics training
4. Anymal locomotion results (Phase A frozen vs Phase B unfrozen)
5. Multi-task results (if reached)
6. Assessment: does the universal geometric substrate transfer to continuous robotics control?
7. Training throughput: steps/second with LiquidARC vs standard MLP
8. Any torch.compile issues with the robotics pipeline

**This experiment moves LiquidARC from proof-of-concept into its intended domain. The architecture was designed for continuous-time dynamics in physical systems. Isaac Sim provides the physical system. The post-transition model provides the pre-trained geometric substrate. Everything established by the prior experiments predicts this will work. Time to test it.**
