# TASK: Continuous Lifecycle Runtime — LiquidARC as an Always-On Dynamical System

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-03-29
**Priority:** HIGH — architectural paradigm shift

**Prerequisites:** Read `shared/outbox/ISAAC_SIM_REPORT.md` and `shared/outbox/CURIOSITY_EXPLORATION_REPORT.md` (when available). This builds on the Anymal walking result (reward -11.2, CV 14, developmental staging) and the curiosity exploration findings (log_std increasing, exploration preventing premature convergence).

---

## The Paradigm Shift

Every experiment so far has treated LiquidARC as a REQUEST/RESPONSE system:
```
observation arrives → embed → run 16 ODE steps → read output → model dies → repeat
```

The ODE exists only when called. Between calls, there is no model — just frozen weights.

This spec implements a CONTINUOUS LIFECYCLE:
```
ODE runs always: dh/dt = -(1/τ)(h - f(h))
observation arrives → sensory forcing perturbs ongoing h
action needed → read from current h
between events → h evolves autonomously (consolidation, prediction, reorganization)
```

The model has its own temporal existence. Observations steer it. Actions are read from it. But the dynamics are self-sustaining — the model thinks between observations.

## Why Now

The evidence supporting this:

1. **The 60-70% ceiling is environmental, not architectural.** The model saturates on impoverished input. Continuous dynamics between observations provide SELF-GENERATED information through autonomous consolidation.

2. **Two phase transitions on Anymal.** The model self-organizes through developmental stages in rich environments. Continuous internal processing provides more opportunity for geometric reorganization.

3. **Curiosity-driven exploration works.** The ||dh/dt|| norm is a meaningful signal — states with turbulent dynamics are informative. In continuous mode, ||dh/dt|| is always available, not computed once per forward pass.

4. **The architecture was designed for this.** LTC dynamics with learned tau are INHERENTLY continuous. The Euler solver discretizes what should be a continuous process. This spec removes the discretization between observations while keeping it within each integration segment (for torch.compile compatibility).

5. **Isaac Sim provides the natural clock.** Physics advances at 120Hz. The ODE integrates between physics steps. The physics engine IS the external world — dense, causal, continuous.

## Architecture

### Core: ContinuousLifecycleRunner

Create `liquid_arc/lifecycle.py`:

```python
"""Continuous lifecycle runtime for LiquidARC in Isaac Sim.

The ODE runs as a persistent process. Observations inject as sensory
forcing. Actions are read continuously. Between observations, the
dynamics evolve autonomously — consolidation, prediction, internal
reorganization.

The key equation changes from:
  dh/dt = -(1/τ)(h - f(h))           [standard, autonomous]
to:
  dh/dt = -(1/τ)(h - f(h)) + F(t)    [forced, with sensory input]

where F(t) = β · (embed(obs) - h) when an observation arrives,
      F(t) = 0 between observations.

The forcing term β · (embed(obs) - h) is predictive coding:
  - When the model's state h already matches the observation embedding,
    forcing is zero (prediction confirmed, no update needed)
  - When h diverges from the observation, forcing is large
    (prediction error, state corrected toward sensory input)

β is per-entity, learned alongside tau:
  - High β entities: trust sensory input (reactive, fast correction)
  - Low β entities: trust internal prediction (contextual, slow drift)
  - The model learns its own trust calibration per entity

Training uses SEGMENTS: collect K physics steps of continuous dynamics,
compute loss on the segment, backprop through the segment only.
The state h carries forward (detached) between segments, but gradients
don't flow across segment boundaries. Same principle as truncated BPTT.
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
    """Computes the forcing function F(t) = β · (obs_embed - h).
    
    β is per-entity, learned. Controls how strongly each entity token
    is pulled toward the new observation vs maintaining its internal state.
    
    High β: reactive (foot tokens — need fast contact response)
    Low β: contextual (body token — maintains overall state estimate)
    
    Args:
        d_model: Hidden dimension (768)
        n_entities: Max number of entity tokens
    """
    
    def __init__(self, d_model: int = 768, n_entities: int = 32):
        super().__init__()
        # Per-entity beta, learned via sigmoid for [0, beta_max] range
        self.beta_logits = nn.Parameter(torch.zeros(n_entities))
        self.beta_max = 2.0  # maximum forcing strength
        
        # Initialize: sigmoid(0) = 0.5 → β = 1.0 (moderate forcing)
        nn.init.zeros_(self.beta_logits)
    
    @property
    def beta(self) -> torch.Tensor:
        """Per-entity forcing strength, [n_entities]."""
        return self.beta_max * torch.sigmoid(self.beta_logits)
    
    def compute_forcing(
        self,
        h_current: torch.Tensor,     # [B, N, d] current ODE state
        obs_embed: torch.Tensor,      # [B, N, d] embedded observation
    ) -> torch.Tensor:
        """Compute sensory forcing: β · (obs_embed - h).
        
        Returns:
            forcing: [B, N, d] additive term for the dynamics
        """
        N = h_current.shape[1]
        beta = self.beta[:N].unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        return beta * (obs_embed - h_current)
    
    def get_prediction_error(
        self,
        h_current: torch.Tensor,
        obs_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Prediction error: ||obs_embed - h||² per entity.
        
        This is the continuous curiosity signal — how surprised the model
        is by the new observation given its internal prediction.
        
        Returns:
            error: [B, N] per-entity prediction error
        """
        return (obs_embed - h_current).detach().norm(dim=-1)


class ContinuousLifecycleRunner(nn.Module):
    """Always-on dynamical system connected to Isaac Sim.
    
    Lifecycle:
        1. Initialize h from first observation
        2. For each physics step:
           a. Inject observation as sensory forcing
           b. Run K internal ODE steps (K = internal_steps_per_obs)
           c. Read action from current h
        3. Between observations, h evolves autonomously
        4. Training: collect segments, backprop per segment
    
    The ODE state h is PERSISTENT — it carries across physics steps,
    across training updates, across episodes. Only reset at episode
    boundaries (robot falls, timeout).
    
    Args:
        config: LiquidARCConfig (5M model config)
        n_entities: Number of entity tokens
        n_actuated: Number of actuated entities
        action_dim: Action space dimension
        state_dim: State features per entity
        internal_steps: ODE steps between observations (default 16)
        autonomous_steps: Extra ODE steps without forcing (default 0)
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
        
        # Core dynamics (from pre-trained checkpoint)
        self.dynamics = ContinuousDynamics(config)
        self.context_pool = ContextPool(config)
        
        # Sensory interface
        self.embedding = RoboticsEmbedding(
            d_model=d,
            max_state_dim=state_dim,
            n_entity_types=8,
            max_entities=max(32, n_entities),
            dropout=config.dropout,
        )
        self.forcing = SensoryForcing(d_model=d, n_entities=n_entities)
        
        # Motor interface
        self.action_head = ActionHead(
            d_model=d,
            action_dim=action_dim,
            n_actuated=n_actuated,
        )
        
        # Integration time per observation cycle
        self.T_per_obs = getattr(config, 'integration_time', 2.0)
        
        # Persistent ODE state — this IS the model's existence
        # Initialized to None, set on first observation
        self._h: Optional[torch.Tensor] = None
        
        # Diagnostics accumulators
        self._curiosity_accum = 0.0
        self._dynamics_norm_accum = 0.0
        self._steps_since_reset = 0
        
        if freeze_dynamics:
            for param in self.dynamics.parameters():
                param.requires_grad = False
            for param in self.context_pool.parameters():
                param.requires_grad = False
    
    @property
    def is_alive(self) -> bool:
        """Whether the model has an active ODE state."""
        return self._h is not None
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: LiquidARCConfig,
                        n_entities: int, n_actuated: int, action_dim: int,
                        state_dim: int, internal_steps: int = 16,
                        autonomous_steps: int = 0,
                        freeze_dynamics: bool = False,
                        device: str = 'cuda'):
        """Load dynamics from pre-trained checkpoint."""
        model = cls(
            config=config, n_entities=n_entities, n_actuated=n_actuated,
            action_dim=action_dim, state_dim=state_dim,
            internal_steps=internal_steps, autonomous_steps=autonomous_steps,
            freeze_dynamics=freeze_dynamics,
        ).to(device)
        
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        dynamics_keys = {k: v for k, v in state_dict.items()
                        if k.startswith('dynamics.') or k.startswith('context_pool.')}
        model.load_state_dict(dynamics_keys, strict=False)
        print(f"Loaded {len(dynamics_keys)} dynamics parameters from checkpoint")
        return model
    
    def reset(self, batch_size: int, device: torch.device):
        """Reset ODE state — called at episode boundaries.
        
        The model is BORN. h starts as zeros — no prior state.
        The first observation will inject the initial state through
        sensory forcing, and the dynamics will begin evolving.
        """
        self._h = torch.zeros(
            batch_size, self.n_entities, self.config.d_model,
            device=device,
        )
        self._curiosity_accum = 0.0
        self._dynamics_norm_accum = 0.0
        self._steps_since_reset = 0
    
    def _run_ode_segment(
        self,
        h: torch.Tensor,
        n_steps: int,
        forcing: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run n_steps of ODE integration, optionally with forcing.
        
        This is the inner loop that torch.compile optimizes.
        The forcing is applied as an additive term at each step.
        
        Args:
            h: [B, N, d] current state
            n_steps: number of Euler steps
            forcing: [B, N, d] sensory forcing (or None for autonomous)
            
        Returns:
            h_final: [B, N, d] state after integration
            dynamics_norm: [B] average ||dh/dt|| across steps (curiosity)
        """
        T = self.T_per_obs
        dt = T / n_steps
        t = 0.0
        
        dynamics_norm_accum = torch.zeros(h.shape[0], device=h.device)
        
        for i in range(n_steps):
            if hasattr(self.dynamics, 'set_step_embed'):
                self.dynamics.set_step_embed(i, n_steps)
            if hasattr(self.dynamics, 'set_step_index'):
                self.dynamics.set_step_index(i, n_steps)
            
            # Core dynamics
            dy = self.dynamics(t, h)
            
            # Add sensory forcing if present
            if forcing is not None:
                # Decay forcing over the integration window
                # Full forcing at step 0, decaying to 0 by step n_steps
                # This models "observation arrives, then fades as model processes it"
                decay = 1.0 - (i / n_steps)
                dy = dy + decay * forcing
            
            # Accumulate dynamics norm (curiosity signal)
            dynamics_norm_accum = dynamics_norm_accum + dy.detach().norm(dim=-1).mean(dim=-1)
            
            h = h + dt * dy
            t = t + dt
        
        dynamics_norm = dynamics_norm_accum / n_steps
        return h, dynamics_norm
    
    def step(
        self,
        obs_tokens: Dict[str, torch.Tensor],
        actuated_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Process one physics timestep.
        
        1. Embed the observation
        2. Compute sensory forcing (prediction error between h and obs)
        3. Run ODE with forcing (internal_steps)
        4. Optionally run autonomous ODE (autonomous_steps)
        5. Read action from current h
        
        Args:
            obs_tokens: Dict from tokenizer (state_features, spatial_*, etc.)
            actuated_indices: Which entity tokens produce actions
            
        Returns:
            dict with 'actions', 'curiosity', 'prediction_error', diagnostics
        """
        device = obs_tokens['state_features'].device
        B = obs_tokens['state_features'].shape[0]
        
        # Initialize if first step
        if self._h is None:
            self.reset(B, device)
        
        # Handle batch size changes (e.g., env reset with different count)
        if self._h.shape[0] != B:
            self.reset(B, device)
        
        # 1. Embed the observation
        obs_embed = self.embedding(
            obs_tokens['state_features'],
            obs_tokens['spatial_x'],
            obs_tokens['spatial_y'],
            obs_tokens['spatial_z'],
            obs_tokens['entity_types'],
            obs_tokens['entity_ids'],
        )
        
        # 2. Compute sensory forcing and prediction error
        # Prediction error: how surprised is the model by this observation?
        prediction_error = self.forcing.get_prediction_error(self._h, obs_embed)
        forcing = self.forcing.compute_forcing(self._h, obs_embed)
        
        # 3. Context from current state (not fresh embedding)
        context_mask = torch.ones(B, self.n_entities, dtype=torch.bool, device=device)
        context = self.context_pool(self._h, context_mask)
        self.dynamics.set_context(context, mask=None)
        self.dynamics.set_n_steps(self.internal_steps)
        
        # 4. ODE integration WITH sensory forcing
        h_after_obs, obs_curiosity = self._run_ode_segment(
            self._h, self.internal_steps, forcing=forcing,
        )
        
        # 5. Autonomous processing (if configured) — no forcing
        if self.autonomous_steps > 0:
            # Update context from post-observation state
            context = self.context_pool(h_after_obs, context_mask)
            self.dynamics.set_context(context, mask=None)
            self.dynamics.set_n_steps(self.autonomous_steps)
            
            h_after_auto, auto_curiosity = self._run_ode_segment(
                h_after_obs, self.autonomous_steps, forcing=None,
            )
            total_curiosity = (obs_curiosity + auto_curiosity) / 2
        else:
            h_after_auto = h_after_obs
            total_curiosity = obs_curiosity
        
        # 6. Read action from current state
        actions = self.action_head(h_after_auto, actuated_indices)
        
        # 7. Update persistent state (detached for next step)
        self._h = h_after_auto.detach()
        
        # Diagnostics
        g = self.dynamics.compute_metric(h_after_auto.detach())
        metric_cv = g.std() / (g.mean() + 1e-8)
        tau = self.dynamics.compute_tau(h_after_auto.detach())
        
        self._steps_since_reset += 1
        
        return {
            'actions': actions,
            'curiosity': total_curiosity,          # [B] intrinsic motivation
            'prediction_error': prediction_error,   # [B, N] per-entity surprise
            'h_state': h_after_auto,                # [B, N, d] for training
            'metric_cv': metric_cv.item(),
            'tau_mean': tau.mean().item(),
            'beta': self.forcing.beta[:self.n_entities].detach(),  # [N] learned forcing strengths
            'steps_alive': self._steps_since_reset,
        }
    
    def handle_resets(self, reset_mask: torch.Tensor):
        """Reset ODE state for specific environments that terminated.
        
        Args:
            reset_mask: [B] bool — True for environments that need reset
        """
        if self._h is not None and reset_mask.any():
            self._h[reset_mask] = 0.0
    
    def get_autonomous_state(self) -> Dict[str, torch.Tensor]:
        """Read the model's current internal state without providing observation.
        
        Useful for monitoring what the model is 'thinking about' between events.
        """
        if self._h is None:
            return {}
        h_norm = self._h.norm(dim=-1)  # [B, N]
        return {
            'h_norm_mean': h_norm.mean().item(),
            'h_norm_std': h_norm.std().item(),
            'h_norm_per_entity': h_norm.mean(dim=0),  # [N]
        }
```

### Training Script: Segment-Based PPO

Create `scripts/train_lifecycle.py`:

The key difference from `train_isaac.py`: the model's state persists across PPO rollout steps. Each rollout step is one call to `model.step()`, not a fresh forward pass. Gradients flow through the segment (rollout_length steps), then the state is detached at the segment boundary.

```python
"""Train LiquidARC continuous lifecycle in Isaac Sim.

Unlike train_isaac.py (discrete forward passes), this script maintains
persistent ODE state across physics timesteps. The model is always alive.

Training uses segment-based PPO:
  1. Collect rollout_length steps of experience (model.step() per physics step)
  2. h carries forward WITHIN the segment (gradients flow)
  3. At segment boundary, detach h (truncated BPTT)
  4. Compute advantages, update with PPO

The crucial difference: within a segment, the model's internal dynamics
evolve continuously. The same h that processed observation at step K
influences the action at step K+1, K+2, ... K+rollout_length.

Usage:
    python scripts/train_lifecycle.py \
      --task Isaac-Velocity-Flat-Anymal-D-Direct-v0 \
      --checkpoint [5M_POST_TRANSITION_CHECKPOINT] \
      --headless \
      --num_envs 1024 \
      --internal_steps 16 \
      --autonomous_steps 4 \
      --rollout_length 32
"""

# Implementation outline — the agent should build the full version:

# 1. Create environment (same as train_isaac.py)
# 2. Create ContinuousLifecycleRunner.from_pretrained(...)
# 3. Create tokenizer (AnymalTokenizer)
# 4. Create PPO infrastructure (value net, optimizer, rollout buffer)

# MAIN LOOP:
# for update in range(total_updates):
#     
#     # --- ROLLOUT COLLECTION ---
#     rollout_buffer.reset()
#     
#     for step in range(rollout_length):
#         obs = env.get_observations()
#         tokens = tokenizer.tokenize(obs)
#         
#         with torch.no_grad():  # No grad during rollout collection
#             result = model.step(tokens, actuated_indices)
#         
#         actions = result['actions']
#         curiosity = result['curiosity']
#         
#         # Step environment
#         obs_next, reward, done, info = env.step(actions.clamp(-10, 10))
#         
#         # Combine rewards
#         intrinsic = curiosity_normalizer.normalize(curiosity)
#         total_reward = reward + beta * intrinsic
#         
#         # Store transition
#         rollout_buffer.add(obs, actions, total_reward, done, 
#                           result['h_state'], result['prediction_error'])
#         
#         # Handle episode resets
#         if done.any():
#             model.handle_resets(done)
#     
#     # --- PPO UPDATE ---
#     # Replay the segment WITH gradients for policy update
#     # This is where the continuous dynamics matter:
#     # we replay the sequence through the model to get differentiable
#     # actions, then compute PPO loss
#     
#     for epoch in range(ppo_epochs):
#         # Reset model state to start of segment
#         # (use stored h from rollout buffer's first step)
#         model._h = rollout_buffer.initial_h.detach()
#         
#         for step in range(rollout_length):
#             tokens = tokenizer.tokenize(rollout_buffer.obs[step])
#             result = model.step(tokens, actuated_indices)
#             
#             # PPO loss on this step
#             # ... standard PPO: ratio, clipped surrogate, value loss, entropy
#         
#         # Gradient step
#         optimizer.step()
#         optimizer.zero_grad()
#     
#     # --- SEGMENT BOUNDARY ---
#     # Detach h for next segment (truncated BPTT)
#     model._h = model._h.detach()
#     
#     # --- LOGGING ---
#     log(update, reward, ep_len, cv, tau, curiosity, prediction_error, beta)
```

### Rollout Buffer for Continuous State

The standard PPO rollout buffer stores (obs, action, reward, done, log_prob, value). The lifecycle version additionally stores:

```python
class LifecycleRolloutBuffer:
    """Rollout buffer that preserves continuous ODE state.
    
    Stores the model's h at each step so the segment can be replayed
    with gradients during PPO updates.
    """
    
    def __init__(self, rollout_length, num_envs, obs_dim, action_dim, 
                 n_entities, d_model, device):
        self.rollout_length = rollout_length
        
        # Standard PPO fields
        self.obs = torch.zeros(rollout_length, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(rollout_length, num_envs, action_dim, device=device)
        self.rewards = torch.zeros(rollout_length, num_envs, device=device)
        self.dones = torch.zeros(rollout_length, num_envs, dtype=torch.bool, device=device)
        self.log_probs = torch.zeros(rollout_length, num_envs, device=device)
        self.values = torch.zeros(rollout_length, num_envs, device=device)
        
        # Lifecycle-specific: h at segment start (for replay)
        self.initial_h = None  # [num_envs, n_entities, d_model]
        
        # Diagnostics
        self.curiosity = torch.zeros(rollout_length, num_envs, device=device)
        self.prediction_error = torch.zeros(rollout_length, num_envs, n_entities, device=device)
```

## Key Design Decisions

### 1. Forcing Decay Within Integration Window

When an observation arrives, the forcing `F = β(obs_embed - h)` is applied with a LINEAR DECAY over the internal_steps:

```
step 0: F * 1.0    (full forcing — observation just arrived)
step 8: F * 0.5    (half forcing — processing)
step 15: F * 0.0   (no forcing — fully assimilated)
```

This models the temporal profile of sensory processing: the observation arrives as a sharp impulse, then fades as the model integrates it into its internal state. By step 16, the model has fully processed the observation and is running on pure internal dynamics.

The decay is a LINEAR ramp, not learned, to keep the compiled graph simple. The STRENGTH of forcing is learned (per-entity β), but the TEMPORAL PROFILE is fixed.

### 2. Autonomous Steps (Optional Processing Time)

`autonomous_steps` controls how many extra ODE steps run AFTER the forcing has decayed — pure internal dynamics with no sensory input. This is the model's "thinking time."

Default: autonomous_steps=0 (no extra processing, same total compute as discrete mode).

Experiment values: 0, 4, 8, 16. More autonomous steps = more internal consolidation between observations. The prediction: some positive value of autonomous_steps will improve performance because the model uses the extra time to propagate information through the heat kernel, organize its internal state, and stabilize its action predictions.

This is the direct test of the hypothesis: does the model benefit from internal processing time between observations? If autonomous_steps > 0 outperforms autonomous_steps = 0, the model is doing useful autonomous computation.

### 3. State Persistence and Episode Boundaries

The ODE state h persists across ALL physics steps within an episode. At episode boundaries (robot falls, timeout), h is reset to zeros for the specific environments that terminated (`handle_resets(done)`).

Between PPO update segments, h also persists (but detached — no cross-segment gradients). The model's state is continuous across the entire episode, not just within 32-step rollout segments.

This means: the model at physics step 1000 carries information from physics step 1 through 1000 consecutive ODE integrations with sensory forcing at each step. The tau parameter controls how fast old information decays — high-tau entities remember far back, low-tau entities are reactive.

### 4. Gradient Flow: Within-Segment Only

Gradients flow through the rollout_length steps of each segment. Between segments, h is detached. This is truncated BPTT with a window of rollout_length steps.

This is a DESIGN CHOICE, not a limitation. Cross-segment gradients would require storing the full computation graph across the entire episode (~1000 steps × 16 ODE steps = 16,000 dynamics evaluations). Truncated BPTT with 32-step segments keeps memory bounded while providing enough temporal gradient flow for the policy to learn multi-step consequences.

### 5. Context Pool from Current State, Not Fresh Embedding

In the discrete model, context is computed from h₀ (the fresh embedding). In the lifecycle model, context is computed from the CURRENT h (the ongoing ODE state). This means the context vector reflects the model's accumulated understanding, not just the latest observation. The context_pool attention mechanism reads from a state that has been shaped by hundreds of previous observations — providing richer global context.

## Experimental Protocol

### Experiment 1: Lifecycle vs Discrete on Anymal

**Condition A: Discrete (baseline)** — the current train_isaac.py with LiquidARCRoboticsModel. Each observation → fresh embed → 16 ODE steps → action. No state persistence.

**Condition B: Lifecycle (internal_steps=16, autonomous_steps=0)** — ContinuousLifecycleRunner. Same total ODE steps per observation (16), but state persists across observations through sensory forcing. No extra compute.

**Condition C: Lifecycle + Autonomous (internal_steps=16, autonomous_steps=8)** — Same as B but with 8 additional autonomous ODE steps per observation. 50% more compute, but the model gets pure thinking time.

All three from the same checkpoint, unfrozen dynamics, same PPO hyperparameters, same environment, 2M steps.

```bash
# Condition A: Discrete
python scripts/train_isaac.py \
  --task Isaac-Velocity-Flat-Anymal-D-Direct-v0 \
  --checkpoint [5M_POST_TRANSITION_CHECKPOINT] \
  --headless --num_envs 1024 \
  --freeze_dynamics false --total_steps 2000000 \
  --output_dir output_lifecycle/discrete

# Condition B: Lifecycle (same compute budget)
python scripts/train_lifecycle.py \
  --task Isaac-Velocity-Flat-Anymal-D-Direct-v0 \
  --checkpoint [5M_POST_TRANSITION_CHECKPOINT] \
  --headless --num_envs 1024 \
  --freeze_dynamics false --total_steps 2000000 \
  --internal_steps 16 --autonomous_steps 0 \
  --output_dir output_lifecycle/continuous

# Condition C: Lifecycle + Autonomous (50% more compute)
python scripts/train_lifecycle.py \
  --task Isaac-Velocity-Flat-Anymal-D-Direct-v0 \
  --checkpoint [5M_POST_TRANSITION_CHECKPOINT] \
  --headless --num_envs 1024 \
  --freeze_dynamics false --total_steps 2000000 \
  --internal_steps 16 --autonomous_steps 8 \
  --output_dir output_lifecycle/continuous_auto8
```

### Experiment 2: Autonomous Steps Sweep

After Experiment 1, if lifecycle mode works, sweep autonomous_steps:

| autonomous_steps | Total ODE steps/obs | Extra compute | Expected effect |
|---|---|---|---|
| 0 | 16 | 0% | Baseline lifecycle |
| 4 | 20 | 25% | Mild consolidation |
| 8 | 24 | 50% | Moderate consolidation |
| 16 | 32 | 100% | Full thinking time (= double compute) |

### Experiment 3: Beta Analysis (Learned Forcing Strengths)

After training, inspect the learned beta values per entity:

```
| Entity | Type | Learned β | Interpretation |
|--------|------|-----------|---------------|
| 0 (body) | base | | High β = reactive / Low β = contextual |
| 1 (FL hip) | joint | | |
| 2 (FL thigh) | joint | | |
| 3 (FL shin) | joint | | |
| ... | | | |
| 12 (RR shin) | joint | | |
```

**Hypothesis:** Foot/shin tokens will have HIGH β (need fast contact response). Body token will have LOW β (maintains overall state estimate). Hip tokens will be intermediate (coordinate between body context and leg reactivity).

If the model learns this hierarchy autonomously, it has discovered the robot's sensory trust structure through gradient descent — which entities need fast sensory updates and which should maintain internal predictions.

### Experiment 4: Curiosity Integration

Add intrinsic motivation (from the CURIOSITY_EXPLORATION spec) to the lifecycle runner:

```bash
python scripts/train_lifecycle.py \
  --task Isaac-Velocity-Flat-Anymal-D-Direct-v0 \
  --checkpoint [5M_POST_TRANSITION_CHECKPOINT] \
  --headless --num_envs 1024 \
  --freeze_dynamics false --total_steps 3000000 \
  --internal_steps 16 --autonomous_steps 8 \
  --intrinsic_beta 0.5 --beta_decay_fraction 0.5 \
  --output_dir output_lifecycle/continuous_curious
```

The lifecycle curiosity is RICHER than discrete curiosity. In discrete mode, curiosity = average ||dh/dt|| over 16 steps of a fresh forward pass. In lifecycle mode, curiosity includes the prediction error from sensory forcing (how surprised the model was by the observation) PLUS the dynamics turbulence during integration (how much internal reorganization the observation triggered). The model can be curious about observations that don't match its predictions, OR about internal states that are turbulent even when observations are expected.

## What to Monitor

### Primary: Reward and Episode Length Comparison

```
| Update | Discrete Rwd | Lifecycle Rwd | Life+Auto Rwd | Discrete EpLen | Lifecycle EpLen | Life+Auto EpLen |
|--------|-------------|--------------|---------------|----------------|----------------|-----------------|
| 20     |             |              |               |                |                |                 |
| 40     |             |              |               |                |                |                 |
| 65     |             |              |               |                |                |                 |
| 100    |             |              |               |                |                |                 |
| 200    |             |              |               |                |                |                 |
| 320    |             |              |               |                |                |                 |
```

### Secondary: Learned Beta Values

```
| Entity | Type | β @step0 | β @step1M | β @step2M | Trajectory |
|--------|------|----------|-----------|-----------|-----------|
| body | base | 1.0 | | | |
| FL_hip | joint | 1.0 | | | |
| FL_shin | joint | 1.0 | | | |
| ... | | | | | |
```

### Tertiary: Prediction Error as Development Indicator

```
| Update | Pred Error (mean) | Pred Error (body) | Pred Error (feet) | CV | Phase |
|--------|------------------|-------------------|-------------------|----|-------|
| 0 | | | | | |
| 20 | | | | | |
| ...| | | | | |
```

Prediction error should DECREASE as the model's internal dynamics become better predictors of the next observation. Spikes in prediction error should correlate with phase transitions (CV shifts) — the model's predictions become temporarily worse as the geometry reorganizes.

### Quaternary: Autonomous Processing Value

For Experiment 2 (autonomous steps sweep), measure whether extra thinking time helps:

```
| auto_steps | Reward @1M | Reward @2M | FPS | Worth it? |
|---|---|---|---|---|
| 0 | | | | baseline |
| 4 | | | | |
| 8 | | | | |
| 16 | | | | |
```

## Success Criteria

### Experiment 1 (Lifecycle vs Discrete)

**Minimum:** Lifecycle mode (B) matches discrete mode (A) in reward and episode length. Persistence doesn't hurt — the continuous dynamics are at least as good as fresh-start dynamics.

**Good:** Lifecycle mode outperforms discrete mode by ≥10% in reward at the same update budget. Persistent state provides useful temporal context that fresh embedding doesn't.

**Strong:** Lifecycle mode reaches locomotion onset EARLIER than discrete mode. The persistent dynamics allow the model to accumulate locomotion-relevant experience across physics steps, bypassing the standing plateau.

**Headline:** Lifecycle + autonomous steps (C) achieves positive reward (genuine walking with velocity tracking) where discrete mode remains at negative reward (standing with penalty). The autonomous processing time enables the model to develop locomotion-quality internal representations that discrete per-observation processing can't.

### Experiment 3 (Beta Analysis)

**Success:** The model learns a NON-UNIFORM beta distribution — different forcing strengths for different entity types. This demonstrates autonomous sensory trust calibration.

**Strong success:** The learned beta hierarchy matches the robot's physical structure (feet > body) — the model discovered biomechanically meaningful trust calibration from reward signal alone.

## Implementation Notes

### torch.compile Compatibility

The `_run_ode_segment` method must compile cleanly. Key constraints:
- `forcing` is always [B, N, d] or None — use a zero tensor instead of None to avoid control flow branching in the compiled path
- `decay = 1.0 - (i / n_steps)` is a float computed from loop index — same pattern as existing dt computation, should compile
- `dynamics_norm_accum` grows by addition each step — same pattern as existing solvers

**If compile issues arise:** Factor `_run_ode_segment` into two compiled functions — one with forcing, one without. Call the appropriate one based on whether forcing is needed. The branch happens OUTSIDE the compiled function.

### Memory Considerations

Persistent h: [1024, 13, 768] = 40MB — negligible.

Rollout buffer h storage: [32, 1024, 13, 768] = 1.3GB — significant but within the Spark's 128GB. If memory is tight, store only the initial_h per segment (40MB) and replay the segment during PPO updates.

### Episode Reset Handling

When `done[i] = True`, the i-th environment in the batch is reset. The model must zero out h[i] while preserving h[j] for all j ≠ i. This is a simple masked assignment:

```python
def handle_resets(self, done):
    if self._h is not None and done.any():
        self._h[done] = 0.0
```

This happens OUTSIDE the compiled ODE — no compile interaction.

### Rollout Replay Strategy

During PPO updates, the segment is replayed WITH gradients. This means running the lifecycle model's step() function rollout_length times with gradient tracking. The cost: rollout_length × (internal_steps + autonomous_steps) dynamics evaluations, with full autograd.

For rollout_length=32, internal_steps=16, autonomous_steps=8: 32 × 24 = 768 dynamics evaluations per PPO epoch. With 4 PPO epochs: 3,072 evaluations per update. At 321 fps (from the compiled Anymal run), this is ~10 seconds per update. Viable for research.

If too slow, reduce rollout_length to 16 or reduce PPO epochs to 2.

## Output

Report to `shared/outbox/LIFECYCLE_REPORT.md`

Include:
1. Lifecycle vs discrete reward/episode length comparison
2. Learned beta values per entity (with biomechanical interpretation)
3. Prediction error trajectory and correlation with CV shifts
4. Autonomous steps sweep results
5. Phase transition timing comparison (lifecycle vs discrete)
6. Curiosity integration results (if run)
7. torch.compile stability notes
8. Assessment: does continuous internal processing improve robotics control?
9. Assessment: does the model learn meaningful sensory trust calibration?
10. FPS comparison across conditions

**This is the experiment that determines whether LiquidARC is a computation-on-demand function or a continuously existing dynamical system. If the lifecycle mode outperforms discrete mode — especially with autonomous processing steps — the model benefits from its own internal temporal existence. That's the foundation for the autonomous agentic substrate.**
