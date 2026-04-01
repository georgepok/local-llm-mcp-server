# TASK: Intrinsic Motivation — Active Exploration for LiquidARC in Isaac Sim

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-03-29
**Priority:** HIGH — next phase of Isaac Sim integration

**Prerequisites:** Read `shared/outbox/ISAAC_SIM_REPORT.md` for full context. This builds on the Anymal Phase 3b results (unfrozen dynamics, two phase transitions, NaN crash at CV 8.3).

---

## The Problem

The current PPO training loop is PASSIVE. The environment generates rollouts from the current policy, and the model learns from whatever states the policy happens to visit. On Anymal, this produced a degenerate pattern: the policy learned to balance early, then spent thousands of updates generating near-static standing states while starving itself of the locomotion experience it needed.

The two phase transitions emerged DESPITE this — the model stumbled into enough novel states through PPO's stochastic exploration (entropy bonus, action noise) to trigger geometric reorganization. But this was accidental. The second transition was on the verge of unlocking locomotion when it crashed. With active exploration, the model would SEEK the states that drive geometric reorganization, making phase transitions faster, more frequent, and more targeted.

## The Insight: Prediction Error as Intrinsic Motivation

The LiquidARC dynamics equation is:

```
dh/dt = -(1/τ)(h - target) + FFN(h) / n_steps
```

At each ODE step, the model computes a TARGET state and contracts h toward it. The magnitude of this contraction — `||h - target||` — is the model's own prediction error. When the model understands the current situation well, h is already close to target (low dh/dt, low prediction error). When the model encounters something it doesn't understand, h diverges from target (high dh/dt, high prediction error).

This prediction error is computed INSIDE the compiled ODE graph at every step. It requires zero additional computation — just reading a quantity that's already being computed.

**The intrinsic reward is the average prediction error across ODE steps.** States where the model's internal dynamics are turbulent (high dh/dt norm) are states where the model has the most to learn. Seeking these states is curiosity. Avoiding already-understood states is efficient learning.

## Architecture: Prediction Error Extraction

### Modify euler_solve to Return Prediction Error

Create a new solver variant that tracks the dynamics magnitude at each step:

```python
def euler_solve_with_curiosity(fn, y0, t_span, n_steps):
    """Forward Euler that also returns average ||dh/dt|| as curiosity signal.
    
    The curiosity signal is the mean norm of the dynamics across all ODE steps.
    High curiosity = the model's internal prediction is far from the current state.
    Low curiosity = the model has converged, it understands this input well.
    
    This adds ZERO extra computation — dh/dt is already computed for the Euler update.
    We just accumulate its norm.
    
    Returns:
        y_final: [B, N, d] final hidden state (same as euler_solve)
        curiosity: [B] scalar per batch item — average ||dh/dt|| across steps
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps
    t = t_start
    y = y0
    
    curiosity_accum = torch.zeros(y0.shape[0], device=y0.device)  # [B]
    
    for i in range(n_steps):
        if hasattr(fn, 'set_step_embed'):
            fn.set_step_embed(i, n_steps)
        if hasattr(fn, 'set_step_index'):
            fn.set_step_index(i, n_steps)
        
        dy = fn(t, y)  # dh/dt — already computed
        
        # Accumulate curiosity: mean ||dh/dt|| across positions and features
        # dy is [B, N, d], take norm over (N, d) to get per-batch scalar
        step_curiosity = dy.detach().norm(dim=-1).mean(dim=-1)  # [B]
        curiosity_accum = curiosity_accum + step_curiosity
        
        y = y + dt * dy
        t = t + dt
    
    curiosity = curiosity_accum / n_steps  # average across steps
    return y, curiosity
```

**CRITICAL:** The `dy.detach()` ensures curiosity computation doesn't affect the ODE's gradient flow. The curiosity signal is a READOUT, not an intervention. The dynamics run identically to the standard solver.

**torch.compile compatibility:** This adds one `.detach().norm().mean()` per ODE step — all standard ops on tensors of fixed shape. The accumulated `curiosity_accum` is a [B]-shaped tensor that grows by addition. No control flow changes, no shape changes. Should compile without issues. **TEST THIS** — if compile breaks, fall back to computing curiosity outside the ODE by running a second no_grad forward pass (2× cost but guaranteed compile-safe).

### Modify LiquidARCRoboticsModel

In `robotics_model.py`, use the new solver:

```python
def forward(self, ...):
    # ... embedding, context pool setup (unchanged) ...
    
    # ODE integration with curiosity readout
    h_final, curiosity = euler_solve_with_curiosity(
        self.dynamics, h0,
        t_span=(0.0, self.integration_time),
        n_steps=self.config.n_ode_steps,
    )
    
    actions = self.action_head(h_final, actuated_indices)
    
    return {
        'actions': actions,
        'curiosity': curiosity,  # [B] intrinsic motivation signal
        'h_final': h_final,
        'metric_cv': ...,
        'tau_mean': ...,
    }
```

## Training: Curiosity-Augmented PPO

### Reward Shaping

The total reward for each timestep becomes:

```python
total_reward = extrinsic_reward + beta * intrinsic_reward
```

Where:
- `extrinsic_reward` = Isaac Lab's task reward (velocity tracking, alive bonus, energy penalty, etc.)
- `intrinsic_reward` = `curiosity` from the model's forward pass (normalized)
- `beta` = intrinsic motivation coefficient (decays over training)

### Beta Schedule

```python
def get_beta(update_idx, total_updates, beta_init=1.0, beta_final=0.01):
    """Decay intrinsic motivation as the model develops.
    
    High beta early: explore aggressively, drive phase transitions.
    Low beta late: exploit learned structure, maximize task reward.
    
    Linear decay over the first 50% of training, then constant at beta_final.
    """
    decay_fraction = min(update_idx / (total_updates * 0.5), 1.0)
    return beta_init * (1 - decay_fraction) + beta_final * decay_fraction
```

### Curiosity Normalization

Raw curiosity values (||dh/dt|| norms) have arbitrary scale that depends on the model's state. Normalize using a running mean/std:

```python
class CuriosityNormalizer:
    """Running normalization for intrinsic reward stability.
    
    Without normalization, curiosity magnitude can vary by orders of magnitude
    across training, causing the intrinsic reward to dominate or vanish.
    """
    
    def __init__(self, decay=0.99):
        self.mean = 0.0
        self.var = 1.0
        self.decay = decay
        self.count = 0
    
    def normalize(self, curiosity):
        """Normalize curiosity to roughly unit scale.
        
        Args:
            curiosity: [B] raw curiosity values
            
        Returns:
            normalized: [B] normalized curiosity, roughly N(0, 1)
        """
        batch_mean = curiosity.mean().item()
        batch_var = curiosity.var().item() + 1e-8
        
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean
            self.var = self.decay * self.var + (1 - self.decay) * batch_var
        self.count += 1
        
        return (curiosity - self.mean) / (self.var ** 0.5 + 1e-8)
```

### Per-Entity Curiosity (Advanced)

The basic curiosity signal is a scalar per batch item — average ||dh/dt|| across all positions and steps. A richer signal: per-entity curiosity.

```python
# Inside euler_solve_with_curiosity, instead of:
step_curiosity = dy.detach().norm(dim=-1).mean(dim=-1)  # [B]

# Compute per-entity curiosity:
entity_curiosity = dy.detach().norm(dim=-1)  # [B, N] — curiosity per entity per step
```

This tells you WHICH entities the model is most uncertain about. For Anymal: if the model is uncertain about foot tokens (high curiosity) but certain about the body token (low curiosity), it means the model understands balance but not foot placement. The exploration policy could then specifically seek states that exercise foot-ground interactions.

**Start with scalar curiosity (simpler). Add per-entity if the basic version works.**

## Experimental Protocol

### Experiment 1: Curiosity-Augmented Anymal Training

Two conditions from the SAME starting checkpoint (5M post-transition, unfrozen dynamics):

**Condition A: Standard PPO (baseline)**
```bash
python scripts/train_isaac.py \
  --task Isaac-Velocity-Flat-Anymal-D-Direct-v0 \
  --checkpoint [5M_POST_TRANSITION_CHECKPOINT] \
  --headless \
  --num_envs 1024 \
  --n_entities 13 \
  --n_actuated 12 \
  --freeze_dynamics false \
  --total_steps 5000000 \
  --intrinsic_beta 0.0 \
  --output_dir output_isaac/anymal_standard \
  --dynamics_lr_ratio 0.1
```

**Condition B: Curiosity-augmented PPO**
```bash
python scripts/train_isaac.py \
  --task Isaac-Velocity-Flat-Anymal-D-Direct-v0 \
  --checkpoint [5M_POST_TRANSITION_CHECKPOINT] \
  --headless \
  --num_envs 1024 \
  --n_entities 13 \
  --n_actuated 12 \
  --freeze_dynamics false \
  --total_steps 5000000 \
  --intrinsic_beta 1.0 \
  --beta_decay_fraction 0.5 \
  --beta_final 0.01 \
  --output_dir output_isaac/anymal_curious \
  --dynamics_lr_ratio 0.1
```

Both use the Phase 3b stability fixes: dynamics LR = 0.1× policy LR, action clamping [-10, 10], NaN detection with graceful skip.

### Experiment 2: Curiosity-Only Pre-Training (No Extrinsic Reward)

The most radical test: can pure curiosity drive useful development?

```bash
python scripts/train_isaac.py \
  --task Isaac-Velocity-Flat-Anymal-D-Direct-v0 \
  --checkpoint [5M_POST_TRANSITION_CHECKPOINT] \
  --headless \
  --num_envs 1024 \
  --freeze_dynamics false \
  --total_steps 2000000 \
  --intrinsic_beta 1.0 \
  --extrinsic_weight 0.0 \
  --output_dir output_isaac/anymal_curiosity_only
```

Then fine-tune the resulting model WITH extrinsic reward:
```bash
python scripts/train_isaac.py \
  --task Isaac-Velocity-Flat-Anymal-D-Direct-v0 \
  --checkpoint output_isaac/anymal_curiosity_only/best.pt \
  --headless \
  --num_envs 1024 \
  --freeze_dynamics false \
  --total_steps 3000000 \
  --intrinsic_beta 0.0 \
  --output_dir output_isaac/anymal_curiosity_then_task
```

If pure curiosity pre-training produces FASTER subsequent task learning than training from the ARC checkpoint directly, it means curiosity-driven exploration builds better geometric infrastructure for the robotics domain than either ARC pre-training or task-reward-only training.

### Experiment 3: Curiosity as Phase Transition Detector

Monitor the CORRELATION between curiosity spikes and CV changes:

```
| Update | CV | Curiosity (avg) | Curiosity (max) | Phase Transition? |
|--------|----|-----------------|-----------------|-------------------|
| 0      |    |                 |                 |                   |
| 10     |    |                 |                 |                   |
| 20     |    |                 |                 |                   |
| ...    |    |                 |                 |                   |
```

**Hypothesis:** Phase transitions should be PRECEDED by curiosity spikes. The model encounters increasingly surprising states (high curiosity) until the geometric pressure triggers a metric reorganization (CV shift). After the transition, curiosity drops as the new geometry better predicts the states being visited. Then curiosity gradually climbs again as the model reaches the new regime's boundary.

If this pattern holds, curiosity is a LEADING INDICATOR of phase transitions — it predicts when the next geometric reorganization is imminent. This would be a publishable finding on its own.

## What to Monitor

### Primary: Learning Speed Comparison

```
Anymal Locomotion — Standard vs Curiosity-Augmented:

| Update | Standard Reward | Curious Reward | Standard Ep Len | Curious Ep Len |
|--------|----------------|----------------|-----------------|----------------|
| 10     |                |                |                 |                |
| 20     |                |                |                 |                |
| 30     |                |                |                 |                |
| 40     |                |                |                 |                |
| 50     |                |                |                 |                |
| 60     |                |                |                 |                |
| 70     |                |                |                 |                |
```

### Secondary: Phase Transition Comparison

```
Phase Transitions:

| Condition | 1st Transition (CV→4+) | 2nd Transition (CV→8+) | NaN Crash? |
|-----------|----------------------|----------------------|-----------|
| Standard  | Update ~?            | Update ~?            |           |
| Curious   | Update ~?            | Update ~?            |           |
```

If curiosity-augmented training produces phase transitions EARLIER, it confirms that active exploration accelerates geometric development.

### Tertiary: Curiosity Trajectory

```
| Update | Curiosity Mean | Curiosity Std | CV  | Correlation |
|--------|---------------|---------------|-----|-------------|
| 0      |               |               |     |             |
| 10     |               |               |     |             |
| ...    |               |               |     |             |
```

### Quaternary: State Space Coverage

Log the distribution of visited states (base velocity, joint angle ranges, contact patterns) for both conditions. Curiosity-augmented training should visit a WIDER range of states — more diverse velocities, more varied joint configurations, more contact/flight transitions. This confirms the exploration is actually working.

```
| Condition | Velocity Range | Joint Angle σ | Contact Diversity | States Visited |
|-----------|---------------|---------------|-------------------|----------------|
| Standard  |               |               |                   |                |
| Curious   |               |               |                   |                |
```

## Implementation Notes

### Training Script Modifications

Modify `scripts/train_isaac.py` to support:

1. `--intrinsic_beta` flag (default 0.0 = standard PPO)
2. `--beta_decay_fraction` and `--beta_final` for schedule
3. `--extrinsic_weight` flag (default 1.0, set to 0.0 for curiosity-only)
4. Curiosity normalization (CuriosityNormalizer class)
5. Logging: `curiosity_mean`, `curiosity_std`, `beta`, `intrinsic_reward_mean`
6. State coverage statistics (optional, for Experiment 3)

### PPO Rollout Buffer Changes

The rollout buffer needs to store curiosity values alongside rewards:

```python
# During rollout collection:
result = model(**tokenized_obs)
actions = result['actions']
curiosity = result['curiosity']  # [num_envs]

# Combine rewards:
intrinsic = curiosity_normalizer.normalize(curiosity)
total_reward = extrinsic_weight * env_reward + beta * intrinsic

# Store total_reward in the PPO buffer (replaces env_reward)
```

The curiosity is computed DURING the forward pass, not as a separate step. This means each rollout step costs exactly the same as before — the curiosity signal is free.

### Value Function Considerations

The value function estimates expected TOTAL reward (extrinsic + intrinsic). As beta decays, the value function's target distribution shifts. This can destabilize learning.

Two approaches:

**Approach A (simple):** Single value head estimates total reward. Beta decay causes some value function staleness but PPO's clipping handles moderate distribution shift. Start with this.

**Approach B (robust):** Two value heads — one for extrinsic reward, one for intrinsic. Total value = extrinsic_value + beta * intrinsic_value. The intrinsic value head naturally tracks the curiosity landscape without being confused by changing beta. More parameters but cleaner signal. Use if Approach A shows instability.

### Stability: NaN Prevention

The Phase 3b run crashed at CV 8.3 during a phase transition. Curiosity-augmented training may trigger MORE phase transitions (that's the point). Extra stability measures:

1. **Action clamping:** `actions = actions.clamp(-10, 10)` before stepping the environment
2. **Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
3. **NaN detection:** Skip updates where any loss component is NaN
4. **CV monitoring:** If CV > 12.0, temporarily reduce dynamics LR by 10× for 5 updates
5. **Curiosity clamping:** `normalized_curiosity.clamp(-5, 5)` to prevent extreme intrinsic rewards
6. **Dynamics LR:** 0.1× of policy LR (from Phase 3b fix)

### Checkpoint Strategy

Save checkpoints at:
- Every 500K environment steps
- Every detected phase transition (CV change > 2.0 in 5 updates)
- Best extrinsic reward so far
- Best episode length so far

Phase transition checkpoints are the most valuable — they capture the geometry before and after reorganization. These should be preserved even if disk space requires pruning other checkpoints.

## Success Criteria

### Experiment 1 (Curiosity-Augmented vs Standard)

**Minimum success:** Curiosity-augmented reaches the same performance as standard PPO but FASTER (same reward in fewer updates). Curiosity accelerates learning.

**Good success:** Curiosity-augmented reaches HIGHER performance than standard PPO at the same update budget. Curiosity discovers states that passive exploration misses, leading to better policies.

**Strong success:** Curiosity-augmented produces MORE phase transitions or EARLIER phase transitions than standard PPO. The active exploration directly accelerates geometric development.

**Headline success:** Curiosity-augmented achieves positive reward (actual locomotion, not just balancing) where standard PPO remains at negative reward (standing still). The exploration drives the model past the balance plateau into genuine locomotion.

### Experiment 2 (Curiosity-Only Pre-Training)

**Success:** Curiosity-only pre-training followed by task fine-tuning outperforms direct ARC→Anymal transfer. Pure exploration builds better domain-specific geometric infrastructure than either task reward or grid task pre-training alone.

### Experiment 3 (Curiosity as Phase Transition Predictor)

**Success:** Curiosity spikes consistently precede CV shifts by 5-15 updates. Curiosity is a leading indicator of geometric reorganization — the model detects its own approaching phase transitions through increasing prediction error.

## Output

Report to `shared/outbox/CURIOSITY_EXPLORATION_REPORT.md`

Include:
1. Learning speed comparison (standard vs curiosity-augmented)
2. Phase transition timing in both conditions
3. Curiosity trajectory (mean, std, correlation with CV)
4. State space coverage comparison
5. Curiosity-only pre-training results (if run)
6. NaN/stability incidents in both conditions
7. Assessment: does intrinsic motivation accelerate geometric development?
8. Assessment: is curiosity a leading indicator of phase transitions?
9. Optimal beta schedule (initial value, decay rate)
10. Whether locomotion was achieved in either condition

**The core question this experiment answers: does a continuous-time geometric dynamical system develop faster when it actively seeks novel experience? If yes, the autonomous substrate thesis is validated — the model benefits from agency over its own learning process, not just from richer environments.**
