# TASK: Adaptive Autonomy — Tau as Self-Regulated Processing Depth

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-03-30
**Priority:** HIGH — replaces fixed autonomous_steps with intrinsic self-regulation

**Prerequisites:** Read `shared/outbox/LIFECYCLE_REPORT.md` for full context. This builds directly on two findings: (1) the lifecycle model achieved equivalent performance with CV<1 and tau→0.997, and (2) the strategic death discovery showed the model already uses tau as a self-regulation mechanism.

---

## The Problem with Fixed Autonomous Steps

The lifecycle spec introduced `autonomous_steps` — extra ODE steps with no sensory forcing, giving the model "thinking time." But fixed autonomous steps create an inverse paradox: when the environment is slow (boring, predictable), the model has plenty of thinking time it doesn't need. When the environment is fast (surprising, dangerous), the model has no spare steps — exactly when deep processing matters most.

The strategic death log proved the model already self-regulates through tau. It pushed tau to 0.997, effectively shutting down its own dynamics. It didn't need an external mechanism — it used the parameter it already has. The problem is that tau currently operates as a uniform dampener: high tau means ALL positions process slowly, low tau means ALL positions process fast. There's no mechanism for the model to say "this observation is surprising — process deeply" versus "this is routine — process minimally."

## The Insight: Tau Already IS Adaptive Processing Depth

The LTC dynamics per ODE step:

```
Δh = (dt/τ) · (target - h) + (dt/n_steps) · FFN(h)
```

The effective update magnitude per step is `dt/τ`. When τ is high (→1.0), each step barely changes h. When τ is low (→0.5), each step aggressively contracts h toward the target. Over 16 steps, low τ produces a state that has converged deeply toward the attractor. High τ produces a state that's barely moved from h₀.

**Currently:** TauNet computes tau from h once (at the start or at each step, depending on implementation), and the same tau applies uniformly throughout the 16 steps. The model can't modulate processing depth WITHIN a forward pass based on how much processing has been done.

**The change:** TauNet already receives h at each ODE step (it's inside the dynamics). If h carries information about how much processing has occurred (which it does — h evolves across steps), TauNet can produce DIFFERENT tau values at different steps. Early steps (h is far from attractor, large ||dh/dt||) → low tau (process aggressively). Late steps (h has converged, small ||dh/dt||) → high tau (minimal updates, effective no-op).

This is adaptive processing depth within the fixed 16-step compiled graph. The model automatically allocates more effective computation to situations that need it, without changing the number of steps.

## Architecture Changes

### Change 1: Dynamics Norm Regularizer

Add a small loss term that penalizes total dynamics magnitude:

```python
L_efficiency = λ_eff · mean(||dh/dt||² across all ODE steps)
```

This gives TauNet direct gradient pressure to minimize unnecessary processing. Without this regularizer, TauNet has no reason to increase tau when processing is complete — the task reward doesn't penalize wasted computation. With it, TauNet learns: "when h is near the attractor, increase tau to reduce ||dh/dt|| and avoid the penalty."

**Implementation in the solver:**

```python
def euler_solve_adaptive(fn, y0, t_span, n_steps, return_efficiency=False):
    """Forward Euler with optional dynamics norm tracking for efficiency loss.
    
    Returns the sum of ||dh/dt||² across steps for the efficiency regularizer.
    This is NOT curiosity (which was reward-based and caused NaN).
    This is a LOSS term that penalizes unnecessary dynamics — the opposite direction.
    
    Curiosity: reward high ||dh/dt|| → model seeks instability → NaN
    Efficiency: penalize high ||dh/dt|| → model seeks convergence → stability
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps
    t = t_start
    y = y0
    
    dynamics_sq_sum = torch.tensor(0.0, device=y0.device)
    
    for i in range(n_steps):
        if hasattr(fn, 'set_step_embed'):
            fn.set_step_embed(i, n_steps)
        if hasattr(fn, 'set_step_index'):
            fn.set_step_index(i, n_steps)
        dy = fn(t, y)
        
        if return_efficiency:
            # Track dynamics magnitude for efficiency loss
            # Use .mean() not .sum() to keep scale independent of sequence length
            dynamics_sq_sum = dynamics_sq_sum + (dy ** 2).mean()
        
        y = y + dt * dy
        t = t + dt
    
    if return_efficiency:
        return y, dynamics_sq_sum / n_steps
    return y
```

**Integration into training loss:**

```python
# In the training loop, after the forward pass:
h_final, efficiency_cost = euler_solve_adaptive(
    dynamics, h0, t_span=(0.0, T), n_steps=16, return_efficiency=True
)

# Total loss:
loss = task_loss + lambda_eff * efficiency_cost
```

**CRITICAL:** This is a LOSS (penalize), not a REWARD (encourage). The curiosity experiments crashed because they rewarded high ||dh/dt|| — the model sought instability. The efficiency regularizer penalizes high ||dh/dt|| — the model seeks convergence. Opposite direction, opposite stability properties. The model is PRESSURED toward efficient processing, not toward turbulence.

`lambda_eff` should be small (0.001-0.01) — enough to nudge TauNet toward efficiency but not so large that it dominates the task reward.

### Change 2: Step-Aware TauNet (Optional Enhancement)

Currently TauNet computes tau from h alone. Adding an explicit step-progress signal would let TauNet know WHERE in the 16-step integration it is:

```python
# In ContinuousDynamics.forward(), the tau computation:

# CURRENT:
tau = self.tau_min + (self.tau_max - self.tau_min) * F.softplus(self.tau_net_linear2(
    F.gelu(self.tau_net_linear1(h_normed))
))

# ENHANCED: add step progress signal
step_progress = self._current_step_index_buf.float() / self._current_n_steps_buf.float()
# Inject into TauNet input by concatenating or by additive embedding
# Option A: Concatenate (requires changing tau_net_linear1 input dim from d to d+1)
tau_input = torch.cat([h_normed, step_progress.expand(B, N, 1)], dim=-1)
tau = self.tau_min + (self.tau_max - self.tau_min) * F.softplus(self.tau_net_linear2(
    F.gelu(self.tau_net_linear1(tau_input))
))
```

This lets TauNet explicitly learn "at step 14 of 16, h is usually converged, so produce high tau." Without it, TauNet must infer step progress from the h values themselves (which change over steps, so the information is available but implicit).

**WARNING:** Changing TauNet's input dimension (d → d+1) breaks checkpoint compatibility. Use Option B instead:

```python
# Option B: Additive step embedding (no dimension change, checkpoint-compatible)
# Add a small step embedding to the TauNet hidden layer

# In __init__:
self.tau_step_embed = nn.Embedding(20, d_met)  # 20 max steps
nn.init.zeros_(self.tau_step_embed.weight)  # starts as no-op

# In forward, tau computation:
tau_hidden = F.gelu(self.tau_net_linear1(h_normed))
step_idx = self._current_step_index_buf.clamp(0, 19)
tau_hidden = tau_hidden + self.tau_step_embed(step_idx)  # additive, starts as no-op
tau = self.tau_min + (self.tau_max - self.tau_min) * F.softplus(
    self.tau_net_linear2(tau_hidden)
)
```

Zero-initialized → loads from existing checkpoint with no behavior change. The step embedding learns over training to modulate tau based on step position. This is a new parameter (`tau_step_embed`: 20 × 192 = 3,840 params) that needs `strict=False` on checkpoint load.

### Change 3: Forcing-Aware Tau (Lifecycle-Specific)

In the lifecycle model, the sensory forcing magnitude tells TauNet exactly how surprising the current observation is. Large forcing = large prediction error = need deep processing. Small forcing = predicted correctly = light processing.

```python
# In _run_ode_segment, when forcing is available:
# Pass the forcing magnitude as additional context to TauNet

# Option: Set a buffer before the ODE that TauNet can read
if forcing is not None:
    forcing_magnitude = forcing.norm(dim=-1, keepdim=True)  # [B, N, 1]
    self.dynamics.set_forcing_signal(forcing_magnitude)
else:
    self.dynamics.set_forcing_signal(None)
```

Then in ContinuousDynamics, TauNet reads this signal:

```python
# In forward():
if self._forcing_signal is not None:
    # Low tau when forcing is large (process deeply)
    # High tau when forcing is small (coast)
    forcing_bias = -0.5 * self._forcing_signal  # negative = lower tau = more processing
    tau_hidden = tau_hidden + forcing_bias
```

This creates the adaptive behavior directly: surprising observations drive tau down (aggressive processing), expected observations leave tau high (minimal processing). The forcing magnitude decays over the integration window (from the lifecycle's linear decay), so tau naturally increases as the observation is assimilated — early steps process deeply, late steps coast.

**NOTE:** This coupling is ONLY for the lifecycle model. The discrete model doesn't have forcing. Guard with a simple None check.

## Experimental Protocol

### Experiment 1: Efficiency Regularizer on Anymal Lifecycle

Three conditions from the same lifecycle checkpoint (the one that reached reward -9.8):

**Condition A: Baseline lifecycle (no regularizer)**
```bash
python scripts/train_lifecycle.py \
  --task Isaac-Velocity-Flat-Anymal-D-Direct-v0 \
  --checkpoint [LIFECYCLE_CHECKPOINT] \
  --headless --num_envs 1024 \
  --freeze_dynamics false \
  --total_steps 2000000 \
  --lambda_eff 0.0 \
  --output_dir output_adaptive/baseline
```

**Condition B: Mild efficiency pressure**
```bash
python scripts/train_lifecycle.py \
  ... \
  --lambda_eff 0.001 \
  --output_dir output_adaptive/eff_001
```

**Condition C: Moderate efficiency pressure**
```bash
python scripts/train_lifecycle.py \
  ... \
  --lambda_eff 0.01 \
  --output_dir output_adaptive/eff_01
```

### Experiment 2: Step-Aware TauNet

Add the tau_step_embed (Option B) and train with efficiency regularizer:

```bash
python scripts/train_lifecycle.py \
  ... \
  --lambda_eff 0.005 \
  --tau_step_embed true \
  --output_dir output_adaptive/step_aware
```

### Experiment 3: Forcing-Aware Tau

Add the forcing_signal coupling in the lifecycle model:

```bash
python scripts/train_lifecycle.py \
  ... \
  --lambda_eff 0.005 \
  --forcing_aware_tau true \
  --output_dir output_adaptive/forcing_aware
```

### Experiment 4: Perturbation Response Test (Diagnostic)

After training, test the model's adaptive behavior with controlled perturbations:

1. **Steady walking** (routine): Record tau trajectory across 16 steps. Expected: high tau (coasting).
2. **Sudden push** (300N lateral force): Record tau trajectory. Expected: tau drops sharply (deep processing), then recovers (assimilation).
3. **Ground removal** (foot loses contact): Record tau trajectory. Expected: tau drops for the affected foot tokens, remains high for unaffected tokens.
4. **New velocity command** (direction change): Record tau trajectory. Expected: tau drops globally (new plan needed), recovers over subsequent observations.

This is the definitive test of adaptive autonomy: the model modulates its own processing depth in response to situation demands, with per-entity granularity.

```python
"""Perturbation response test — measure tau adaptation to controlled surprises.

Run a trained lifecycle model in evaluation mode. Inject controlled perturbations
at specific timesteps and record the per-entity, per-step tau values.
"""

def perturbation_test(model, env, tokenizer, perturbation_schedule):
    """
    perturbation_schedule: list of (timestep, perturbation_type, magnitude)
    
    Returns: dict of per-timestep tau trajectories (all 16 steps × 13 entities)
    """
    tau_trajectories = {}
    
    obs = env.reset()
    model.reset(batch_size=obs.shape[0], device=obs.device)
    
    for t in range(total_steps):
        tokens = tokenizer.tokenize(obs)
        
        # Hook into TauNet to record per-step tau
        tau_per_step = []
        original_forward = model.dynamics.forward
        
        def hooked_forward(t_ode, h):
            tau = model.dynamics.compute_tau(h)
            tau_per_step.append(tau.detach().cpu())
            return original_forward(t_ode, h)
        
        model.dynamics.forward = hooked_forward
        result = model.step(tokens, actuated_indices)
        model.dynamics.forward = original_forward
        
        # Apply perturbation if scheduled
        if t in [s[0] for s in perturbation_schedule]:
            perturbation = [s for s in perturbation_schedule if s[0] == t][0]
            env.apply_perturbation(perturbation[1], perturbation[2])
        
        obs, reward, done, info = env.step(result['actions'].clamp(-10, 10))
        
        if t in perturbation_schedule_timesteps:
            tau_trajectories[t] = {
                'tau_per_step': torch.stack(tau_per_step),  # [16, B, N, 1]
                'prediction_error': result['prediction_error'],
                'perturbation_type': perturbation[1],
            }
    
    return tau_trajectories
```

## What to Monitor

### Primary: Tau Adaptation Pattern

The key diagnostic: does tau vary across ODE steps within a single observation?

```
Per-Step Tau Profile (average across batch, body token):

| ODE Step | Baseline τ | Efficiency τ | Step-Aware τ | Forcing-Aware τ |
|----------|-----------|-------------|-------------|-----------------|
| 0        |           |             |             |                 |
| 4        |           |             |             |                 |
| 8        |           |             |             |                 |
| 12       |           |             |             |                 |
| 15       |           |             |             |                 |
```

**Expected pattern for adaptive behavior:**
- Steps 0-4: Low tau (aggressive processing of new observation)
- Steps 5-10: Tau climbing (state converging)
- Steps 11-15: High tau (converged, effectively coasting)

**Expected pattern for surprising observation:**
- Steps 0-8: Low tau sustained (deep processing needed)
- Steps 9-15: Tau climbing (finally converging)

**Expected pattern for routine observation:**
- Steps 0-2: Moderate tau (light processing)
- Steps 3-15: High tau (already converged)

### Secondary: Per-Entity Tau Differentiation

```
Per-Entity Tau (step 0, surprising observation):

| Entity | Type | Tau (routine) | Tau (surprise) | Difference |
|--------|------|--------------|---------------|-----------|
| Body   | base | | | |
| FL hip | joint | | | |
| FL shin | joint | | | |
| FR hip | joint | | | |
| ... | | | | |
```

Entities experiencing large prediction error (foot that lost contact) should show lower tau than entities with small prediction error (unaffected joints).

### Tertiary: Efficiency Cost Trajectory

```
| Update | Task Reward | Efficiency Cost | Tau Mean | ||dh/dt|| Mean |
|--------|------------|----------------|----------|----------------|
| 0      |            |                |          |                |
| 50     |            |                |          |                |
| 100    |            |                |          |                |
| ...    |            |                |          |                |
```

The efficiency cost should decrease over training as TauNet learns to suppress unnecessary dynamics. If it decreases WITHOUT task reward degradation, the model is learning to be efficient. If task reward also drops, lambda_eff is too high.

### Quaternary: Strategic Death Prevention

The strategic death exploit used tau→0.997 to shut down dynamics. The efficiency regularizer should make this HARDER, not easier, because dying quickly means the policy must produce large, sudden actions — which require low tau (large ||dh/dt||) in the steps immediately before death. The regularizer penalizes this. Monitor whether the strategic death behavior reappears with the regularizer.

## Implementation Notes

### Checkpoint Compatibility

**Efficiency regularizer:** Zero code changes to the model. Only the solver and loss computation change. Fully checkpoint-compatible.

**Tau step embed (Option B):** New parameter `tau_step_embed` (3,840 params). Zero-initialized → no behavior change on load. Requires `strict=False` when loading checkpoint. The existing TauNet weights are unchanged.

**Forcing-aware tau:** New buffer `_forcing_signal` set before the ODE. No parameter changes. The bias is additive to the existing tau_hidden. Checkpoint-compatible (the forcing signal is None when not in lifecycle mode, producing identical behavior).

### torch.compile Compatibility

All changes are within the existing compiled graph structure:
- Efficiency tracking: one `(dy ** 2).mean()` per step — same as curiosity tracking but as a loss not reward
- Tau step embed: one embedding lookup per step — same pattern as existing step_embeds in MetricNet
- Forcing signal: one buffer read + addition per step — same pattern as existing _metric_overlay

None of these add control flow or change tensor shapes. Should compile without issues.

### Training Stability

The efficiency regularizer has a STABILIZING effect (unlike curiosity which was destabilizing):
- It penalizes large ||dh/dt||, which means it penalizes dynamics near the NaN boundary
- It encourages tau to increase when processing is complete, which DAMPS the ODE
- It creates pressure AGAINST the phase-transition-like CV spikes that caused previous NaN crashes

This should make the lifecycle model MORE stable, not less. The strategic death exploit also becomes harder because it requires sustained low tau for the final death action, which the regularizer penalizes.

### Integration with Lifecycle and Linguistic Mind

The adaptive autonomy is a PROPERTY of the dynamics, not a separate module. Once trained, it works in any context:
- **Isaac Sim:** TauNet adapts per physics step based on observation surprise
- **Linguistic Mind:** TauNet adapts per conversation event based on message novelty
- **Discrete mode:** TauNet adapts per forward pass based on task difficulty
- **Autonomous processing:** TauNet at high tau during autonomous steps (no forcing, converged state) → autonomous steps are naturally cheap

The forcing-aware tau coupling specifically means: when the lifecycle model receives a surprising message (large forcing), tau drops and the model processes deeply. When it receives a routine update (small forcing), tau stays high and the model barely changes. When running autonomously between events (zero forcing), tau maximizes and the 16 steps are effectively no-ops — the model idles efficiently until the next event.

This dissolves the fixed-autonomous-steps problem entirely. The model doesn't need a separate parameter for thinking time. The dynamics THEMSELVES encode thinking depth through tau, and tau adapts to the situation through the efficiency pressure and the forcing signal.

## Success Criteria

### Does Tau Adapt Within Forward Passes?

**Minimum:** With efficiency regularizer, tau mean increases over training (model learns to be efficient). Per-step tau shows ANY variation across the 16 steps.

**Good:** Clear descending tau profile across steps: low at step 0, high at step 15. The model processes aggressively early and coasts late.

**Strong:** Tau profile CHANGES based on observation surprise. Routine observations show fast convergence (tau high by step 4). Surprising observations show sustained low tau (through step 10+). The model allocates processing depth based on information content.

**Headline:** Per-entity tau differentiation during perturbations. Foot tokens that lose contact show low tau while unaffected joints show high tau. The model allocates processing depth per-entity based on per-entity prediction error. This is the adaptive autonomy: each part of the system processes at its own depth based on its own surprise level.

### Does Efficiency Regularizer Prevent Strategic Death?

**Success:** With lambda_eff > 0, the strategic death exploit no longer emerges (or takes significantly longer to discover). The per-step penalty on ||dh/dt|| makes the sharp actions required for deliberate death expensive.

### Does Task Performance Degrade?

**Success:** Task reward with efficiency regularizer is within 5% of baseline. The model learns to be efficient WITHOUT sacrificing performance. If reward drops >10%, lambda_eff is too high.

## Output

Report to `shared/outbox/ADAPTIVE_AUTONOMY_REPORT.md`

Include:
1. Per-step tau profiles (baseline vs efficiency vs step-aware vs forcing-aware)
2. Tau response to controlled perturbations (the diagnostic test)
3. Per-entity tau differentiation during surprises
4. Efficiency cost trajectory over training
5. Task reward comparison across conditions
6. Strategic death behavior under efficiency pressure
7. Assessment: does the model learn adaptive processing depth?
8. Assessment: does per-entity, per-step tau modulation emerge?
9. Optimal lambda_eff value
10. Implications for the linguistic mind (does adaptive tau transfer to conversation events?)

**The core question: can the model learn to allocate its own computational budget — 16 steps of fixed-graph ODE — adaptively based on situation demands, using nothing but the tau mechanism it already has plus a small efficiency nudge? If yes, the model has INTRINSIC adaptive autonomy — it doesn't need external scheduling of thinking time. It schedules its own.**
