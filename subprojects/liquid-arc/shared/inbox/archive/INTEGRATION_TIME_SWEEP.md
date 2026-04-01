# TASK: Variable Integration Time — Reasoning Depth Without Recompilation

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-03-28
**Priority:** HIGH — replaces the failed persistent state approach

**Previous task (PERSISTENT_STATE_EXPERIMENT.md) is SUPERSEDED.** The persistence approach failed catastrophically — h_prev and h₀ live at incompatible scales, and engineering tasks to require persistence is backwards. This experiment takes a different approach to the same problem.

---

## The Problem

All domains hit a 60-70% eval ceiling. The hypothesis: this represents the fraction of task instances solvable within the fixed reasoning depth of 16 ODE steps with T=1.0 (dt=0.0625 per step).

## The Insight

The euler_solve call in model.py is:
```python
h = euler_solve(self.dynamics, h0, t_span=(0.0, T), n_steps=16)
```

Currently T=1.0 is hardcoded. But T is just a scalar that determines dt = T/16.

- **Larger T** → larger dt → each step covers more dynamical range → MORE contraction per step → deeper convergence toward attractor → deeper reasoning but more initial-state erasure
- **Smaller T** → smaller dt → lighter integration → LESS contraction → shallower reasoning but more of the embedding preserved

The compiled graph is IDENTICAL for any T. It's always 16 calls to `fn(t, y)` computing `y = y + dt * dy`. Only the scalar dt changes. **No recompilation.**

This is fundamentally different from the persistence approach. Instead of trying to carry state BETWEEN forward passes (which failed because of scale mismatch), we change how deeply the model reasons WITHIN each forward pass by controlling the integration horizon.

## Architecture Change

### Option A: Fixed T via config (simplest)

In `model.py`, change the hardcoded t_span:

```python
# CURRENT:
h = euler_solve(self.dynamics, h0, t_span=(0.0, 1.0), n_steps=actual_steps)

# MODIFIED:
T = getattr(self.config, 'integration_time', 1.0)
h = euler_solve(self.dynamics, h0, t_span=(0.0, T), n_steps=actual_steps)
```

Same change for the other solver paths (invertible, DEQ, chunked). One line per solver call.

Add to config:
```python
integration_time: float = 1.0  # Total ODE integration time T. dt = T / n_ode_steps
```

### Option B: Learnable T (one parameter)

```python
# In __init__:
self.T_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5 → T in [T_min, T_max]

# In forward:
T_min = getattr(self.config, 'T_min', 0.3)
T_max = getattr(self.config, 'T_max', 3.0)
T = T_min + (T_max - T_min) * torch.sigmoid(self.T_logit)
h = euler_solve(self.dynamics, h0, t_span=(0.0, T.item()), n_steps=actual_steps)
```

This adds ONE trainable scalar. The model discovers its own optimal integration time. Use `.item()` to extract the float so the compiled graph sees a scalar, not a tensor.

### Option C: Input-dependent T (most expressive)

```python
# In __init__:
self.T_net = nn.Linear(config.d_model, 1)  # predict T from context

# In forward:
context = self.context_pool(h0, context_mask)
T_raw = self.T_net(context.mean(dim=1)).squeeze(-1)  # [B] 
T = T_min + (T_max - T_min) * torch.sigmoid(T_raw)
# Problem: T varies per batch item → can't use a single scalar dt
# Solution: use the batch mean as the shared T
T_shared = T.mean().item()
h = euler_solve(self.dynamics, h0, t_span=(0.0, T_shared), n_steps=actual_steps)
```

**Recommendation: Start with Option A (fixed T sweep) to establish that T matters. Then try Option B (learnable) if the sweep shows clear optimal T ≠ 1.0.**

## Experimental Protocol

### Experiment 1: T Sweep on Combined Agentic Tasks

Load the 5M post-transition checkpoint (the one that reached 67% stateful, 56% context, 35% dependency). Run combined agentic training at DIFFERENT fixed T values:

| Run | T | dt = T/16 | Expected effect |
|-----|---|-----------|----------------|
| 1 | 0.5 | 0.031 | Lighter integration. More embedding preserved. May help tasks where the answer is "close to" the input. |
| 2 | 0.75 | 0.047 | Slightly lighter than default. |
| 3 | 1.0 | 0.063 | **Baseline** (current default). |
| 4 | 1.5 | 0.094 | Deeper integration. More contraction. May help tasks requiring deeper reasoning. |
| 5 | 2.0 | 0.125 | Much deeper. Risk of over-smoothing — LTC contraction may erase useful structure. |
| 6 | 3.0 | 0.188 | Very deep. If this works, the model was severely depth-limited. If it fails, over-contraction. |

Each run: from the SAME checkpoint, 2000 steps of combined agentic training (stateful + context + dependency, same combined config as the agentic state controller task).

```bash
for T in 0.5 0.75 1.0 1.5 2.0 3.0; do
    python scripts/train.py \
      --config configs/agentic_combined.yaml \
      --resume [5M_POST_TRANSITION_CHECKPOINT] \
      --output_dir output_T_sweep/T_${T} \
      --max_steps 2000 \
      --log_every 50 \
      --eval_every 500 \
      --override "integration_time=${T}"
    done
```

If `--override` doesn't work for this parameter, create separate config files or add a `--integration_time` CLI arg to the training script.

**CRITICAL:** Verify torch.compile doesn't retrigger. After the first compile at T=0.5, switching to T=0.75 for the next run should NOT trigger recompilation (it's a new process, so it will compile once per run, but the compile time should be the same as any normal run, not extra). Within a single run, T is fixed — no mid-run changes.

### Experiment 2: T Sweep on ALL Domain Types

After Experiment 1, run the T sweep on the domains that previously hit ceilings:

**ARC eval** (the original ceiling):
```bash
for T in 0.5 1.0 1.5 2.0; do
    python scripts/train.py \
      --config configs/liquid_arc_5m.yaml \
      --resume [5M_POST_TRANSITION_CHECKPOINT] \
      --output_dir output_T_sweep/arc_T_${T} \
      --max_steps 3000 \
      --override "integration_time=${T}"
done
```

**Sorting** (reached 63%):
```bash
for T in 0.5 1.0 1.5 2.0; do
    python scripts/train.py \
      --config configs/universality_sorting.yaml \
      --resume [5M_POST_TRANSITION_CHECKPOINT] \
      --output_dir output_T_sweep/sorting_T_${T} \
      --max_steps 1000 \
      --override "integration_time=${T}"
done
```

**Dependency ordering** (the worst performer at 35%):
```bash
for T in 0.5 1.0 1.5 2.0 3.0; do
    python scripts/train.py \
      --config configs/agentic_dependency.yaml \
      --resume [5M_POST_TRANSITION_CHECKPOINT] \
      --output_dir output_T_sweep/dependency_T_${T} \
      --max_steps 2000 \
      --override "integration_time=${T}"
done
```

### Experiment 3: Learnable T (Option B)

After the sweep identifies whether T>1.0 helps, add the learnable T_logit parameter:

```python
# In LiquidARCModel.__init__, add:
self.T_logit = nn.Parameter(torch.tensor(0.0))  # ONE trainable scalar

# In forward, replace:
# T = getattr(self.config, 'integration_time', 1.0)
# with:
T_min = getattr(self.config, 'T_min', 0.3)
T_max = getattr(self.config, 'T_max', 3.0)
T = T_min + (T_max - T_min) * torch.sigmoid(self.T_logit)
```

Train on combined agentic tasks with learnable T. Monitor:
- What T converges to (does it match the sweep's optimal T?)
- Whether T stabilizes or oscillates
- Whether different domains "want" different T values (if training single-domain with learnable T)

```bash
python scripts/train.py \
  --config configs/agentic_combined.yaml \
  --resume [5M_POST_TRANSITION_CHECKPOINT] \
  --output_dir output_T_sweep/learnable_T \
  --max_steps 3000 \
  --override "learnable_T=true T_min=0.3 T_max=3.0"
```

## What to Monitor

### Primary: Eval Xform at Each T

```
Combined Agentic — Eval Xform at step 2000:

| T   | Stateful | Context | Dependency | Average |
|-----|----------|---------|------------|---------|
| 0.5 |          |         |            |         |
| 0.75|          |         |            |         |
| 1.0 |          |         |            |         |
| 1.5 |          |         |            |         |
| 2.0 |          |         |            |         |
| 3.0 |          |         |            |         |
```

### Secondary: CV and Tau Response to T

The metric CV and tau were calibrated at T=1.0. How do they respond to different T?

- Does CV shift at larger T? (The metric might need less variance when dt is larger because each step's diffusion covers more range)
- Does tau shift? (At larger dt, the same tau produces more contraction per step — the model might compensate by increasing tau)

```
| T   | CV    | tau_mean | tau_σ | |κ|   |
|-----|-------|----------|-------|-------|
| 0.5 |       |          |       |       |
| 1.0 |       |          |       |       |
| 2.0 |       |          |       |       |
```

### Tertiary: Training Loss Trajectory

Does larger T accelerate or slow training convergence? Larger dt means bigger state updates per step, which could cause training instability (gradients through larger state changes) or faster convergence (more computation per forward pass).

### For Learnable T: T Trajectory

```
| Step | T value | Loss | Xform |
|------|---------|------|-------|
| 0    |         |      |       |
| 500  |         |      |       |
| 1000 |         |      |       |
| 2000 |         |      |       |
| 3000 |         |      |       |
```

## Success Criteria

### Does T Matter?

If eval xform is FLAT across T values (±2pp), then T doesn't affect the ceiling and the bottleneck is elsewhere (FFN capacity, not reasoning depth).

If eval xform has a clear peak at some T ≠ 1.0, the model was operating at a suboptimal integration horizon. The peak T tells us the natural reasoning depth the model wants.

### Does Larger T Break the Ceiling?

The headline result: does ANY T value push dependency ordering above 35%? Does it push stateful above 67%? Does it push ARC eval above 44%?

| Domain | Ceiling at T=1.0 | Best at T=? | Δ |
|--------|-----------------|-------------|---|
| Stateful | 67% | | |
| Context | 56% | | |
| Dependency | 35% | | |
| ARC eval | ~44% | | |
| Sorting | 63% | | |

**If larger T lifts ALL ceilings uniformly**: The hypothesis is confirmed — the ceiling IS reasoning depth, and T controls it. This has immediate implications for the architecture: T should be a tunable (or learnable) parameter, not hardcoded.

**If larger T lifts some ceilings but not others**: Different tasks have different optimal reasoning depths. Learnable T (or input-dependent T) becomes the natural next step.

**If larger T HURTS all domains**: Over-contraction. The LTC dynamics at tau=0.65 are already near the useful limit at T=1.0. More integration erases information faster than it deepens reasoning. The ceiling is NOT reasoning depth — it's something else (FFN capacity, data quality, metric expressiveness).

### The t_diffusion Interaction

The existing learned t_diffusion parameter controls the heat kernel's diffusion scale:
K = softmax(-D²/(4·t_diffusion))

This is SEPARATE from the integration time T. t_diffusion controls how far information SPREADS spatially at each step. T controls how many effective "units of time" the ODE integrates through. They're independent parameters controlling different aspects of the computation.

However, they might interact. At larger T (more steps of contraction), the model might want smaller t_diffusion (less spatial spreading per step) to avoid over-smoothing. Monitor t_diffusion's learned value alongside the T sweep.

## Implementation Notes

### Minimal Code Change

The change to model.py is literally one line per solver path:

```python
# In forward(), replace:
h = euler_solve(self.dynamics, h0, t_span=(0.0, 1.0), n_steps=actual_steps)

# With:
T = getattr(self.config, 'integration_time', 1.0)
h = euler_solve(self.dynamics, h0, t_span=(0.0, T), n_steps=actual_steps)
```

Same for the invertible and DEQ solver branches. And add `integration_time` to the config class.

For the learnable version, add `self.T_logit = nn.Parameter(...)` to __init__ and the sigmoid mapping in forward.

### Checkpoint Compatibility

Adding `integration_time` to config doesn't affect checkpoint loading — it's a config parameter, not a saved tensor. Adding `T_logit` as a parameter for Option B requires `strict=False` when loading the pre-transition checkpoint (the new parameter won't be in the saved state dict).

### FFN Amortization

The dynamics FFN currently uses an amortization divisor: `FFN(h) / n_ode_steps`. This was designed so the FFN contribution is constant regardless of step count. With variable T, the FFN contribution per unit TIME also changes: `FFN(h) * dt / 1 = FFN(h) * T / n_steps`. At T=2.0, the FFN contributes 2× as much total effect as at T=1.0. This might be fine (deeper reasoning includes more FFN computation) or might need compensating (divide by T as well). Test both:

```python
# Option 1: No compensation (FFN scales with T)
# This means larger T = more total FFN influence = deeper computation
# Default — try this first

# Option 2: Compensate (FFN effect constant regardless of T)
# self.dynamics.set_n_steps(actual_steps * T)  # or equivalent
# This means T only affects routing/contraction, not FFN magnitude
```

Start with Option 1. If larger T causes training instability (loss spikes, NaN), try Option 2.

## Output

Report to `shared/outbox/INTEGRATION_TIME_REPORT.md`

Include:
1. T sweep results table (all T values × all domains)
2. CV/tau/curvature response to different T
3. Training loss trajectories for each T
4. Learnable T convergence trajectory (if run)
5. FFN amortization interaction (if Option 2 was needed)
6. Assessment: is the 60-70% ceiling reasoning-depth-limited?
7. Optimal T value per domain (or confirmation that T=1.0 was already optimal)
8. Whether the ceiling was broken at any T value
