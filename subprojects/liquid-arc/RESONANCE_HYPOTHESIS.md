# Resonance Hypothesis for Phase Transitions in LiquidARC

## The Observation That Prompted This

The phase transition is not reproducible even with the exact documented recipe. Two runs on presumably the same codebase succeeded; a clean reproduction on a clean codebase failed. The transition depends on something beyond the static training configuration.

## The Resonance Model

### Mechanical Resonance
A bridge has a natural frequency. Wind or traffic at that frequency creates forcing that adds energy constructively — each cycle builds on the previous one. Small forces at the right frequency produce growing oscillations that eventually cause structural failure/reorganization.

### In LiquidARC
The model's internal dynamics (MetricNet weight updates under AdamW) have a natural frequency determined by:
- Learning rate (3e-4)
- AdamW momentum (β1=0.9 → ~10-step integration window)
- Weight decay (0.01)
- Loss landscape curvature at the current weight configuration
- All of which depend on the exact numerical state (random seed, precision, compilation)

The training batch sequence provides forcing:
- Procedural batches: BUILD signal (push MetricNet toward structured routing)
- ARC batches: DISRUPT signal (push MetricNet away from over-specialized routing)
- The 30/70 ratio creates a specific forcing rhythm: ~3.3 steps between ARC disruptions on average

### The Resonance Condition
When the build-disrupt rhythm matches the optimizer's natural integration timescale, each build phase adds CONSTRUCTIVELY to the previous one. The net accumulated signal ratchets CV upward. Over 5000 steps, this constructive accumulation pushes CV from 3.0 to the critical threshold.

When the rhythm DOESN'T match (different numerical environment → different natural frequency), the same 30% ratio produces a forcing that's slightly out of phase. Build phases don't reinforce each other. CV stays flat.

### Why Each Ratio Fails
```
15% ARC (period ~6.7): Forcing period too long. Build phases over-fit between disruptions.
                        The model specializes to procedural before disruption can redirect.
                        No constructive accumulation of GENERAL structure.

30% ARC (period ~3.3): Forcing period matches AdamW first moment timescale (~10 steps).
                        3 build steps accumulate in m, 1 disrupt step partially decays.
                        NET: constructive accumulation of task-invariant structure.
                        But ONLY when the exact natural frequency matches — fragile.

50% ARC (period ~2.0): Forcing period too short. Every build immediately disrupted.
                        AdamW first moment oscillates without accumulating direction.
                        Destructive interference — WORST performance of all ratios.

70% ARC (period ~1.4): Forcing dominated by disruption. No coherent build phase.
                        m tracks diverse ARC gradients without accumulation.
                        Continuous adaptation without resonance.
```

## Why Reproducibility Failed

The natural frequency depends on the EXACT numerical trajectory of the optimizer state. This trajectory depends on:
1. Random seed → different batch sequences → different gradient history
2. torch.compile behavior → different numerical precision → different gradient values
3. CUDA version → different floating-point accumulation order → different gradient values
4. Code changes → different autograd graph → different backward pass numerics

Any of these shifts the natural frequency by even a fraction of a step. If the resonance window is narrow (as the 50% failure suggests), even a tiny frequency shift breaks the constructive accumulation.

The two original successful runs happened to have natural frequencies that fell within the resonance window of the 30% forcing. The clean reproduction had a different natural frequency (different numerical trajectory from different starting conditions) and the resonance condition wasn't met.

## Testable Predictions

### 1. Seed Sweep Should Find Resonant Seeds
If the hypothesis is correct, some random seeds should produce transitions and others shouldn't, even with identical configs. A sweep of 50-100 seeds should show a bimodal distribution: some seeds transition at ~step 5400, most don't.

Expected: ~5-20% of seeds produce transitions (those whose numerical trajectories create natural frequencies matching 30% forcing).

### 2. Ratio Fine-Tuning Per Seed
For a seed that DOESN'T transition at 30%, there should exist a nearby ratio (e.g., 28% or 33%) where the transition DOES fire. The optimal ratio is seed-dependent because each seed produces a different natural frequency.

### 3. CV Oscillation Frequency Should Predict Transition
Before the transition, CV oscillates slightly around the floor (~3.0-3.5). The FREQUENCY of this oscillation should correlate with whether the transition fires:
- Seeds where CV oscillation period ≈ 3.3 steps: transition fires
- Seeds where CV oscillation period ≠ 3.3 steps: transition doesn't fire

### 4. Adaptive Ratio Should Be More Robust
A controller that monitors CV oscillation frequency in real-time and adjusts the ARC ratio to maintain resonance should produce transitions more reliably than a fixed 30% ratio.

## Implications for the Adaptive Controller

The adaptive criticality controller we built maintained CV in a TARGET ZONE (4.5-6.0). This is the wrong control variable. Instead, it should have maintained RESONANCE between the forcing rhythm and the system's natural frequency.

A resonance-maintaining controller would:
1. Monitor CV (or gradient norm) time series for oscillation frequency
2. Compute the current natural period from autocorrelation
3. Set ARC probability = 1 / natural_period to match forcing to natural frequency
4. Continuously adjust as the natural frequency evolves during training

This addresses the fragility problem: instead of hoping the fixed 30% ratio matches the natural frequency (which depends on unknowable numerical factors), the controller FINDS the matching ratio for whatever the current natural frequency happens to be.

## Connection to Biological Critical Periods

Biological critical periods also have resonance-like properties:
- The visual cortex's critical period requires structured input at specific temporal frequencies
- Stroboscopic illumination at the wrong frequency doesn't trigger orientation column development
- The "right" frequency depends on the neural circuit's intrinsic dynamics (GABA maturation, etc.)
- The critical period is a NARROW WINDOW where intrinsic and extrinsic frequencies align

The LiquidARC phase transition may be the computational analog: a narrow window where the training dynamics' natural frequency aligns with the build-disrupt forcing frequency, enabling constructive accumulation that crosses the critical threshold.

## Next Steps

1. **Seed sweep:** Run 20-50 seeds with identical 30% config. Count how many transition. If 2-8 out of 50 transition, the resonance hypothesis is supported.

2. **Frequency analysis:** For seeds that transition, measure the CV oscillation frequency during the pre-transition plateau. For seeds that don't, measure the same. Compare.

3. **Adaptive resonance controller:** Build a controller that detects natural frequency from CV autocorrelation and adjusts ARC ratio to maintain resonance. Test on seeds that don't transition at fixed 30%.

Each of these is a 30-45 minute experiment (one training run) and could be run in parallel.
