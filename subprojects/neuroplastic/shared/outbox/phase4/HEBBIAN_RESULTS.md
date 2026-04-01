# Phase 4: Hebbian Learning Results

**Date:** 2026-03-11
**Model:** NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
**Target layers:** 44, 46, 48, 50 (Mamba layers, attention-free tail)
**Learning rate:** 0.01
**Homeostasis:** L2 norm preservation + 0.999 global decay

---

## Summary

**The Hebbian experiment produced a negative result.** Activity-driven Mamba A modifications via per-head `add_slice` operations degraded the model from 20% baseline to 16% during adaptation, and catastrophically destroyed coherent generation by Phase D (0/12 capability eval).

| Phase | Accuracy | Description |
|-------|----------|-------------|
| **A (Baseline)** | 20.0% (10/50) | Unmodified model on 50 state-tracking problems |
| **B (Hebbian)** | 16.0% (8/50) | With Hebbian updates after each problem (-4%) |
| **C (Retention)** | 20.0% (4/20) | 20 new problems, post-adaptation weights |
| **D (Cross-domain)** | 0.0% (0/12) | Full capability eval — model producing garbage |

---

## Phase A: Baseline Accuracy by Difficulty

| Difficulty | Pass | Total | Accuracy |
|-----------|------|-------|----------|
| 1 (easy) | 6 | 10 | 60% |
| 2 | 2 | 15 | 13% |
| 3 | 2 | 15 | 13% |
| 4 (hard) | 0 | 10 | 0% |

The model can solve easy single-entity state tracking but fails on multi-entity, multi-step problems.

---

## Phase B: What Happened

The Hebbian update loop executed correctly:
1. **TRACE** each problem (capture per-head activation norms via `head_norm_mean`)
2. **Compute** per-head deltas: active heads get negative delta (slower decay = preserve), dormant heads get positive delta (faster decay = prune)
3. **Apply** via `add_slice` operations (64 heads x 4 layers = 256 API calls per problem)
4. **Homeostasis**: normalize each layer back to original L2 norm

Over 50 problems, the L2 norms barely drifted:
- Layer 44: 16.3864 -> 16.3700 (stable)
- Layer 46: 1073.27 -> 1072.20 (0.1% drift)
- Layer 48: 21785.43 -> 21763.66 (0.1% drift)
- Layer 50: 7378.73 -> 7371.34 (0.1% drift)

Despite small norm changes, **the direction of weight vectors changed substantially**. By Phase D, the model output was entirely degenerate: repeated Unicode replacement characters with occasional English words ("structured", "that", "buffer").

---

## Phase D: Catastrophic Degradation

All 12 capability tests returned garbage. Example response for "What is A?" (should be 14):
```
structured that [hundreds of replacement characters]
```

The model lost all coherent generation ability while the Hebbian updates were still applied.

**After weight restoration** (experiment protocol restores pre-adaptation checkpoint), the model immediately returned to normal: "2 + 3 = 5" — confirming the degradation was caused by the Hebbian weight modifications, not API/container issues.

---

## Analysis: Why It Failed

### 1. Direction vs. Magnitude
Homeostatic L2 normalization preserves magnitude but allows direction to drift. After 50 rounds of Hebbian redistribution (each followed by re-normalization), the cumulative directional change was enough to destroy the model's learned representations even though norms stayed within 0.1%.

### 2. The Hebbian Signal Was Too Noisy
`head_norm_mean` (mean activation magnitude per head) is a very coarse signal. It tells you which heads were active on average, but not whether that activity was related to state-tracking computation. The Hebbian rule strengthened heads that happened to be active on state-tracking prompts — but those same heads may be critical for other computations (like coherent text generation).

### 3. The Update Rule Was Monotonic
The simplified Hebbian rule (active = preserve, dormant = prune) has no mechanism to distinguish *useful* activity from *incidental* activity. Every active head gets the same treatment regardless of whether its activity contributed to a correct answer.

### 4. Homeostasis Was Necessary But Insufficient
Without homeostasis, norms would have exploded (runaway strengthening). With homeostasis, norms were preserved but the direction change was unconstrained. We needed directional homeostasis (e.g., constraining the angle of change per step), not just magnitude homeostasis.

---

## Comparison: Hebbian vs. Score-Driven (Phase 3)

| Metric | Phase 3 (Score-driven) | Phase 4 (Hebbian) |
|--------|----------------------|-------------------|
| Peak accuracy | 91.7% (11/12 eval) | 20% (baseline, no improvement) |
| State tracking best | 3/3 (once) | Never improved |
| Degradation | Gradual, recoverable | Catastrophic, total |
| Update mechanism | LLM-reasoned scalar scaling | Activity-driven per-head add |
| # modifications to break | ~15-20 | ~50 (accumulated) |

Score-driven modification (Phase 3) was dramatically more effective. The LLM could reason about which parameters to change and by how much. Hebbian updates are blind — they follow activity patterns regardless of task relevance.

---

## Key Insight

The CURRENT_TASK hypothesis was: "The structure learns because it was active. Not because it was evaluated." The experiment falsifies this for the current setup:

**Activity-driven modification without task-relevant signal selection is worse than random.** The Hebbian rule doesn't know which activations are relevant to the task. It strengthens whatever happened to fire, which includes irrelevant computation. The cumulative effect is noise injection that degrades the model.

Biological Hebbian learning works because:
1. The learning rule operates on individual synapses during forward pass (not post-hoc)
2. Neuromodulators (dopamine, etc.) gate which synapses actually update (reward signal)
3. Local inhibition provides competitive dynamics (not just global L2 normalization)

None of these are available through the neuroplastic API's external trace + modify interface.

---

## What Would Be Needed

For activity-driven learning to work, we would need:
1. **Finer-grained traces**: Per-token, per-head activation vectors (not just mean norms)
2. **Contrastive signal**: Compare activations on correct vs. incorrect problems to identify task-relevant heads
3. **Directional constraints**: Limit angle of change per update step (not just L2 norm)
4. **In-model updates**: Modifications during forward pass, not between forward passes
5. **Selective gating**: Only update heads whose activity correlates with correct answers

Items 1-3 could potentially be implemented with the existing API. Items 4-5 would require changes to the neuroplastic plugin itself.

---

## Files

```
phase4_hebbian/
  hebbian_engine.py          # Core Hebbian update engine
  training_inputs.json       # 50 state-tracking problems
  experiment_protocol.py     # Four-phase experiment runner
  results/
    phase_a_results.json     # Baseline: 20% (10/50)
    phase_b_results.json     # Hebbian: 16% (8/50)
    phase_c_results.json     # Retention: 20% (4/20)
    phase_d_results.json     # Cross-domain: 0% (0/12) -- catastrophic
    phase_d_eval/            # Full eval harness output
    phase4_summary.json      # Combined results
```
