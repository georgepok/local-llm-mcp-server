# LiquidARC Phase Transition — Foundation Conditions

## Purpose

This document records the EXACT conditions under which the LiquidARC self-organizing phase transition was observed, what has been proven to break it, and the practical consequences for future work. 

**CRITICAL: The phase transition is NOT REPRODUCIBLE.** A clean reproduction attempt with the exact documented recipe on a clean codebase FAILED (March 2026). CV stayed at ~3.0 for 15K steps with no acceleration toward threshold. The two original successful transitions depended on conditions we cannot identify — possibly specific PyTorch version, CUDA state, random seed interaction, or other environmental factors.

**All existing post-transition checkpoints are IRREPLACEABLE ARTIFACTS.** They cannot be recreated. Back them up. Never overwrite them.

**Practical consequence: All future experiments MUST resume from existing post-transition checkpoints. The transition itself is a historical observation, not a reproducible procedure.**

---

## Key Rules (from all experiments)

1. **Post-transition checkpoints are irreplaceable.** The transition cannot be reproduced from scratch.
2. **Improvements come from modifying training recipe on existing checkpoints**, not from add-on modules.
3. **Small auxiliary modules (~40-70K params) on frozen base do NOT work.** Tested 4 memory variants, correction nets, metric overlays — all fail or produce noise.
4. **The phase transition is task-contingent.** Fires on ARC (global spatial patterns) but NOT on cellular automata (local rules satisfiable at CV ~3.0).
5. **The transition, when it occurred, was a generalization event** — the model found routing that worked across task types, not just for one type.

---

## 1. The Transition Event (Historical)

Observed in two runs on what was presumably the same codebase state. At approximately step 5,300-5,400:

```
Before (step 5000):   CV ≈ 3.5-5.0, loss ≈ 2.30, eval xform ≈ 15-22%
During (step 5400):   CV crosses ≈ 5.5-6.0, loss begins collapse
After  (step 7000):   CV ≈ 5.0-5.3, loss ≈ 1.20, eval xform ≈ 42-48%

Learning rate during transition: 16 pp / 1K steps
Learning rate post-transition:   0.75 pp / 1K steps
```

**Reproduction status:**
- Original run: transition fired ✓
- First reproduction (same codebase): transition fired ✓
- Metric freeze V2 run (modified codebase, freeze disabled): FAILED ✗
- Clean reproduction (clean codebase, exact recipe, March 2026): FAILED ✗
- Cellular automata run (clean codebase, CA data): No transition (task doesn't demand it)

**The two successes and three failures suggest the transition depends on factors beyond the documented recipe — possibly including specific software versions, compilation behavior, or numerical precision characteristics of the exact runtime environment.**

---

## 2. Architecture (Reference)

```
ARCEmbedding → ContextPool → euler_solve(ContinuousDynamics, h₀, 16 steps) → OutputHead
Total: 572,238 parameters at d_model=256, d_metric=64, d_ffn=512
```

ContinuousDynamics applied 16× via Euler ODE integration:
1. MetricNet → Riemannian metric g (order parameter carrier)
2. SDPA Heat Kernel → routing via K = softmax(-D²/4τ)
3. Value projection → W_v
4. Output projection → W_o (residual: target = h + W_o(SDPA(q,k,V)))
5. TauNet → per-position time constant
6. LTC contraction → dh/dt = -(1/τ)(h - target)
7. FFN residual → dh/dt += FFN(h)/n_steps

---

## 3. Training Recipe (Historical — Not Proven Sufficient)

The following recipe was used for both successful transitions. A clean reproduction with this exact recipe FAILED, so it is necessary but NOT sufficient:

```
Loss: 5.0 × CE(changed) + 0.05 × CE(unchanged) + 0.05 × |κ| - 0.001 × Var(τ) + 0.1 × max(0, 3.0-CV)²
Optimizer: AdamW, lr=3e-4, weight_decay=0.01, warmup 500 steps, cosine decay
Data: 70% procedural (13 rules, infinite stream) + 30% real ARC (400 tasks, augmented)
Batch size: 16, max_seq_len: 2048
geo_loss_enabled: false, tau_freeze_steps: 0
```

---

## 4. What Was Learned from the Transition

Despite non-reproducibility, the transition produced FUNCTIONAL checkpoints that demonstrate:

### 4.1 The Build-Disrupt Ratcheting Mechanism

The 30/70 mix creates a rhythm: procedural batches BUILD coherent metric structure, ARC batches DISRUPT task-specific structure. Only task-INVARIANT structure survives → CV ratchets upward over ~5000 steps.

Evidence: 15% ARC (too little disruption) → no ratcheting. 50% ARC (too much disruption) → destructive interference. 70% ARC (all disruption) → continuous adaptation without accumulation.

### 4.2 Task-Contingent Triggering

The transition fires only when the task demands routing complexity that exceeds CV ~3.0. CA rules (local, neighbor-dependent) are satisfiable at CV 3.0 → no transition needed → model achieves 46% xform without transitioning. ARC rules (global, whole-grid) require CV > 5.5 for non-trivial routing → transition required.

### 4.3 Post-Transition Generalization

The transition eliminated the CV oscillation between task types. Pre-transition: CV crashes 1.5 units on ARC batches. Post-transition: CV crashes only 0.5 units. The model found routing that generalizes across task types.

### 4.4 Verified TTT Bimodal Capability

Post-transition, 36% of ARC tasks are "comprehended" (base 64.8%, TTT improves to 77.1%). 64% are "not comprehended" (TTT destructive). The verification gate perfectly separates these.

---

## 5. Post-Transition Experiments (Complete Record)

### What Works

| Approach | Mechanism | Result |
|----------|-----------|--------|
| Sequential 30%→50% | Changed ratio after transition | 47.8% peak (BEST STABLE) |
| Adaptive controller | Dynamic ratio based on CV | 55.6% peak (HIGH VARIANCE) |
| Verified TTT | Per-task adaptation with gate | 48.5% overall |

### What Doesn't Work

| Approach | Result |
|----------|--------|
| Working Memory v1 (h residual) | -14.4% xform (copy bias) |
| Working Memory v2 (h + overlay) | -6.6% xform (copy bias) |
| Working Memory v3 (overlay only) | Noise (±2%) |
| Working Memory v4 (observe + correct) | Noise (±10%) |
| Metric Freeze | ≈0% (model compensates) |
| Multi-Pass Inference | -7% per pass (reinforces errors) |
| Enhanced TTT (500 steps) | Overfitting, not learning |

### Key Conclusions

- Add-on modules on frozen base NEVER work (tested 7 approaches)
- Training recipe changes on post-transition checkpoints DO work
- The model's representation is too tightly co-adapted for external modules
- The ~48% ceiling is in computation (FFN), not routing (MetricNet)

---

## 6. Known-Good Checkpoints

**THESE ARE IRREPLACEABLE. BACK THEM UP.**

```
/workspace/liquid-arc/output_reproduce/checkpoints/step_7500.pt    (30% ARC, post-transition)
/workspace/liquid-arc/output_30to50/checkpoints/best.pt            (sequential, peak ~47.8%)
/workspace/liquid-arc/output_adaptive_criticality/checkpoints/     (adaptive controller run)
```

---

## 7. Open Directions

1. **Multi-domain ratcheting:** Resume from checkpoint, train with diverse spatial tasks (procedural + CA + conditional transforms) + 30% ARC disruption. Tests whether computational generalization follows the same ratcheting principle as routing generalization.

2. **Post-transition fine-tuning:** Freeze MetricNet/TauNet, unfreeze FFN + output head, train on 100% real ARC. Specializes computation while preserving routing.

3. **Scaling:** The 5M model (d=768) transitioned at CV ~7.0. The sequential curriculum on the 5M checkpoint is untested.

4. **Non-ARC applications:** The spatial routing at CV ~5.0 is genuinely useful for spatial perception tasks (UI layout, change detection) even without reasoning capability.

5. **Understanding reproducibility failure:** Systematic investigation of what environmental factors enabled the original transitions. PyTorch version, CUDA version, specific random seeds, exact git commit.

---

## 8. Version History

- **2026-03-17 v4**: Clean ARC reproduction FAILED. Transition confirmed non-reproducible.
  - Rewrote document framing from "reproducible phenomenon" to "historical observation"
  - Added CA domain transfer results (task-contingent triggering)
  - Consolidated all post-transition experiment results
  - Emphasized checkpoint preservation

- **2026-03-17 v3**: Working memory v1-v4 results (all failed)
- **2026-03-16 v2**: Reproduction failure from code changes, checkpoint preservation policy
- **2026-03-16 v1**: Initial foundation document
