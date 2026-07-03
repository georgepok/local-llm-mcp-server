# LAYER-WISE PERTURBATION RESULTS — Analysis & Next Steps

## What Phase 1 Validated

The perturbation architecture WORKS mechanically:
- Correction ratio stabilizes at 1.3-2.9% of residual norm (well under 0.5 threshold)
- B_within/B_across separation emerges at mid/late depth (structurally correct)
- No D² runaway from the ODE itself — the 146K D² is the residual stream through MetricNet
- Architecture is stable across 36 layers

## What Phase 1 Shows

The pre-transition checkpoint (CV=0.4) is the bottleneck, not the architecture:

1. MetricNet AMPLIFIES (g > 1) → residual D² grows 80,000× through depth, regardless of ODE
2. MetricNet doesn't DIFFERENTIATE (CV=0.4 ≈ uniform) → no structured routing at any ε
3. At ε=0.1, correction is 2% of residual → structurally correct but too weak to change generation
4. Even at higher ε, an amplifying uniform MetricNet can't produce the structured compression that routing needs

## Recommended Next Steps (ordered)

### Step 1: ε sweep (quick experiment, ~1 hour)

Run the causal chain + parallel chains test suite at ε = {0.2, 0.5, 1.0}:

```
ε=0.2:  correction ~5% of residual. Does B_across improve?
ε=0.5:  correction ~12% of residual. Does the 5-hop chain pass?
ε=1.0:  correction ~25% of residual. Maximum perturbation before full coupling.
```

Track: correction_ratio, D²_corrected (not D²_residual), B_within, B_across, causal chain score.

If ε=0.5 produces 6/6 on causal chains → we have a working architecture at this checkpoint. Publish results and move to criticality training for refinement.

If ε=1.0 still produces 5/6 → the pre-transition MetricNet CAN'T produce useful routing at any coupling strength. Move to Step 2.

### Step 2: Train d=2048 checkpoint with sustained criticality (if ε sweep fails)

Use the validated criticality scaffolding from the d=768 ARC experiments:
- D²/4τ loss (λ=0.01, target=18 or scale to d=2048)
- tau_quality_loss (λ=0.05, mean=1.0, log_spread=0.6)
- Convergence coupling (beta=1.0)
- CV floor/ceiling (3.0 / 8.0)
- NO tau_var_loss (replaced by tau_quality)

Train on ARC data for ~6000 steps (same as d=768 experiments).
Target metrics: CV≈7, D²/4τ≈18 (or d=2048 equivalent), tau_mean≈1.0.

Then re-run the layer-wise architecture with the post-transition checkpoint.
Expected: MetricNet COMPRESSES (amp < 1.0) → D² stays bounded through depth.
The perturbation at ε=0.1 would carry structured routing signal because the MetricNet differentiates which dimensions matter.

### Step 3: Layer-wise criticality training (longer term)

Train with the layer-wise hooks active — the criticality system operates per-layer during Qwen3-4B forward passes. This produces a MetricNet that's specifically adapted to the depth-varying residual stream, not just to ARC input distributions.

This is the end-state architecture: MetricNet trained for depth-appropriate routing, layer-wise co-processing, perturbation-bounded, sustained criticality at every depth.

## The Structural Finding Worth Preserving

The B_within/B_across pattern through depth:

```
             Early     Mid      Late
B_within:   -0.10    +0.14    +0.13   (turns positive — within-event cohesion)
B_across:   -0.22    -0.30    -0.26   (stays negative — cross-event separation)
```

This shows the architecture DISCOVERS event boundaries through depth, even with a pre-transition checkpoint at 2% coupling. The signal is real. It just needs amplification — either through higher ε or through a MetricNet that produces stronger differential routing.

This is the validation that the layer-wise architecture adds something no single-hook approach provides: progressive geometric evolution through computation depth.
