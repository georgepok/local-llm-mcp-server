# Geometric Engine v2.1 — Patch Instructions

## Context
- **Date**: 2026-02-08
- **Prerequisite**: Read `VALIDATION_RESULTS.md` — the engine disabled itself
- **Reference**: `GEOMETRIC_ENGINE_SPEC.md`, `IMPLEMENTATION.md`, `DEV_AGENT_TASKS.md`
- **Engine File**: `/home/pokazge/models/fluid_geometry.py`

---

## What Happened

The stability monitor detected the engine was degrading generation quality and decayed `confidence_override` to 0.06, effectively disabling itself. This is the safety system working correctly. The problem is upstream — three cascading issues in the signal processing chain.

---

## Fix 1: Smooth Entropy Before Differentiating

**Priority**: CRITICAL — this is the root cause

**Problem**: Raw entropy values are spiky (many H=0.000 tokens interspersed with occasional H=0.5–1.4). Taking second derivatives of a spiky signal produces noise, not curvature. The trace shows temperature oscillating between 0.1 and 1.76 on consecutive tokens — that's the engine responding to differentiation artifacts, not genuine geometry.

**What to do**: Apply an exponential moving average to entropy BEFORE computing first and second derivatives. The EMA should have a window equivalent to roughly 8–16 tokens — enough to smooth single-token spikes into gradual transitions while preserving genuine shifts between confident and uncertain generation.

The derivatives (dH, d²H) should be computed on the smoothed signal, not the raw signal. The raw H can still be stored for diagnostics but should not feed the curvature calculation.

**Why this works**: The interesting signal IS there — transitions from H≈0 to H≈1 mark boundaries between committed and exploratory generation. Smoothing converts δ-function spikes into bell curves centered on these transitions. The curvature of a smoothed entropy curve at a transition point tells you "how quickly is the model shifting mode" — a genuinely useful geometric signal.

**Verify by**: After fix, run 10 prompts with FG_TRACE=1 and inspect the trace. Temperature should vary smoothly over spans of 5–20 tokens, not oscillate wildly between consecutive tokens. κ values should be smaller in magnitude and more sustained.

---

## Fix 2: Fix Stability Monitor Calibration

**Priority**: CRITICAL — this is what kills the engine

**Problem**: `baseline_perplexity = 1.29` is artificially low. The current estimation uses `max(softmax(logits))` which captures peak probability, not the probability of the token actually selected. Real generation perplexity is higher than 1.29, so the 5% tolerance band (threshold ≈ 1.35) is impossibly tight. Any temperature modulation raises perplexity above this, triggering pullback.

**What to do**: Replace the hard-threshold comparison with drift detection. Instead of "is current perplexity more than 5% above baseline," use "has the rolling average perplexity shifted significantly relative to its own variance." Concretely:

- Track rolling perplexity mean AND variance (EMA of both)
- Trigger pullback only when rolling mean exceeds baseline by more than 2 standard deviations of the rolling variance
- This makes the monitor adaptive to the actual noise level rather than anchored to an absolute threshold

Additionally, the baseline itself should use the same estimation method as the rolling window. If both use the same biased estimator, the bias cancels in the comparison. The problem was that the baseline was set during low-confidence operation (engine doing nothing) and compared against high-confidence operation (engine actively modulating). Make sure baseline updates continue throughout operation so it tracks the engine's effect, not just the raw model's behavior.

**Verify by**: After fix, run convergence test (60+ prompts). `confidence_override` should remain above 0.8 throughout. If it drops below 0.5, the fix didn't work.

---

## Fix 3: Normalize Curvature by Entropy Scale

**Priority**: IMPORTANT — prevents over-reaction in low-entropy regime

**Problem**: This model operates at very low entropy (0–1.4 bits). A κ of 0.5 when average entropy is 1.0 bit represents a massive relative change. The same κ of 0.5 when average entropy is 8.0 bits is negligible. Without normalization, the engine over-reacts because it treats absolute curvature as meaningful regardless of the entropy regime.

**What to do**: Scale the curvature-to-temperature mapping by the current entropy regime. When entropy is low, the same absolute κ should produce a smaller temperature adjustment than when entropy is high. One approach: divide κ by a running estimate of typical entropy magnitude before applying the temperature law. Another: scale the gain factor in the temperature law by something proportional to mean entropy.

The goal is that at operational confidence, temperature should typically vary in a range like 0.85–1.15 for normal generation, with excursions to 0.7–1.5 only during genuine mode transitions. The current range of 0.1–1.76 is far too aggressive.

**Verify by**: After fix, trace should show T values clustering between 0.85–1.15 with occasional wider excursions. The floor (T=0.1) should almost never be hit during normal generation.

---

## Fix Order

Apply fixes in order: 1 → 2 → 3. Each fix reduces the severity of the downstream problem.

After all three fixes:

1. Reset state: delete the state file so engine starts fresh
2. Re-run convergence test (Task 2 from DEV_AGENT_TASKS.md) — 60+ prompts
3. Inspect converged state (Task 3) — confirm `confidence_override` stays above 0.8
4. Run trace (Task 4) — confirm temperature varies smoothly and moderately
5. Run A/B comparison (Task 5) — this is the test that was blocked before

---

## What NOT to Change

- The phase space model (H, dH, d²H → κ) is sound. The issue is signal quality, not the framework.
- The confidence ramp (C = 1 - exp(-t/10000)) is fine. Don't change it.
- The think-token bias logic is untested but architecturally correct. Leave it alone until temperature modulation is validated.
- State persistence, token ID resolution, and the vLLM integration are all working. Don't touch them.

---

## Success Criteria for v2.1

After all fixes and re-validation:

1. `confidence_override` remains above 0.8 after 20K+ tokens
2. Temperature trace shows smooth, moderate variation (0.85–1.15 typical)
3. κ values are sustained over token spans, not single-token spikes
4. A/B comparison shows engine-active responses are at least as good as engine-disabled
5. No container errors or state file corruption

Write results to `VALIDATION_RESULTS_V2.1.md` in this directory.
