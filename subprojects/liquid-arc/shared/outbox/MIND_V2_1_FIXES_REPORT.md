# Mind v2.1 Critical Observations — Fix Report

## Changes Made

### 1. Fixed: Delta normalization destroying magnitude variation
**Root cause:** LayerNorm normalized every delta to unit variance (L2 norm ≈ √d ≈ 50). All tokens became equidistant.

**Fix:** Replaced LayerNorm with RMS normalization (mean-subtract + divide by RMS). Preserves relative magnitudes — function words stay small, content words stay large.

**Impact:**

| Metric | Before (LayerNorm) | After (RMS norm) |
|--------|-------------------|-------------------|
| Delta norm range | 49-50 (all same) | 12-214 (varies) |
| D² distribution | UNIFORM (p90/p50=1.28) | HEAVY TAIL (p90/p50=155) |
| MetricNet amplification | 0.99× (identity) | 11.7× (active!) |
| tau_std | 2.5e-7 (zero) | 0.45-1.20 (differentiating) |
| CV | 0.5 | 0.59-0.93 |

### 2. Fixed: D²/4τ diagnostic producing negative/broken values
**Root cause:** SDPA-factored bias `B_ij = q·k/(2t) - ||k||²/(4t)` can produce positive B values, making reconstructed D² negative.

**Fix:** Added actual sampled pairwise D² computation independent of SDPA factorization. Also added tau_logit diagnostics (pre-sigmoid TauNet activations).

**New diagnostic output:**
```
[bias] [126x126] CV=0.59 D²=73.7±1955 D²/4τ=9.0 tau=0.38±0.53 tau_logit=-2.31±2.32
```

### 3. Fixed: h_norm_per_position not tracked
**Fix:** Added `h_norm / sqrt(n_tokens)` to observe_event logs and get_diagnostics. Shows per-position norm stays bounded at ~63 (homeostasis working).

### 4. Fixed: Thinking trace leakage
**Fix:** Post-processing in QwenBridge strips `<think>` tags and common reasoning preambles ("Okay, the user is asking..."). Not perfect — some traces still leak on edge cases.

### 5. Added: Adaptive D² EMA tracking
**Purpose:** Track actual D² distribution via exponential moving average for future adaptive criticality target.

### 6. Verified: Checkpoint geometry intact on ARC
**Result:** ARC inputs → CV=6.2, D²/4τ=12, tau=2.05. The MetricNet is at criticality for ARC. Text deltas are the issue, not model degradation.

### 7. Characterized: Text delta distribution
**Key finding:** With RMS normalization, text deltas produce:
- D² median=73.7 (metric-weighted), D²/4τ=9.0
- Heavy-tailed distribution (p90/p50=155) — natural clustering
- MetricNet amplifies 11.7× (actively shaping geometry)

This is dramatically different from ARC (D²/4τ≈12, CV≈6.2). The MetricNet trained on ARC's factored embeddings (color+position+role) amplifies text deltas differently — it finds structure but at a different scale.

## Remaining Issues

1. **CV still sub-critical (0.59 vs ARC's 6.2):** The MetricNet amplifies text deltas but the overall metric differentiation is lower than ARC. This may require online adaptation of MetricNet to text inputs.

2. **D² median = 0 in live deployment:** Many token pairs within the same event have near-zero D² (after ODE processing they converge). The distribution is bimodal: near-zero for same-event pairs, large for cross-event pairs. Use mean or p75 instead of median for the diagnostic.

3. **Responses include context echoing:** "Temporal: Mind initialized. ODE encoder active." appears in responses because the context prompt includes system events. Should filter system events from context.

4. **Relevance scoring still flat:** Readout heads untrained. Lower priority — requires feedback signal.

## Architecture Status After Fixes

| Component | Status | Evidence |
|---|---|---|
| Token extraction (Δh) | Working | Heavy-tailed distribution, magnitude variation preserved |
| Token buffer management | Working | 8→109→286→391 tokens |
| MetricNet on text | ACTIVE | 11.7× amplification, CV=0.6-0.9 |
| Tau differentiation | ACTIVE | tau_std=0.45-1.20 (was 2.5e-7) |
| PE discrimination | Working | 55.6→53.2→5.0 (topic-sensitive) |
| Bias dimensions | Working | [9×9]→[124×124]→[289×289] |
| Post-hoc ODE feedback | Working | Generated tokens feed back |
| Criticality regime | Partial | D²/4τ=9 on text vs 12 on ARC |
| Generation quality | Fair | Qwen3-4B limitations, some trace leakage |
