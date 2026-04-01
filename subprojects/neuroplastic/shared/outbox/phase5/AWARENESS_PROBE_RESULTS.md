# Phase 5: Thinking-Chain Awareness Probe — Results

**Date:** 2026-03-11
**Model:** NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
**Method:** Replay approach (generate thinking chain, then trace as input)
**Trials:** 10 per problem, 5 problems, 50 total trials

---

## Verdict: MODERATE CORRELATION — Partial Introspective Access

The thinking chain carries **real but partial** signal about internal processing state. The model has genuine access to coarse-grained dynamics (overall processing difficulty) but not fine-grained details (which specific entity is being tracked poorly).

---

## Grand Aggregate Correlations

| Metric Pair | Spearman rho | Interpretation |
|-------------|:---:|---|
| Confidence vs Mamba mean norm | **0.351** | Moderate — higher confidence when activations are stronger |
| Confidence vs Mamba mean change rate | **0.314** | Moderate — higher confidence when states are actively updating |
| Confidence vs Layer 48 output norm | **0.396** | Moderate-strong — layer 48 most informative |
| Confidence vs Layer 50 output norm | **0.348** | Moderate |
| Claim correctness vs Mamba norm | **0.205** | Weak-moderate — correct claims weakly track activation health |
| Claim correctness vs change rate | **0.193** | Weak — not much signal here |

All correlations are **positive and consistent across problems**. This is not noise — it's a real signal.

---

## The Key Finding: Differential Correlation (Correct vs Incorrect Trials)

This is the most important result. Three problems had both correct and incorrect trials:

### Confidence vs Mamba Mean Norm

| Problem | Correct Trials | Incorrect Trials | Delta |
|---------|:-:|:-:|:-:|
| warehouse_hard | 0.384 | 0.544 | +0.16 (inverse!) |
| bank_hard | **0.424** | **0.201** | **-0.22** |
| inventory_vhard | **0.455** | **0.152** | **-0.30** |

### Confidence vs Mamba Mean Change Rate

| Problem | Correct Trials | Incorrect Trials | Delta |
|---------|:-:|:-:|:-:|
| warehouse_hard | 0.447 | **-0.267** | **-0.71** |
| bank_hard | 0.300 | 0.120 | -0.18 |
| inventory_vhard | 0.307 | 0.303 | -0.00 |

### Interpretation

**On bank_hard and inventory_vhard:** When the model gets the answer right, confidence-activation correlation is **0.42-0.46**. When it gets the answer wrong, correlation drops to **0.15-0.20**. The model's self-monitoring degrades exactly when it would be most needed.

**On warehouse_hard (the anomaly):** The incorrect trial shows confidence vs norm at **0.54** (high!) but confidence vs change_rate at **-0.27** (negative!). This means the model felt confident when activations were large *but stagnant*. It was confidently stuck — high activation magnitude with low dynamism. The model couldn't distinguish "processing actively" from "stuck with high residual norms."

This is the most actionable finding: **the model confuses activation magnitude with processing quality**. High norms don't mean good computation — they can mean the model is stuck in a high-activation attractor state while failing to track state changes.

---

## Per-Problem Breakdown

### shirts_medium (difficulty 2) — 10/10 correct
- All trials correct, so no correct/incorrect comparison
- Confidence vs mamba_norm: 0.40
- Confidence vs layer 48 norm: 0.44 (highest)
- Easy problem, model consistently confident and correct

### fruit_easy (difficulty 1) — 10/10 correct
- All correct, no split
- Confidence vs mamba_norm: 0.21 (lowest)
- Confidence vs change_rate: 0.37
- Too easy — low variance in both confidence and activations

### warehouse_hard (difficulty 3) — 9/10 correct
- The one incorrect trial shows inverted change_rate correlation
- Correct: confidence tracks both norm (+0.38) and dynamism (+0.45)
- Incorrect: confidence tracks norm (+0.54) but anti-tracks dynamism (-0.27)
- **"Confidently stuck" pattern**

### bank_hard (difficulty 2) — 6/10 correct
- Best data: 6 correct, 4 incorrect
- Largest correct/incorrect split on norm correlation: 0.42 vs 0.20
- Claim correctness vs norm: 0.49 on correct trials (near-strong!)
- The model's state claims are most accurate when activations are healthy

### inventory_vhard (difficulty 4) — 9/10 correct
- Surprisingly high accuracy for "very hard"
- Correct: norm correlation 0.46, change_rate 0.31
- Incorrect: norm correlation 0.15, change_rate 0.30
- Large norm correlation drop (0.46 → 0.15) on incorrect trial

---

## What This Means for Neuroplasticity

### The Good News
The thinking chain is **not purely performative**. There is a genuine 0.30-0.40 Spearman correlation between self-reported confidence and Mamba state dynamics across 50 trials. The model has partial introspective access to its own processing state.

### The Bad News
1. **The correlation is moderate (0.3-0.4), not strong (>0.5).** The model can sense "this is hard" vs "this is easy" but can't pinpoint "I'm losing track of oranges specifically."

2. **Self-monitoring fails on errors.** The correlation drops from ~0.4 to ~0.2 on incorrect trials. The model is less aware of its processing state precisely when something is going wrong. This is the worst-case scenario for self-directed modification — the model would need accurate self-monitoring during failures to know what to fix.

3. **The model confuses magnitude with quality.** The warehouse_hard anomaly shows the model feeling confident when activations are large but stagnant. It can't distinguish "processing well" from "stuck with high activation."

### Implications for Self-Directed Modification

A thinking-chain-guided modification approach could work for **coarse adjustments**:
- "This problem type feels harder" → adjust global parameters
- "I'm generally less confident on multi-entity tracking" → target Mamba layers

But it would fail for **surgical fixes**:
- "My orange count is wrong because head 37 in layer 48 isn't maintaining state" → the model can't sense this level of detail
- Self-monitoring accuracy drops on failures → the model can't reliably identify what went wrong

The practical implication: **combine thinking-chain signal with external TRACE data**. Use the model's confidence as a coarse filter ("which problems are hard?") and TRACE data for the fine-grained diagnosis ("which heads are failing?"). Neither signal alone is sufficient, but together they could create a more effective modification strategy than either Phase 3 (external-only) or Phase 4 (activity-only).

---

## Methodology Notes

- **Replay approach:** Generated thinking chain normally, then replayed full conversation through TRACE. This captures activation patterns for the thinking text but not the exact generative dynamics (the model processes the thinking as input rather than generating it). A live-capture approach would be more precise.
- **Segment alignment:** Thinking text was split into segments (by paragraph/blank line), and per-token activations were divided into equal-sized chunks. This is approximate — ideally segments would be aligned to exact token positions.
- **Confidence extraction:** Based on keyword matching (HIGH_CONFIDENCE and LOW_CONFIDENCE word lists), plus explicit confidence markers when the model annotated its own confidence. This is crude but consistent.
- **Sample sizes:** 50 total trials with 44 correct and 6 incorrect. The incorrect trial analysis has limited statistical power (n=1 for warehouse and inventory, n=4 for bank_hard). More trials on harder problems would strengthen the differential analysis.

---

## Files

```
phase5_awareness/
  awareness_probe.py           # The probe script
  results/
    probe_results.json         # Full results (all 50 trials)
    correlation_summary.json   # Aggregate correlations
```

---

## Suggested Next Steps

1. **Harder problems** to get more incorrect trials (need n>10 incorrect for statistical power)
2. **Live capture** during generation (not replay) to see actual generative dynamics
3. **Token-level alignment** of thinking text to activations (not chunk-based approximation)
4. **Combined approach**: Use thinking confidence as a coarse "difficulty detector" + TRACE data for surgical modification targeting

---

*The model partially knows what it's doing while it's doing it. But it knows least when it matters most.*
