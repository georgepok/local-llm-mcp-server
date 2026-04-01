# Multi-Pass Inference Diagnostic — LiquidARC 572K

## Summary

Multi-pass inference **does not help** — it actively degrades performance. The model's errors are fundamental (wrong rule), not refinable (imprecise output).

```
============================================================
Multi-Pass Inference Results
============================================================
Checkpoint: output_30to50/checkpoints/best.pt (step 15000, 30%→50% sequential)
Tasks evaluated: 298
Tasks skipped (seq_len): 121
Time: 24.1s (0.08s/task)

Pass 1: xform_acc=47.2%, cell_acc=16.1%, tasks_solved=1/298 (0.3%)
Pass 2: xform_acc=41.4%, cell_acc=13.7%, tasks_solved=1/298 (0.3%)
Pass 3: xform_acc=40.2%, cell_acc=13.2%, tasks_solved=1/298 (0.3%)

Cell changes 1→2: 9384 changed, 662 improved, 1527 worsened (net: -865)
Cell changes 2→3: 5019 changed, 399 improved, 569 worsened (net: -170)

Pass 1→2: 0 new solves, 0 regressions
Pass 2→3: 0 new solves, 0 regressions
```

## Interpretation

### The errors are fundamental, not refinable

- **47.2% → 41.4% → 40.2%**: Each pass makes things worse
- **Net -865 cells** degraded on Pass 1→2: model changes ~9,400 cells but worsens 2.3× more than it improves
- **Zero new solves**: Not a single task was rescued by self-refinement
- **Pass 2→3 is smaller change** (5,019 vs 9,384): the model is converging toward a fixed point — its own confident (but wrong) prediction

### What this means

The model knows **WHERE** cells need to change (geometric routing is working — 47% of transform cells correct) but applies the **WRONG transformation** to many of them. Seeing its own wrong output doesn't help because:

1. The wrong predictions look plausible to the model — it doesn't recognize its errors
2. Adding the wrong prediction as a "demo" teaches the model the wrong rule
3. The model then applies that wrong rule more confidently, degrading accuracy

### Implications for next steps

| Finding | Implication |
|---------|------------|
| Multi-pass hurts | Refinement mechanisms (Level 2) have low ceiling |
| Errors are wrong-rule, not imprecise | Need more computational capacity, not better iteration |
| Only 1/298 tasks solved | Task solve requires ALL cells correct — even 47% xform acc is far from solving |
| 121 tasks skipped (seq_len) | 30% of eval tasks exceed 2048 tokens — longer context would help coverage |

**Recommended direction**: Parallel computation pathway (Level 3) — the model needs to consider multiple candidate transformations and select the best one, not iteratively refine a single guess. Alternatively, scaling d_model to increase the model's rule-encoding capacity.

## Experimental Details

- **Model**: LiquidARC 572K (d=256), post-phase-transition, 30%→50% sequential curriculum
- **Checkpoint**: step 15000, peak eval xform ~47.8%
- **Multi-pass method**: Previous prediction added as extra demo pair (test_in → predicted_out) before the test. If sequence too long, drops earliest original demo.
- **No training, no gradients, no TTT** — pure frozen inference
- **298 tasks evaluated** (121 skipped due to sequence length > 2048)
