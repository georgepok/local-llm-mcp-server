# Verified TTT Evaluation Report

```
============================================================
Verified TTT Results
============================================================
Checkpoint: /workspace/liquid-arc/output_30to50/checkpoints/best.pt
Tasks evaluated: 279
Tasks skipped: 121
Time: 1179.4s (4.2s/task)

--- Transform Accuracy ---
Base model                              : 47.1%
Standard TTT                            : 40.4%
Verified TTT (loss<0.01)                : 48.5%
Partial TTT (loss<0.1)                  : 48.5%

--- Task Solve Rate ---
Base model                              : 1/279 (0.4%)
Standard TTT                            : 2/279 (0.7%)
Verified TTT                            : 2/279 (0.7%)
Partial TTT                             : 2/279 (0.7%)

--- Gate Statistics ---
TTT converged (loss < 0.01): 101 (36.2%)
TTT partially converged (loss < 0.1): 48 (17.2%)
TTT failed (loss >= 0.1): 130 (46.6%)

--- Verified TTT vs Base ---
New solves (verified TTT solves, base doesn't): 2
Regressions (base solves, verified TTT doesn't): 1
Net new solves: 1

--- Standard TTT vs Base ---
New solves (TTT solves, base doesn't): 2
Regressions (base solves, TTT doesn't): 1
Net new solves: 1

--- Per-Gate Accuracy (TTT result only, not gated) ---
verified_ttt (101 tasks): base=64.8%, ttt=77.1%, delta=+12.4%
partial_ttt (48 tasks): base=39.5%, ttt=35.8%, delta=-3.7%
base_fallback (130 tasks): base=39.2%, ttt=25.7%, delta=-13.4%
```
