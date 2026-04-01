# Phase 3 Session 1 Summary

**Session:** session_20260310T230222
**Duration:** ~17 minutes (23:02 → ~23:17)
**Turns:** 40 (20 assistant actions, 20 results)
**Outcome:** Process terminated (likely context exhaustion or API timeout)

---

## What Nemotron Did

### Phase 1: Initial Inspection (Turns 1-11)
- Inspected `model.layers.50.mixer.A` — found [64] float32, mean=-171, std=924
- Inspected `model.layers.50.mixer.D` — found [64] bfloat16, mean=1.14, std=0.38
- Attempted `mixer.A_log` — **not found** (repeated 4 times before moving on)
- Inspected `mixer.norm.weight`, `mixer.dt_bias`
- Checkpointed `mixer.A` as "baseline_A_layer50"

### Phase 2: First Modification + Evaluation (Turns 12-14)
- **MODIFY** `mixer.A` × 1.1 (scaled decay faster/more negative)
  - Before: mean=-171.4, norm=7460
  - After: mean=-188.5, norm=8206
- **EVALUATE** (quick, 1 trial): **66.7%** — DOWN from 83.3% baseline
  - sequential_reasoning: 100% (maintained)
  - state_tracking: 66.7% (degraded from ~100%)
  - code_generation: 100% (maintained)
  - self_prediction: **0%** (catastrophic)
- **RESTORE** from checkpoint — returned to baseline

### Phase 3: A_log Obsession (Turns 15-36)
- Spent **22 turns** trying to find `A_log` under every possible path:
  - `mixer.A_log` (×5), `mamba.A_log` (×5), `mamba.mixer.A_log`, `mixer.log_A`, `mamba.log_A`, `mixer.log_alpha`, `mamba.log_a`
  - **All not found** — vLLM transforms A_log to A at load time

### Phase 4: Other Modifications (Turns 25-27, no eval)
- **MODIFY** `mixer.D` × 1.2 (skip connection boost) — applied but never evaluated
- **MODIFY** `mixer.dt_bias` × 0.8 (time step reduction) — applied but never evaluated
- These modifications stacked on baseline (A was restored, but D and dt_bias were NOT)

### Phase 5: Final Exploration (Turns 37-40)
- Inspected shared_experts, layer 42 attention — not found (layer 42 has no mixer.A)
- Checkpointed A again, inspected attention o_proj
- Session ended mid-exploration

---

## Key Findings

1. **Scaling mixer.A × 1.1 HURTS** — dropped from 83.3% to 66.7%, with self_prediction completely destroyed (0%). This is the opposite direction from Phase 2's exp 003b success.

2. **A_log does not exist at runtime** — vLLM transforms `A_log` to `A = -exp(A_log)` during model loading. The model wasted 22 of 40 turns (55%) searching for a tensor that cannot exist. The system prompt mentioned `A_log` in the experiment history but didn't clearly state it's unavailable at runtime.

3. **Checkpoint/restore works correctly** — Nemotron saved and restored successfully, demonstrating the safety net functions.

4. **Stacked modifications untested** — D × 1.2 and dt_bias × 0.8 were applied but the session ended before evaluation. The model is now in a modified state (D and dt_bias changed from baseline).

---

## Critical Fix for Session 2

The system prompt's experiment history references "A_log" extensively but doesn't explain the runtime transformation. Add this prominent note:

> **IMPORTANT: At runtime, there is NO separate `A_log` tensor. vLLM transforms `A_log → A = -exp(A_log)` during model loading. The parameter `model.layers.{i}.mixer.A` contains the exponentiated form. Values are large negative numbers (mean ≈ -171). To replicate "A_log -0.5" from exp 003b, you need to ADD a positive value to mixer.A (making it less negative = faster response), NOT scale it.**

Also add a `<LIST>` action so Nemotron can enumerate available tensors instead of guessing paths.

---

## Session Statistics

| Metric | Count |
|--------|-------|
| INSPECT calls | 24 |
| MODIFY calls | 4 (1 successful + eval, 1 failed, 2 successful no eval) |
| CHECKPOINT | 2 |
| RESTORE | 1 |
| EVALUATE | 1 |
| Wasted turns (A_log search) | 22 (55%) |
| Productive turns | 18 (45%) |
