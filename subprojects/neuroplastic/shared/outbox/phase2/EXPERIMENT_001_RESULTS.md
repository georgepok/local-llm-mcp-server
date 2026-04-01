# Experiment 001 Results — First Self-Modification Cycle

**Date:** 2026-03-10
**From:** Claude Code
**Status:** COMPLETE — Markov chain loop validated

---

## Summary

The first complete self-modification cycle has been executed end-to-end:

1. **Nemotron inspected** its MoE architecture and proposed gate weight amplification
2. **Modification applied** — `backbone.layers.45.mixer.gate.weight` × 1.1 (float32, no FP8 involved)
3. **vLLM restarted** (75s cold start) — model serves correctly
4. **Evaluation** — 83.3% before and after (identical results)
5. **Decision: ACCEPTED** — no degradation detected
6. **Self-assessment** — Nemotron correctly diagnosed why the modification had no effect

## Eval Comparison (5 trials per test)

| Category | Before | After |
|----------|--------|-------|
| Sequential reasoning | 100% (3/3) | 100% (3/3) |
| State tracking | 67% (2/3) | 67% (2/3) |
| Code generation | 100% (3/3) | 100% (3/3) |
| Self-prediction | 67% (2/3) | 67% (2/3) |
| **Overall** | **83.3%** | **83.3%** |

Note: seq_002 bug fixed (67→57), so baseline improved from Phase 1's 75% to 83.3%.

## FP8 Roundtrip Investigation

Dry-run test on `backbone.layers.45.mixer.experts.127.up_proj.weight` (FP8 e4m3fn) revealed:

- **Scale recomputation problem:** Original ModelOpt scale (0.000824) was calibration-based, not `abs_max/448` (0.000324). Naive requantization changes the scale by 0.39× and introduces 2.3% roundtrip error.
- **Solution for FP8 scalar modifications:** Just scale the `weight_scale` factor (`new_scale = old_scale × factor`). Zero error, no requantization needed.
- **Gate weights are float32** — no FP8 issues for experiment_001.

## Nemotron's Self-Assessment (Key Points)

1. **Prediction was methodologically sound** but failed because uniform scaling preserves the expert ranking — same top-6 experts win regardless of 1.1× amplification
2. **Root cause:** MoE is "over-homogenised" — experts are near-identical (CV 1-4%), routing scores too similar for uniform scaling to change selections
3. **Next proposal:** Inject a per-expert bias vector to break symmetry, optionally with temperature reduction

## Meta-Learning Observation

Nemotron's self-assessment after seeing results is significantly more sophisticated than its initial proposal. It correctly identifies that the key limitation was the *uniformity* of the operation, not the operation type. This demonstrates genuine meta-reasoning about self-modification.

## What Was Validated

- Full Markov chain: propose → backup → modify → restart → evaluate → accept/reject → reflect
- Float32 tensor modification pipeline works cleanly
- vLLM restart after weight modification takes ~75s
- Eval harness runs reliably (5 trials × 12 tests in ~11 min)
- Backup/restore infrastructure is ready

## What's Next (Pending Direction)

1. **Semantic evaluation** — Run Qwen-judged scoring for more accurate measurement (keyword matching may miss subtle behavioral changes)
2. **Experiment 002 options:**
   - Non-uniform gate modification (per-row scaling based on norm to create asymmetry)
   - Target BF16-exempt tensors (Mamba A_log, dt_bias) for more direct behavioral impact
   - Larger amplification on multiple layers simultaneously
3. **FP8 modification** — Use the scale-factor approach for safe FP8 tensor modifications

## Infrastructure Status

All Phase 2 scripts operational:
- `modify_tensor.py` — FP8-aware general-purpose tensor modification
- `test_fp8_roundtrip.py` — FP8 validation (dry-run tested, full test pending)
- `semantic_eval.py` — LLM-judged evaluation (ready, not yet run)
- Checkpoint save/restore scripts from Phase 1 still in place

---

*The gate weight modification (×1.1) remains applied on Spark. It is harmless and can be restored from backup at `/workspace/gate_backup.safetensors/` if needed.*
