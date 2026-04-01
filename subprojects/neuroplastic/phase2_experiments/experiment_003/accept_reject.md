# Experiment 003: Accept/Reject Decision

## Sub-experiment 003a: A_log + 0.5 (increase memory)

**Decision: ACCEPTED (no change)**

- Target: `backbone.layers.50.mixer.A_log` — shift +0.5 (slower decay, longer memory)
- Result: 83.3% (10/12) — identical to baseline
- No degradation, no improvement

## Sub-experiment 003b: A_log - 0.5 (increase forgetting)

**Decision: ACCEPTED (IMPROVEMENT DETECTED)**

- Target: `backbone.layers.50.mixer.A_log` — shift -0.5 (faster decay, more responsive)
- Result: **100% (12/12)** — first improvement over baseline!

### Detailed Comparison

| Category | Baseline | 003a (+0.5) | 003b (-0.5) |
|----------|----------|-------------|-------------|
| sequential_reasoning | 100% (3/3) | 100% (3/3) | 100% (3/3) |
| state_tracking | 67% (2/3) | 67% (2/3) | **100% (3/3)** |
| code_generation | 100% (3/3) | 100% (3/3) | 100% (3/3) |
| self_prediction | 67% (2/3) | 67% (2/3) | **100% (3/3)** |
| **Overall** | **83.3%** | **83.3%** | **100%** |

### Key Finding

**Faster SSM decay in deep Mamba layers improves state tracking and self-prediction.** The state_001 (bag inventory) test went from consistently failing to passing majority vote (3/5). Self-prediction tests all pass majority vote.

This is counter-intuitive: *more forgetting* improved *state tracking*. Possible explanations:
1. Layer 50 is second-to-last Mamba layer — at this depth, the SSM may benefit from being more responsive to recent tokens rather than carrying stale state
2. Faster decay reduces interference from distant tokens, allowing more accurate local state computation
3. The thinking model's reasoning chain provides the long-range context; the SSM just needs accurate local processing

### A_log Statistics

| Metric | Original | 003a (+0.5) | 003b (-0.5) |
|--------|----------|-------------|-------------|
| Mean | 0.978 | 1.478 | 0.478 |
| Std | 2.685 | 2.685 | 2.685 |
| Range | [-5.2, 9.4] | [-4.7, 9.9] | [-5.7, 8.9] |

**Note:** The exp 002 MoE gate modifications are still applied (layers 43, 45, 47, 49). The 003b improvement is ON TOP of those. The A_log -0.5 modification is the first to produce a measurable behavioral change.
