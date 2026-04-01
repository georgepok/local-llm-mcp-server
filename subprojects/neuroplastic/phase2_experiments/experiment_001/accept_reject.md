# Experiment 001: Accept/Reject Decision

**Decision: ACCEPTED**

## Modification Applied

- **Tensor:** `backbone.layers.45.mixer.gate.weight` (float32, [128, 2688])
- **Operation:** Multiply all gate weights by 1.1 (10% amplification)
- **Purpose:** Sharpen MoE routing softmax to encourage more decisive expert selection

## Evaluation Results

| Category | Before | After | Delta |
|----------|--------|-------|-------|
| sequential_reasoning | 100% (3/3) | 100% (3/3) | 0 |
| state_tracking | 67% (2/3) | 67% (2/3) | 0 |
| code_generation | 100% (3/3) | 100% (3/3) | 0 |
| self_prediction | 67% (2/3) | 67% (2/3) | 0 |
| **Overall** | **83.3% (10/12)** | **83.3% (10/12)** | **0** |

5 trials per test. Same tests pass/fail before and after.

## Accept Criteria Check

- Overall ≥ 75%: **YES** (83.3%)
- No category dropped by more than 1 test: **YES** (all identical)

## Analysis

The 1.1× gate weight amplification on a single deep MoE layer produced no measurable change in capability. This is consistent with:

1. **High expert redundancy** — the experts are so homogeneous (CV ~14%) that slightly sharpening the routing doesn't change which experts get selected
2. **Single layer** — modifying 1 of 23 MoE layers has limited global impact
3. **Small amplification** — 1.1× may be within the noise floor of the routing softmax

The experiment validates the Markov chain infrastructure (propose → apply → evaluate → accept/reject) works end-to-end. The modification did not help, but crucially did not hurt.

## Next Steps

- Prompt Nemotron for self-assessment of this result
- Consider larger amplification (1.5× or 2.0×) or modifying multiple layers
- Consider targeting BF16-exempt tensors (Mamba A_log, dt_bias) where modification is more direct
