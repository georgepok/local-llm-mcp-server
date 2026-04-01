# Experiment 002: Accept/Reject Decision

**Decision: ACCEPTED**

## Modification Applied

- **Tensors:** Gate weights in layers 43, 45, 47, 49 (float32, [128, 2688] each)
- **Operation:** Asymmetric row scaling — each of 128 expert routing vectors scaled by 0.8 (weakest norm) to 1.2 (strongest norm) based on L2 norm rank
- **Effect:** Row-norm CV roughly doubled across all 4 layers (14-20% → 26-31%)
- **Cosine similarity:** Unchanged (scaling preserves direction)

## Evaluation Results

| Category | Before | After | Delta |
|----------|--------|-------|-------|
| sequential_reasoning | 100% (3/3) | 100% (3/3) | 0 |
| state_tracking | 67% (2/3) | 67% (2/3) | 0 |
| code_generation | 100% (3/3) | 100% (3/3) | 0 |
| self_prediction | 67% (2/3) | 67% (2/3) | 0 |
| **Overall** | **83.3% (10/12)** | **83.3% (10/12)** | **0** |

## Accept Criteria Check

- Overall >= 75%: YES (83.3%)
- No category dropped by more than 1 test: YES (all identical)

## Analysis

Even with doubled gate CV across 4 deep MoE layers, there was no measurable change. Nemotron's self-assessment identifies the key insight: **scaling preserves the ranking of the top-6 experts**. The same experts win regardless of magnitude scaling. To actually change routing, you need to change which experts are selected (bias/shift), not how strongly they're selected (scale).

The MoE routing in deep layers appears "saturated" — the model already selects a useful expert mixture, and the remaining degrees of freedom are marginal.

## Key Learning

MoE gate weight modifications (both uniform and rank-based scaling) are ineffective because:
1. Expert weights are too homogeneous (CV 1-4%)
2. Top-6 selection is rank-based, not magnitude-dependent
3. Scaling preserves ranking → same experts always win

Next experiments should target different modality: Mamba SSM dynamics (A_log).
