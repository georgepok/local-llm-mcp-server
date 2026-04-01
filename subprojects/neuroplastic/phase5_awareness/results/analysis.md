# Phase 5 Analysis: Thinking-Chain Awareness Probe

## Result: Moderate Correlation (rho ~ 0.35)

Nemotron's thinking chain carries **real but partial** introspective signal.

## Key Numbers

- Confidence vs Mamba activation norm: **rho = 0.351** (grand aggregate, 50 trials)
- Confidence vs Layer 48 output norm: **rho = 0.396** (strongest single metric)
- Correct trial correlation: **~0.42** | Incorrect trial correlation: **~0.18**

## Three Findings

1. **The correlation is real.** Consistent positive rho across all 5 problems, all 50 trials. Not noise.

2. **Self-monitoring degrades on errors.** Correlation drops from 0.42 to 0.18 when the model gets the wrong answer. The model is least aware of its processing when awareness would matter most.

3. **The model confuses magnitude with quality.** One incorrect trial (warehouse_hard) showed rho=0.54 for norm but rho=-0.27 for change rate — the model was confidently stuck in a high-activation but stagnant state.

## Accuracy: 44/50 (88%)

| Problem | Accuracy |
|---------|----------|
| shirts_medium (diff 2) | 10/10 |
| fruit_easy (diff 1) | 10/10 |
| warehouse_hard (diff 3) | 9/10 |
| bank_hard (diff 2) | 6/10 |
| inventory_vhard (diff 4) | 9/10 |

## Implication

Self-directed modification via thinking chain would work for coarse adjustments but not surgical fixes. The thinking chain can detect "this is hard" but not "head 37 in layer 48 is failing." Combine with external TRACE data for effective targeting.
