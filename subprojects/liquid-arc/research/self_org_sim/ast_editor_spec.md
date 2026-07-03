# AST Editor Substrate — Phase 0 spec

Sibling to `multi_substrate_toy.py`. Same hypothesis (K=2 coupled substrates
differentiate functional roles under task pressure), different head structure
(three emit heads instead of one regression target).

## Hypothesis

When a task naturally decomposes into "where" (pointer) and "what" (op + payload)
sub-tasks, K=2 substrates with lateral coupling specialise on the two roles and
outperform a parameter-matched single-substrate model. If true, this is exactly
the head structure the LiquidARC AST editor needs and the mechanism ports cleanly.

## Task — sequence editing as AST analog

- Source `S ∈ V^N`, target `T ∈ V^N`. Both are random sequences.
- Corruption: K_corrupt random non-trivial edits applied to T → S.
- Edit op vocabulary (3): `nop`, `set(p, y)`, `swap(p, y)` where swap targets
  `(p + y + 1) mod N`.
- Ground-truth fix script: greedy leftmost-mismatch policy. At each step, find
  the first position where state ≠ target. If a swap-pair partner exists in the
  remaining mismatches, emit `swap`; else emit `set`. Pad with `nop`.

The greedy policy is deterministic given (state, target), so per-step
cross-entropy has a clean target. The policy genuinely uses both ops (set and
swap) so the (where, what) decomposition has real signal.

## Heads

Three independent linear projections from the concatenated substrate output
`h_concat ∈ R^{B × K·d_per}`:
- pointer: `→ R^N` (which position to edit)
- op: `→ R^{N_OPS}` (which op)
- payload: `→ R^{PAYLOAD_DIM}` where `PAYLOAD_DIM = max(V, N-1)` covers both
  token values and swap offsets.

## Substrate

Re-uses the `Substrate` MLP from `multi_substrate_toy.py` (in_dim → hidden →
out_dim_per_sub) and the `MultiSubstrate` lateral-coupling structure
(stacked-mean-of-others as auxiliary input across `n_inner_steps`).

Input encoding: one-hot of `S` and `T` flattened — `in_dim = 2·N·V`.
The substrate has to internally compute per-position match/mismatch from its
own MLP weights. With hidden ≥ 64 this is comfortably learnable.

## Conditions

| name | K | coupled | hidden | comment |
|---|---|---|---|---|
| K1 | 1 | n/a | 64 | small baseline |
| K1_wide | 1 | n/a | 128 | param-matched control |
| K2_isolated | 2 | no | 64 | param-matched, no coupling |
| K2_coupled | 2 | yes | 64 | proposed mechanism |

## Pass criteria

Following the multi_substrate_toy convention. Averaged over 3 seeds:

1. **Mechanism**: `EM(K2_coupled) ≥ EM(K1_wide) + 5pp` AND
   `EM(K2_coupled) ≥ EM(K2_isolated) + 5pp` where EM is exact-match
   recovery rate (full edit script applied to S equals T).
2. **Differentiation**: cosine similarity between substrates' final outputs
   averaged over the eval set < 0.5 (substrates ARE different).
3. **Asymmetric ablation**: zeroing substrate-0's contribution to one head and
   substrate-1's contribution to a different head produces asymmetric drops —
   `sum_h |drop(sub0, h) - drop(sub1, h)| ≥ 0.05`.

## Out of scope

Real AST structure, LiquidARC ODE substrate, halting, environment loop —
those are Phase 1+ once the mechanism is validated.
