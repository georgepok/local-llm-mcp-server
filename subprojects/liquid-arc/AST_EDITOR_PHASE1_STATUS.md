# Phase 1 — Synthetic AST + LiquidARC: Status report

Status: **INCONCLUSIVE** — the (where, what) head-decomposition mechanism that
passed Phase 0 (NumPy MLP toy) does not reproduce on LiquidARC's ODE
substrate via either head architecture I tested. Stopping to reassess.

## What was run

All on DGX Spark inside `fgn-train`. Code in:
- `subprojects/fgn-v3/fgn/tasks/synthetic_ast.py`
- `subprojects/fgn-v3/fgn/ast_editor_model.py`
- `subprojects/fgn-v3/scripts/train_ast_phase1.py`

Two head architectures tried:

### A) AR token-stream (3 tokens per edit, teacher-forced training)

Single LM head over a 37-token vocab. Each edit encoded as `(PTR, OP, PAY)`
tokens at consecutive positions. Standard AR causal-LM convention.

- depth-2 tree (7 nodes), K_corrupt=2, K_fix=3, d=48 K=1 / d=32 K=2:
  **K=1 hits 100% EM by step 100** of a 200-step smoke. K=2 follows.
- depth-3 tree (15 nodes), K_corrupt=3, K_fix=4, d=48 K=1 / d=32 K=2:
  K=1 100% EM by step 400, K=2 98%.
- depth-3 tree, K_corrupt=6, K_fix=8, d=32 K=1 / d=24 K=2:
  Both saturate at 98% EM.

**Diagnosis:** under teacher forcing, each per-token prediction is a trivial
1-step lookup (find first mismatch / leaf-or-inner / target value). The (where,
what) decomposition is never tested because the predictions are individually
trivial. AR-rollout eval gives the same numbers because the model memorised
perfectly — the AR rollout produces the same predictions teacher-forcing did.

### B) Parallel 3-head emission (single SLOT token per edit)

Custom `ASTEditorModel` with three independent heads (pointer, op, payload)
reading from K_FIX hidden states at SLOT positions. No AR easy path — three
edits emitted in parallel from one substrate forward pass.

- depth-3 tree, K_corrupt=2, K_fix=3, d=48 K=1 / d=32 K=2:
  Both stall at **EM ~1%, ptr_acc ~41%, op_acc ~70%** after 1500 steps.
- Sanity check: d=128 K=1 (425K params, 3× larger): same plateau —
  EM ~0%, ptr ~40%, op ~74% at 800 steps.

**Diagnosis:** the parallel architecture forces three simultaneous edit
decisions from one ODE forward pass. With K_FIX slots sharing the same SLOT
token + only positional embeddings to differentiate them, the model can't
extract per-slot reasoning info. It collapses to predicting marginal
distributions (frequent NOPs, common pointer biases). This isn't a tuning
issue; even 3× capacity doesn't move it.

## Why neither architecture tests the mechanism

**A) AR + teacher forcing**: predictions are trivial lookups. The K=2 vs K=1
comparison degenerates — nothing to specialise on.

**B) Parallel 3-head**: per-slot signal is too weak to learn at any model
size I tested. Not a fair test of K=2 either; both K=1 and K=2 fail
identically.

The Phase 0 result was specific to a small MLP substrate where the (where,
what) sub-tasks were genuinely competing for limited capacity. LiquidARC's
ODE substrate has *much* more per-pass compute (8 Euler iterations × full
SDPA + MetricNet + FFN per step), so either the task is trivial (and K=1
suffices) or the task is hard for completely different reasons than capacity
specialisation.

## Three possible next steps

I think the right move is **option 3** but want explicit direction before
spending more cycles.

### 1. Per-step deep-supervision Phase 1b

Use the existing halting + deep-supervision machinery in
`liquid_model.py`. Each ODE step emits a candidate edit; halt distribution
chooses among candidates. This is the "ODE-step-as-edit-step" mapping from
the original design (H2). Aligns with LiquidARC's iterative-refinement
paradigm and may produce different K=1 vs K=2 dynamics because halting +
deep sup gives K=2 more granular places to specialise.

Risk: still task-bottlenecked. If the synthetic AST is fundamentally too
easy for the ODE substrate, deep sup won't help.

### 2. Variable-structure synthetic task

Random k_corrupt per example, mixed tree depths, padded sequences. Forces
the model to maintain dynamic per-example state rather than memorising
fixed-shape patterns. Should saturate K=1 less trivially.

Risk: more code, same fundamental "ODE is too powerful for synthetic" issue.

### 3. Skip to Phase 2 directly

Run on real Python code repair (TFix or similar). Real code distributions
are naturally complex enough that K=1 won't trivially saturate. The K=2
mechanism gets a fair test in the regime it was designed for: tasks that
genuinely need substrate specialisation.

This is the path I'd recommend. Phase 0 already validated the mechanism on
its native substrate (MLPs). Phase 1 synthetic was meant to bridge to real
data; if it can't bridge cleanly, going straight to real data is more
honest than tuning a synthetic until it shows an effect.

## What I won't do without explicit direction

- Tune more synthetic-AST configurations. I've tried four task variants ×
  three model sizes × two head architectures and the K=1/K=2 comparison
  has been identical every time. More tuning is unlikely to produce signal.
- Conclude Phase 1 "passed" or "failed" — neither is supported by what I ran.
- Write up the K=0% finding as a published-quality result. It would need
  per-step ablations and a different head structure to be conclusive.

## Asks

Pick a direction:
- **A** = "Try Phase 1b with halting + deep-sup before giving up on synthetic"
- **B** = "Variable-structure synthetic task; revise the task module"
- **C** = "Skip to Phase 2 — real code, this is where the mechanism gets a
  fair test"
- **D** = something else
