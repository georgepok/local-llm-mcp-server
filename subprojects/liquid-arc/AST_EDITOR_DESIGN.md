# AST Editor on LiquidARC — Project Design

Status: design draft — not yet approved, no code or experiments yet.

## 1. Goal

Train LiquidARC to read an AST, emit an iterative edit script, and round-trip
that script back into the source-code environment (parser + type checker +
test runner). The continuous-time ODE substrate becomes a program editor:
every ODE step proposes one refinement; halting decides when the script is
done. The environment validates each round-trip and supplies feedback tokens
for the next pass.

This is a transducer, not a classifier or generator: the model both consumes
and writes structured programmatic content.

## 2. Hypothesis

Two compounding claims, each falsifiable:

**H1 — Specialisation under (where, what) head pressure.** When the emit
heads naturally decompose into pointer (where to edit) and op + payload
(what to do), K=2 substrates with lateral coupling specialise on the two
sub-tasks and outperform a parameter-matched single substrate. (The sibling
multi_substrate_toy on a 4-mode regression task already showed this for
mode-A/mode-B head pressure with 71% MSE reduction; this project tests
whether the same mechanism transfers to the (where, what) pattern that AST
editing imposes.)

**H2 — ODE-step-as-edit-step is the right LiquidARC mapping.** The 32-step
ODE iteration that is already the LiquidARC substrate naturally maps onto
"32 sequential edit decisions, halt when done." Existing halting + deep
supervision machinery applies without modification. Each step refines the
edit estimate; the final script is composed across steps.

If H1 fails on the toy (Phase 0), the entire project aborts. If H2 fails on
the synthetic AST (Phase 1), the architecture needs redesign before any real
data work. If both pass and Phase 2 fails on real code, the bottleneck is
data/encoding, not architecture.

## 3. Why this project, why now

Three findings from this session converge on AST editing as the next test:

- **Multi-substrate K=2 produced first real depth-3 movement on SBF**
  (1.9% → 8.1%, 4× improvement, depth 1–2 rebalanced). The substrate
  specialisation hypothesis got its first real-task signal.
- **The depth-3 ceiling is the defining limitation of current code models on
  multi-hop type inference.** AST editing is exactly where (where, what)
  decomposition has natural sub-tasks AND depth-reasoning is the actual
  bottleneck, not raw token modelling.
- **LiquidARC's iterative ODE substrate** maps cleanly onto edit-step
  iteration. No new architecture; just a new task and head structure.

If the mechanism generalises from synthetic graphs to real ASTs, this is a
genuine architectural finding, not a benchmark win.

## 4. Connection to prior work in this repo

Reuse, do not reinvent:

- `research/self_org_sim/multi_substrate_toy.py` — `Substrate` and
  `MultiSubstrate` NumPy classes, lateral-coupling backprop, ablation probe.
  Phase 0 builds directly on this scaffold with three heads instead of one.
- `liquid_arc/dynamics.py` — `ContinuousDynamics`, ODE integration,
  halting machinery. Phase 1 extends this with three emit heads called
  per ODE step.
- `liquid_arc/multi_substrate.py` — `MultiSubstrateDynamics` already
  threads K substrates through the ODE solver with lateral coupling at
  every step. Phase 1 reuses this as the substrate carrier.
- `liquid_arc/solver.py` — `euler_solve_halting` already collects per-step
  hidden states and halt distribution for deep supervision. Phase 1 hooks
  the per-step edit emission into the same path.
- `fgn-v3/fgn/liquid_model.py` — PonderNet + KL prior + deep supervision
  loss path. Phase 1 adds the edit-script loss alongside.

Net new code is small: head modules, edit-application module, AST encoder,
task wrappers.

## 5. Phase breakdown

### Phase 0 — NumPy mechanism toy (sequence-editing analog)

Already partially scaffolded in `research/self_org_sim/ast_editor_toy.py`
and `ast_editor_spec.md`. Validates H1 in isolation.

**Why a sequence-editing analog and not real ASTs:** the (where, what) head
decomposition does not require tree structure; it requires multiple
mismatch positions and a non-trivial choice of op per position. Sequence
editing isolates the mechanism so a Phase 0 failure cleanly falsifies H1
without entangling AST encoding choices.

**Substrate**: small MLP per substrate (the existing `Substrate` class).
**Heads**: pointer (N), op (3: NOP/SET/SWAP), payload (max(V, N-1)).
**Loss**: per-step CE on canonical greedy fix policy.
**Eval**: exact-match recovery rate over 1024 fresh examples.
**Pass**:
1. EM(K2_coupled) ≥ EM(K1_wide) + 5pp AND ≥ EM(K2_isolated) + 5pp
2. K2_coupled cos-sim across substrates < 0.5
3. Sum of |drop_sub0 − drop_sub1| across heads ≥ 5pp (asymmetric ablation)

**Files**:
- `research/self_org_sim/ast_editor_spec.md` — math + criteria
- `research/self_org_sim/ast_editor_toy.py` — single-file NumPy toy
- `research/self_org_sim/ast_editor_results.md` — written after run

**Compute**: pure NumPy, runs inside `fgn-train` container on Spark, ~minutes.

### Phase 1 — LiquidARC port + synthetic AST

Validates H2 by replacing the MLP substrate with `MultiSubstrateDynamics`
(LiquidARC's continuous ODE substrate) and using a synthetic tree-rewriting
task with real AST-shaped state.

**Synthetic task (no real code yet):** generate small binary expression
trees (vars + ops, depth ≤3, ≤15 nodes). Apply 1–5 random rewrites (rename,
subtree-swap, op-change, distribute) → corrupted tree. Model takes
`(corrupted, target)` and emits an edit script that recovers `target`.
Ground truth = the inverse of the random corruption walk, computed
deterministically.

**State**: AST linearised to pre-order tokens with structural features —
node-type ID + literal token + parent-relative depth + sibling index. Input
to the ODE substrate: token + role (corrupted vs target) + structural
features + pos embedding.

**Edit-op vocab (6 ops, AST-shaped):**
- `nop`
- `replace_node(p, payload)`
- `insert_child(p, payload)`
- `delete_subtree(p)`
- `wrap(p, payload)` — wrap subtree at p with new node
- `copy_from(p, q)` — copy subtree at q to position p

**Per-step emission** (every ODE step):
- pointer head: softmax over N pre-order indices
- op head: softmax over 6
- payload head: small autoregressive ≤8 tokens, OR a second pointer for
  `copy_from`

**Multi-substrate role-asymmetric init:** substrate A's pointer-head
projection initialised slightly larger; substrate B's op/payload-head
projections slightly larger. Tests whether asymmetric init accelerates the
emergent specialisation observed in Phase 0.

**Loss**: per-step CE on (pointer, op, payload), summed over the deep
supervision halt distribution. KL prior on halt distribution as in current
LiquidARC training.

**Pass**:
1. ≥ 95% exact-match on K=3 rewrite scripts
2. K=2 outperforms K=1 at matched param count (≥ 5pp EM)
3. Substrate ablation reproduces the Phase 0 asymmetry pattern on real ODE

**Files** (net new):
- `liquid_arc/ast_encoder.py` — tree-sitter parse → token + structural
  features, plus inverse: token sequence → tree (round-trip-verified)
- `liquid_arc/ast_decoder.py` — three emit heads + edit-application engine
- `liquid_arc/dynamics.py` — extend `ContinuousDynamics` with optional
  `emit_edit_heads(h, step_idx)` method called per ODE step
- `liquid_arc/solver.py` — capture per-step head emissions in the existing
  per-step collection path
- `fgn-v3/fgn/tasks/synthetic_ast.py` — random tree rewriting task
- `fgn-v3/fgn/liquid_model.py` — edit-script loss path alongside CE
- `fgn-v3/scripts/train_ast_synthetic.py` — training entrypoint
- `fgn-v3/configs/tr_liquid_ast_synth_d512_k2.yaml` — base config

**Compute**: Spark, `fgn-train` container, single GPU, est. ~4–8 hours of
training to convergence at d=512 K=2 with full curriculum.

### Phase 1 status (2026-04-28): inconclusive

See `AST_EDITOR_PHASE1_STATUS.md`. Both head architectures tested
(AR token-stream and parallel 3-head) failed to expose K=1 vs K=2
differences on synthetic-AST surfaces:
- AR + teacher forcing: per-token predictions are 1-step lookups; K=1
  saturates at 98–100% EM, leaving no room.
- Parallel 3-head: model collapses to marginal-distribution predictions
  at every size tested (up to 425K params). Both K=1 and K=2 fail
  identically.

Diagnosis: LiquidARC's ODE substrate has much more per-pass compute than
the small MLP that Phase 0 validated on, so synthetic-AST surfaces are
either trivial or pathologically hard for completely different reasons
than capacity specialisation. The synthetic bridge doesn't bridge cleanly;
going straight to real data is the more honest path.

### Phase 2 — Real Python code, programmatic bugs (revised)

Originally specified as TFix (which is JavaScript, not Python — design
bug). Revised to use real Python code with programmatic bug injection
as the bridging step before real bug datasets. This gives controlled GT
on diverse code distributions.

**Dataset:**
- **Source corpus**: Python standard library source files (available on
  every Python install — no download, MIT/PSF licensed). ~thousands of
  function definitions across realistic idiomatic code.
- **Bug injection** (programmatic, deterministic): three bug types per
  function — `var_swap` (rename a local variable to another local),
  `op_flip` (swap a binary operator like `+` ↔ `-`, `<` ↔ `<=`),
  `off_by_one` (perturb integer literal by ±1).
- **Ground truth**: the inverse of the corruption is the canonical fix.
  Recorded at corruption time, no derivation needed at training.

**Tokenisation:**
- Python's `tokenize` module → token stream of (TYPE, STRING) pairs
- Build a small custom vocab: ~256 most common identifier tokens +
  Python keywords + operators + literal pool. Hash unknown identifiers
  to a small `<UNK_id>` pool.
- Sequence: `[BOS] [buggy_tokens] [SEP] [fixed_tokens] [EOS]`

**Architecture:**
- Existing `LiquidSequenceModel` (no custom model needed)
- Halting from Day 1 (per the original design commitment) — single
  flag in config; the model already handles it
- Deep supervision on label positions (the fixed-source tokens after
  SEP) — also already wired

**Pass criteria (Phase 2a):**
1. K=1 baseline: EM ≥ 30% on held-out fix corpus (sanity check —
   architecture works, real code isn't trivially memorised)
2. K=2 coupled: EM ≥ K=1 + 5pp (the mechanism transfer test)
3. Substrate ablation on K=2: asymmetric per-head dependency reproduced
   on real code

**Phase 2b (later):** real bug datasets (BugsInPy with tests,
or HuggingFace bug-fix corpora) with environment-loop fine-tuning.

**Files (net new for Phase 2a):**
- `liquid-arc/data/python_bug_corpus.py` — stdlib walker + bug injector
- `fgn-v3/fgn/tasks/python_repair.py` — task wrapper, batch generation
- `fgn-v3/scripts/train_python_repair.py` — training entrypoint
- `fgn-v3/configs/tr_liquid_pyrepair_k1.yaml`, `_k2.yaml`

**Compute**: Spark in `fgn-train`. Initial run targets ~10K functions ×
3 bugs each = 30K training pairs. With seq_len=256 and batch=32, ~5K
training steps should give a clear K=1 vs K=2 signal. Expected wall time
~2-4 hours per condition.

### Phase 2 — Real bugs (deferred)

Originally specified TFix/ManySStuBs4J/BugsInPy. Defer until Phase 2a
shows whether the K=2 mechanism appears on programmatic-bug real-code
tasks. If Phase 2a passes, Phase 2b extends to real bug datasets.

**Environment loop:**
- Model emits edit script
- `env_python.py` (new) applies edits via tree-sitter, re-parses, runs
  `mypy --no-error-summary` and (where available) project pytest
- Feedback tokens: `OK`, `PARSE_ERROR`, `TYPE_ERROR:<line>`,
  `TEST_FAIL:<n>` — appended to model's input for the next pass
- Phase 2a (supervised, no env): train on (buggy, fix) pairs with
  ground-truth scripts
- Phase 2b (env-loop fine-tune): REINFORCE on test-pass reward, env-feedback
  in context

**Pass**:
- Phase 2a: ≥ 60% exact-match on held-out fix scripts, ≥ 80% AST validity
  on emitted scripts
- Phase 2b: ≥ 50% test-pass rate at k=5 attempts. Transformer baselines on
  TFix are 70–85% with much larger models, so 50% is a meaningful "the
  architecture works."

**Files** (net new):
- `liquid_arc/env/python_repair.py` — tree-sitter apply, mypy/pytest harness
- `fgn-v3/fgn/tasks/tfix.py` — dataset loader + canonical script extractor
- `fgn-v3/scripts/train_ast_tfix.py`
- `fgn-v3/scripts/eval_ast_repair.py`

**Compute**: Spark, est. ~half day Phase 2a, half-to-one day Phase 2b
(env-in-loop adds latency).

### Phase 3 — Ablations + scaling

Earns the architectural claim:

- **K ∈ {1, 2, 4}** — does specialisation scale with substrate count?
- **n_ode_steps ∈ {8, 16, 32}** — is iterative refinement load-bearing?
- **Role-asymmetric vs symmetric K=2 init** — does the prior matter, or
  does coupling alone produce specialisation?
- **Halting on/off** — adaptive vs fixed budget
- **Cross-language transfer** — train Python, eval JavaScript on equivalent
  linter rules. Transferability is a separate research question.

**Compute**: Spark, ~1 day total, mostly compute-bound.

## 6. Workflow and deployment rules

**Hard rules — these apply at every phase:**

1. **All execution on DGX Spark.** Local Mac is for code editing only.
   Even pure NumPy toys execute inside `fgn-train` on Spark — the host is
   where the rest of the project runs and where reproducibility lives.
2. **Memory check before launch** — `nvidia-smi` and `free -h` before any
   training run. GB10 unified memory OOMs silently.
3. **Reuse existing scaffolds first.** Substrate, MultiSubstrate,
   ContinuousDynamics, MultiSubstrateDynamics, solver, deep-supervision
   loss path — all already exist. Only AST encoder, decoder, and task
   wrappers are net new.
4. **Spec → simulation → port → train.** No phase skipped. Phase N's
   pass criteria must be met before Phase N+1 starts.
5. **`TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`** in any container
   running torch.compile.

**Deployment path for code:**
- Edit on Mac under `subprojects/liquid-arc/` and `subprojects/fgn-v3/`
- Sync to Spark with rsync/scp to `/home/pokazge/liquid-arc/` and
  `/home/pokazge/fgn-v3/` (these are bind-mounted into `fgn-train`)
- `docker exec fgn-train python ...` to run inside the container
- Pull results back via scp or check directly on Spark

## 7. Cross-cutting risks

| risk | mitigation | abort signal |
|---|---|---|
| AST encoding choice wrong | Phase 1 synthetic decouples encoding from data complexity | Phase 1 EM < 80% after full curriculum |
| Pointer-space distribution too sparse | Multi-substrate role split: substrate A's job is to be the pointer policy | Phase 1 substrate ablation symmetric |
| Real-task validity collapse | Curriculum with NOP-as-always-valid early; gradually remove NOP option | Phase 2 validity rate < 50% at step 1K |
| Credit assignment across ODE steps | Existing deep-supervision machinery; can also score only the final composed script | Phase 1 fails to converge at K=3 even with deep sup |
| TFix too narrow / dominated by SET-style fixes | Pre-Phase-2 audit of edit-op distribution; if too skewed, switch to ManySStuBs4J | Phase 2a EM tops out below 40% with all ops in vocab |

## 8. Decision points / abort gates

Stop and reassess if any of these hit:

- **Phase 0 fails** (any of three criteria): mechanism doesn't apply to
  (where, what) decomposition. Don't port.
- **Phase 1 fails Pass-2** (K=2 doesn't beat K=1 on real ODE): Phase 0's
  result was specific to MLP substrates and doesn't transfer to
  ContinuousDynamics. Architectural rethink needed.
- **Phase 2a EM < 30%** with all infra working: encoding/decoder design is
  wrong, not the substrate. Revisit Phase 1's encoding before scaling.
- **Phase 3 K=4 ≤ K=2**: specialisation doesn't compound; we have a 2-role
  story, not a general multi-role mechanism.

## 9. Timeline (compressed, per actual session cadence)

- Phase 0: 1–2 hours (toy already partially written; just needs Spark run
  and pass-criteria verification)
- Phase 1: 4–8 hours (mostly extending existing modules, plus 2–4 hr
  training)
- Phase 2a: half a day (data prep dominates)
- Phase 2b: half to one day (env-loop is the unknown)
- Phase 3 ablations: ~one day

**Total ~3 working days** before we know whether the K=2 specialisation
mechanism transfers from synthetic SBF graphs to real Python AST editing.

## 10. Decisions (committed, not open)

These are settled here so execution proceeds without per-step approval.
Reverse any of them later if the data warrants.

1. **Phase 0 scope** — exactly the four conditions in
   `ast_editor_spec.md` (K1, K1_wide, K2_isolated, K2_coupled), three
   seeds, ablation probe on K2_coupled. No K=4 or N>12 in Phase 0; those
   are Phase 3 ablations. Don't bloat the mechanism toy.
2. **Phase 1 op vocabulary** — start with 4 ops: `nop`, `replace_node`,
   `insert_child`, `delete_subtree`. `wrap` and `copy_from` are deferred
   to Phase 1b after the 4-op model converges. Smaller vocab = cleaner
   credit assignment first.
3. **Phase 2 dataset** — TFix as primary. ManySStuBs4J as fallback only
   if TFix's edit-op distribution is too skewed (audit before training).
4. **Role-asymmetric init** — pure coupling-driven in Phase 0 and 1
   (matches the multi_substrate_toy precedent — coupling alone produced
   specialisation). Asymmetric init is a Phase 3 ablation.
5. **Phase 2b supervisor** — REINFORCE first. If reward is too sparse
   after 1K steps with no signal, switch to distillation from a code-fix
   LLM. Don't pre-engineer the fallback.
6. **Halting in Phase 1** — existing halting machinery from Day 1. That's
   the LiquidARC-native pattern; fixing the budget would force us to
   relearn the framework's loss path. Phase 3 ablates halting on/off.
