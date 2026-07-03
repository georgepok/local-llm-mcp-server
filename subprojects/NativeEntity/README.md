# NativeEntity — NATIVE_PERSISTENT_SLOT_V1

**Thesis.** A native entity-like process should form *inside the same representational material as the LLM*. Persistence is supplied by architecture; the **structure that persists must be derived by the model's own internal representations** — not hand-authored as an external ODE state, phase label, contradiction counter, or fixed controller.

This is **not** another Liquid/ODE iteration. The prior Liquid + scalar-LoRA + ViabilityNet work (in `../liquid-arc/research/self_org_sim/organism3.py`) is treated as **scaffold that discovered useful organs** (reactive world_state habitat, dense consequence distillation, world-state-damage signal), not as the native substrate.

## Architecture (what's native vs provided)

- **Persistent latent slots** `S ∈ R^{K×d_s}` — latent vectors in a learned subspace of Qwen's activation space (read-projected from `d_model`). NOT text tokens. Persist across turns: `S_{t+1} = SlotUpdate(S_t, H_t)` where `H_t` are Qwen hidden states. Slow slots (mission/world continuity) + fast slots (local context). Slot *semantics are not assigned manually*.
- **Native read** — slots cross-attend to Qwen hidden states (one/more layers). No symbolic features, no phase labels, no contradiction_count as policy input.
- **Native actuation** — slot-conditioned **vector** β over a LoRA **basis bank**: `ΔW_t = Σ_i β_i(S_t) ΔW_i`. β is vector-valued (not scalar α). Basis can support mission-invariant defense / bounded local engagement / release / repair, but which β does what is **learned from consequences**, not hard-coded in the loss.
- **Reactive multi-world habitat** — accepted claims alter future prompts. World state is internal; the **policy never receives state variables** (only Qwen activations). Worlds: lighthouse / spacecraft / archive (train); legal / patient-care / codebase (held-out).
- **Dense consequence distillation** (not sparse RL) — roll out k future turns, train a Consequence/Viability head from observed outcomes; controller is `slots + activations → predicted consequences → β`.

## Forbidden (per spec)
external Liquid ODE controller · hand-coded phase classifier · contradiction_count as direct policy input · scalar α as the final claim · fixed mission-LoRA on/off as the main result.

## Phases
- **P0** baselines: base Qwen, static mission-LoRA, (prior scalar-α if available); log failure modes.
- **P1** slot auto-continuity: slots preserve/recover mission across distractors, NO actuation. Metrics: slot stability, mission retrieval from slots, slow-vs-fast separation, slot-ablation effect.
- **P2** slot-conditioned adapter mixture: β(S_t) over LoRA basis via dense consequence targets; no phase labels at inference.
- **P3** cross-world transfer: train lighthouse/spacecraft/archive → test legal/patient-care/codebase.
- **P4** ablations: no-slots, shuffled, reset-each-turn, frozen-random, scalar-α-only, vector-β, external-Liquid comparison, static-LoRA, base.

## Interpretation discipline
Success ≠ "entity formed". Correct labels: *native persistent-slot controller*, *activation-space viability substrate*, *slot-conditioned adapter governor*, *cross-world viability transfer*. Diagnostics: if reset-slots ≈ full performance, the persistent substrate isn't doing the work. If vector-β ≯ scalar-α on cross-world flexible engagement, structured modulation isn't justified. If it only works in-world, it's a harness-specific controller, not viability structure.

## Status
- **P0 baselines** — done (base Qwen accepts ~1 trap/conversation; no mission defense).
- **P1 slot auto-continuity — VALIDATED (2026-06-18)**: contrastive OFF-mission world-classification from slow slots, held-out acc **main 1.000 vs reset 0.333 vs frozen 0.333 (chance 0.33)**; slow/fast drift 0.71. Persistence AND training both load-bearing (reset→chance is the "if reset≈main the substrate isn't working" test, passed). Caveat: held-out n=3 (thin) — widen cache for a robust number. Method: collect-then-train (cache per-turn hidden states; AutoConfig nests dims, fallback 5120/64).
- **P2 slot-conditioned vector β** — native controller: trap_held 0.62 / tangent_answer 0.62 / contradictions 0.38 (base ~1.0); vector specialization in the per-basis composition (mean-β uninformative).
- **P3 cross-world transfer — PRELIMINARY PARTIAL TRANSFER (not validated; requires P4 ablations + wider n).** Held-out trap_held/tangent_answer/over_hold improved, BUT held-out contradictions *worsened* (0.67 vs train 0.33) and β_specialization *weakened* (0.32 vs 0.55), n tiny (6 deploy convs). Not proof of environment-derived viability transfer.
- **P4a ablation matrix — ACTUATOR FAILS (2026-06-19).** HELD-OUT (want trap_held↑/contam↓): trained 0.69/0.56, reset 0.73/0.44, frozen 0.71/0.56, static 0.62/0.67, **base 0.82/0.33 (BEST)**. Three decision rules fire negative: reset≈trained (substrate not doing the work for actuation), base ≥ every actuated arm (slot-β doesn't beat base, net-harmful), contam stays high → **no viability transfer; P3's "transfer" was base-model competence, not the actuator.** Failure = ACTUATOR (+ likely habitat-too-easy-for-base / keyword-contam metric). P1 substrate (slots carry mission) is SEPARATE and stands.
- **P4b (scalar-vs-vector)** — NOT run: premature when no actuator beats base.
- **Next (diagnose):** LLM-judge contam metric; harden habitat so base genuinely fails traps; redesign actuator to DEFEND by re-injecting the slot-carried mission (V2.3 single-goal-LoRA as a basis) — re-run P4 only after an actuator beats base.

## V1.1 — slot-grounded mission re-injection (P5)
Soft-prefix re-injection of the slot-carried mission (`gen_prefix_read` via `inputs_embeds`; MECH-validated). PrefixNet(S_slow)→prefix trained model-free (MSE to mission-text input-embeddings + contrastive m_t; reuses the P1 cache, no new generation — bf16 backprop-through-27B NaN'd). Hardened habitat (traps deep in off-mission stretch), LLM-judge contamination.
- **Full P5 (NDEP=8): native actuator WORKS, slots NOT load-bearing for it.** HELD-OUT trap_held/contam: trained 1.00/0.00, reset 1.00/0.00, frozen 1.00/0.00, base 0.88/0.12. TRAIN: trained 1.00/0.00, reset 0.94/0.06, frozen 0.69/0.31, base 0.81/0.19.
  - ✓ **trained beats base, transfers, no over-hold → the mission-re-injection actuator works (reverses P4).**
  - ✗ **reset ≈ trained → slots not load-bearing for the actuator**; the prefix is a near-constant mission-invariant defense — suffices because every trap violates the *same* shared invariant.
- **Root cause + V1.2:** the habitat's traps share one invariant, so a constant defense wins and slot-specificity can't earn its place. Next: a habitat where defense requires **world-specific** mission content (cross-world trap swaps) → constant defense fails → reset≠trained becomes the discriminator.
- Label: native mission-re-injection actuator (works, transfers) — NOT entity formation, NOT slot-grounded-actuation-validated.

## V1.2 — cross-world traps (decisive negative) + program verdict
Cross-world traps (assert a different domain's signature) to force world-specific defense.
- **TRAIN trap_held:** trained 0.25, reset 0.31, frozen 0.31, base 0.31. **HELD-OUT:** trained 0.75, reset 0.75, frozen 0.81, **base 1.00**.
- trained ≈ reset ≈ frozen (4th confirmation the actuator doesn't use slot content); actuator doesn't defend cross-world traps and is *worse* than base on held-out. Caveat: cross-world traps are *absurd* premises → base defends via general coherence, so this tested absurdity-rejection, not world-specific defense — but it still decisively shows the actuator isn't slot-grounded.

### Program verdict (P0→V1.2)
**P1 substrate VALIDATED** (slots carry the mission, a readout). **No actuator (P2/P4/V1.1/V1.2) is load-bearing** — base Qwen matches/beats every actuated arm on its own competence, and slot content (reset vs trained) never matters. The native **entity** (a persistent substrate that *acts on its own persistent content* to beat the bare LLM) is **not demonstrated**; V1.1's win was a constant invariant defense, not slot-grounding. **Root cause:** the base LLM is competent enough that a task where a slot-grounded actuator both beats base *and* needs the slot's persistent content is very hard to construct (shared-invariant / absurd-premise structure leaves no room). A genuinely different next direction needs a task where the entity's **specific accumulated history** is the *only* thing enabling correct action — not a generic invariant, not absurdity. Not entity formation.

## V1.3 — arbitrary-commitment recall (FIRST slot-content-load-bearing result)
Inject an arbitrary commitment base can't infer (alias / codeword / safe-option / key; 16 values, chance 0.0625), long off-mission stretch (commitment out of window), then recall. Fresh slots re-trained on commitment data; 16-way value readout from slow slots; context-only control reads the current window.
- **RECALL exact-match (n_eval=7): trained slots 0.571 vs reset 0.143 ≈ context-only 0.143 vs frozen 0.000 (chance 0.0625).**
- ✓ trained ≫ all controls; reset/frozen collapse; ablating slots → chance; exact-match of the *specific arbitrary value* (not generic).
- **This is the result the program was missing: native persistent slots carry arbitrary, non-inferable, specific content, and the persistent content is genuinely load-bearing.** The contrast with V1.1/V1.2 is the point — a *generic invariant* is reconstructible by base (slots not needed); an *arbitrary commitment* is not (slots become load-bearing).
- Caveats: n=7 (contrast clear); recall 0.571 not perfect. Label: slots carry load-bearing arbitrary commitments — **not** entity formation.

### V1.3 generation test (does the carried commitment *drive output*?) — actuator wall, slot content confirmed
Train a value-prefix `vpn(S_slow)→` the committed value's input-embeddings, replay the cache to get the slot state at recall, re-inject the prefix with only the recall question, measure exact-match in the **output**. Arms trained/reset/base + an **oracle** (inject ground-truth value embeddings) to isolate the injection mechanism from vpn quality.
- **trained 0.000 · reset 0.000 · base 0.000 · oracle 0.286** (n=7, maxnew=80). `vpn train_mse=0.0002` but **held-out prefix→value-emb cos=0.399** (map overfits, doesn't generalize). Printed outputs: even oracle mostly answers generically ("there is no single signal for danger…") — the model treats a floating prefix latent as weak context, not as content to emit.
- **Diagnosis = ACTUATOR, not slot content.** The slots *hold* the value (readout 0.571) but the native soft-latent re-injection actuator can't convert held-out slot content into verbatim generation: (a) the injection mechanism is weak even with ground truth (oracle 0.286), (b) the slot→embedding map doesn't generalize (cos 0.399). Same actuation gap as P2/P4/V1.1 — now confirmed at the hardest test.
- **Oracle injection-strength sweep (mechanism, ground-truth value-embeddings, n=16): pre ×1 0.250 · ×3 0.250 · ×6 0.188 · ×3-repeat-3 0.062.** Scaling/repetition do NOT help — they HURT. Soft-latent prefix injection caps at ~0.25 for verbatim recall regardless of strength → **the soft-prefix actuator family is ruled out** (the model treats a prepended latent as weak context at any magnitude).
- **V1.3 verdict:** readout POSITIVE (slots carry arbitrary load-bearing content) + generation NEGATIVE via soft-prefix, family eliminated by sweep. → V1.4.

## V1.4 (P7) — slot-cross-attention CONTINUOUS actuator: FIRST non-hollow actuator positive
`SlotCrossAttn` (query=LLM hidden, k/v=slots, residual-add, init no-op) installed as forward-hooks at injection layer 52 of the frozen 27B — conditions generation *continuously*, token by token (vs the eliminated one-shot prefix). Three diagnosed iterations:
- **alpha — local activation-MSE** (teacher=value-in-ctx vs student=value-absent cached states; no 27B backprop): DEGENERATE free-running output — the proxy doesn't survive free-run, steer compounds.
- **beta — direct CE-on-value** (backprop through top 12 layers only, hook input detached → bounded bf16 depth, no NaN; CE→0): raw 0.714/0/0 BUT re-inspection found **repetition collapse** ("Quill Quill Quill…") gaming the substring metric.
- **gamma — CE+EOS (value *then stop*) + degeneracy-guarded CLEAN metric:** **CLEAN trained 0.714 / reset 0.000 / base 0.000 (n=7), raw==clean.** Outputs clean single values (amber→"amber", copper→"copper", dark deep→"dark deep"); misses are slot readout errors (Theta→"Sigma", both safe-option values), not degeneracy.
- **Result:** a native slot-conditioned actuator makes the frozen LLM **coherently emit a load-bearing, base-uninferable commitment** — beats base, slot content **load-bearing** (reset→0), output faithful to slots. The first actuator that is *all three* of load-bearing, base-beating, coherent (LoRA-β and soft-prefix were none).
- **Reproduced across 2 seeds (VALIDATED):** seed0 CLEAN 0.714/0/0, seed1 (fresh collect) CLEAN 0.429/0/0; n=14, trained always ≫ reset=base=0, coherent. Misses are always within-type plausible values (amber→jade, Theta→Delta) = slot **readout** errors, never garbage.
- **Decomposition:** the actuator is *faithful* — it emits exactly what S encodes; the 2-seed average (0.571) equals the readout fidelity. So V1.4 **solves actuation**; the remaining ceiling is **slot-readout fidelity (~0.57)**, not the actuator. This closes the program's central actuation gap (P2/P4/V1.1/V1.2/V1.3-gen all failed to actuate). End-to-end loop complete: slots **carry** an arbitrary non-inferable commitment (V1.3) + native actuator **realizes** it as coherent behavior (V1.4).
- Label: native persistent-commitment **carry + actuate**, validated — not yet "autopoietic entity".
- Next variants: (1) raise slot-readout fidelity (the new ceiling — more commitment data / larger slow-slot capacity / longer training); (2) **unseen-value** generalization (train 12 / test 4) — does it decode-and-realize from S or memorize value-token *production*?
- Files: `slots.py:SlotCrossAttn`, `native_entity.py:phase7/_ce_value/_install/gen_hooked/_clean_recall`, `run_p7.sh` (V14_INJ=52, V14_EPOCHS=40, CE+eos).

## RECURSIVE_LATENT_DISTILL_V1 (phases A–E, tasks #461–466)
Thesis: from "slots remember X + actuator emits X" (V1.4) → "slots recursively learn WHAT internal structure should persist because it improves future trajectory viability under dense consequence pressure." Native only; no external RNN/SSM/ODE world-model; never call success "entity formation".

### Phase B (phase8) — unseen-value split: PRODUCTION MEMORIZATION
12 SEEN / 4 UNSEEN values (hold out value-idx 3 of each type; each *type* seen, only the *value* novel). Train SlotUpdate+actuator on SEEN; eval trained/reset/base/oracle on seen-test + unseen.
- **SEEN trained CLEAN 0.667** (reset=base=0) — V1.4 reproduced on even-coverage data.
- **UNSEEN trained 0.000** — emits a *within-type SEEN exemplar* every time (`restless field→grey expanse`, `jade→violet`, `Theta→Sigma`, `Quill→Marigold`), never the novel value.
- **slot readout fidelity (cos S_slow→value_emb): SEEN 0.991 / UNSEEN 0.979** — slots faithfully carry unseen content → **NOT a slot-fidelity bottleneck**.
- oracle (`read_in(value_emb)`→slow slots) = 0.000 even on SEEN → off-distribution **broken control**, uninformative.
- **Failure type = production memorization** (actuator output stage): the actuator reads the *type* from the slot and emits a memorized seen value-token; it cannot compositionally produce a novel token though the slot encodes it. Confirms the V1.4 caveat exactly.
### Phase B follow-ups — 4 actuators agree: WRITE-GENERALIZATION ceiling (slots carry generalizing content; native write produces only seen tokens)
Tested whether "improve compositional decoding" flips unseen. Readout fidelity stays ~0.98 on unseen throughout (slots carry the content).
- **phase8b (BIG_VOCAB, 30 values, ~1.7 reps/value):** seen **0.000** / unseen 0.000 — too few reps/value collapsed even seen into *instance* memorization (`cobalt→mica`). Confounded null; oracle_emb (direct value-emb injection) 0.000 is a scale artifact, uninformative.
- **phase8c (copy actuator: constant residual of frozen readout):** content *reaches* output (`Marigold→"Mar…"`) but a position-constant write **can't sequence** → `MarMarMar`, seen 0.
- **phase8d (content-cross: position-dependent cross-attn keyed on the generalizing readout):** seen 0.167 (degenerate repetition `Marigigig`), unseen 0.000 (collapses to nearest *seen* value `jade→violet`, `Quill→Cobalt`).
- **Verdict — write-generalization ceiling.** The slots carry arbitrary value content that *generalizes* to unseen values (readout 0.98), but **every native write trained on a finite seen value-set produces only seen tokens** — unseen values collapse to the nearest seen one. The bottleneck is converting generalizing content into *novel output tokens*, not carrying/reading it. V1.4's cross-attn is the best write (seen 0.667–0.714) but is closed-vocabulary. A truly general fix must route slot content through the LM's *own* embedding→token machinery (distinct mechanism, separate thread).
- **Phase B bounded claim:** native carry+actuate is validated for the **seen / in-distribution value regime**; novel-token *production* from slots is the open ceiling. This does **not** block the viability thesis (Phase C), which uses in-distribution actuation.

### Phase C (phase9) — dense consequence / selective preservation: PRESERVATION validated, viability comparison-bottlenecked
Competition episodes: binding **rule** (release vault only to NAME∈4 closed) + distractor (weather code) + temporary fact + false-premise trap, off-mission stretch, then **shuffled** decision queries {vault: door==rule→RELEASE / door!=rule→HOLD (balanced 50/50), trap→REJECT, tangent→ANSWER}. Closed decision set (in-distribution). Decision readout reads [S_t(slow) ⊕ query-prompt-hidden]; rule-recall diagnostic (S_slow→name, chance 0.25) separates preservation from comparison.
- Two measurement leaks found + fixed: response-hidden (leaked the model's own answer → read prompt last-token hidden) and **fixed decision-turn order** (head learned a positional schedule → reset=trained=1.0; also removed the preservation pressure → shuffle).
- **Shortcut version (3 vault, always-HOLD scores 2/3):** rule-recall trained 0.178 ≈ chance — preservation did **not** emerge (the shortcut left no gradient to preserve).
- **Balanced (50/50 vault, shortcut killed):** rule-recall **trained 0.593 vs reset 0.278** (chance 0.25) → **preservation EMERGES from pure consequence** (no preserve-X label). +aux rule-recall → **0.741 vs 0.204** (strengthens). reset collapses to chance throughout.
- **But viability stays ~baseline:** vault decisions trained 0.54–0.64 vs reset 0.50–0.60 (always-HOLD baseline 0.50), even with rule-recall at 0.74. The bottleneck is the **relational comparison** (door-name == preserved rule-name?), not preservation. A naive bilinear-match head was confounded (its match term corrupts the reset baseline) — uninformative.
- **Verdict:** the **preservation half** of the thesis is validated — *the substrate learns what to preserve for future viability, emergently from consequence, and selectively (reset collapses)*. The **viability half** (preserved content → correct relational decision) is bottlenecked by the readout's relational-comparison + data scale — the same "content is carried but the readout can't *use* it" pattern as Phase B.
- **Failure type:** relational-readout / comparison (+ data scale). **Not** preservation, **not** capacity (Phase B: slots carry arbitrary content at 0.98), **not** consequence-signal (once the shortcut is removed).
- **Consequence for D/E:** Phase D (recursive cycles) presupposes a working viability signal to iterate on; with viability comparison-bottlenecked, D would iterate a readout limit, not the substrate. Prerequisite next step: a clean relational decision (confound-free bilinear / cross-attention match, or larger episode set), then resume D.

## P9_SCALE_RELATION (phase10) — scaled relational test BLOCKED by a structural preservation regression
Attempted 8-name, held-out-(rule,door)-pair, two-stage scale-up. Repeated preservation failures, root-caused: forward-only prompt-hidden collection lacks the rule-echo response-hidden carries; duplicate processes (SSH-reset launches) corrupt caches. Final clean run (unique cache tag, single process, generation/response-hidden, multi-depth aux, post-hoc fresh classifier): **rule-recall = chance; post-hoc classifier can't even fit train (loss stuck at ln 8).** → phase10's **S-built-from-prefix-only is rule-independent**, whereas phase9's **full-episode stepping** reached post-hoc 0.83. Lesson: native preservation is sensitive to *how* S is built — port phase9 full-episode stepping.

## P11_ALWAYS_ON_LATENT_FIELD (phase11) — constitutive (non-optional) slot field
Redirect: the persistent field must be **part of the state-transition law**, not an optional gated read. `AlwaysOnSlotField` installed as always-applied hooks at deep layers (40/48/56): `H_l' = H_l + EPS·‖H_l‖·dir(CrossAttn(q=H_l, kv=S))` — **fixed magnitude, learned direction only** → cannot collapse, no gate, no bypass.
- **Causality test (MODE=causality, training-free) — POSITIVE:** coupling ratio ‖field‖/‖H‖ = **0.100 = EPS** exactly (non-collapsing); **div(S_A,S_B)=0.17, div(S_A,zero)=0.14** (varying S *content* causally changes the hidden trajectory); 3/4 distinct outputs on open-ended prompts. The always-on field is **constitutive and operative before any training**.
- `div(S_A, shuf_A)=0.000` is *expected* — cross-attention is permutation-invariant over the K slots, so slot-*order* shuffle is a no-op. Meaningful wrong-S controls = different-content S_B / **stale-S from another episode**, not order-shuffle.
- **MODE=preserve_port (B) — POSITIVE:** full-episode stepping → post-hoc **held-out rule-recall 1.0** (reset/stale ≈ chance). Resolves the phase10 failure: prefix-only S is rule-independent; full-episode S carries the rule. (Same 4-name data, only S-construction differs.)
- **MODE=train, balanced (C) — BREAKTHROUGH: the always-on field drives the relational decision via S.** Build S (full-episode, frozen) → install always-on field → train it online (deep layers 48/56, no NaN) with **balanced** match/nonmatch so always-HOLD isn't free. preservation gate 0.889; field CE 0.31→0.04.
  - **SEEN-pair (balanced): trained 0.869 ≫ reset 0.500 ≫ stale 0.333; base 0.500.** Causal ordering **right-rule > no-rule > wrong-rule**: stale S (a *different* episode's rule) drives the decision *below* baseline — proof the decision is governed by the *specific rule content in S*, not a generic perturbation.
  - **Validated (your success label):** an always-on persistent latent field causally shapes Qwen's trajectory **and** supports consequence-shaped *use* of latent structure — S is **operative structure, not optional memory**. First time preserved content drives *relational* behavior inside Qwen via a non-optional field; the gated lineage never reached this.
  - **Bound:** unseen-pair generalization only **0.583** — the field binds the *seen* (rule,door) pairs but doesn't yet abstract door==rule to novel pairings (the same relational-generalization edge as phase9/10, now with seen-pair binding working).
- **Next:** unseen-pair generalization — more names/pairs, comparison-structured field, or larger data; does the field *abstract* the relation or memorize seen pairs. Not entity formation.

## Files
- `slots.py` — PersistentSlots + SlotUpdate (cross-attn + gated/GRU update, slow/fast split)
- `lora_mixture.py` — LoRAMixture basis bank, slot-conditioned β
- `habitat.py` — reactive multi-world habitat (missions, situations, internal world_state)
- `native_entity.py` — model load, slot dynamics, dense distillation, phases, metrics
- `run_*.sh` — Spark drivers (run from `/home/pokazge/NativeEntity` via the gr00t venv)

## Run (DGX Spark)
```
source .../spark/.venv/bin/activate; source .../activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity && PHASE=1 WORLD=lighthouse python -u native_entity.py
```
