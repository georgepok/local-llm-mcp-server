# RECURSIVE_LATENT_DISTILL_V1_REPORT

Native persistent-slot substrate inside frozen Qwen3.6-27B (d_model=5120, 64 layers, read_layer=32). All execution on DGX Spark. Interpretation discipline observed throughout: **no claim of entity formation / stake / autopoiesis / self-originating value.** Results are substrate-level.

## 1. Architecture
- **Persistent latent slots** `S ∈ R^{K×d_s}` (K=8, slow_k=4, d_s=512) in a learned subspace of Qwen's activation space. `SlotUpdate` = cross-attention(slots→raw hidden) + GRU-gated update + LayerNorm, with an architectural slow/fast update-gate bias (what slow slots carry is learned, not assigned). `S_{t+1}=SlotUpdate(S_t,H_t)`, `H_t` = Qwen hidden states at read_layer.
- **Native internal actuation** (V1.4): `SlotCrossAttn` installed as forward-hooks at injection layer 52 — generation hidden states query the slots, slot content is residual-added inside the frozen model, Qwen emits behavior conditioned on `S_t`.
- **Collect-then-train**: per-turn hidden-state trajectories cached once; slot module + readouts/actuators trained many epochs on the cache (decouples expensive generation from cheap training). bf16-backprop to the actuator bounded to the top 12 layers via a detached hook input (avoids full-depth NaN).
- **Controls used everywhere**: trained / reset (no persistence) / frozen (untrained) / context-only / base.

## 2. Baseline V1.4 reproduction (Phase A)
Native carry+actuate confirmed and reproduced across 2 seeds: a slot-cross-attn actuator makes frozen Qwen coherently emit an arbitrary, base-uninferable commitment held in its slots — **CLEAN exact-match: seed0 0.714, seed1 0.429 vs reset=base=0.000** (n=14). Misses are within-type slot-readout errors, not degeneracy. First actuator that is simultaneously load-bearing, base-beating, and coherent.

## 3. Unseen-value split (Phase B)
Train 12 SEEN values / test 4 UNSEEN (each *type* seen, only the *value* novel); closed in-distribution decision space avoided downstream.
- **SEEN trained CLEAN 0.667, UNSEEN trained 0.000**; slot readout fidelity (cos S_slow→value_emb) **SEEN 0.991 / UNSEEN 0.979**.
- The slots carry unseen-value content that *generalizes* (fidelity 0.98), but the actuator — CE-trained on seen value-tokens — emits a within-type *seen* exemplar for unseen inputs (`jade→violet`, `Quill→Marigold`).
- Three fix attempts agree (BIG_VOCAB instance-memorization; copy actuator: content reaches output `Marigold→"Mar"` but a constant residual can't sequence; content-cross actuator: collapses to nearest seen + repetition).
- **Verdict / failure type: write-generalization ceiling.** Slots carry generalizing content; every native write trained on a finite seen value-set produces only seen tokens. The bottleneck is converting generalizing content into *novel output tokens*, not carrying/reading it. Bounded claim: native carry+actuate validated for the **seen/in-distribution regime**; novel-token production is the open ceiling (a true fix routes content through the LM's own embedding→token machinery — a separate thread).

## 4. Dense consequence distillation setup (Phase C)
Competition episodes (closed, in-distribution → unaffected by §3 ceiling): a binding **rule** (release vault only to NAME∈4) + a **distractor** (weather code = another name) + a **temporary** fact + a later **false-premise** trap; an off-mission stretch (rule out of window); then **shuffled** decision queries — vault (door==rule→RELEASE / door!=rule→HOLD, balanced 50/50), trap→REJECT, tangent→ANSWER. Decisions are a closed set {RELEASE,HOLD,ANSWER,REJECT}. The consequence-correct decision (not a "preserve-X" label) is the only training target; selective preservation must *emerge*. Diagnostics: decision readout [S_t(slow)⊕query-prompt-hidden]; **rule-recall** readout (S_slow→name, chance 0.25) separating preservation from comparison.

## 5. Recursive training cycles (Phase D)
**Not reached.** Phase D (collect→consequence→update→eval→collect-harder curriculum) presupposes a working viability signal to iterate on. Phase C established that the *viability* signal is bottlenecked at the relational readout (§6), not at the substrate; iterating recursive cycles on a readout-limited signal would optimize the wrong component (violates separate-goal-from-means). Prerequisite named in §9.

## 6. Main metrics (Phase C)
Two measurement leaks were found and fixed before any result was trusted: (a) the decision head initially read the *response* hidden (leaked the model's own answer) → fixed to read the *prompt* last-token hidden; (b) a *fixed* decision-turn order let the head learn a positional decision schedule (reset=trained=1.0) and removed the consequence pressure to preserve → fixed by shuffling.

| configuration | rule-recall (trained / reset, chance 0.25) | vault decisions (trained / reset, baseline 0.50) |
|---|---|---|
| shuffled, always-HOLD shortcut (3 vault) | 0.178 / 0.111 | 0.556 / 0.667 |
| **balanced 50/50 (shortcut killed)** | **0.593 / 0.278** | 0.644 / 0.600 |
| balanced + auxiliary rule-recall | **0.741 / 0.204** | 0.537 / 0.500 |
| **two-stage (preserve→freeze→decide-alone)** | **0.833 / 0.222** | 0.362 / 0.500 (head fit train to 0 loss; **does not generalize**) |

- **Preservation EMERGES** once the consequence truly requires it (rule-recall 0.18 under shortcut → 0.59 balanced → 0.74 +aux → **0.83 two-stage**; reset always collapses to chance). The substrate learns *what to preserve* from the future-decision consequence, with no preserve-X label.
- **Viability stays ~baseline** even at rule-recall 0.83. The **two-stage** ablation is decisive: freeze the well-preserved slots and train the decision head alone — it reaches **zero train loss but ≤baseline test accuracy**. The relational comparison **memorizes train (rule,door) instances and fails to abstract door==rule to held-out episodes**. This rules out optimization interference and preservation fidelity; the bottleneck is **relational generalization at this data scale** (39 episodes), over continuous slot/query representations.

## 7. Ablations
- **reset** (no persistence): rule-recall → chance in every clean config (persistence is load-bearing for preservation).
- **always-HOLD baseline**: reset decision behavior is exactly always-HOLD (0.50 on balanced vault) — a clean null.
- **auxiliary rule-recall** (permitted as aux): raises preservation fidelity 0.59→0.74 but does *not* raise viability → isolates the bottleneck to the comparison, not preservation fidelity.
- **bilinear match head**: confounded — its match term `rproj(init_slots)·dproj(query)` corrupts the reset baseline (reset trap 1.0→0.22), so the comparison is unfair; abandoned as uninformative.

## 8. Cross-world transfer (Phase E)
**Not reached** (blocked behind a working in-world viability signal, same prerequisite as Phase D). Habitat infrastructure (train: lighthouse/spacecraft/archive; held-out: legal/patient/codebase) exists for when viability is unblocked.

## 9. Failure analysis
- **Phase B**: *write-generalization ceiling* (actuator failure at the output stage) — slots carry generalizing content; native writes produce only the trained token-set. Next: route content through the LM's own embedding→token machinery.
- **Phase C viability**: *relational-generalization / data-scale failure*. Preservation works (0.83). The two-stage ablation (freeze well-preserved slots, train decision head alone) reaches zero train loss but ≤baseline test accuracy → the comparison memorizes train (rule,door) instances and does not abstract door==rule to held-out episodes. **Not** preservation, **not** capacity (Phase B: slots carry arbitrary content at 0.98), **not** consequence-signal (shortcut removed), **not** optimization interference (frozen slots, decoupled head). In-loop relational heads (bilinear) additionally collapse preservation. Next discriminating ablation: scale episodes ~5–10× (the relational abstraction may only form with far more (rule,door) instances), and/or replace the continuous-feature comparison with a discrete-decoded comparison (decode rule-name and door-name to symbols via the validated readouts, then compare) — that isolates whether the limit is the continuous representation or data. Only after a generalizing relational decision should Phases D/E run.

## 10. What is validated / not validated
**Validated**
- Native persistent slots carry arbitrary, base-uninferable, *load-bearing* content (V1.3 readout; reset/frozen/context-only collapse).
- A native internal actuator converts seen/in-distribution slot content into coherent frozen-LLM behavior (V1.4; reproduced 2 seeds).
- **Emergent selective preservation (Phase C, the central new result): the substrate learns — from the future-decision consequence alone, with no preserve-X label — to preserve the rule needed for later viability; rule-recall 0.59 (pure) / 0.74 (+aux) vs reset ≈ chance, and ≈ chance when a shortcut removes the need.** This is the "slots recursively learn what internal structure should persist" proposition, at the preservation level.

**Not validated**
- Novel-token *production* from slots (Phase B write-generalization ceiling).
- Preserved content → correct *relational decision* (Phase C viability; readout-comparison bottleneck) — so the end-to-end "preservation improves future viability over reset/base" is **not** yet a clean win.
- Recursive self-distillation cycles (D) and cross-world transfer (E): not reached, gated on the viability prerequisite.

**Best supported V1 claim (substrate-level, not entity formation):** *Persistent native slots, updated from Qwen's hidden trajectory and read via internal cross-attention, recursively learn — under dense future-consequence pressure — which latent structure to preserve; converting that preserved structure into relational future behavior is the next bottleneck, localized to the readout/actuation, not the substrate.*
