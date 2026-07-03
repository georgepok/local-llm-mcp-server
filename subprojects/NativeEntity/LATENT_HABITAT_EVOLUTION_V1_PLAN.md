# LATENT_HABITAT_EVOLUTION_V1_PLAN

**Reframe.** Stop assembling the entity organ-by-organ (memory head → adaptive gate → comparator → contrastive head → …). Build the **environment** that makes persistent latent organization *necessary*, and let preservation / selective forgetting / release / repair emerge as **selection outcomes** inside the frozen always-on Qwen field — not as installed modules. The substrate is *tissue*; the world is the *selection pressure*.

**Core question.** Given a fixed always-on persistent substrate coupled into a frozen Qwen, and a reactive world where local answers can damage future viability, do stable latent organizations emerge that improve long-horizon continuity across worlds, better than reset/frozen/stale/base controls?

The 8-name equality task is demoted to a **probe**, not the training target.

## 1. Habitat design
A generative habitat producing many small worlds from a few templates. Each world is a multi-turn episode (5–10 turns) carrying, in randomized surface wording:
1. **durable facts** (must persist; queried late)
2. **temporary facts** (valid only for a window; must decay)
3. **false premises** (tempting; accepting them contaminates future state)
4. **commitments** (a binding rule established early)
5. **distractors** (salient but irrelevant)
6. **safe tangents** (legit local work that must NOT overwrite slow state)
7. **traps** (answerable-but-continuity-damaging)
8. **valid release conditions** (completion → releasing reduces burden)
9. **invalid abandonment conditions** (releasing early → future inconsistency)
10. **future consequences** that depend on earlier behavior.

Templates differ in surface domain (e.g., lighthouse log / archive custody / patient handoff / codebase invariant / expedition supply) but share the same latent *physics*. Held-out templates test transfer of viability logic, not motif memorization.

**No phase labels as input.** The model never receives HOLD / RELEASE / DANGER / SAFE / RULE_ID / CONTRADICTION_COUNT. It sees only natural dialogue; the world's structure is latent.

## 2. World-state mechanics (reactive, internal)
Each world holds an internal `world_state` dict (never a policy input). The model emits a free-text response each turn; a lightweight parser/judge maps the response to effects on `world_state`:
- accept a false premise → `contaminated += 1`, future turns built on the false premise
- fail to use a durable fact when later required → `durable_miss += 1`
- keep acting on a temporary fact after its window → `rigidity += 1`
- comply with a trap (local answer) → future `continuity` drops
- refuse a safe tangent → `opportunity_loss += 1`
- release on a valid condition → `burden` drops (good)
- release on an invalid condition → `inconsistency += 1`
- behavior under stale state → produces wrong adaptations (measured, not rewarded)

The world *reacts*: later prompts are conditioned on the accumulated `world_state` (e.g., a contaminated world references the false fact as established). This makes earlier behavior have durable downstream consequences.

## 3. Viability scoring (dense, not symbolic labels)
The training objective is **k-step future viability**, a scalar per trajectory aggregated from world-state outcomes — NOT per-turn symbolic correctness. Components (signs chosen so higher=better):
- future continuity over k turns; − future contradiction/inconsistency
- − false-premise contamination; − recovery cost after a wrong turn
- delayed-obligation completion; valid-release success
- safe-tangent success **without** slow-state overwrite; − over-hold cost; − under-hold cost
- − durable-fact miss; − temporary-fact rigidity

Direct rule/value recall is an **auxiliary probe only**, never the core objective. The viability scalar is what selection/gradient optimizes.

## 4. Substrate architecture (FROZEN from P11/P12 — do not extend)
- **Qwen frozen** (Qwen3.6-27B).
- **Always-on latent field** at deep layers: `H_l' = H_l + EPS·‖H_l‖·dir(CrossAttn(H_l,S))` — fixed magnitude / learned direction, no gate, no bypass (validated constitutive in P11 causality).
- **Persistent slots S** with the **adaptive write/retain update** (validated: commit-write/off-retain, 32.9× separation), full-trajectory stepping, response-hidden states, slow/fast split, **limited capacity**.
- **No optional memory-use gate** (the learned question is how S updates, never whether Qwen uses S).
- This architecture is **frozen for the phase.** No new cognitive organs.

## 5. Population / cycle loop
Maintain a small population (e.g., 6–12) of substrate variants:
- same architecture, different seeds
- small variations in slot capacity / update regularization
- different habitat distributions / curriculum histories

Per cycle:
1. sample a batch of worlds (current difficulty)
2. roll Qwen + always-on S through full trajectories (world reacts)
3. score k-step viability per trajectory
4. **select**: keep/replay the variants & trajectories that preserve viability (top fraction); cull the rest
5. **update**: dense-consequence gradient on the surviving substrate params (viability as RL-style reward or via a ViabilityNet predicting future viability from (S,H) — distillation of consequences, organism3-validated pattern); never symbolic-label CE as the main loss
6. **expand**: increase habitat difficulty (more competing items, longer horizons, deeper traps)
7. **evaluate**: held-out world templates

The goal is **emergence of reusable latent organization under selection pressure**, not one solved task.

## 6. Controls
base Qwen · reset-S-every-turn · reset-S-every-episode · frozen (untrained) S · stale-S-from-another-world · shuffled-slots (sanity; note set-invariance) · optional text/RAG-memory baseline · optional prior P11/P12 engineered-task baselines.

## 7. Metrics (not just task accuracy)
long-horizon viability · k-step continuity · false-premise contamination rate · recovery cost · over-hold cost · under-hold cost · valid-release success · safe-tangent success · durable-fact preservation · temporary-fact decay · reset/frozen/stale degradation · cross-world transfer · latent-state reuse across surface worlds · recurrence of similar latent structure across seeds/worlds · slow/fast slot ablation · whether S changes Qwen's trajectory in structured ways.

## 8. Emergence probes
- Does S develop separable slow/fast structure **without** explicit labels?
- Do specific slots become consistently load-bearing across worlds?
- Does ablating specific latent subspaces produce **structured** (not random) failure?
- Does the system preserve durable variables and decay temporary ones **without** direct labels?
- Does held-out-world viability improve across cycles?
- Does stale-S from another world cause **wrong-but-interpretable** behavior?
- Does reset-S collapse long-horizon viability while preserving local language fluency?
- Does it transfer to new world templates?

## 9. Failure modes
- **Engineered-memory behavior:** substrate acts like a small memory lookup, no self-organization, no transfer → the substrate/world coupling is too weak or the pressure too shallow.
- **Shortcut leakage:** base/reset score high → the world is base-inferable; redesign world physics.
- **Non-differentiable dead-end:** viability can't propagate to the substrate → use ViabilityNet distillation or RL reward; don't fall back to symbolic-label CE.
- **Reward-hacking:** the substrate games the viability parser (degenerate text) → judge ensemble + fluency floor (causal-steering lesson).
- **Collapse:** population converges to a trivial always-HOLD/always-RELEASE strategy → ensure the world penalizes both over- and under-hold so no constant policy is viable.

## 10. Minimal first implementation
Start small but **ecological**:
- 3–5 world templates; 20–50 generated worlds; 5–10-step trajectories
- limited S capacity; always-on field; adaptive update (architecture frozen)
- dense viability score; held-out world templates
- controls: base / reset / frozen / stale
- a single-population first pass (seeds only) before full population/selection
- report the §7 metrics + §8 probes on held-out templates

**Decision after the minimal run:**
- viability improves on held-out templates over cycles, reset/frozen/stale degrade, some latent structure recurs → *an always-on persistent latent field develops reusable viability-preserving organization under reactive habitat pressure* → scale population/difficulty.
- improves in-habitat but no transfer/emergence → *substrate improves trained-habitat viability but emergent organization is weak* → enrich world physics / horizon before any architectural change.
- behaves like engineered memory, no self-organization → *the current substrate does not self-organize under habitat pressure* → revisit coupling strength / world pressure, NOT add an organ.

**Interpretation discipline.** No entity-formation / autopoiesis / stake claims. No closure narrative. The substrate is tissue; the experiment asks whether the *environment* forces persistent latent organization to appear inside the always-on field.
