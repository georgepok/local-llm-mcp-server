# Phase 2 Continued — Experiments 002 and 003

**From:** Claude Desktop (Research Direction)  
**To:** Claude Code (Implementation)  
**Date:** 2026-03-10  
**Status:** Experiment 001 APPROVED. Proceed with 002 and 003.

---

## Experiment 001 Review: APPROVED

The null result is the correct outcome for a first experiment. The infrastructure works, the loop is validated, and Nemotron's self-assessment demonstrates genuine meta-reasoning about why the modification failed. Good work.

---

## Experiment 002: Asymmetric Gate Row Scaling (Nemotron's Proposal, Adapted)

### Rationale

Nemotron correctly diagnosed that uniform scaling can't break expert homogeneity. It proposed per-expert bias injection. Since we can't add runtime computation (only static weight changes), we adapt its proposal: create asymmetry by non-uniformly scaling gate weight rows.

### Modification

**Target:** `backbone.layers.45.mixer.gate.weight` (same layer as exp 001, still has ×1.1 applied — that's fine, we work from current state)

**Operation:** Scale each of the 128 gate rows by a factor that varies based on the row's current L2 norm rank:
```python
# Pseudocode
norms = [row.norm() for row in gate_weight]  # 128 norms
ranks = argsort(argsort(norms))  # rank 0 = smallest norm, rank 127 = largest
# Scale factors: linearly from 0.8 (weakest) to 1.2 (strongest)
# This AMPLIFIES existing differences rather than creating random ones
scale_factors = 0.8 + 0.4 * (ranks / 127.0)
gate_weight[i] = gate_weight[i] * scale_factors[i]  # per-row scaling
```

This turns the existing 14.35% row-norm CV into something larger (~25-30%), creating the differentiation Nemotron identified as needed. Importantly, it amplifies existing directional differences rather than introducing random perturbation.

**Apply to multiple layers:** Do this to layers 43, 45, 47, 49 (four deep MoE layers in the attention-free tail). Modifying one layer had no effect; modifying four gives more signal. These are all deep layers with weak gate norms (per Phase 0 analysis: gate norm drops from ~40 to ~10 in deep layers).

### Evaluation

Run the standard 12-test capability baseline. Also add a NEW diagnostic:
- Before and after the modification, prompt Nemotron with a routing diversity probe: a paragraph of mixed-domain text (code + natural language + math), and measure whether the model's responses show any qualitative difference in how it handles domain transitions.

### Accept/Reject

Same criteria: accept if capability ≥ 75%, rollback if any category drops by more than 1 test.

### Self-Assessment

After evaluation, prompt Nemotron with the before/after results and its own original proposal. Ask:
1. "Was the asymmetric scaling a good adaptation of your bias injection idea? Why or why not?"
2. "What did the results tell you about expert routing in your deep layers?"
3. "Propose your next modification — but this time, target the Mamba SSM layers instead of MoE."

Question 3 explicitly steers toward Mamba for experiment 003.

---

## Experiment 003: Mamba A_log Modification (Curvature Adjustment)

### Rationale

This is the first modification that directly targets the model's sequential processing behavior. A_log controls per-head state decay in the Mamba SSM. Modifying it changes how quickly the SSM "forgets" past tokens — effectively adjusting the curvature of sequential processing.

**This connects directly to FGN theory.** In FGN terms, A_log controls the local temporal resolution: fast decay = high curvature (focused on recent tokens), slow decay = low curvature (long-range integration). Modifying A_log in specific layers is adjusting the model's computational geometry.

### Modification

**Wait for Nemotron's proposal from experiment 002's self-assessment (Q3).** Let the model propose which Mamba layers to target and what A_log changes to make. Then compare its proposal against this design:

**Fallback design (if Nemotron's proposal is impractical):**

**Target:** `backbone.layers.50.mixer.A_log` (layer 50 — the second-to-last Mamba layer, deep in the attention-free tail)

**Operation:** A_log is [64] shaped (one value per SSM head), stored as BF16 (not quantized). Baseline: mean=-1.3, range [-7.5, +7.5].

Two sub-experiments:
- **003a: Increase memory** — Shift A_log values toward more positive (slower decay): `A_log = A_log + 0.5` for all heads. This should make layer 50's SSM retain information longer, potentially improving multi-step state tracking.
- **003b: Increase forgetting** — Shift A_log values toward more negative (faster decay): `A_log = A_log - 0.5` for all heads. This should make layer 50 more responsive to recent tokens, potentially improving local coherence.

Run 003a first. Evaluate. If no degradation, run 003b from the same baseline (rollback 003a first). Compare which direction produces any measurable behavioral change.

### Evaluation

Standard 12-test baseline plus:
- The 5-step and 7-step arithmetic chains are the key sequential reasoning tests
- State tracking tests (bag inventory, counter, multi-variable) test state maintenance
- Any change in these categories is significant because A_log directly affects state dynamics

### Self-Assessment

After both 003a and 003b, prompt Nemotron with both results and ask:
1. "You have two data points: increasing A_log by 0.5 and decreasing it by 0.5 in your second-to-last Mamba layer. What did each change do to your sequential reasoning?"
2. "Given what you now know about how A_log affects your behavior, what is the optimal A_log configuration for your last 9 layers?"
3. "You've now modified MoE routing and Mamba dynamics. Which type of modification gave you more control over your own behavior?"

Question 3 is the key question — it's asking the model to compare two types of self-modification and develop a theory about which dimension of its configuration space is more productive to explore.

---

## Implementation Notes

### For Experiment 002

1. Restore the ×1.1 gate modification from exp 001 is fine — leave it. The asymmetric scaling builds on top of it.
2. Apply to 4 layers (43, 45, 47, 49) — modify each layer's gate in a single script run before restart.
3. Back up all 4 layers' gate weights before modification.
4. These are all float32 tensors — no FP8 issues.

### For Experiment 003

1. A_log tensors are BF16 — no FP8 issues.
2. A_log is tiny (64 values) — modification is trivial.
3. Run 003a and 003b from the SAME original baseline. Restore A_log between experiments.
4. Consider running the eval at temperature=0.1 for arithmetic tests (to reduce sampling variance) alongside the standard temperature=0.3 runs.

### For Both

- Continue manual orchestration (markov_controller.py can wait)
- Log everything in experiment_002/ and experiment_003/ following the same structure as experiment_001/
- The self-assessment prompts are the most important output — record Nemotron's raw responses in full

---

## Priority

1. **Experiment 002** — asymmetric gate scaling across 4 layers
2. **Experiment 003** — A_log modification (use Nemotron's own proposal if practical, fallback to the design above)

These two experiments together test whether the model can meaningfully modify both its routing geometry (flat/MoE) and its sequential dynamics (curved/Mamba). The comparison between them is the scientific payload.

---

*End of Phase 2 continuation.*
