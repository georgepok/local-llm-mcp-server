---
name: claim-laddering
description: >-
  Use when stating a conclusion from AI/ML experiments — especially a NEGATIVE result
  ("X doesn't work / is exhausted / hits a ceiling") or a positive capability claim.
  Enforces matching claim strength to evidence: distinguish not-measured vs
  optimization-failure vs fundamental-limit, scope to the tested regime, require a
  robustness package before strong claims, and always name the next variant instead
  of declaring a search space closed.
---

# Claim Laddering

Match the strength of the claim to the strength of the evidence. Most research errors here
are **over-closing** (declaring a direction dead) or **over-claiming** (declaring a
capability) from a single run or an unvalidated metric.

## The three-rung ladder for a negative result
Before writing "X does not work" / "the route is exhausted", classify which rung you are on:

1. **Not measured** — the test crashed, the control failed, the metric was an artifact, the
   encoding was illegible, the loss was diluted, or the baseline was contaminated. This is a
   MEASUREMENT failure. You have learned nothing about X. Fix and re-run.
2. **Optimization failure** — under THIS objective/data/scale, training found a degenerate or
   shortcut solution (mode collapse, constant bias, memorization). X might work under a
   different signal. Claim only: "under objective O at scale N, the model learned <shortcut>."
3. **Fundamental limit** — even a control that DIRECTLY forces the target behavior (e.g. a
   contrastive/oracle objective) cannot produce it, across seeds and a fair positive control.
   Only here may you write "X cannot do Y in this regime", and still scope it (n, seeds, model).

Do not jump to rung 3 from rung 1 or 2. A debunked positive returns you to "not measured",
NOT to the opposite negative — retract in both directions.

## Requirements before a STRONG claim (either sign)
- **Content/causal control passed** (see `causal-experiment-controls`): the effect survives
  input-scrambling; the metric is not label frequency.
- **Robustness package**: ≥2 seeds, a fair positive control, and either a scale or
  layer/config sweep. One breakthrough run + one control = signal, not proof.
- **Ceiling and floor named**: report the trivial baseline (floor) and the
  information-in-context / oracle result (ceiling); locate the claim between them.
- **Scope stated**: task, n, seeds, model, and what was NOT tested.

## Never close a search space
A negative result narrows the space; it does not exhaust it. Whenever you report a failure,
**name the next distinct variant** (different objective, injection point, representation,
adaptation, or environment) and why it could differ. "Architecture/approach doesn't help" is
a meaningless framing — architectures are choices in a search, not binary theorems.

## Language discipline for mechanism claims
- Prefer mechanism-scoped statements: "under first-token NLL the substrate learned a
  content-independent presence-bias" over "the substrate reasons / remembers / fails".
- Separate what the component DID from what you HOPED it does. "presence-effect" ≠ "uses the
  information"; "memorizes train" ≠ "generalizes"; "teacher-forced fit" ≠ "generates it".
- If a result would change the program direction, get a second independent check (independent
  judge/seed/metric) before acting on it.

## Cautionary example (this project, 2026-07)
A latent-memory interface was called "weak-but-real" (rung-2 claim from an artifact metric),
then "exhausted for this architecture class" (rung-3 claim) — both from the SAME number that
turned out to be mode collapse (rung 1: not measured). The correct path was: debunk the
metric → drop to "not measured" → run a content-forcing control to separate rung 2 from
rung 3 → then, and only then, scope the claim. When unsure which rung you are on, you are on
rung 1.
