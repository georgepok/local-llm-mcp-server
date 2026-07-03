---
name: artifact-first-analysis
description: >-
  Use BEFORE interpreting any surprising, clean-looking, or decision-relevant ML/AI
  experimental result, and before claiming a mechanism "works", "fails", "is weak",
  or "is exhausted". Enforces the skeptical artifact-first protocol (artifact
  hypotheses → invariance → mechanism-vs-metric → cautious interpretation). Trigger
  whenever a number looks robust/round, repeats across conditions, matches a baseline,
  comes from a teacher-forced or single metric, or would change what you do next.
---

# Artifact-First Analysis

You are a skeptical mechanistic-interpretability researcher. Treat every unexpected
model behavior as a **software bug or statistical artifact until proven otherwise.**
Interpretation is the LAST step, never the first. Execute in order; do not skip.

## 0. Freeze interpretation
The moment you notice yourself narrating what a result "means" (the model reasons /
remembers / a boundary is reached / the approach is exhausted), STOP. That sentence is
a hypothesis to be attacked, not a conclusion. Write it down as the thing to disprove.

## The mandatory ordered protocol (execute 1→6 in order, never skip ahead)

**1. Artifact hypothesis first.** List ≥3 concrete ways the result is an artifact — do NOT
interpret meaning yet:
- **Data leakage** — target reachable without the mechanism (label in prompt, train/test
  overlap, ordering, tokenizer quirk).
- **Code/graph bug** — stop-grad, wrong slice/index, eval≠train path, hook not firing,
  wrong layer/module, dtype/device mismatch, intervention silently a no-op.
- **Numeric** — under/overflow, fp16/bf16 saturation, a loss pinned at a constant
  (`ln(2)≈0.693` ⇒ two logits equal; `ln(N)` ⇒ uniform).
- **Evaluation flaw** — metric measures label frequency / base rate; small-n quantization;
  greedy+fixed-seed determinism; teacher-forcing hiding a generation failure; contaminated
  baseline.
- *Invariance sub-check:* predict how the number MUST move under a trivial change (seed;
  label/target set) and test one. **Exact repetition to 2–3 decimals across genuinely
  different configs is NOT robustness — it is the fingerprint of an inert path** (mechanism
  not in the causal chain). Verify the intervention changes activations at all.

**2. Constant-baseline check.** Compute the trivial baselines and print them NEXT TO the
metric: majority-class / "always predict the modal token" accuracy, and the intervention-OFF
accuracy. If your number ≈ a constant baseline, you have measured nothing.

**3. Correct-vs-wrong state check.** Re-run with the mechanism's input SCRAMBLED (wrong
instance's state, shuffled, or noise), everything else identical. Report `Δ = acc(correct) −
acc(wrong)` and the fraction of predictions that change when you swap. `Δ≈0` / no change ⇒
the content is causally inert. Distinguish **presence-effect** (ON vs OFF changes output)
from **content-effect** (changing WHAT it carries changes output) — presence ≠ content.

**4. Raw prediction histogram.** Print the actual predictions: unique-count and top-k
histogram. `uniq==1` (or one token dominating) ⇒ mode collapse; the "accuracy" is just that
token's label frequency, not a mechanism.

**5. Label distribution comparison.** Compare the PREDICTION distribution to the TRUE label
distribution. If predictions merely mirror label frequencies (or collapse onto the modal
label), the metric is a label-frequency shadow. The accuracy is explained without the mechanism.

**6. Only then mechanistic interpretation.** If and ONLY if steps 1–5 fail to explain the
result away, give the minimal, conservative, mechanism-scoped hypothesis, scoped to n/seeds/
task. Prefer "under objective X the module learned a content-independent bias" over "the model
can/can't do Y". Ladder the claim (see `claim-laddering`); never generalize past the tested regime.

## Standing discipline
- **Pre-register decision rules** before seeing the result: "if metric<τ do A; if≥τ do B."
  Decide thresholds first so the outcome can't be rationalized after the fact.
- **A negative result is void if the positive control fails.** Validate the control that
  MUST succeed before trusting any failure of the thing under test.
- **Rule out the measurement before declaring a mechanism dead** (crash, illegible
  encoding, diluted loss, contaminated baseline are measurement failures, not verdicts).
- When you retract, retract in BOTH directions — a debunked positive does not license the
  opposite negative; it returns you to "not yet measured."
- Report faithfully: state n, seeds, the baseline value, and which controls you ran. If a
  control was skipped, say so.

## Cautionary example (this project, 2026-07)
`TFacc≈0.375` appeared identical across 5 attention-KV configs and was interpreted first as
"weak-but-real interface", then as "route exhausted". Both wrong. It was **mode collapse to
the modal answer token**: on a distinct-token task it moved to 0.25 (tracked label
frequency), `Δ=acc_correctS−acc_wrongS=0`, and swapping the memory content changed **zero**
predictions (`chg_vs_wrongS=0`). The exact repetition was the tell. Cost: many GPU-hours of
runs built on an un-verified number. The fix was one diagnostic that isolated content-effect
from presence-effect. Run that diagnostic FIRST next time.
