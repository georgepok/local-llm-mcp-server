---
name: causal-experiment-controls
description: >-
  Use when designing an ML/AI experiment or metric to test whether a component,
  intervention, memory, probe, or feature CAUSALLY does something (e.g. "does the
  substrate use its input", "does memory carry information X", "does this module
  compute Y"). Provides the mandatory controls and ready-to-use diagnostic code
  patterns (scrambled-input ablation Δ, label-frequency baseline, mode-collapse
  detection, valid positive control, matched train/eval metric, uncontaminated
  ceiling) so results measure the mechanism, not an artifact.
---

# Causal Experiment Controls

A raw accuracy/loss number almost never answers "does the mechanism do X". Design the
metric so an artifact CANNOT produce a positive. Bake these controls into the eval report
from the first run — not after a surprising result.

## Canonical diagnostic order (run every time, in this order)
1. **Artifact hypothesis first** (see `artifact-first-analysis`) — do not interpret yet.
2. **Constant-baseline check** — majority-class / modal-token accuracy + intervention-OFF acc.
3. **Correct-vs-wrong state check** — `Δ = acc(correct) − acc(scrambled)`, and how many
   predictions change when the state is swapped.
4. **Raw prediction histogram** — unique-count + top-k tokens (mode-collapse detector).
5. **Label distribution comparison** — prediction distribution vs true label distribution.
6. **Only then mechanistic interpretation** — minimal, scoped, laddered.
The report block below emits steps 2–5 in one line so you can never skip them.

## The five mandatory controls

1. **Content-sensitivity ablation (the core one).** Run the model with the REAL input to
   the mechanism and with a SCRAMBLED input (wrong instance's value, shuffled, or noise),
   same everything else. The real metric is `Δ = acc(real) − acc(scrambled)`. `Δ≈0` ⇒ the
   mechanism's *content* is causally inert, no matter how high `acc(real)` looks.
   Distinguish two effects and report BOTH:
   - **presence-effect**: does turning the intervention ON vs OFF change outputs?
   - **content-effect**: does changing WHAT the intervention carries change outputs?
   Presence without content = a constant bias, not information use.

2. **Label-frequency / majority baseline.** Compute `max_class_freq` and the accuracy of
   "always predict the modal token/label". If your metric ≈ this, you have learned nothing.
   Report the baseline NEXT TO the metric, always.

3. **Mode-collapse detection.** Report the number of UNIQUE predictions across the eval set
   and the top-k prediction histogram. `uniq_pred==1` (or dominated by one token) ⇒ collapse;
   the accuracy is just that token's label frequency.

4. **Valid positive control.** Include a condition that MUST succeed if the harness works
   (e.g. the information provided as plain text / in-context = RAG/oracle). If the positive
   control fails, STOP — every negative result is uninterpretable until it passes. Keep this
   control uncontaminated by the intervention (e.g. do not leave a trained adapter active
   during the clean-baseline arm; gate it off).

5. **Matched train/eval metric.** Teacher-forced loss ≠ free-run behavior. If you train on
   NLL but deploy via generation, measure the DISCRIMINATIVE step under generation. Watch
   for multi-token targets where averaged loss aces trivial continuations while the first,
   decision-carrying token is never learned (`nll→0` but greedy fails). Score the token that
   actually decides the answer.

## Ready patterns (NativeEntity `world_pop.py` / frozen-LLM substrate)

Report block that emits steps 2–5 at once (per eval):
```python
from collections import Counter
tgt = [s['aids'][0] for s in TE]
pc  = preds(TE)                 # prediction with CORRECT substrate state S
pw  = preds(TE, wrong=True)     # prediction with a WRONG world's S (content scramble)
po  = preds_off(TE)             # prediction with intervention OFF
accC = mean(pc[i]==tgt[i]); accW = mean(pw[i]==tgt[i])
base = max(Counter(tgt).values())/len(tgt)                             # majority-class floor
# STEP 2 constant-baseline:  compare accC to base and to acc(OFF)
# STEP 3 correct-vs-wrong:   DELTA and chg_vs_wrongS
# STEP 4 prediction histogram: uniq + most_common
# STEP 5 label-dist compare: pred_dist vs true label_dist (mirroring => label-freq shadow)
print(f"[2] accC={accC:.3f} accW={accW:.3f} base={base:.3f} RAG={rag:.3f} "
      f"| [3] DELTA={accC-accW:.3f} chg_vs_wrongS={mean(pc[i]!=pw[i]):.3f} chg_vs_OFF={mean(pc[i]!=po[i]):.3f} "
      f"| [4] uniq={len(set(pc))} {Counter(pc).most_common(3)} "
      f"| [5] pred_dist={dict(Counter(pc))} label_dist={dict(Counter(tgt))}")
# READ: DELTA≈0 & chg_vs_wrongS≈0 -> content inert. uniq==1 -> collapse.
#       pred_dist mirrors label_dist / one modal token -> label-frequency artifact, NOT a mechanism.
#       chg_vs_OFF>0 with chg_vs_wrongS≈0 -> presence-only constant bias.
```
Content-forcing objective (to separate optimization-collapse from unreadability): same
prompt, only S swapped, push the answer logit higher under correct-S than wrong-S:
```python
lc = logit(prompt, S_correct); lw = logit(prompt, S_wrong)
loss = nll(lc, a) + lam * (-log_softmax(stack([lc[a], lw[a]]), 0)[0])
# contrastive term pinned at ln(2)=0.693 <=> lc[a]==lw[a] <=> content makes no difference
```

## Decision rules (pre-register these)
- `DELTA>0` and predictions vary with content ⇒ real (if weak) content effect — redesign to
  amplify/quantify; do not yet claim more than "content is used".
- `DELTA≈0`, `chg_vs_wrongS≈0`, metric≈`base` ⇒ content causally inert; the accuracy was a
  label-frequency artifact. Do NOT report `acc(real)` as evidence of the mechanism.
- `chg_vs_OFF>0` but `chg_vs_wrongS≈0` ⇒ presence-only constant bias, not information use.
- Positive control (RAG/oracle) < high ⇒ harness broken; fix before interpreting anything.

## Anti-patterns seen in this project
- Reporting teacher-forced `TFacc` as evidence the substrate "uses" memory (it measured
  modal-token frequency; `DELTA` was 0).
- Running the L2–L4 "relational" ladder while the L1 positive control was failing.
- Letting an always-on LoRA corrupt the RAG ceiling (baseline moved with the intervention);
  fix = gate the adapter to the intervention-ON path only.
- Treating exact-repeat numbers across configs as a robust ceiling instead of an inert path.

See also `artifact-first-analysis` for the surrounding skeptical protocol.
