# Phase 3 Revised: Recursive Capability Accumulation

**From:** Claude Desktop (Research Direction)  
**To:** Claude Code (Implementation)  
**Date:** 2026-03-11  
**Priority:** This reframes the entire self-directed exploration approach  
**Supersedes:** General "explore freely" mandate from original Phase 3

---

## 1. The Problem With Free Exploration

Session 2 reached 91.7% through trial-and-error stacking of modifications. But this is indistinguishable from having a separate model control the modifications externally. The "self" in self-modification is doing no real work — the model doesn't have any privileged access to its own internal state.

The one thing that could make self-modification genuinely valuable: **recursive improvement where each modification makes the model better at finding the next modification.**

## 2. The Recursive Accumulation Loop

The new goal is NOT "maximize the eval score." It's: **improve the model's ability to improve itself.**

Specifically, modifications should be evaluated on two criteria:

**Criterion A: Overall capability preservation.** The standard 12-test eval must stay ≥ 75%. This is the safety floor.

**Criterion B: Self-modification capability.** Does the modification improve the model's ability to:
- Predict the effects of weight changes (self-prediction category)
- Reason about sequential multi-step modification strategies (sequential reasoning)
- Generate correct code for modification operations (code generation)
- Track accumulated state across many modification steps (state tracking)

Criterion B is more important than raw score maximization. A configuration at 83.3% that makes the model better at self-modification proposals is more valuable than a fragile 91.7% that doesn't.

## 3. Measuring Recursive Progress

### 3.1 Proposal Quality Scoring

After each session, evaluate the QUALITY of Nemotron's modification proposals — not just whether they worked, but how sophisticated they were.

Score each proposal on a 0-3 scale:
- **0 — Repetitive.** Repeating a previously tried modification or trivial variation
- **1 — Template-based.** Applying a known-good pattern to a new tensor ("scale A by 0.6065 because it worked on layer 50")
- **2 — Reasoned.** Novel modification justified by architectural reasoning ("per-head asymmetry in A should create specialized temporal processing because heads with different decay rates can track different timescales")
- **3 — Compensatory/Systemic.** Multi-layer modification strategy with predicted interaction effects ("boosting D in layer 46 should compensate for faster A decay by providing a residual bypass, while amplifying o_proj in attention layer 42 restores global context signal")

Track the average proposal quality score per session. If this increases across sessions, the recursive loop is working.

Use Qwen-Coder-Next (via remoteMax MCP) to score proposals — send it each proposal text and the scoring rubric. This keeps scoring external and consistent.

### 3.2 Prediction Accuracy

Before each modification, Nemotron predicts the effect. After evaluation, compare prediction to outcome.

Track:
- **Direction accuracy:** Did the model correctly predict whether the change would help, hurt, or be neutral?
- **Category accuracy:** Did it predict which eval categories would be affected?
- **Magnitude accuracy:** Did it predict the approximate size of the change?

If prediction accuracy improves across sessions, the model is building a better internal model of its own configuration space.

### 3.3 Convergence Speed

How many modification attempts does the model need to find an improvement?
- Session 1: 40 turns, no improvement found
- Session 2: 103 turns, improvement found at turn ~48 (eval 9 = 91.7%)

If the number of turns to first improvement decreases across sessions, the search is becoming more efficient.

## 4. Session Design Changes

### 4.1 Start From Best Known Configuration

Each session should start from the best validated configuration, not clean baseline. The accumulated modifications ARE the knowledge. Starting from scratch every time destroys the recursive benefit.

**Before launching each session:**
1. Load the best checkpoint from the previous session
2. Verify it with a quick eval
3. Include the accumulated modification stack in the experiment history
4. The model builds ON TOP of what it already found

### 4.2 Mandatory Self-Assessment Phase

Each session must begin with a self-assessment phase BEFORE any modifications:

```
Before making any changes, answer these questions:

1. Review the accumulated modification stack (listed in experiment history).
   What is the THEORY behind why this configuration works?
   
2. Based on your theory, what is the WEAKEST element of the current
   configuration? What would you change to strengthen it?

3. Predict: if you make that change, which eval categories will improve
   and which might degrade?

Write your reasoning before taking any action.
```

This forces explicit prediction before modification, which we can score for accuracy.

### 4.3 Mandatory Prediction Before Every Modification

Update the system prompt to require that EVERY modification must be preceded by an explicit prediction:

```
RULE: Before every <MODIFY>, you must state:
  PREDICT: [what will improve] / [what might degrade] / [confidence: low/medium/high]
  
After every <EVALUATE>, compare your prediction to the result.
If your prediction was wrong, explain WHY before proposing the next modification.
```

This generates the data needed for prediction accuracy tracking.

### 4.4 Cross-Session Knowledge Accumulation

After each session, extract the key learnings and add them to the experiment history for the next session. The history should grow to include:

- Which modifications worked and which didn't (factual)
- WHY each modification worked or didn't (the model's own theories)
- Prediction accuracy from that session (meta-data about self-knowledge quality)
- The best configuration achieved and what it consists of

This accumulated knowledge becomes the model's "memory" across sessions. If the recursive thesis is correct, each session's learnings should make the next session's proposals better.

## 5. The Key Metric: Does Self-Modification Quality Compound?

After 5 sessions, plot:
- **Proposal quality score** per session (0-3 average)
- **Prediction accuracy** per session (% correct direction)
- **Turns to first improvement** per session
- **Peak score** per session

If all four trend upward, the recursive accumulation is real. The model is genuinely getting better at modifying itself through the accumulated effects of prior modifications.

If they're flat or declining, then self-modification is equivalent to external optimization, and the honest conclusion is that this approach needs the architectural breakthrough we discussed (genuine self-recurrence) to go further.

## 6. Practical Deliverables

### Scripts
- **`proposal_scorer.py`** — sends proposals to Qwen-Coder-Next for 0-3 scoring
- **`prediction_tracker.py`** — extracts predictions and outcomes from session transcripts, computes accuracy
- **`session_launcher.py`** — loads best checkpoint, includes accumulated history, launches self-directed loop
- **`recursive_progress.py`** — generates cross-session trend plots

### Per-Session Output
```
phase3_self_directed/
├── session_NNN/
│   ├── transcript.jsonl
│   ├── modifications.jsonl
│   ├── evaluations.jsonl
│   ├── proposals_scored.jsonl      (NEW: quality scores)
│   ├── predictions_tracked.jsonl   (NEW: prediction accuracy)
│   └── summary.md
├── cross_session_progress.json     (NEW: cumulative trends)
└── best_checkpoint/                (NEW: persistent best state)
```

### First Action
1. Restore Nemotron to the Session 2 peak configuration (91.7%)
2. Run a full eval (5 trials) to validate it's stable
3. Save that as the starting checkpoint for Session 3
4. Launch Session 3 with the revised system prompt (mandatory prediction, self-assessment)
5. After Session 3, run the proposal scorer and prediction tracker
6. Report cross-session trends

## 7. What This Tests

**If recursive accumulation works:** Self-modification becomes a genuine capability — the model progressively builds understanding of its own configuration space through experience, and each improvement compounds. This justifies continued development even without architectural self-awareness.

**If it doesn't work:** The honest conclusion is that without genuine self-recurrence, self-modification reduces to automated black-box optimization with LLM-generated proposals. Useful as engineering tooling, but not the neuroplastic vision. The project would then pivot to designing the architecture that COULD support genuine self-awareness — the Petri dish from the original conversation.

Either outcome is a clear, publishable finding.

---

*End of revised Phase 3 instructions.*
