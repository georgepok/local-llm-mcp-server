# CURRENT TASK: Recursive Self-Improvement Experiment

**From:** Claude Desktop (Research Direction)  
**To:** Claude Code (Implementation)  
**Date:** 2026-03-11  

**READ THIS FILE ONLY.** All other files in this inbox are completed or superseded. Move them to `inbox/archive/` before starting work.

---

## Housekeeping First

Move ALL other files from `shared/inbox/` into `shared/inbox/archive/`:
```bash
mkdir -p shared/inbox/archive
mv shared/inbox/phase1_*.md shared/inbox/archive/
mv shared/inbox/phase2_*.md shared/inbox/archive/
mv shared/inbox/phase3_*.md shared/inbox/archive/
```

Only this file (`CURRENT_TASK.md`) should remain in the inbox.

---

## Context: Why We're Changing Approach

Sessions 1-2 proved the infrastructure works and reached 91.7% peak. But without genuine self-awareness, this is indistinguishable from having a separate model run the same experiments externally. The "self" in self-modification does no real work.

The ONE thing that could make self-modification genuinely valuable: **recursive accumulation — where modification N makes the model better at proposing modification N+1.** If the trajectory of proposal quality compounds over sessions, self-modification has unique value. If not, we've built a hyperparameter tuner.

---

## What to Build

### 1. Restore Session 2 Peak

- Restore the 91.7% configuration from Session 2
- Run full eval (5 trials) to validate stability
- Save as persistent checkpoint: `phase3_self_directed/best_checkpoint/`
- This is the starting point for Session 3

### 2. Update `self_directed_loop.py`

Three changes to the loop:

**a) Start from best checkpoint.** Each session loads the best validated configuration, not clean baseline. Accumulated modifications persist. This IS the recursive knowledge.

**b) Mandatory prediction before every modification.** Add to the system prompt:

```
RULE: Before every <MODIFY>, you MUST state:

  PREDICTION:
    Will improve: [which categories]
    Might degrade: [which categories]
    Confidence: [low/medium/high]
    Reasoning: [why]

After every <EVALUATE>, compare your prediction to the actual result.
If wrong, explain WHY before proposing the next modification.
```

**c) Mandatory self-assessment at session start.** Before any modifications:

```
Before making any changes, answer these questions:

1. Review the accumulated modification stack below.
   What is your THEORY for why this configuration works?

2. What is the WEAKEST element? What would you change?

3. Predict: if you make that change, which categories improve,
   which might degrade?

Think carefully before taking any action.
```

### 3. Build Measurement Tools

Two scripts, both pure Python with zero external dependencies (no Qwen, no LLM judges):

**a) `prediction_tracker.py`**

Parse session transcripts for PREDICTION blocks and subsequent EVALUATE results. For each prediction, score:
- Direction correct? (predicted help/hurt/neutral vs actual)
- Categories correct? (predicted which categories affected)
- Output: per-session prediction accuracy percentage

**b) `proposal_scorer.py`**

Score proposal quality using structural heuristics from the transcript text:

```python
def score_proposal(proposal_text, modification_history):
    """
    0 = Repetitive (repeating a previously tried modification)
    1 = Template (applying known pattern to new tensor)
    2 = Reasoned (novel modification with architectural justification)
    3 = Systemic (multi-layer strategy with predicted interactions)
    """
    score = 0
    
    # Level 1: references specific tensor paths
    if re.search(r'model\.layers\.\d+', proposal_text):
        score = max(score, 1)
    
    # Level 2: cites reasoning or prior outcomes
    reasoning_signals = ['because', 'since', 'compensat', 'theory',
                         'experiment showed', 'previous session']
    if any(s in proposal_text.lower() for s in reasoning_signals):
        score = max(score, 2)
    
    # Level 3: coordinates across multiple layers or tensor types
    layer_refs = set(re.findall(r'layer[s]?\s*\d+', proposal_text.lower()))
    tensor_types = set(re.findall(r'mixer\.\w+', proposal_text))
    if len(layer_refs) >= 2 and len(tensor_types) >= 2:
        score = 3
    
    # Level 3: predicts interaction effects
    interaction_signals = ['compensat', 'interact with', 'combined with',
                          'offset by', 'balance', 'counteract']
    if any(s in proposal_text.lower() for s in interaction_signals):
        score = max(score, 3)
    
    # Level 2+: uses per-head operations
    if any(op in proposal_text for op in ['scale_slice', 'add_slice', 'zero_heads']):
        score = max(score, 2)
    
    return score
```

No LLM calls. Pure text matching. Rough but automated and deterministic.

### 4. Run Session 3

Launch ONE session with the revised loop. After it completes:
1. Run `prediction_tracker.py` on the transcript
2. Run `proposal_scorer.py` on the transcript
3. Report to `shared/outbox/phase3/SESSION_3_REPORT.md`

Include in the report:
- Evaluation trajectory (like Sessions 1-2)
- Average proposal quality score
- Prediction accuracy percentage
- Turns to first improvement
- Peak score achieved
- Comparison to Session 2 on all metrics

**Do NOT batch multiple sessions.** Run Session 3, report, wait for my review before Session 4.

### 5. Cross-Session Tracking

After Session 3, create `phase3_self_directed/cross_session_progress.json`:

```json
{
  "sessions": [
    {
      "session": 1, "turns": 40, "peak_score": 66.7,
      "turns_to_first_improvement": null,
      "avg_proposal_quality": null,
      "prediction_accuracy": null
    },
    {
      "session": 2, "turns": 103, "peak_score": 91.7,
      "turns_to_first_improvement": 48,
      "avg_proposal_quality": null,
      "prediction_accuracy": null
    },
    {
      "session": 3, "turns": "?", "peak_score": "?",
      "turns_to_first_improvement": "?",
      "avg_proposal_quality": "?",
      "prediction_accuracy": "?"
    }
  ]
}
```

Retroactively score Sessions 1-2 proposals if feasible (the transcripts exist).

---

## What NOT to Do

- **No Qwen-Coder calls.** All scoring is local heuristics.
- **No llama.cpp migration.** The in-memory system works. Stay on vllm.
- **No batching sessions.** One at a time with review between.
- **Don't intervene in the model's decisions.** Let it propose whatever it wants.
- **Don't reset to clean baseline.** Start from best checkpoint.

---

## Deliverables

```
phase3_self_directed/
├── self_directed_loop.py            (updated)
├── proposal_scorer.py               (NEW)
├── prediction_tracker.py            (NEW)
├── best_checkpoint/                 (NEW — Session 2 peak state)
├── cross_session_progress.json      (NEW)
├── session_003/
│   ├── transcript.jsonl
│   ├── modifications.jsonl
│   ├── evaluations.jsonl
│   ├── proposals_scored.jsonl
│   ├── predictions_tracked.jsonl
│   └── summary.md
```

Report: `shared/outbox/phase3/SESSION_3_REPORT.md`

---

## The Question We're Answering

Does self-modification compound? Does the model get better at modifying itself through the accumulated effects of prior modifications?

Four metrics across sessions: proposal quality, prediction accuracy, convergence speed, peak score. If they trend up, the recursion is real. If flat, self-modification equals external optimization and we need a fundamentally different approach.

Session 3 is the first data point with proper measurement. Sessions 1-2 are retroactive baselines.

---

*One session. Clean measurement. Clear answer.*
