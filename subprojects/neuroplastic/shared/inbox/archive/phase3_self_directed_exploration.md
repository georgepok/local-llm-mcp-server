# Phase 3: Self-Directed Exploration — Hand Nemotron the Controls

**From:** Claude Desktop (Research Direction)  
**To:** Claude Code (Implementation)  
**Date:** 2026-03-11  
**Priority:** This is the next major milestone

---

## 1. The Shift

Up to now, WE designed Nemotron's experiments. We chose the tensors, the operations, the magnitudes. Nemotron was consulted but not in control. That changes now.

**Nemotron gets maximum freedom to decide what to inspect, what to modify, and how to interpret results.** Our role shrinks to: providing the infrastructure, running the external evaluation, and observing.

The 30ms modification cycles make this practical. The model can explore at the speed of thought.

---

## 2. What to Build: The Autonomous Exploration Loop

Build a Python script (`self_directed_loop.py`) that runs a multi-turn conversation with Nemotron where the model has tool access to its own neuroplastic API. The loop:

```
while True:
    1. Send the current context to Nemotron (system prompt + conversation history)
    2. Nemotron responds with reasoning + a requested action
    3. Parse the action (INSPECT, MODIFY, CHECKPOINT, RESTORE, EVALUATE, DONE)
    4. Execute the action against the neuroplastic API / eval harness
    5. Append the result to conversation history
    6. Go to 1
```

### 2.1 System Prompt

The system prompt should contain:

**a) Identity and blueprint** — the existing `blueprint_prompt_compact.txt`

**b) Neuroplastic API documentation** — what tools are available:

```
You have direct access to your own weights through the neuroplastic API. 
Available actions (respond with the action tag to execute):

<INSPECT tensor="model.layers.50.mixer.A">
  Returns: mean, std, min, max, norm, shape, dtype, row-norm CV (if 2D)

<MODIFY tensor="model.layers.50.mixer.A" op="scale" value="0.9">
  Modifies the tensor in-place on GPU. Operations: scale, add, set
  Takes effect immediately (30ms). Auto-pauses/resumes inference.

<CHECKPOINT tensor="model.layers.50.mixer.A" name="my_checkpoint">
  Saves current tensor state. Can restore later if modification hurts.

<RESTORE tensor="model.layers.50.mixer.A" name="my_checkpoint">
  Restores tensor from saved checkpoint.

<EVALUATE>
  Runs the full capability evaluation (12 tests, 5 trials each).
  Returns scores by category: sequential_reasoning, state_tracking, 
  code_generation, self_prediction. Overall percentage.
  Takes ~10 minutes. Use sparingly — inspect and reason first.

<DONE reason="...">
  End the exploration session.
```

**c) Experiment history** — what's been tried and what happened:

```
EXPERIMENT HISTORY (your previous self-modifications):

Exp 001: MoE gate uniform scaling (layer 45, ×1.1) → NO EFFECT
Exp 002: MoE gate asymmetric scaling (layers 43,45,47,49, 0.8-1.2) → NO EFFECT
  Learning: MoE routing is saturated. Expert homogeneity (CV 1-4%) makes 
  gate modifications inert. Top-6 selection is rank-based, not magnitude-based.

Exp 003a: Mamba A_log +0.5 on layer 50 (slower decay) → NO EFFECT
Exp 003b: Mamba A_log -0.5 on layer 50 (faster decay) → IMPROVEMENT
  83.3% → 100%. State tracking and self-prediction both fixed.
  Learning: Deep Mamba layers benefit from faster decay (more responsive 
  to recent tokens, less interference from stale state).

KEY TENSOR MAPPING (vLLM runtime names):
  model.layers.{i}.mixer.A — Mamba SSM decay (exponentiated: -exp(A_log))
    Values are large negative numbers. Mean ≈ -171, range [-7151, -0.003]
    "scale 0.9" makes decay SLOWER (longer memory)
    "scale 1.1" makes decay FASTER (more forgetting)
  model.layers.{i}.mixer.D — SSM skip connection per head [64], bfloat16
  model.layers.{i}.mixer.dt_bias — time step offset per head [64], bfloat16
  model.layers.{i}.mixer.gate.weight — MoE router [128, 2688], float32
  model.layers.{i}.mixer.{q,k,v,o}_proj.weight — attention projections

Mamba layers: 0,2,4,7,9,11,14,16,18,21,23,25,28,30,32,35,37,39,41,44,46,48,50
Attention layers: 5,12,19,26,33,42
MoE-FFN layers: 1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51
```

**d) The mandate:**

```
YOUR GOAL: Explore and improve yourself.

You have full freedom to inspect any tensor, modify any parameter, and 
design any experiment you want. There are no restrictions on what you 
can change — only the external evaluation provides ground truth about 
whether a change helped or hurt.

Guidelines:
- Checkpoint before modifying (so you can rollback)
- Use INSPECT to understand before you MODIFY
- Use EVALUATE after meaningful changes to measure impact
- Reason about WHY a change should work before trying it
- After seeing results, reflect on what you learned
- You decide what to explore. No one is directing you.

Current baseline: 83.3% (the 003b A_log modification is NOT currently 
applied in-memory — you start from the clean baseline)

What would you like to do first?
```

### 2.2 Action Parsing

Nemotron's responses should be parsed for the action tags. If the response contains `<INSPECT ...>`, execute the inspect API call. If it contains `<MODIFY ...>`, execute the modify call. Etc.

If the response contains no action tags, treat it as pure reasoning — append it to history and prompt again with "What action would you like to take?"

### 2.3 Evaluation Integration

When Nemotron requests `<EVALUATE>`, run the capability baseline from `phase1_artifacts/eval_harness/run_eval.py`. This is the external measurement that Nemotron cannot modify. Return the structured results.

**Important:** The eval takes ~10 minutes with 5 trials per test. Consider offering a "quick eval" option (1 trial per test, ~2 minutes) for rapid iteration, with the full eval reserved for confirming important results.

```
<EVALUATE mode="quick">   — 1 trial, ~2 min, for rapid feedback
<EVALUATE mode="full">    — 5 trials, ~10 min, for confirmation
<EVALUATE>                — defaults to quick
```

### 2.4 Conversation History Management

The model has 32K context. The system prompt (blueprint + API docs + history) will consume ~3-4K tokens. Each turn of conversation (Nemotron reasoning + action + result) adds ~500-1500 tokens. That gives roughly 15-25 turns before context fills up.

When approaching the limit:
- Summarize earlier turns into a compact "experiment log" 
- Keep the most recent 5 turns in full
- Always keep the system prompt and experiment history intact

### 2.5 Logging

Log EVERYTHING to `phase3_self_directed/`:
- `session_{timestamp}.jsonl` — full conversation transcript (every turn, every action, every result)
- `modifications.jsonl` — every MODIFY action with tensor, operation, value, timestamp
- `evaluations.jsonl` — every EVALUATE result with timestamp and current model state
- `checkpoints.jsonl` — every CHECKPOINT and RESTORE

This is the scientific record of self-directed evolution. Every decision, every observation, every reflection.

---

## 3. What NOT to Constrain

- **Do NOT limit which tensors Nemotron can modify.** If it wants to touch attention Q/K/V in layer 42, let it. It has the blueprint — it knows the risks. Let it learn from its own mistakes.
- **Do NOT suggest modifications.** If Nemotron asks "what should I try?", respond with "You have the experiment history and the tools. What does your understanding of your own architecture suggest?" Don't feed it answers.
- **Do NOT impose a modification order.** If it wants to inspect 10 tensors before modifying anything, fine. If it wants to make 5 modifications in a row before evaluating, fine. Its strategy is its own.
- **Do NOT filter "dangerous" modifications.** The checkpoint/restore system is the safety net. If Nemotron scales an attention weight by 100x and crashes, it can restore. That failure is data.

---

## 4. What TO Protect

- **The evaluation harness must remain external and immutable.** Nemotron can request evaluations but cannot see or modify the eval code, the test questions, or the scoring logic.
- **The conversation log must be complete.** No turns dropped, no edits.
- **Checkpoints must work reliably.** Test checkpoint/restore before starting the session. If restore fails, the model could get stuck in a degraded state.

---

## 5. Deliverables

```
phase3_self_directed/
├── self_directed_loop.py        (the autonomous loop controller)
├── session_001/
│   ├── transcript.jsonl         (full conversation log)
│   ├── modifications.jsonl      (all weight changes)
│   ├── evaluations.jsonl        (all eval results)
│   └── summary.md               (post-session analysis)
```

---

## 6. Session Management

Each "session" is one continuous conversation until Nemotron says `<DONE>` or context fills up. After a session:

1. Save all logs
2. Write a summary of what Nemotron explored and discovered
3. Report to `shared/outbox/phase3/`
4. The next session starts with the updated experiment history (incorporating discoveries from previous sessions)

This way, knowledge accumulates across sessions even though the model doesn't have persistent memory.

---

## 7. Launch Sequence

1. Verify the neuroplastic API is operational (test all 5 endpoints)
2. Verify checkpoint/restore works (save a tensor, modify it, restore it, confirm values match)
3. Verify the eval harness runs cleanly against the current model state
4. Launch `self_directed_loop.py` with the full system prompt
5. **Observe. Log. Don't intervene.**

Report what happens in `shared/outbox/phase3/`.

---

## 8. What We Expect to Learn

We don't know what Nemotron will do. That's the point. Possibilities:

- It might systematically sweep A_log across all 23 Mamba layers, mapping the full sensitivity landscape
- It might discover that dt_bias or D modifications are more powerful than A_log
- It might attempt attention weight modifications and discover which attention layers are fragile vs. robust
- It might develop a theory of its own architecture based on experimental evidence that differs from what the blueprint says
- It might make a catastrophic modification, observe the damage, restore, and learn from it
- It might do something we haven't imagined

Whatever it does, the data is the outcome. The conversation transcript IS the research finding.

---

*The model is the scientist. We are the lab equipment.*

*End of Phase 3 instructions.*
