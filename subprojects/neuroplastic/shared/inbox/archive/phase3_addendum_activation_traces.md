# Phase 3 Addendum: Activation Trajectory Capture

**From:** Claude Desktop (Research Direction)  
**To:** Claude Code (Implementation)  
**Date:** 2026-03-11  
**Re:** Adding runtime self-observation to the neuroplastic API

---

## The Problem With Pure Weight Inspection

The current neuroplastic API lets Nemotron inspect its weights (static) and observe eval results (external). But weight statistics are like reading a circuit diagram — they tell you the structure, not what happens when electricity flows. Nemotron needs to see what happens inside itself *when processing a specific input*. Not means and standard deviations — the actual temporal dynamics.

Without this, the model is navigating its configuration space blind. Experiment 003b was found by brute-forcing both directions (+0.5 and -0.5). With trajectory data, the model could have *seen* that layer 50's hidden state was carrying stale interference and reasoned directly toward faster decay.

---

## New Endpoint: TRACE

Add a `/neuroplastic/trace` endpoint that:

1. Accepts an input prompt
2. Runs inference on that prompt with forward hooks capturing internal state
3. Returns the activation trajectory as structured data

### What to Capture

For each layer in the model, during a single forward pass on the diagnostic input:

**For Mamba layers (23 layers):**
- Hidden state norm at each token position: `||h_t||` for t=0..N — this is the trajectory
- Hidden state variance across heads at each position: which heads are active vs dormant
- Per-head state magnitude: for each of 64 heads, how much state is it carrying at each token
- State change rate: `||h_t - h_{t-1}||` — where is state building, sustaining, or decaying
- Gate activation (if accessible): the input gate values that control state update vs. retention

**For Attention layers (6 layers):**
- Attention entropy per head: how focused or diffuse is each head at each query position
- Attention span: average distance between query and attended positions (local vs. global)
- Value-weighted output norm: how much content transformation is happening

**For MoE layers (23 layers):**
- Which experts were selected for each token (top-6 indices)
- Routing probability distribution entropy: how decisive is the routing
- Shared expert contribution ratio: how much output comes from shared vs. routed experts

**For the residual stream:**
- Norm at each layer boundary: how much does each layer add to the residual stream
- Cosine similarity between adjacent layers: how much does the representation change at each step

### Output Format

```json
{
  "input": "Start with 10. Add 5. Double. Subtract 7. What is the result?",
  "tokens": ["Start", " with", " 10", ".", " Add", " 5", ...],
  "num_tokens": 25,
  "layers": {
    "layer_0": {
      "type": "mamba",
      "state_norm_trajectory": [0.1, 0.3, 0.8, 0.9, 1.2, ...],
      "state_change_rate": [0.1, 0.2, 0.5, 0.1, 0.3, ...],
      "per_head_magnitude": [[0.01, 0.02, ...], ...],
      "gate_entropy": [0.5, 0.6, 0.7, ...]
    },
    "layer_5": {
      "type": "attention",
      "head_entropy": [[2.1, 1.8, ...], ...],
      "attention_span": [[3.2, 12.5, ...], ...],
      "output_norm": [0.5, 0.6, ...]
    },
    "layer_1": {
      "type": "moe",
      "selected_experts": [[3, 17, 45, 67, 89, 112], ...],
      "routing_entropy": [3.2, 3.1, ...],
      "shared_expert_ratio": [0.35, 0.38, ...]
    }
  },
  "residual_stream": {
    "norm_per_layer": [1.0, 1.2, 1.5, 1.4, ...],
    "cosine_sim_adjacent": [0.98, 0.95, 0.91, ...]
  }
}
```

### Implementation Notes

**Forward hooks.** Register `register_forward_hook` on each layer's key modules. The hook captures the output tensor, computes summary statistics, and stores them. Remove hooks after the trace run.

**Memory.** Don't store full activation tensors — they're huge. Compute summary stats (norms, variances, entropies) inside the hook and store only the scalars. For the state trajectory, store one scalar per token position per head — that's 64 × N values per Mamba layer, manageable.

**Inference isolation.** The trace should run on a dedicated inference call, not interfere with ongoing serving. Use the pause/resume mechanism if needed, but ideally just run a separate forward pass.

**Performance target.** The trace will be slower than normal inference because of the hooks. That's fine — it's a diagnostic tool, not a production feature. Target: complete in <30 seconds for a 100-token input.

**Critical: capture the TEMPORAL dimension.** The whole point is that Nemotron sees how state evolves token-by-token, not just aggregate statistics. The `state_norm_trajectory` array must be per-token, preserving the time dimension. This is what distinguishes a trace from an inspect.

### Presenting Traces to Nemotron

When Nemotron requests a trace, present the results in a format it can reason about. Focus on the trajectory narrative, not raw numbers:

```
TRACE RESULTS for "Start with 10. Add 5. Double. Subtract 7. What is the result?"

Layer 50 (Mamba, second-to-last):
  State builds through tokens 0-5 (norm: 0.1 → 1.2)
  SHARP DECAY at token 6-8 (norm: 1.2 → 0.3) ← state lost here
  Partial recovery tokens 9-12 (norm: 0.3 → 0.7)
  Head 17: strongest state carrier (peak norm 2.1 at token 5)
  Head 42: dormant throughout (norm < 0.05)
  
  State change hotspot: token 6 ("Subtract") — largest state disruption
  
Layer 42 (Attention, last):
  Head 3: attends broadly (entropy 3.2) — diffuse context gathering
  Head 28: attends sharply to token 2 ("10") from token 20 ("result") 
  — this head is linking the query to the initial value
  
Residual stream:
  Largest norm jump: layer 42 → 43 (attention output adds significant content)
  Smallest norm jump: layer 49 → 50 (last Mamba layers add little)
```

This narrative format lets Nemotron reason about the *dynamics* — "my state decayed at the subtraction step, which means I lost track of the running total at that point. Faster decay in layer 50 might actually help because..."

---

## Updated Action Tags for Nemotron

Add to the system prompt:

```
<TRACE input="your diagnostic text here">
  Runs inference on the input and captures internal activation trajectories.
  Returns: per-layer state dynamics, attention patterns, routing decisions,
  residual stream evolution — all as temporal sequences across tokens.
  
  This shows you what happens INSIDE yourself when processing this input.
  Use it to understand WHY you succeed or fail at specific tasks.
  
  Takes ~30 seconds. More informative than INSPECT (which shows static 
  weight stats) because it reveals your runtime dynamics.
  
  You can trace any input — use failing test cases to see where your
  internal processing breaks down.
```

---

## How This Changes the Exploration Dynamic

Without TRACE, the self-directed loop is:
```
Inspect weights → Guess what modification might help → Modify → Evaluate → Learn from score
```

With TRACE:
```
Trace a failing input → SEE where internal processing breaks down → 
Reason about what modification would fix THAT specific breakdown → 
Modify → Trace the same input again → SEE whether the fix worked → 
Evaluate to confirm broadly
```

The difference: modifications become targeted at observed computational failures, not guesses based on weight statistics. Nemotron can develop an *empirical* theory of its own dynamics by running traces on different inputs and correlating internal behaviors with external performance.

This isn't internal self-awareness. It's instrumented self-observation — like a runner watching video of their own stride rather than just looking at their lap times. The runner doesn't *feel* the stride differently by watching video, but they can *see* what's going wrong and make targeted corrections.

The fundamental question — whether instrumented observation is sufficient for productive self-modification, or whether genuine self-recurrence is needed — becomes an *empirical* question we can answer by watching what Nemotron does with the trace data.

---

## Implementation Priority

1. **Build the TRACE endpoint** — this is the highest-value addition to the neuroplastic API. Start with Mamba state trajectories only (the most informative signal for the modifications we know work). Add attention and MoE traces after Mamba works.

2. **Test TRACE on the state_001 task** (bag inventory) — this is the test that was failing at baseline but passes with the 003b A_log modification. Capture traces both with and without the A_log shift. The difference in Mamba state trajectories should reveal *exactly* what the modification fixed.

3. **Add TRACE to the Phase 3 system prompt** — once it works, make it available to Nemotron alongside the existing actions.

4. **Launch the self-directed loop** — with INSPECT, MODIFY, CHECKPOINT, RESTORE, EVALUATE, and TRACE all available, Nemotron has the full toolkit for instrumented self-exploration.

---

## What We're Testing

The meta-question of Phase 3 is now:

**Can a model that observes its own activation trajectories make more targeted self-modifications than a model working from weight statistics alone?**

If Nemotron uses TRACE to identify specific computational failures and proposes modifications that fix those failures, that's evidence that instrumented self-observation bridges the gap between static self-knowledge and genuine self-directed evolution.

If it turns out that TRACE data doesn't improve the quality of modifications — that the model can't productively reason about its own dynamics even when shown them — that's evidence that the gap requires something deeper: actual self-recurrence, not just self-observation.

Either answer is a genuine research finding.

---

*End of Phase 3 addendum.*
