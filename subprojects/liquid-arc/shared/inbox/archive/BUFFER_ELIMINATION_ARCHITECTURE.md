# BUFFER ELIMINATION — Architecture Recommendation

## The Problem Dev Identified

The token buffer is a redundant, worse context window:
- 512 tokens, 91.5% curriculum noise, 8.5% conversation signal
- Filled by autonomous loop BEFORE user starts talking
- ODE routes between accordion tokens and bridge tokens → meaningless bias
- Buffer management (dropping, priority, bootstrap overhead) adds complexity without value

Dev's diagnosis is correct. The buffer must go.

## Dev's On-The-Fly Proposal

Process only the current generation prompt through the ODE each call. No persistence. Eliminates all buffer problems.

**Trade-off:** Loses persistent geometric state (PE trajectory, accumulating B_across alignment, autonomous loop processing).

## Recommended Architecture: Persistent State + Per-Call Context

Don't eliminate persistence entirely. Eliminate the TOKEN buffer. Keep a small, fixed-size ODE state.

```
PERSISTENT (survives across calls):
  h_state ∈ ℝ^{1, K, d}     where K = 32-64 (fixed, NOT growing)
  This is the ODE's "working memory" — accumulated geometric structure

PER-CALL (rebuilt each call):
  context_prompt → Qwen3 forward → layer 18/24 deltas → [1, N_prompt, d]
  These are the CONVERSATION tokens Qwen3 will generate over

EACH CALL:
  1. Build context prompt from conversation event list (user + assistant only)
  2. Extract deltas: Qwen3 forward(context_prompt) → Δh → [1, N_prompt, d]
  3. Combine: h_combined = [h_state ; Δh_prompt]  → [1, K + N_prompt, d]
  4. Run ODE over h_combined (16 Euler steps, heat kernel routes between ALL positions)
     - State positions interact with prompt positions through heat kernel
     - Accumulated structure from h_state influences routing of current prompt
  5. Compute bias over N_prompt positions ONLY → B[N_prompt × N_prompt]
     - Exclude state positions from bias — they're not in Qwen3's prompt
     - But state positions DID influence prompt positions through ODE routing
  6. Update h_state from the ODE output (first K positions evolve)
     - h_state ← ODE_output[:, :K, :]  (persistent state updated by current context)
  7. Inject bias into Qwen3 attention → generate response
  8. Optionally: extract deltas from generated response, do one more ODE step to update h_state
```

## What This Preserves

**From the buffer approach:**
- Persistent geometric state across turns (h_state accumulates)
- PE signal: displacement of h_state when new context arrives
- The B_across trajectory: state positions develop cross-event alignment over multiple calls

**From Dev's on-the-fly approach:**
- Clean conversation-only tokens for bias computation
- No curriculum pollution
- No buffer management, dropping, bootstrap overhead
- Bias perfectly aligned with Qwen3's actual prompt

## What Changes

| Aspect | Current buffer | Proposed |
|---|---|---|
| ODE input | Token buffer [1, 512, d] (all events) | h_state [1, K, d] + prompt deltas [1, N, d] |
| Bias source | State cosine over ALL 512 tokens | State cosine over N_prompt tokens only |
| Persistence | Token buffer grows until full, drops old tokens | h_state is fixed size K, always updated |
| Curriculum | Injected into buffer, pollutes everything | Not in the pipeline at all during conversation |
| Autonomous loop | Processes buffer continuously | Updates h_state during idle time (optional) |
| Buffer management | Priority dropping, event_id tracking, bootstrap | None needed |

## The h_state Design

h_state is NOT a summary of past events. It's the ODE's internal state — the hidden representation that the dynamics maintain. It starts as learned initial state (from the checkpoint) and evolves as conversation events are processed.

Fixed size K = 32-64 positions. These are abstract "working memory" positions, not specific tokens. The MetricNet routes information from prompt tokens TO these positions during ODE integration, and FROM these positions back to prompt tokens. The positions develop structure that reflects the accumulated conversation.

This is analogous to:
- A transformer's KV cache (fixed size, updated per token)
- An RNN's hidden state (fixed size, accumulated over sequence)
- The brain's working memory (limited capacity, actively maintained)

The difference: the ODE's state positions participate in heat kernel routing with the current prompt. They're not passive storage — they actively shape how the current prompt tokens interact.

## Init and Training

**h_state initialization:** From the checkpoint's learned initial state, or zero-initialized and learned during the first few conversation turns.

**Training:** The sustained criticality system (D²/4τ loss, tau_quality, convergence coupling) operates on the combined [h_state ; prompt] tensor. The criticality target should be computed over the prompt positions only (since those are what the bias covers).

**Checkpoint:** Save h_state as part of the Mind's persistent state. Load on restart. This preserves conversation continuity across server restarts.

## What Happens to the Autonomous Loop

The autonomous loop currently processes the buffer continuously. Without a buffer, it would process h_state:

During idle time (no user messages):
1. h_state evolves under ODE dynamics (self-routing between state positions)
2. This is "consolidation" — the state settling into stable geometric structure
3. No curriculum injection (curriculum was the source of pollution)
4. Optional: inject a single "reflection" event to probe the state

This is actually BETTER than the current autonomous loop, which floods the buffer with curriculum. Idle consolidation without new input is how the biological brain consolidates memory.

## Migration Path

1. **Phase 1:** Implement per-call delta extraction (already exists in delta_extractor). Build the combine-route-split pipeline.
2. **Phase 2:** Add h_state as a persistent [1, K, d] tensor. Initialize from zeros or from the current buffer's ODE output (compressed to K positions via mean pooling over events).
3. **Phase 3:** Remove the token buffer entirely. Remove all buffer management code (dropping, event_id tracking, bootstrap injection).
4. **Phase 4:** Simplify the autonomous loop to h_state consolidation only.

## The Key Insight

The buffer tried to make the ODE into a CONTEXT STORE — accumulating all tokens the system has ever seen. But the LLM already has a context store (its prompt + context window). The ODE's value is GEOMETRIC ROUTING — discovering structural relationships. It needs access to current tokens (to route between them) and persistent state (to bring accumulated structure). It does NOT need to store the tokens themselves.

The ODE is a DYNAMICS engine, not a DATABASE. Treat it as such.
