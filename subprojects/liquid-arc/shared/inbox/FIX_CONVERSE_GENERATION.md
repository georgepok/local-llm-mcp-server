# TASK: Fix `converse` Tool — h_norm Collapse During INBOUND Step

**From:** Claude Desktop (Research Direction)
**To:** Claude Code (Implementation)
**Date:** 2026-04-05
**Priority:** HIGH — the primary interaction tool is broken

---

## The Problem

The `converse` MCP tool produces hallucinated garbage (Star Wars, movie reviews, random biographies) while `query_qwen` produces coherent, correct responses to identical prompts.

**Evidence:**

Same question: "What is a phase transition in the context of complex systems?"
- `query_qwen`: Perfect answer about critical thresholds, nonlinear dynamics, emergent behaviors. h_norm=102.5
- `converse`: "John McTiernan grew up in Chicago..." h_norm=16.875

The difference: `converse` runs INBOUND before OUTBOUND. `query_qwen` runs OUTBOUND only.

## Root Cause

The INBOUND step in `converse` crushes h_norm from ~100+ to ~15-17. The OUTBOUND step immediately follows, generating from this low-magnitude state. W_inject(h_pooled) produces prefix embeddings that are 6× weaker than normal. Qwen3 effectively generates without meaningful prefix context — producing random completions.

The INBOUND pipeline encodes the user message through `_force_geometric_signal`, which runs a full ODE integration cycle. This integration appears to reduce h_norm dramatically — either because the new forcing signal overwhelms the existing state, or because the ODE dynamics contract the state toward a low-norm equilibrium during the encoding pass.

## The Fix

The OUTBOUND step should use the ODE state from BEFORE the INBOUND encoding, or the state should be captured before INBOUND runs. The logic:

```
converse(message):
    # 1. Capture h(t) BEFORE inbound processing
    h_pre_inbound = snapshot(h(t))  # or mean_pool(h(t)).clone()
    
    # 2. INBOUND: encode user message into ODE
    inbound_result = _encode_through_coupling(message)
    # h(t) is now updated (possibly with crushed h_norm)
    
    # 3. OUTBOUND: generate using PRE-INBOUND state
    # The user's message should inform what Qwen3 responds to,
    # but the geometric context comes from the accumulated state
    # BEFORE this specific message disrupted it.
    prefix_embeds = W_inject(h_pre_inbound)  # full-magnitude prefix
    response = generate_through_qwen(message, prefix_embeds)
    
    # 4. FEEDBACK: encode the response into ODE
    feedback_result = _encode_through_coupling(response)
    
    return response
```

**Why pre-inbound state:** The user's message is brand new — the ODE hasn't had time to integrate it meaningfully. The accumulated state from everything BEFORE this message is the meaningful temporal context. The INBOUND step serves to eventually integrate the message for future queries, but for the current generation, the pre-inbound state is the right context.

This also matches how `query_qwen` works — it uses the current state without modifying it first, and it produces coherent output.

## Alternative Fix: Scale the Forcing Signal

If the INBOUND step should update h(t) before generation (so the response can reference the just-received message), then the fix is to prevent the h_norm collapse:

```python
def _force_geometric_signal(self, arc_signal, ...):
    # ... existing code ...
    
    # After ODE integration, restore h_norm to pre-forcing magnitude
    h_norm_before = self._h.norm()
    # ... ODE integration happens ...
    h_norm_after = self._h.norm()
    
    # Prevent more than 50% norm reduction from a single forcing event
    if h_norm_after < 0.5 * h_norm_before:
        scale = (0.5 * h_norm_before) / h_norm_after.clamp(min=1e-8)
        self._h.mul_(scale)
```

This prevents any single INBOUND event from collapsing h_norm. The ODE state evolves but maintains enough magnitude for W_inject to produce meaningful prefix.

## Recommended: Use Both Fixes

1. Capture h_pre_inbound for the OUTBOUND prefix generation (primary fix)
2. Also add the norm floor to `_force_geometric_signal` to prevent collapse in general (the CV collapse that was just fixed may have a related root cause)

This ensures:
- `converse` produces coherent output (using full-magnitude pre-inbound state for prefix)
- Future queries after many INBOUND events don't degrade (norm floor prevents cumulative collapse)
- `query_qwen` continues working as before (no change needed)

## Verification

After the fix:

```
converse("What is a phase transition?")
→ Should produce answer similar to query_qwen's coherent response
→ h_norm during OUTBOUND should be >50 (not 15-17)
→ Response should NOT contain hallucinated content (Star Wars, movie reviews, biographies)
```

Also verify that FEEDBACK still works — the response should enter the ODE state and be visible in subsequent `get_context` calls.

## Files to Modify

| File | Change |
|------|--------|
| `liquid_arc/mind.py` | In the `converse` method: capture h_pooled before INBOUND, use it for OUTBOUND prefix generation. Optionally add h_norm floor in `_force_geometric_signal`. |

One file, one method. The fix is capturing a single tensor before the INBOUND step and using it for the OUTBOUND step.
