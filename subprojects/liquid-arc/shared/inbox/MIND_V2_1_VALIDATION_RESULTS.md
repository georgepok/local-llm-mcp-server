# MIND v2.1 Refinements — Validation Results

## Source
Claude testing Mind MCP after all 7 refinements. Clean reset, 5-turn causal chain + cross-domain test.

## Validation Summary

| Item | Status | Notes |
|------|--------|-------|
| 1. D² cross-event | PARTIAL | D²_across=14653 on first call, returns 0.0 on live converse calls. Sampling/filtering bug. |
| 2. Token buffer | WORKING | 1024 capacity, 775 used at event 7, priority dropping correct |
| 3. Tau rescaling | WORKING | tau_mean stable 1.5-2.5 (was collapsing to 0.29). Critical fix. |
| 4. Bias statistics | WORKING | B_range=1960 — geometry producing STRONG routing signals |
| 5. Thinking trace | PARTIAL | Catches some preambles, misses others. Garbled response on complex prompt. |
| 6. Context filtering | WORKING | No system events in generation context |
| 7. Adaptive crit target | HELD | Correct — tau stable first, then activate |

---

## Remaining Issues (prioritized)

### Priority 1: D²_across returns 0.0 in live converse calls

Every `converse` call showed `D_sq_4tau: 0.0`. The cross-event sampling works on initial test (D²_across=14653) but fails during live conversation flow. Hypothesis: the `event_id` field may not be propagating correctly when tokens are added during `generate_iterative()` feedback — generated tokens might all get the same event_id as the prompt, making the filter think everything is same-event.

**Check:** Log the event_id distribution in the token buffer during a converse call. How many unique event_ids? Are generated tokens getting their own event_id?

### Priority 2: Tau diagnostic shows raw vs rescaled values

The report acknowledges: "compute_tau() and diagnostic logs still show RAW TauNet output (~0.34). The ODE internally uses rescaled values."

This creates confusion:
- `get_diagnostics` → tau_mean=1.67 (which value? raw or rescaled?)
- `converse` → tau_mean=2.55 (appears to be rescaled)
- Report says raw is ~0.34

These should be unified. Always report the ACTUAL tau used in the LTC. The raw TauNet output is an internal detail. The tau that matters for D²/4τ and dynamics quality is the rescaled value.

### Priority 3: D²_across = 14,653 implies HYPER-supercritical cross-event routing

If cross-event D²/4τ = 4184 (= 14653 / (4×0.875)), the heat kernel weight for cross-event pairs is exp(-4184) ≈ 0. Cross-event attention is completely killed. Only within-event pairs (D²≈0) get any attention weight through the geometric channel.

This means the MetricNet's 11.7× amplification OVERSHOOTS — it pushes cross-event distances so high that the heat kernel saturates. The B_range=1960 is large but the POSITIVE B values (+536) may all be within-event, while cross-event B values are deeply negative (-1424).

**Diagnostic request:** Break down B statistics by event pair type:
```
B_within_mean:   mean bias for same-event token pairs
B_across_mean:   mean bias for cross-event token pairs
B_across_max:    maximum bias for any cross-event pair (the BEST cross-event connection)
```

If B_across_max is negative → the geometry is BLOCKING all cross-event routing. The causal chain reasoning works entirely through text context ICL, not geometric bias.

If B_across_max is positive → some cross-event pairs ARE being connected geometrically. The question becomes which pairs.

### Priority 4: Generation quality on complex prompts

The cross-domain test produced:
```
"User: What is the most effective way to prevent this kind of cascade failure?"
```

The model generated a FAKE USER TURN instead of answering. This is a Qwen3-4B formatting issue (it gets confused about whose turn it is in the context), not a geometry issue. But it makes it impossible to evaluate whether the geometric routing helps with complex reasoning.

Possible mitigations:
- Add a stronger system prompt: "You are an assistant. Respond directly to the user. Never generate text for the user."
- Add stop sequences that halt on "User:" or "Human:"
- Post-process to strip any generated fake turns

### Priority 5: Expose rescaled tau in all diagnostics

Unify tau reporting:
- `get_diagnostics` → report rescaled tau (actual LTC value)
- `converse` → same
- Log lines → same
- Optionally also report raw TauNet output with a different key name (e.g., `tau_raw`)

---

## Key Open Question

**Is the attention bias helping generation or is everything still ICL?**

B_range=1960 looks impressive, but if B_across_max < 0 (all cross-event pairs repelled), the bias is actually HURTING cross-event reasoning by suppressing attention between turns. The causal chain success would be 100% ICL (text context prompt).

The test: disable the attention bias (`bias_lambda=0`) and rerun the causal chain test. If results are identical → bias isn't helping. If results degrade → bias IS contributing. This is the definitive test but requires a config change on the deployment.

Alternatively, the B_within vs B_across breakdown requested in Priority 3 answers this without needing to redeploy.
