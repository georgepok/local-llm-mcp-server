# MIND v2.1 — Outstanding Issues Remedy Request

## Context
After five rounds of fixes (RMS normalization, bias normalization, tau rescaling, buffer expansion, generation cleanup), the system produces clean, geometrically-informed responses with correct causal chain reasoning and cross-domain structural recognition. This document covers what remains.

---

## P1 — Repetition loop in generation

**Observed:** Turn 4 (causal chain question) produced a perfect first paragraph then repeated it 4 times verbatim until max_tokens.

**Likely cause:** The normalized attention bias creates a strong routing pattern that Qwen3-4B (4B params) locks onto during autoregressive generation. Once the model produces a sequence that aligns with the bias pattern, it re-enters the same attention basin on each subsequent token and repeats.

**Fix options (try in order):**
1. Add `repetition_penalty=1.2` to the generation config in QwenBridge
2. If that's insufficient, reduce `bias_lambda` from 0.3 to 0.1-0.2 — weaker bias gives Qwen3 more freedom to break out of repetition
3. Add post-generation dedup: if any 30+ token sequence appears twice in the response, truncate at the first occurrence

**Acceptance:** No repeated paragraph in the causal chain test (5-turn bridge→shortage scenario).

---

## P2 — D²/4τ live diagnostic returns 0.0

**Observed:** Every `converse` call returns `D_sq_4tau: 0.0` or `0.001`. Offline test showed `D²/4τ = 63.0`.

**Likely cause:** The cross-event sampling in the bias computation path either:
- (a) Doesn't have access to `event_id` per token during the generate path (only during observe_event), or
- (b) Samples too few pairs and misses cross-event combinations, or
- (c) Uses `D²_within` (which IS 0) instead of `D²_across` for the reported metric

**Fix:** 
- Guarantee cross-event sampling: instead of random pairs, explicitly sample one token from event A and one from event B. With 3+ events this always works.
- Verify that the reported `D_sq_4tau` in the `converse` response uses `D²_across`, not `D²_all` or `D²_within`
- Log the number of cross-event pairs found in each sample for debugging

**Acceptance:** `D_sq_4tau` returns a value > 1.0 on any `converse` call with 2+ prior events.

---

## P3 — Tau diagnostic: raw vs rescaled ambiguity

**Observed:** The report states diagnostics show raw TauNet output (~0.34), while the ODE uses rescaled values (~1.5-2.5). But `converse` returns `tau_mean=2.65` and `get_diagnostics` returns `tau_mean=1.99`. It's unclear which is raw and which is rescaled.

**Fix:** Unify all tau reporting to show the ACTUAL value used in the LTC contraction `dh/dt = -(1/τ)(h - target)`. This is the rescaled value. Optionally expose raw TauNet output as a separate field `tau_raw` for debugging.

Affected endpoints:
- `get_diagnostics` → `tau_mean`, `tau_std` should be rescaled
- `converse` response → `tau_mean` should be rescaled
- `observe_event` log lines → same
- `[bias]` log lines → same

**Acceptance:** `get_diagnostics().tau_mean` and `converse().tau_mean` return the same value (within ODE cycling variance) when called in sequence with no intervening events.

---

## P4 — Thinking trace leakage (remaining cases)

**Observed:** Turn 4 response started clean. Turn 5 (cross-domain) was clean. But some edge cases still produce "Okay, let's see..." or "The user is asking..." preambles.

**Fix:** In addition to the current pattern stripping, try:
1. Check if Qwen3-4B supports `enable_thinking=False` in generation config — this would suppress thinking at the model level rather than post-processing
2. If not, add a negative prompt / logit bias: suppress token IDs for "Okay", "Let me", "Hmm," in the first 5 generated tokens
3. Strengthen the system prompt: "Never begin your response with reasoning about what the user wants. Start directly with your answer."

**Acceptance:** 10 consecutive `converse` calls produce zero thinking preambles.

---

## P5 — B_within vs B_across breakdown in live diagnostics

**Observed:** The offline test confirmed B_across > B_within (geometry favors cross-event routing). But the `converse` response doesn't include this breakdown — only the aggregate D²/4τ (which is broken per P2).

**Fix:** Add to `converse` response:
```json
{
  "B_within_mean": 115.5,
  "B_across_mean": 145.4,
  "B_across_max": 532.6,
  "B_range": 1960
}
```

This is the most informative diagnostic for the geometric routing. B_across > B_within confirms the geometry is connecting events, not isolating them. B_across_max shows the strength of the strongest cross-event connection. B_range shows overall routing dynamic range.

**Acceptance:** `converse` response includes `B_across_mean` and `B_across_max` fields with non-zero values.

---

## P6 — Bootstrap token budget

**Observed:** Bootstrap consumes 389-485 of 1024 buffer slots (38-47%) before any conversation content arrives. After 7 events + 2 generations, buffer reaches 818. For longer conversations (15+ turns), the buffer will fill and start dropping conversation tokens.

**Request:** Characterize what the bootstrap tokens contain and whether they can be reduced:
- How many are from "Mind initialized. ODE encoder active." (the bootstrap text)?
- How many are from autonomous loop initialization?
- Can bootstrap text be shortened to 2-3 tokens?
- Can bootstrap tokens be assigned lowest drop priority so they're evicted first when the buffer fills?

Not urgent — 1024 handles 7+ turn conversations. But for production use with longer sessions, the bootstrap overhead matters.

**Acceptance:** Bootstrap uses < 100 tokens, OR bootstrap tokens are dropped first when buffer fills.

---

## Summary

| # | Issue | Effort | Impact | Blocks |
|---|---|---|---|---|
| P1 | Repetition loop | Small (generation config) | Response quality | Nothing, but visible |
| P2 | D²/4τ live diagnostic | Small (sampling fix) | Correct criticality assessment | Understanding whether geometry is at target |
| P3 | Tau raw vs rescaled | Small (reporting unification) | Diagnostic clarity | Correct interpretation of all tau metrics |
| P4 | Thinking traces | Small-medium | Response quality | Nothing, cosmetic |
| P5 | B breakdown in converse | Small (add fields) | Confirms geometry helps | Understanding routing contribution |
| P6 | Bootstrap budget | Medium | Long conversation support | Nothing currently |

P1-P3 are the functional priorities. P4-P5 are diagnostic/quality. P6 is forward-looking.
