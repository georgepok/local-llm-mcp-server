# MIND v2.1 Post-Fix — Remaining Issues & Refinements

## Source
Claude testing Mind MCP after RMS normalization fix. The fix transformed the system — tau_std went from 2.5e-7 to 0.95, MetricNet activated (11.7× amp), cross-domain reasoning now succeeds. This document covers what still needs work.

---

## 1. D²/4τ diagnostic STILL unreliable in live deployment

The offline characterization showed D²/4τ=9.0 for text. But live deployment returns:

```
Turn 4 (causal question):    D_sq_4tau = 0.001
Turn 5 (cross-domain):       D_sq_4tau = 1.02
```

These are 9× and 9× below the offline measurement. The sampled pairwise D² computation is likely hitting the bimodal distribution: same-event token pairs (near-zero D²) dominate the sample.

**Fix:** When sampling pairs for the D² diagnostic, sample ONLY cross-event pairs. Each token carries a `source` field in the token buffer. Filter: `idx_i.source != idx_j.source`. This gives the cross-event D² that matters for inter-turn routing.

Alternatively, report three D² values:
```
D²_within:  median D² for same-event pairs (should be small — tokens converge)
D²_across:  median D² for cross-event pairs (this is the routing-relevant metric)
D²_all:     median D² for all pairs (the bimodal mix — less useful)
```

`D²_across / (4τ)` is the number that tells us if cross-turn routing is near criticality.

## 2. Token buffer at 512 after 7 events — dropping strategy matters

```
After reset:       389 tokens (bootstrap)
After 3 events:    456 tokens (+67 from 3 user messages)
After 2 generations: 512 tokens (buffer full, dropping starts)
```

The bootstrap alone consumes 389 of 512 slots (76%). After two generate cycles, the buffer is full. Every subsequent event drops old tokens.

**Questions:**
- What's the dropping strategy? FIFO? Salience-based? Random?
- Do bootstrap tokens get dropped first? They should — they carry no conversation content.
- Are structurally important early tokens (the bridge closure from Turn 1) preserved when the buffer fills?

**Recommendation:** 
- Reduce bootstrap tokens or give them lowest drop priority
- Consider a two-tier buffer: protected slots (first N tokens of each user event, never dropped) + recyclable slots (generated tokens, bootstrap, continuations)
- Or increase buffer to 1024 if memory allows — 512 is tight for multi-turn conversation with generation feedback

## 3. tau_mean drops aggressively: 3.0 → 0.29

```
After reset:       tau_mean = 3.00
After 1 event:     tau_mean = 3.00  (tau_std = 0.62)
After 2 events:    tau_mean = 3.00  (tau_std = 0.82)
After 3 events:    tau_mean = 3.00  (tau_std = 0.95)
After generation:  tau_mean = 2.53
After generation:  tau_mean = 1.47
Final diagnostic:  tau_mean = 0.29  (tau_std = 0.24)
```

tau_mean collapses from 3.0 to 0.29 across the conversation. This is 10× below the tau_quality_loss target of ~2.0. The tau_quality_loss should be anchoring tau_mean near its target — is it active in the deployed config?

If tau_mean reaches 0.29 with tau_min=0.1 and log_tau_std target=0.6, positions are in the range [0.05, 0.53]. At tau=0.05, each Euler step moves the state 1/0.05 = 20× toward target — likely overshooting. At tau=0.29 mean, the ODE is in a very aggressive integration regime.

**Check:** Is `tau_quality_loss_enabled: true` in the deployed config? If yes, what is `tau_quality_lambda`? The collapse suggests either the loss is disabled or lambda is too small to counteract the ODE dynamics pushing tau down.

**If tau_quality_loss IS active:** The online dynamics are overpowering the loss. The convergence coupling (high residual → lower tau) may be too aggressive — every new event has high initial residual, which pushes tau down, which makes integration faster, which causes MORE displacement on the next event, which keeps residual high... a positive feedback loop driving tau to minimum.

**Potential fix:** Add a slow tau recovery term — tau drifts back toward tau_mean_target when no events are arriving (during autonomous cycling). Or reduce tau_convergence_beta so the coupling doesn't dominate.

## 4. CV still sub-critical (0.55-1.08 vs ARC's 6.2)

The RMS fix raised CV from dead (0.5) to alive (0.55-1.08), but it's still 6-10× below the ARC critical value. The MetricNet amplifies 11.7× but overall differentiation is low.

**This may not be a bug.** Text deltas have continuous structure, not discrete categorical structure. The critical CV for text may be lower than for ARC. The question: at CV=0.6-0.9 with D²_across/4τ≈9, is the attention bias producing STRUCTURED routing or near-uniform routing?

**Diagnostic request:** During a converse call, log the attention bias matrix statistics:
```
B_max:    maximum bias value (how much the strongest pair's attention is boosted)
B_min:    minimum bias value (how much the weakest pair is suppressed)
B_range:  B_max - B_min (dynamic range of the bias)
B_std:    standard deviation of bias values (spread of routing signal)
```

If B_range is tiny (e.g., 0.01), the bias is near-uniform regardless of CV. If B_range is meaningful (e.g., 0.5-2.0), the routing is structured even at CV=0.6. This is the metric that directly measures "is the geometry doing something useful to attention."

## 5. Thinking trace strip is incomplete

Responses still start with:
```
 Assistant\nOkay, let's see. The user wants to...
```

The regex stripping catches some patterns but not "Assistant\n" prefix or "Okay, let's see" variations.

**Better approach:** Pass `enable_thinking=False` in the Qwen3 generation config (if supported by the model). If not supported, strip everything before the first line that doesn't match common preamble patterns:
```python
preambles = ["okay,", "let me", "i need to", "first,", "so,", "assistant", "hmm,"]
lines = response.split('\n')
for i, line in enumerate(lines):
    if not any(line.strip().lower().startswith(p) for p in preambles):
        return '\n'.join(lines[i:])
```

## 6. Context echoing in responses

From the fixes report: "Temporal: Mind initialized. ODE encoder active." appears in responses because the system bootstrap text is in the context prompt.

**Fix:** Filter event types from the context prompt. System/bootstrap events should not appear in the text context fed to Qwen3 for generation. Only include user_message and assistant_message events (and possibly goal events).

## 7. Adaptive criticality target — ready to implement?

The offline characterization found D²/4τ=9 for text (vs 12 for ARC, vs 18 at d=768). The D² EMA tracking was added. The next step from the original observations was:

```python
D_sq_ema = 0.99 * D_sq_ema + 0.01 * D_sq_actual_median
criticality_target = D_sq_ema / (4 * tau_mean_target)
```

Is this worth activating now? The risk: the adaptive target could chase the D² distribution down as tau collapses (issue #3), creating a moving target that never stabilizes. It might be better to first fix the tau collapse, THEN activate adaptive criticality once tau_mean is stable.

**Recommended order:**
1. Fix tau collapse (ensure tau_quality_loss is active and strong enough)
2. Get stable tau_mean near target (1.0-2.0)
3. Then activate adaptive D²/4τ target based on actual text D² distribution
4. Then add criticality loss to push toward that target

Without stable tau, the criticality target is meaningless — D²/4τ changes because tau is drifting, not because the MetricNet is adapting.

---

## Summary of priorities

| # | Issue | Effort | Impact |
|---|---|---|---|
| 1 | D² diagnostic: cross-event sampling only | Small (filter by source) | Correct metric for criticality assessment |
| 2 | Token buffer: bootstrap bloat + dropping strategy | Medium | Conversation longevity |
| 3 | Tau collapse: investigate tau_quality_loss status | Small (config check) | Prevents runaway dynamics |
| 4 | Bias statistics logging | Small (add to diagnostics) | Answers "is geometry helping?" |
| 5 | Thinking trace: enable_thinking=False | Small | Response quality |
| 6 | Context prompt: filter system events | Small | Response quality |
| 7 | Adaptive criticality target | Hold until tau stable | Future: auto-calibrate to text |

Items 1, 3, and 4 are diagnostic — they tell us whether the system is working. Items 2, 5, 6 are quality-of-life. Item 7 is the path to full geometric engagement on text.

The tau collapse (#3) is the most important: if tau is collapsing to 0.29 with the quality loss supposedly anchoring at 2.0, something is wrong in the loss configuration or the dynamics are overpowering it. Fix that first — everything else depends on stable tau.
