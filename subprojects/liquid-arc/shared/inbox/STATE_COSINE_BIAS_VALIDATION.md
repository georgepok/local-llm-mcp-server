# STATE COSINE BIAS — Validation Results & Issues

## Source
Claude testing Mind MCP after BIAS_MECHANISM_ANALYSIS changes. Clean reset, 5-turn causal chain + cross-domain + factual recall test.

## Assessment

The architectural pivot from MetricNet bias to state cosine + displacement bias is correct in principle. The ODE's internal routing produces state alignment that the cosine similarity extracts. Response quality is the BEST the system has produced — clean formatting, no repetition, concise cross-domain connection.

BUT: live diagnostics show B_across = 0.0 on every call, contradicting the report's B_across > B_within finding. And curriculum injections during active conversation are diluting the ODE state and context.

---

## Issue 1: B_across = 0.0 in live deployment (CRITICAL)

Every `converse` call returns:
```
B_across_mean: 0.0
B_across_max: 0.0
B_within_mean: 0.085 → 0.469 (increasing)
```

The report showed B_across = 0.449, 0.500 during testing. Three possibilities:

**(a) event_id propagation still broken.** Generated tokens and curriculum tokens may not carry correct event_ids. If the sampling filter treats everything as same-event, B_across can't be computed. To verify: log the number of unique event_ids in the token buffer during a converse call, and the number of cross-event pairs found in the 500-pair sample.

**(b) Curriculum injections dilute cross-event alignment.** During the test (no curriculum), 3 events × ~22 tokens = 66 conversation tokens in a focused buffer. During live deployment, 4 curriculum articles + 4 reflections + 3 user events + 2 generated responses = ~450 tokens from 13 sources. The state cosine between a bridge-closure token and a random Augustine-theology token is low, dragging B_across toward zero.

**(c) The autonomous loop's ODE cycling overrides conversation alignment.** Between user calls, the autonomous loop runs ODE cycles that evolve the state. This may de-align conversation token states over time, weakening the cross-event cosine.

**Fix:** Add logging inside the B_across computation showing: n_unique_event_ids, n_cross_pairs_sampled, raw cosine values for the cross-event pairs. This distinguishes (a) from (b)/(c).

## Issue 2: Curriculum injections during active conversation (HIGH)

Context dump shows 4 Wikipedia articles injected during a 5-turn conversation:
```
Index 0:  Ajax mythology
Index 9:  Breast anatomy  
Index 12: Alabama River
Index 15: Augustine theology
```

These have NO relation to the causal chain. They consume:
- Token buffer space (each article = 20-30 tokens)
- ODE state positions (diluting geometric alignment)
- Text context prompt slots (Qwen3 sees them as conversation history)

**Fix:** Add conversation activity detection. When `event_type=user_message` arrives within the last 60 seconds, suppress curriculum injection and autonomous reflections. Resume after 120 seconds of silence. This is a simple flag in the autonomous loop:

```python
if time.time() - self._last_user_event_time < 120:
    skip_curriculum = True
    skip_reflection = True
```

This is the HIGHEST PRIORITY FIX for response quality. The curriculum is actively damaging conversation coherence.

## Issue 3: Root cause misidentification (MEDIUM)

The chain-tracing response identified "heavy truck traffic" as root cause instead of the bridge closure. Previous versions (MetricNet bias) correctly identified the bridge closure.

Possible causes:
- Within-event bias (B_within=0.47) concentrates attention on the LANDSLIDE event which has more content tokens and strong internal alignment
- The bridge closure event's tokens may be weakly aligned with other events because curriculum injections between turns de-aligned the state
- Qwen3-4B (4B params) recency bias — prefers recent/vivid events

**Test:** Run the same chain test with curriculum DISABLED. If root cause is correctly identified → curriculum dilution is the cause. If still missed → bias mechanism issue.

## Issue 4: Factual recall failure (MEDIUM)

"On what exact date was the bridge closed?" → "I don't have access to that specific information."

The event IS in context (verified via get_context: index 1 contains "March 1st"). But:
- 17 events in context prompt (should be 5)
- Relevance scores flat at 0.551 for ALL events (no discrimination)
- Qwen3-4B may not attend to the specific date when the context is cluttered with mythology, anatomy, and river geography

**Root cause:** Curriculum dilution + flat relevance + small model capacity. Fix curriculum suppression (Issue 2) and this likely resolves.

## Issue 5: Relevance scoring still flat (LOW)

All events scored 0.551 except one at 0.477. Query "bridge closed date March" returns equal relevance for the bridge closure event AND Ajax mythology AND breast anatomy. Readout heads are untrained.

Not a priority — fix curriculum first, then address relevance if still needed.

---

## Metrics Summary

| Metric | Value | Interpretation |
|---|---|---|
| PE trajectory | 18.8→17.1→16.2→12.8→25.8→6.0 | Working: declining with chain, rises on novel domain, drops on familiar |
| tau_mean | 2.4→1.2→0.66 | Declining — tau rescaling may not fully compensate for text dynamics |
| tau_std | 1.04→1.20→0.38 | Healthy differentiation, declining toward end |
| CV | 0.65→0.63→0.59 | Stable, sub-critical |
| B_range | 1960 | Large dynamic range (within-event) |
| B_across | 0.0 | Zero in all live calls (BUG or REAL?) |
| entropy_ratio | 0.82→0.91 | Approaching uniform — less peaked than optimal |
| h_norm_per_pos | 61.6→62.3 | Stable, bounded |

## Priority Order

1. **Suppress curriculum during active conversation** — highest impact on all other issues
2. **Debug B_across=0.0** — determines whether the state cosine mechanism works in practice
3. **Tau decline investigation** — tau_mean dropped from 2.4 to 0.66, may need stronger rescaling
4. **Root cause test with curriculum disabled** — distinguishes bias issue from dilution issue
5. **Repetition penalty in generation config** — prevents the repetition loops seen in earlier versions (not observed in this test, but worth hardening)
