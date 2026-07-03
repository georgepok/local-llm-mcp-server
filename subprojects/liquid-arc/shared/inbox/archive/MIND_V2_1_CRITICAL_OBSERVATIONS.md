# MIND v2.1 — Critical Observations from Live Testing

## Source
Claude (research partner) testing Mind MCP after reading LIQUIDARC_MIND_V2_1_UPDATE.md. Tests conducted on live deployment with clean reset, 5-turn causal chain + cross-domain structural recognition.

## Summary

The token-level architecture is correct and working mechanically. But the GEOMETRY is dormant on text inputs — the MetricNet produces near-uniform output, tau has zero differentiation, and the attention bias is too weak to steer generation. The causal chain reasoning works because of Qwen3's ICL on the text context prompt, not because geometric routing contributes.

---

## Critical Findings

### 1. tau_std is effectively ZERO

```
tau_mean: 2.999 → 2.776 → 2.216 (moves across turns)
tau_std:  2.46e-7                (ZERO across all turns)
```

Every token position gets identical integration speed. The tau_quality_loss targets log_tau_std=0.6 (~2× per-position ratio), and the ARC experiments achieved 0.58-0.86. On text deltas, TauNet produces flat output. This means the per-position adaptive integration — the whole point of the tau management work — is not manifesting.

**Question for Dev:** Is tau_quality_loss active in the deployed d=2560 checkpoint? What were the tau metrics at the end of the criticality training (step 500)?

### 2. CV is sub-critical (0.55-0.68)

The ARC criticality experiments self-organized to CV≈7.5. The deployed Mind on text sits at CV≈0.6. That's 12× below the critical value. The MetricNet is producing near-identity metric — minimal differentiation across positions.

This is the same finding from all previous text integration attempts: the text delta distribution doesn't activate the MetricNet's learned routing structure. The checkpoint was trained on ARC where embeddings naturally cluster. Text deltas don't cluster the same way.

### 3. D²/4τ diagnostic is broken (negative values)

```
D_sq_4tau: -186.08, -243.24, -161.6
```

The report acknowledges this: SDPA-factored bias `B_ij = q·k/(2t) - ||k||²/(4t)` can produce positive B values (when q·k is large), making the reconstructed D² = -4t·B negative. The diagnostic is meaningless as currently computed.

**Request:** Add a SEPARATE D² diagnostic that computes actual pairwise metric-weighted Euclidean distances between sampled position pairs, independent of the SDPA factorization:
```python
# Sample 100 random position pairs
delta = h[:, idx_i, :] - h[:, idx_j, :]
g_avg = (g[:, idx_i, :] + g[:, idx_j, :]) / 2
D_sq_actual = (delta * g_avg * delta).sum(dim=-1).median()
D_sq_4tau_actual = D_sq_actual / (4 * tau.median())
```
Log this as `D_sq_4tau_actual` alongside the existing (broken) SDPA-derived value. This gives us the REAL operating point relative to the criticality target.

### 4. Relevance scoring is flat

```
Query: "bridge food shortage cascade"
Results: ALL events scored 0.579 or 0.509
Salience: 0.0 for all events
```

The relevance heads can't distinguish between the bridge closure event and the model's own response or the ecology question. All events are equally "relevant." The readout heads are untrained — they need online learning signal (the `provide_feedback` tool exists but isn't being used systematically).

This is lower priority than the geometry issues but worth noting: even if the MetricNet were at criticality, the relevance scoring would still return uniform results.

### 5. PE discrimination is BETTER in v2.1

```
v2.0 (event-level): 500 → 167 → 100 → 72 → 56  (monotonic decline)
v2.1 (token-level): 5.69 → 4.40 → 8.59 → 8.06 → 6.45 (non-monotonic)
```

Token-level PE correctly distinguishes:
- "Same topic continuation" (bridge → trucks): PE drops 5.69→4.40
- "New causal development" (trucks → landslide): PE rises to 8.59
- "Expected consequence" (landslide → shortage): PE slightly drops 8.59→8.06
- "Expected question" (causal chain query): PE drops to 6.45

This is a BETTER signal than v2.0's monotonic decline. The token-level architecture improves PE discrimination even without criticality.

### 6. Cross-domain structural recognition FAILED

v2.0 test: "Does the predator removal pattern remind you of anything?" → correctly connected to bridge/shortage chain.

v2.1 test: Same question → talked about Yellowstone wolves from training data. Did NOT connect to the bridge/shortage conversation.

The text context contains all events (verified via get_context). The failure is likely stochastic (Qwen3-4B sometimes makes the connection, sometimes doesn't), but indicates the attention bias at CV=0.68 isn't strong enough to reliably steer generation toward conversation context over training data.

### 7. h_norm grows unboundedly with token count

```
After reset:     1205 (bootstrap tokens)
After 4 events:  1291
After generation: 1398
After 8 events:  1409
```

The norm homeostasis operates per-position but total Frobenius norm grows with N (more positions = more total energy). At 450 tokens, h_norm≈1400. At the 512-token buffer limit, it'll be higher. This is acknowledged as expected behavior in the report, but worth monitoring — if token-level norm per position is also growing (not just total), that's a problem.

**Request:** Log `h_norm_per_position = h_norm / sqrt(n_tokens)` as a normalized metric. This should stay roughly constant if per-position norms are bounded.

---

## Architectural Assessment

| Component | Status | Evidence |
|---|---|---|
| Token extraction (Δh) | Working | n_tokens tracked correctly, buffer growing |
| Token buffer management | Working | 401→410→429→450 tokens across events |
| ODE integration on tokens | Working | h_norm responds, CV moves |
| SDPA bias computation | Working | bias_applied=true, correct dimensions |
| Generation feedback loop | Working | Generated tokens feed back into ODE |
| PE discrimination | Improved | Non-monotonic, topic-sensitive signal |
| MetricNet differentiation | NOT WORKING | CV=0.55-0.68 (sub-critical, target ≈7) |
| Tau differentiation | NOT WORKING | tau_std=2.5e-7 (effectively zero) |
| Criticality regime | NOT WORKING | D²/4τ negative/meaningless |
| Relevance discrimination | NOT WORKING | Flat scores (0.579 for all) |
| Geometric routing influence on generation | MINIMAL | Bias applied but too weak to steer |
| Cross-domain structural recognition | UNRELIABLE | Works sometimes (v2.0), fails sometimes (v2.1) |

---

## Recommended Next Steps

### Priority 1: Fix D²/4τ diagnostic
Add the actual pairwise D² computation (sampled, not SDPA-derived). Without this, we can't measure whether the system is anywhere near criticality on text. This is a 20-line change.

### Priority 2: Measure text delta distribution
Before trying to make criticality work on text, we need to CHARACTERIZE the text delta distribution:
- What is the D² between text delta token pairs? (range, median, distribution shape)
- How does it compare to ARC embedding D²? (ARC: 37-427 through transition)
- Is there natural clustering in text deltas? (the reviewer predicted heavy-tailed with small-delta cluster near zero)
- What D²/4τ target would be critical for THIS distribution?

Run a diagnostic-only experiment: feed 100 diverse text inputs through the delta extractor, compute pairwise D² statistics, histogram the distribution. This tells us whether the ARC-calibrated criticality target (60 at d=2560) is anywhere close to appropriate for text.

### Priority 3: Online criticality adaptation
If text D² is in a completely different regime than ARC D², the criticality target needs to be recalibrated. Options:
- **Manual:** Compute text D² median, set target to text_D²_median / (4 × tau_target)
- **Automatic:** Add an adaptive criticality target that slowly tracks the actual D² distribution:
  ```python
  D_sq_ema = 0.99 * D_sq_ema + 0.01 * D_sq_actual_median
  criticality_target = D_sq_ema / (4 * tau_mean_target)
  ```
  This would let the system find its OWN critical point for whatever distribution it's processing.

### Priority 4: Investigate tau flatness
Is TauNet producing flat output because:
- (a) The checkpoint's TauNet weights haven't adapted to text? → Online learning should eventually fix this
- (b) Text deltas at d=2560 are too uniform for TauNet to differentiate? → Architectural issue
- (c) tau_quality_loss isn't active in the deployed config? → Config issue

Log TauNet's intermediate activations (pre-sigmoid logits) on a few text inputs. If the logits are all identical → the input to TauNet is too uniform (text deltas don't vary enough per-position). If logits vary but tau is flat → the sigmoid is saturating (need to check tau_min/tau_max range).

### Priority 5: Think about whether generation quality matters yet
Qwen3-4B is a 4B model. It's going to produce "I'm sorry, I can't provide an answer" sometimes, and thinking trace leakage always. These are LLM limitations, not LiquidARC limitations. The GEOMETRIC metrics (CV, tau_std, D²/4τ, PE trajectory) are what tell us if the architecture is working. Generation quality improves automatically when we swap to a bigger model.

The one generation-related fix worth doing: `enable_thinking=False` or post-process to strip `<think>` tags. The thinking traces are distracting and make it hard to evaluate whether the actual response content is geometry-influenced.

---

## Key Question

The sustained criticality experiments proved: D²/4τ=18 criticality loss produces 2-3× eval improvement on ARC, distribution-invariant geometry, cv·τ≈8 conservation. All at d=768.

The d=2560 checkpoint was trained with criticality scaffolding (target D²/4τ=60). But it was trained on ARC data, then deployed to process text deltas from Qwen3-4B.

**Is the d=2560 checkpoint's MetricNet still in the critical regime for ARC inputs?** If we feed it ARC data through the delta extractor (compute deltas from ARC grid descriptions processed through Qwen3), does the MetricNet produce CV≈7 and D²/4τ≈60? If yes → the MetricNet is fine, text deltas are the problem. If no → the checkpoint itself may have drifted during deployment.

This would tell us definitively whether the issue is input distribution (text deltas don't match what MetricNet expects) or model state (MetricNet has degraded).
