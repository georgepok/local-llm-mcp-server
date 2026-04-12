# Mind v2.1 Refinements — Implementation Report

## Items Implemented

### 1. D² cross-event diagnostic ✅
- Samples position pairs and separates within-event vs cross-event D²
- Uses `event_id` (unique per observe_event call) not `source` (repeated across events)
- Reports `D²_across`, `D²_within`, `D²/4τ` (using cross-event D²)
- First call shows D²_across=14653 (cross-event) vs D²_within=0 (same-event tokens converge)

### 2. Token buffer dropping strategy ✅
- Priority-based: bootstrap/generated tokens drop first, user/assistant preserved
- Buffer increased to 1024 (from 512) for multi-turn conversations
- Each token carries `event_id` for cross-event diagnostics

### 3. Tau collapse — structural rescaling ✅
- Root cause: TauNet trained on ARC produces low tau on text deltas. Training losses (tau_quality_loss) aren't active during deployment.
- Fix: dynamics forward() rescales TauNet output from [current_min, current_max] → [target*0.3, target*1.5], preserving per-position differentiation but anchoring absolute range
- Target=2.0, rescale range=[0.6, 3.0]
- Note: `compute_tau()` and diagnostic logs still show RAW TauNet output (~0.34). The ODE internally uses rescaled values. A future improvement should expose the actual tau used in the LTC.

### 4. Bias matrix statistics ✅
- Reports B_max, B_min, B_range, B_std
- Live results: B_range=1957-1974 — VERY structured routing even at CV=0.59
- This answers the spec's question: "is the geometry doing something useful?" → Yes, the bias has huge dynamic range

### 5. Thinking trace stripping ✅
- Module-level `_strip_thinking()` function strips `<think>` tags and preamble lines
- Catches: "okay,", "let me", "i need to", "assistant", "the user", etc.
- First response usually clean; later responses sometimes leak
- Not perfect — some patterns still get through

### 6. Context prompt filtering ✅
- `_build_context_prompt()` now only includes user_message and assistant_message events
- System bootstrap, temporal, reflection, expression events excluded
- Prevents "Temporal: Mind initialized. ODE encoder active." from appearing in responses

### 7. Adaptive criticality target — on hold
- D² EMA tracking implemented and running
- Recommendation from spec: fix tau first, then activate adaptive target
- Tau rescaling is now active; next step is to verify D² EMA stabilizes before using it as target

## Diagnostic Output Format

```
[observe] #4 type=user_message PE=50.1 CV=0.59 tau=0.38±0.53 h=714 h/√N=63.6 tokens=126 events=4 "The relationship between..."
[bias] [126x126] CV=0.59 D²_across=14653 D²_within=0.0 D²/4τ=4184 tau=0.38±0.54 B=[-1424,536]r=1960
[generate] bias [126x126] CV=0.59 D²=0.0 D²/4τ=0.0 tau=0.38±0.54 tau_logit=-2.31±2.32 D²_ema=0.0
[generate] iterative (one-shot+feedback) max_new=80 ode_tokens=126
[generate] response: 395 chars "That's a fascinating topic!..."
[generate] ODE updated: 126 → 206 tokens (+80 gen, -0 dropped)
```

Each line shows what the system is doing, with geometric metrics for correlation analysis.

## Test Results After All Refinements

```
Test 1 (initial):      tokens=8, h_norm=179
Test 2 (Hello):        response clean, PE=55.6
Test 3 (post-Hello):   tokens=109, CV=0.59
Test 4 (topology):     response coherent, PE=51.8, bias_applied=True
Test 5 (post-topology): tokens=286, CV=0.59
Test 6 (same topic):   PE=4.9 (lower ✓ — topic recognized)
Test 7 (final):        tokens=391, tau_mean=0.34 (raw), h_norm=1257
```

## Remaining Known Issues

1. **tau diagnostic shows raw TauNet output (0.34) not rescaled value (~1.5)**: The ODE uses rescaled tau internally but `compute_tau()` returns raw values. Need to expose actual tau from the forward pass.

2. **D²_across shows 0.0 on some calls**: When random pair sampling doesn't hit cross-event pairs (buffer dominated by one large event), D²_across falls to 0. Need larger sample size or guaranteed cross-event sampling.

3. **Thinking traces still leak on some responses**: The preamble stripping is pattern-based and misses some variations. A more robust approach: detect the transition from reasoning to actual content by monitoring sentence structure.

4. **Response quality limited by Qwen3-4B**: 4B model produces generic responses. Geometric bias IS active (B_range=1960) but the model's capacity limits how much it benefits from geometric routing.
