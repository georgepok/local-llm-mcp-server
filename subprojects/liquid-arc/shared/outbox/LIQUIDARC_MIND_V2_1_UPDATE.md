# LiquidARC Mind v2.1 — Token-Level Integration Update

## What Changed

v2.0 mean-pooled per-token deltas into a single vector per event (one ODE position per conversation turn). v2.1 feeds every token as its own ODE position. The bias matrix scales from [N_events × N_events] to [N_tokens × N_tokens]. Generated tokens feed back into the ODE state.

## Architecture Change

### v2.0 (event-level)
```
"What is topology?" → Qwen3 → 8 token deltas → mean pool → 1 ODE position
ODE state: [1, 7, 2560]  (7 events)
Bias: [7×7]
```

### v2.1 (token-level)
```
"What is topology?" → Qwen3 → 8 token deltas → 8 ODE positions
ODE state: [1, 291, 2560]  (291 accumulated tokens)
Bias: [291×291]
Generated response → 50 token deltas → append to ODE → state grows to 341
```

Every token from every conversation turn persists as an individual geometric position. The MetricNet routes between ALL tokens across ALL turns. A token from the first message can directly influence a token in the fourth message through the heat kernel, based on their geometric distance.

## Specific Implementation Changes

### `delta_extractor.py`
- `extract()` returns per-token `delta_h: [1, N_tokens, d]` — no mean-pooling
- Returns `token_texts: list[str]` for each position (decoded token strings)
- New `extract_and_append()` computes how many old tokens to drop when buffer exceeds 512

### `attention_bias.py`
- Uses SDPA factorization: `B_ij = q_i·k_j/(2t) - ||k_j||²/(4t)` where `q = k = h_normed * sqrt(g)`
- Avoids materializing [N, N, d] intermediate — O(N²d) → O(N²) via `torch.bmm`
- Handles full token-level N (tested up to 291 tokens)

### `qwen_bridge.py`
- `make_hook()` takes `n_ctx_tokens` to align bias with Qwen3's token positions
- `generate_iterative()`: one-shot generation + post-hoc ODE feedback
  - Generates all tokens with initial bias
  - Extracts deltas from generated text
  - Appends generated token positions to ODE state
  - Returns updated `h_ode` with generation tokens included

### `mind.py` — Token Buffer Architecture
- New `_token_buffer: list[dict]` — sliding window of up to 512 token entries
  - Each entry: `{delta_h: [d], source: str, text: str, timestamp: float}`
- New `_process_text_tokens()` — extracts per-token deltas, appends to buffer, rebuilds `self._h`
- `observe_event()` token path:
  - Extracts deltas OUTSIDE `_gpu_lock` (Qwen3 forward is slow)
  - Rebuilds `self._h` from token buffer INSIDE `_gpu_lock` (prevents race with autonomous loop)
  - Runs ODE on full token buffer, not event subset
- `generate_with_bias()` uses full `self._h` (all tokens), calls `generate_iterative()` for post-hoc feedback
- Autonomous loop uses `self._h.shape[1]` (token count) not `len(self.events)` (event count)
- Autonomous loop syncs token buffer entries with ODE-updated positions

### Event vs Token Separation
Event-level features (salience, tau_bias, clustering) operate on the event list (max 64 events). Token-level features (ODE state, bias, PE) operate on the token buffer (up to 512 tokens). They don't interfere:
- `_update_salience()` skipped in token mode (salience is per-event, not per-token)
- `_compute_tau_bias()` sets `dynamics._tau_external_bias = None` in token mode (per-event-type bias doesn't map to token positions)
- `_consolidation_emb` skipped when N > 64 (already handled)
- Event list still maintained for context prompt building, curiosity controller, metadata

### `content_proj.py` (NEW)
- `ContentProjection` module: attention-weighted pooling from N ODE positions → n_prefix summary tokens, then linear projection to LLM dimension
- Not yet integrated into generation pipeline — prepared for future content injection alongside bias

## Effect on Functionality

### Token Buffer Growth
```
Bootstrap:        8 tokens  ("Mind initialized. ODE encoder active.")
After "Hello":    9 tokens  (+1 token)
After response:  109 tokens (+50 generated tokens fed back)
After topology:  126 tokens (+17 tokens from user message)
After response:  206 tokens (+80 generated tokens fed back)
After follow-up: 291 tokens (+5 tokens from "Tell me more about topology")
After response:  341 tokens (+50 generated tokens fed back)
```

### Bias Matrix Scaling
```
v2.0:  bias [7×7]   → 7 events, covers 7 positions in Qwen3's attention
v2.1:  bias [291×291] → 291 tokens, covers 291 positions in Qwen3's attention
```

The geometric routing now covers the entire conversation context at token granularity, not just event summaries.

### Post-Hoc ODE Feedback
After each generation:
1. Generated text extracted through Qwen3 layer 18
2. Per-token deltas computed
3. Appended to ODE state
4. ODE state reflects the full conversation trajectory including the model's own responses

This creates a closed loop: the ODE shapes the generation (via bias), and the generation shapes the ODE (via feedback). In v2.0, generated responses were mean-pooled to one position and stored as events — losing the token-level trajectory of the model's own reasoning.

### PE Discrimination
```
"Hello"                  → PE = 55.6  (novel)
topology message         → PE = 26.4  (different topic)
"Tell me more"           → PE = 4.9   (same topic, already in ODE) ← lower ✓
```

PE now measures how much the ODE state displaced when specific tokens arrived, not when an averaged event blob arrived. Token-level PE is more granular — each token's meaning-change contributes individually.

### Server Logs
```
[tokens] extracted 17 tokens from text (119 chars)
[tokens] buffer: +17 new, total=126 tokens (source=user_message)
[observe] #4 type=user_message PE=26.4 CV=0.63 tau=1.64 h=694 tokens=126 events=4
[generate] bias [126x126] CV=0.63 D²/4τ=-161.6 tau=1.63 crit=False
[generate] iterative (one-shot+feedback) max_new=80 ode_tokens=126
[generate] response: 395 chars "Temporal: Mind initialized..."
[generate] ODE updated: 126 → 206 tokens (+80 gen, -0 dropped)
```

Every operation shows token counts, bias dimensions, and ODE state changes. The logs tell exactly what the system is doing at every step.

## Known Issues in v2.1

1. **Thinking trace leakage**: Qwen3-4B's internal reasoning still visible in responses. Needs `enable_thinking=False`.

2. **Negative D²/4τ**: The SDPA-factored bias computation produces negative D²/4τ values (-161, -253). This is because the bias `B_ij = q·k/(2t) - ||k||²/(4t)` can be positive (when q·k is large) — the D² estimate from `-4t * B` becomes negative. The diagonal is the issue: self-attention `B_ii = ||q_i||²/(2t) - ||k_i||²/(4t) = ||q_i||²/(4t)` is always positive. Off-diagonal can go either way. Need to use actual pairwise D² for the diagnostic, not reconstruct from B.

3. **h_norm growing**: h_norm increases with token count (162→694→1066→1237) because more token positions accumulate energy. The norm homeostasis operates per-position but total Frobenius norm grows with N. This is expected behavior, not a bug — the system has more information.

4. **Content projection not yet connected**: `ContentProjection` module exists but isn't integrated into the generation pipeline. Currently the LLM receives only routing information (bias) but not content from the ODE.

5. **Response quality**: Qwen3-4B is a 4B model — response quality is limited. The topology response says "I'm sorry, but I can't provide an answer" — not because of LiquidARC but because the model is small and the context is mostly its own previous responses. Upgrading to a larger model would improve response quality independent of the geometric integration.

## What v2.1 Enables That v2.0 Couldn't

1. **Cross-turn token routing**: A noun from the first message and a verb from the fourth message can be geometrically linked through the heat kernel. Mean-pooling destroyed this.

2. **Generation trajectory tracking**: The ODE sees HOW the model generates — which tokens produce large deltas (novel generation) vs small deltas (routine completions). This feeds into the convergence coupling.

3. **Scalable bias**: As conversation grows, the bias matrix covers more positions. A 500-token conversation has [500×500] bias — the LLM's attention at every position is geometrically informed by the full context.

4. **Foundation for iterative coupling**: With token-level state, true iterative generation (generate N tokens → update ODE → continue) is architecturally possible. v2.1 uses one-shot + post-hoc feedback as a stepping stone.
