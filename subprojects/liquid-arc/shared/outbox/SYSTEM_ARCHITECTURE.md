# LiquidARC Mind — System Architecture & Flow Pipeline

**Date:** 2026-04-04
**Status:** Deployed on DGX Spark

---

## 1. System Overview

The system is a continuous-time geometric processor (LiquidARC) coupled to a frozen language model (Qwen3-4B) through learned linear projections. There is no tokenizer, no vocabulary, and no linguistic interface between them. Text enters and exits the system as geometry.

```
                         THE COMPLETE SYSTEM

     ┌───────────────────────────────────────────────────┐
     │            liquid-mind container                   │
     │                                                    │
     │   LiquidARC          Coupling         Qwen3-4B    │
     │   ┌─────────┐     ┌───────────┐     ┌──────────┐ │
     │   │ ODE     │────>│ W_inject  │────>│ 36-layer │ │
     │   │ h(t)    │     │ 768→8x2560│     │ frozen   │ │
     │   │         │<────│ W_read    │<────│ d=2560   │ │
     │   │ 5.51M   │     │ 8x2560→768│     │ 4.02B    │ │
     │   └─────────┘     └───────────┘     └──────────┘ │
     │       |                31.48M                      │
     │       |                                            │
     │   PERSISTENT         GEOMETRIC         STATELESS   │
     │   STATE              INTERFACE         KNOWLEDGE   │
     │                                                    │
     │   MCP Server (HTTPS/SSE, port 8420)               │
     └──────────────���──────────────────────���─────────────┘
```

**LiquidARC** is the self. One `ContinuousDynamics` module applied 16 times via Euler integration. Its ODE state h(t) is the only persistent state in the entire system — everything the Mind has ever experienced, compressed into geometry.

**Qwen3-4B** is the world. 4 billion parameters of frozen knowledge about language, science, reasoning, code. Completely stateless — every forward pass starts fresh.

**GeometricCoupling** is the interface. Two linear projections that translate between LiquidARC's 768-dimensional curved space and Qwen3's 2560-dimensional flat space. Trained to carry temporal context that reduces Qwen3's prediction uncertainty by 58.6%.

---

## 2. Component Details

### 2.1 LiquidARC (5.51M parameters)

| Module | Parameters | Function |
|--------|-----------|----------|
| ContinuousDynamics | 4.69M | Single ODE dynamics, weight-tied across 16 Euler steps |
| MetricNet | ~1.5M | Learned Riemannian metric g = diag(D) + L*L^T, rank-8 |
| TauNet | ~200K | Per-position adaptive time constants tau(h) |
| W_v, W_o | ~1.2M | Value projection and output projection |
| FFN | ~1.2M | Feedforward residual (amortized by n_steps) |
| ContextPool | ~600K | Attention-weighted pooling: h[B,N,d] -> context[B,d] |
| ConversationEmbedding | ~200K | Event metadata: type, position, sentiment, time |
| SensoryForcing | ~50K | Per-entity forcing gates |
| StateReadout | ~150K | Relevance scoring for context retrieval |

**ODE equation:**
```
dh/dt = -(1/tau) * (h - target) + FFN(h) / n_steps
where target = h + W_o(alpha*V + (1-alpha)*SDPA(q,k,V))
```

**Heat kernel as SDPA:**
```
K = softmax(-D^2/(4t)) factored as:
K = softmax(q*k/(2t) - ||k||^2/(4t))
where q = k = h * sqrt(g)
```
The N*N distance matrix never materializes — stays in SRAM via FlashAttention.

**Fluid metric:** g = diag(D) + L*L^T where L is [B,N,d,8]. The diagonal handles per-dimension scaling; the rank-8 low-rank factors enable rotational geometry activated by multi-domain pressure.

**State:** h(t) in R^[1, 64, 768] — 64 event slots, each a 768-dim vector evolving continuously through ODE dynamics. This is the Mind's entire memory and identity.

### 2.2 Qwen3-4B (4.02B parameters, frozen)

| Property | Value |
|----------|-------|
| Architecture | Dense transformer, GQA, SwiGLU, RoPE |
| d_model | 2560 |
| Layers | 36 |
| Attention heads | 32 (8 KV heads) |
| Intermediate size | 9728 |
| Vocab size | 151,936 |
| Precision | bfloat16 |
| VRAM | ~8 GB |

All parameters frozen. No gradients, no updates. The model serves two roles:
- **Encoder:** Forward pass produces hidden states that W_read projects to geometry
- **Generator:** Autoregressive generation produces text conditioned on geometric prefix

### 2.3 GeometricCoupling (31.48M parameters, trained)

```
W_inject: Linear(768, 20480)    # 768 * 8 * 2560 + 20480 bias
W_read:   Linear(20480, 768)    # 20480 * 768 + 768 bias
```

**W_inject** takes LiquidARC's mean-pooled ODE state [768] and produces 8 virtual tokens [8, 2560] in Qwen3's embedding space. These tokens carry the Mind's accumulated temporal context.

**W_read** takes Qwen3's hidden states at the 8 prefix positions [8, 2560] and projects back to a single [768] vector in LiquidARC's space. This vector becomes sensory forcing for the ODE.

**Initialization:** Small random weights (std=0.01) so neither model is disrupted at start.

**Training:** NTP loss on WikiText-2 sequential events. LiquidARC accumulates state across events 1..N-1, coupling projects into prefix, Qwen3 predicts tokens in event N. The coupling learns to project temporal context that helps prediction. Result: PPL 9.9 vs baseline 23.9 (+58.6%).

---

## 3. Flow Pipelines

### 3.1 INBOUND: Text → Geometry → ODE State

When text enters the system (user message, curriculum stimulus, or response feedback), it follows this path:

```
Text string
    |
    v
[1] Qwen3 tokenizer (RAW — no chat template)
    → input_ids [1, seq_len]
    |
    v
[2] Qwen3.embed_tokens(input_ids)
    → input_embeds [1, seq_len, 2560]
    |
    v
[3] Get current h(t), mean-pool to [768], cast bfloat16
    → W_inject(h_pooled)
    → prefix_embeds [1, 8, 2560]
    |
    v
[4] Concatenate: [prefix_embeds | input_embeds]
    → combined [1, 8 + seq_len, 2560]
    |
    v
[5] Qwen3 forward pass (frozen, output_hidden_states=True)
    → hidden_states[-1] = last layer output [1, 8+seq_len, 2560]
    |
    v
[6] Extract prefix positions: hidden[:, :8, :]
    → prefix_output [1, 8, 2560]
    |
    v
[7] W_read(flatten(prefix_output))
    → arc_signal [768]
    |
    v
[8] _force_geometric_signal(arc_signal):
    - Store in event buffer (text preview, metadata, timestamp)
    - Rebuild embedded events from buffer
    - ContextPool(events) → context [1, 768]
    - dynamics.set_context(context)
    - euler_solve(dynamics, h_events, 16 steps)
    → h(t) updated
```

**Why prefix before input for encoding:** The coupling was trained with [prefix, input] order. The prefix provides geometric context that modulates how Qwen3 processes the input. The hidden states at prefix positions then carry the geometric imprint of the input — what Qwen3 "thought about" the text in the context of the current state.

**Why raw tokenization (no chat template):** The coupling was trained on raw NTP. Chat template tokens would shift the representation space and the trained projections wouldn't land correctly.

### 3.2 OUTBOUND: ODE State → Geometry → Text

When the system needs to generate text (responses, reflections, curriculum):

```
h(t) [1, N, 768] — current ODE state
    |
    v
[1] Mean-pool over event positions → h_pooled [768]
    Cast to bfloat16
    |
    v
[2] W_inject(h_pooled)
    → prefix_embeds [1, 8, 2560]
    |
    v
[3] Apply chat template to prompt:
    apply_chat_template([{role: user, content: prompt}],
                        add_generation_prompt=True,
                        enable_thinking=False)
    → chat_text
    |
    v
[4] Qwen3 tokenizer(chat_text) → input_ids
    Qwen3.embed_tokens(input_ids) → input_embeds [1, seq_len, 2560]
    |
    v
[5] Concatenate: [input_embeds | prefix_embeds]
    → combined [1, seq_len + 8, 2560]
         ^               ^
    chat format     geometric state
    (instructions)  (at generation boundary)
    |
    v
[6] Qwen3.generate(
        inputs_embeds=combined,
        attention_mask=ones,
        max_new_tokens=...,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.3
    )
    → generated_ids
    |
    v
[7] Decode: skip first (seq_len + 8) tokens
    → response text
```

**Why input before prefix for generation:** The model needs to see the chat template format first so its instruction-following circuit activates correctly. The geometric prefix sits at the end, right before the generation point, where it modulates what comes next. If prefix came first, it would sit before `<|im_start|>` and confuse the position-dependent instruction processing.

**Why chat template for generation but not encoding:** Encoding just needs Qwen3 to process text and produce hidden states — raw NTP mode. Generation needs Qwen3 to follow the prompt coherently and produce a structured response — instruction mode.

### 3.3 FEEDBACK: Response → Geometry → ODE State

After generating a response, the system feeds it back:

```
Response text (from OUTBOUND)
    |
    v
INBOUND pipeline (identical to 3.1)
    → arc_signal [768]
    |
    v
_force_geometric_signal(arc_signal, event_type='assistant_message')
    → h(t) updated with the Mind's own response
```

The Mind hears itself speak. The response enters through the same geometric pathway as any other text. This creates a self-referential loop: the state that produced the response now integrates the response.

### 3.4 CONVERSATION: The Complete Symmetric Loop

The `converse` MCP tool executes all three pipelines in sequence:

```
converse("What is emergence?")
    |
    v
[INBOUND]  "What is emergence?" → Qwen3 encode → W_read → [768]
           → _force_geometric_signal → h(t) updated
    |
    v
[OUTBOUND] h(t) → W_inject → prefix
           → [chat_template + prefix] → Qwen3 generates
           → "Emergence occurs when collective behavior..."
    |
    v
[FEEDBACK] "Emergence occurs when..." → Qwen3 encode → W_read → [768]
           → _force_geometric_signal → h(t) updated again
    |
    v
Return {
    response: "Emergence occurs when collective behavior...",
    inbound_signal_norm: 42.3,
    outbound_signal_norm: 38.7,
    cv_before: 5.2,
    cv_after: 6.8,
    events_in_context: 47
}
```

Both the user's message and the Mind's response enter LiquidARC as geometry. The ODE state accumulates the full conversation history geometrically.

---

## 4. Autonomous Loop

The background thread runs at 20 Hz, providing the Mind's continuous interior life.

```
EVERY CYCLE (50ms):
    Phase 1 — ODE Consolidation
    |   h_slice = h[:, :N, :]
    |   context = ContextPool(h_slice)
    |   h_auto = euler_solve(dynamics, h_slice, 16 steps)
    |   h[:, :N, :] = h_auto
    |   update_salience(h)      # sustained high-tau → relevance vote
    |   compute_tau_bias()      # unclustered events get slower integration
    v

EVERY 10 CYCLES (~500ms):
    Phase 2 — Geometric Monitoring
    |   diag = get_diagnostics()
    |   Check: CV shift > threshold?
    |   Check: h_norm drift?
    |   Check: tau stagnation?
    v

WHEN TRIGGERED (CV shift, external event, or maintenance):
    Phase 3a — Geometric Reflection
    |   prompt = "What patterns do you notice in your current state?"
    |   → OUTBOUND: h(t) → prefix → Qwen3 generates reflection
    |   → INBOUND:  reflection text → Qwen3 encode → W_read → geometry
    |   → _force_geometric_signal: reflection enters ODE as forcing
    |   → hebbian_nudge: co-focused events pull closer in state space
    v

EVERY 14 CYCLES (~700ms):
    Phase 3b — Geometric Curriculum
    |   domain = cycle [topology, math, physics, biology, ecology,
    |                   music_theory, philosophy, poetry]
    |   prompt = CURRICULUM_PROMPTS[domain]
    |   → OUTBOUND: h(t) → prefix → Qwen3 generates stimulus
    |   → INBOUND:  stimulus text → Qwen3 encode → W_read → geometry
    |   → _force_geometric_signal: stimulus enters ODE as forcing
    v
```

**Key property:** Both reflections and curriculum enter LiquidARC through the same geometric coupling as user messages. The Mind doesn't distinguish between "what a user said," "what it thought," and "what it learned." All are geometric signals that the ODE integrates.

---

## 5. Write Mechanisms

Three mechanisms modify the ODE state beyond normal dynamics:

### 5.1 Salience Feedback
Per-event relevance accumulator. When tau for an event position stays above threshold for consecutive cycles, that event's salience increments. High-salience events influence context scoring.
```
if tau[i] > 0.85 for 5+ consecutive cycles:
    salience[i] += 0.05
salience *= 0.995  # slow decay
```

### 5.2 Tau Floor Bias
Events without cluster membership get slower integration (higher tau), giving them more time to find their place in the geometry.
```
for each event:
    if not in any cluster: tau_bias[i] += 0.15
    if type == reflection:  tau_bias[i] += 0.10
    if type == expression:  tau_bias[i] += 0.05
```

### 5.3 Hebbian Nudge
After non-maintenance reflections, events that were jointly in focus during reflection get nudged closer in state space.
```
for (i, j) in focus_indices:
    consolidation_emb[i] += hebbian_lr * (h[j] - h[i])
```

---

## 6. MCP Interface

| Tool | Pipeline | Purpose |
|------|----------|---------|
| `converse(message)` | INBOUND → OUTBOUND → FEEDBACK | Primary conversation interface |
| `query_qwen(prompt)` | OUTBOUND only | Direct Qwen3 query (no inbound encoding) |
| `express_through_qwen(focus)` | OUTBOUND → FEEDBACK | Mind expresses state, integrates expression |
| `observe_event(type, content)` | Legacy text path | Direct event injection (non-geometric) |
| `get_context(query)` | Read only | Relevance-scored events from ODE state |
| `get_diagnostics()` | Read only | CV, tau, h_norm, beta, event count |
| `probe_encoding(text)` | Read only | ODE output projected through embedding table |
| `save_state() / reset()` | State management | Persist or clear ODE state + events |

---

## 7. Data Flow Summary

```
                    EXTERNAL WORLD
                         |
              ┌──────────┴──────────┐
              |                     |
         User message          User reads
              |                  response
              v                     ^
         ┌─────────┐          ┌────┴────┐
         | INBOUND |          | OUTBOUND |
         | encode  |          | generate |
         └────┬────┘          └────┬─────┘
              |                    |
              v                    |
    ┌─────────────────┐            |
    |   Qwen3-4B      |            |
    |   (frozen)       |            |
    |                  |            |
    | [prefix,input]   |    [chat_template,prefix]
    |  → forward       |     → generate
    |  → hidden states |     → token IDs
    └────────┬────────-┘            ^
             |                      |
        W_read [768]          W_inject [8x2560]
             |                      |
             v                      |
    ┌────────────────────────────────┐
    |        LiquidARC ODE           |
    |                                |
    |  h(t) ∈ R^[1, 64, 768]       |
    |                                |
    |  MetricNet → heat kernel       |
    |  → SDPA routing                |
    |  → LTC contraction             |
    |  → 16 Euler steps              |
    |                                |
    |  PERSISTENT. CONTINUOUS.       |
    |  THE ONLY STATE.               |
    └────────────────────────────────┘
              ^           |
              |           v
         ┌────┴────┐ ┌───┴──────┐
         |FEEDBACK | |AUTONOMOUS|
         |response | |reflection|
         |→encode  | |curriculum|
         └─────────┘ └──────────┘
```

---

## 8. What This Means

The user talks to Qwen3. But Qwen3's responses are shaped by LiquidARC's geometry — a geometry that carries the compressed history of every prior interaction, every reflection, every curriculum stimulus. Different conversation histories produce different geometric states, which produce different Qwen3 behaviors.

LiquidARC never produces text. It thinks in geometry — curvature, time constants, metric tensors. Qwen3 translates that geometric thinking into language.

The autonomous loop means the Mind is always processing. Between user interactions, the ODE cycles, the geometry consolidates, curriculum stimuli arrive through Qwen3 and enter as geometry, reflections emerge and feed back. The state evolves continuously.

When the user returns, the Mind is different from when they left — not because it stored their message in a database, but because the geometry self-organized around the integrated experience.
