# MAMBA STATE CAPTURE — The Natural Input for LiquidARC

## The Realization

Everything we've built (delta extraction, token buffers, sliding windows, h_state accumulation) is simulating what Mamba-2 layers compute natively. We're using an attention model (Qwen3-4B) and manually extracting sequential state updates. Mamba-2 IS a sequential state model — its hidden state is exactly the "meaning trajectory" the ODE is designed to process.

## What Mamba State IS

Each Mamba-2 layer maintains a hidden state s_t that updates recurrently:
```
s_t = A·s_{t-1} + B·x_t    (selective state space model)
y_t = C·s_t                  (output projection)
```

After processing token t, s_t contains the compressed history of tokens 1..t. This is:
- Fixed size regardless of context length
- Natively sequential (not derived from attention)
- Already compressed (no buffer management needed)
- The actual computational state the model uses (not a derivative)

## What This Replaces

| Current (simulated) | With Mamba states |
|---|---|
| Qwen3 forward → Δh extraction | Direct state capture from Mamba layer |
| Token buffer [1, 512, d] | Mamba state [n_heads, state_dim] per layer — fixed size |
| Sliding window over recent turns | State carries full history natively |
| h_state as artificial working memory | Mamba state IS working memory |
| Context prompt scaling problem | No scaling — state is constant size |
| Buffer pollution from curriculum | No buffer to pollute |

## Architecture with Nemotron-Nano-30B

Nemotron has 23 Mamba-2 + 23 MoE + 6 GQA layers.

```
INBOUND (Mamba → ODE):
  Text → Nemotron processes tokens through 23 Mamba-2 layers
       → capture SSM state from selected layer(s) (e.g., layer 12, 18, 23)
       → state enters LiquidARC ODE as input
       → ODE discovers geometric structure via MetricNet + heat kernel

OUTBOUND (ODE → GQA attention):
  ODE state → MetricNet → g(h) → D² → B_ij = -D²/4t
            → inject as attention bias into Nemotron's 6 GQA layers
            → GQA layers generate with geometric routing
  
Mamba layers: unchanged, do sequential processing
MoE layers: unchanged, do token transformation
GQA layers: native attention + geometric bias from ODE
```

LiquidARC sits BETWEEN Mamba (sequential state) and GQA (attention). It receives the sequential trajectory and outputs geometric routing. This is exactly the meta-controller role the reviewer described.

## Why Mamba States Are the Natural ODE Input

1. **Type match**: Mamba states are continuous sequential states. The ODE processes continuous dynamics. Both encode information as trajectories, not as static tokens.

2. **Compression match**: Mamba state is already compressed (fixed-size per layer). The ODE doesn't need to compress — it receives pre-compressed state and discovers geometric structure within it.

3. **Information content**: Mamba state after token t encodes HOW the model's understanding changed across all t tokens. This is the "meaning transformation" signal the reviewer identified as the correct ODE input.

4. **No distribution mismatch**: Mamba states live in the model's native representation space. They're not converted, projected, or adapted — they're the model's own internal state. The MetricNet needs to learn to route THESE states, but they're at least in a natural distribution (unlike random embeddings or extracted deltas).

5. **The phase transition connection**: The reviewer predicted Mamba states would have natural clustering — "most tokens cause small state updates (common words), rare tokens cause large updates (topic shifts)." This heavy-tailed distribution is what the MetricNet needs for the phase transition (D² in the critical regime).

## Engineering: State Capture Options

### Option 1: vLLM KV Connector Plugin (Recommended)
vLLM's KV Connector interface was designed for exporting model internal state. Write a plugin that intercepts Mamba layer state after each forward pass and exports it via shared memory or IPC.

Pros: Nemotron stays in vLLM (optimized, fast generation). State capture is non-invasive.
Cons: KV Connector API may not directly support SSM states (designed for KV caches). May need custom extension.

### Option 2: In-Process Quantized Nemotron
Load Nemotron at INT4 (~8GB) with hooks on Mamba layers. Full access to states. Use same instance for generation (slower than vLLM but simpler).

Pros: Full control over state access. Single instance.
Cons: INT4 quality loss. Slower generation than vLLM. Still 8GB GPU for the model.

### Option 3: Mamba-Only Forward Pass
Extract just the Mamba-2 layers from Nemotron (~30% of params). Run these in-process for state capture. Use vLLM for full-model generation.

Pros: Small memory footprint for state capture. vLLM handles generation.
Cons: Mamba layers without MoE/GQA may produce different states than full model. Significant engineering to extract and run subset of layers.

### Option 4: Two-Phase Processing
Phase 1: Send input text to vLLM Nemotron. Generation completes.
Phase 2: Send same text to in-process Qwen3-4B (current setup) for delta extraction.
Use deltas as PROXY for Mamba states until proper state capture is implemented.

Pros: Works today with current code. No new engineering.
Cons: Approximation, not real Mamba states. All current buffer/window issues remain.

## Recommendation

Option 1 (vLLM KV Connector) is the right long-term path. Option 4 (current Qwen3 deltas) is the bridge.

The current delta+bias architecture VALIDATED the mechanism. The Mamba state capture makes it NATIVE. The transition is:
1. Current: Qwen3 Δh → buffer/window → ODE → bias → Qwen3 generation
2. Bridge: Qwen3 Δh → h_state + window → ODE → bias → Qwen3 generation (current spec)
3. Target: Nemotron Mamba state → ODE → bias → Nemotron GQA generation

Each step the architecture simplifies. Step 3 eliminates: delta extraction, buffers, windows, h_state, Qwen3 as a separate model. What remains: Mamba state → ODE → attention bias → GQA. Clean.

## What Transfers From Current Work

- Sustained criticality system (D²/4τ loss, tau_quality, convergence coupling)
- State cosine + displacement bias computation
- Attention bias injection via forward hooks
- Bias normalization for softmax compatibility
- PE as novelty signal (state displacement)
- The MetricNet + TauNet architecture
- All geometric diagnostics (CV, D²/4τ, B_across, entropy)

What doesn't transfer: token buffer, event_id tracking, dropping strategy, sliding window, delta extraction, bootstrap tokens. All eliminated by native state capture.

## The Connection to the Full Research Arc

FGN → LiquidARC → Mind v1 (prefix, failed) → Mind v2 (delta+bias, working) → Mamba state capture (natural)

Each step discovered what the ODE actually needs:
- FGN: learned Riemannian metric + heat kernel routing
- LiquidARC: phase transitions as computational reorganization
- Mind v1: the ODE can't output tokens (distribution mismatch)
- Mind v2: the ODE should output routing (attention bias)
- Now: the ODE should receive sequential state (Mamba), not static tokens

The ODE sits between sequential processing (Mamba) and parallel processing (attention). It translates temporal structure into geometric structure. That's its computational role.
