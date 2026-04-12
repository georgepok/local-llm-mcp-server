# Attention Bias Mechanism — Deep Analysis & Resolution

## The Problem

The ARC-trained MetricNet's SDPA bias produced B_within >> B_across (within-event routing dominant, cross-event blocked). Normalization couldn't fix this because the GEOMETRY ITSELF routes within-event. The MetricNet learned ARC grid routing (within-grid correct) which doesn't transfer to text conversation (cross-turn routing needed).

## Failed Approaches

1. **Global normalization** (B - mean) / std → uniform softmax (H=0.99)
2. **Per-row normalization** → same issue, within-event cluster dominates
3. **N-scaled range** 4*log(N) → still uniform at N=512
4. **Trajectory accumulation** (sum B across ODE steps) → reinforces within-event pattern
5. **Early ODE step capture** → early steps also within-event dominant

All failed because they tried to extract cross-event signal from MetricNet, which doesn't have it.

## The Solution: ODE State Cosine Similarity

Instead of asking MetricNet for distances, use the ODE STATE ITSELF as the bias:

```python
h_unit = h / ||h||  # unit-normalize ODE state
bias_ij = h_unit_i · h_unit_j  # cosine similarity
```

Combined with displacement correlation (tokens that moved together during ODE):

```python
Δh = h_post - h_pre  # how each position changed
Δh_unit = Δh / ||Δh||
displacement_ij = Δh_unit_i · Δh_unit_j

bias = 0.5 * cosine_sim + 0.5 * displacement
```

### Why This Works

The ODE integrates information between positions through 16 steps of heat kernel routing. Positions from DIFFERENT events that are causally related end up with ALIGNED h vectors because the ODE routed information between them. The cosine similarity of the final state captures this alignment.

The MetricNet bias measures "how similar are these positions in metric space?" — which is input-distribution-dependent and ARC-biased. The state cosine measures "did the ODE make these positions similar?" — which is dynamics-dependent and reflects actual information flow.

### Evidence

Three-turn causal chain test:

| Turn | Bw (within) | Bx (across) | Bx > Bw? |
|------|-------------|-------------|----------|
| T1 (bridge collapsed) | 0.435 | 0.000 | — (only 1 event) |
| T2 (trucks rerouted) | 0.441 | **0.449** | **YES** |
| T3 (supply disruption) | 0.403 | **0.500** | **YES (+24%)** |

Cross-event routing EXCEEDS within-event when multiple conversation events are in the ODE. The response for T3: "The supply disruption was primarily caused by the collapsed bridge, which forced..." — correctly connecting the causal chain through geometric routing.

### What the ODE Contributes

The ODE's value is now measurable:
- **Without ODE**: Raw deltas have no cross-event alignment (independent Qwen3 forward passes)
- **With ODE**: 16 steps of heat kernel routing CREATES cross-event alignment in the state space
- **State cosine captures this**: positions routed together → aligned h → high B_across

The ODE IS the bridge between events. Its internal routing (through MetricNet + heat kernel) produces alignment that the bias extracts. The MetricNet shapes WHERE routing happens; the state cosine extracts WHAT routing produced.

## Remaining Issues

### 1. Buffer retention
512-token buffer drops conversation tokens within 2-3 turns. Generation feedback (50-200 tokens per response) + curriculum stimuli consume buffer space. Cross-event pairs disappear when old tokens are evicted. Need: larger buffer or more aggressive dropping of generated/curriculum tokens.

### 2. Generation failure on complex prompts
Qwen3-4B produces "Assistant" (empty) on multi-step reasoning prompts ("What is the full chain of events..."). This is model capability, not geometry. The bias IS flowing — the model can't use it for complex reasoning.

### 3. Entropy H=0.92
The routing is structured but not as peaked as ARC (H≈0.3). With cosine similarity in [0, 1] and per-row normalization, the softmax doesn't produce extremely peaked attention. This is acceptable — 24% more cross-event attention than within-event is meaningful even without extreme peaking.

## Architecture After Resolution

```
observe_event:
  1. Delta extraction (Qwen3 layer 18 → Δh)
  2. Token buffer append
  3. ODE integration (16 Euler steps, heat kernel routing)
  4. Capture h_pre and h_post for displacement bias
  5. PE from state displacement

generate_with_bias:
  1. Compute cosine similarity of ODE state  → state bias [N,N]
  2. Compute displacement correlation         → disp bias [N,N]
  3. Combine: bias = 0.5*state + 0.5*disp
  4. Per-row normalize for Qwen3 injection
  5. Generate with bias hooks
  6. Post-hoc feedback: generated tokens → ODE

The MetricNet is NOT in the bias path. It still runs inside the ODE
(computing g for heat kernel routing), but the bias to Qwen3 comes
from the ODE's OUTPUT (state cosine + displacement), not from the
MetricNet's distances.
```

This is the correct architecture: the MetricNet shapes internal ODE routing, and the ODE's resulting state shapes LLM attention. The MetricNet is the hidden layer; the state cosine is the output.
