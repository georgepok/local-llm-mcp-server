# Integration Time Sweep Report — Variable T on Combined Agentic Tasks

**Date:** 2026-03-28
**Checkpoint:** 5M model (d=768), step 10000 post-transition
**Question:** Is the 60-70% eval ceiling caused by insufficient ODE integration depth?

---

## Answer: NO — The ceiling is NOT reasoning-depth-limited

Average eval xform is flat (±2pp) across T=0.5 to T=3.0 — a 6× range of integration depth. The 60-70% ceiling is caused by something else: likely FFN capacity, data representation, or the task format itself.

---

## T Sweep Results (eval at step 11500, 1500 training steps)

| T | dt=T/16 | Stateful | Context | Dependency | **Average** |
|---|---------|----------|---------|------------|-------------|
| 0.5 | 0.031 | 62.7% | 56.7% | 37.2% | 52.2% |
| 0.75 | 0.047 | 65.2% | 53.2% | 36.0% | 51.5% |
| **1.0** | **0.063** | **68.3%** | 56.7% | 33.9% | 53.0% |
| 1.5 | 0.094 | 62.1% | 59.7% | 38.4% | 53.4% |
| 2.0 | 0.125 | 59.0% | **64.3%** | 37.7% | **53.7%** |
| 3.0 | 0.188 | 62.1% | 58.5% | **38.9%** | 53.2% |

### Per-Domain Observations

**Stateful execution:** Peaks at T=1.0 (68.3%), declines at higher T. The model was already near-optimal for state tracking depth. Deeper integration doesn't help and slightly hurts (over-contraction erases useful state information).

**Context relevance:** Improves with T. Peaks at T=2.0 (64.3%), a +8pp gain over T=0.5. Content-based filtering benefits from deeper diffusion — more ODE steps allow the query to propagate further through the context items. This is the only domain where T clearly matters.

**Dependency ordering:** Slight improvement at higher T (33.9% → 38.9%, +5pp from T=1.0 to T=3.0). Graph traversal marginally benefits from deeper integration but the effect is small.

**Average:** Essentially flat at 52-54%. No T value breaks the ceiling.

## Assessment

### The ceiling is NOT reasoning depth

If the ceiling were caused by 16-step ODE depth being insufficient:
- Higher T (which gives each step more dynamical range) should lift all domains
- The lift should be proportional to task difficulty (dependency should benefit most)
- There should be a clear inflection point where "enough depth" kicks in

None of these hold. The ceiling is stable across 6× integration range.

### What the ceiling likely IS

1. **FFN capacity bottleneck**: The single shared ContinuousDynamics module (applied 16×) has limited representational capacity. Adding more integration time doesn't add new parameters — it just applies the same transformation more times.

2. **Task format ceiling**: The grid-based token format may limit expressiveness. Some tasks (dependency ordering) require relational reasoning that 10-color, (x,y)-coordinate tokens can't efficiently encode.

3. **Training data diversity**: With infinite procedural data but fixed task structure, the model memorizes surface patterns (the "fast acquisition" we see in the first 200 steps) and plateaus once those are exhausted.

### Interesting finding: context benefits from deeper T

The context relevance task uniquely benefits from T=2.0 (+8pp over baseline). This makes sense: the query token needs to "search" through all context rows to find matching categories. More integration time = more diffusion steps = the query's influence can spread further through the sequence. This is the heat kernel doing what it was designed for — spatial information propagation.

The other tasks don't benefit because their bottleneck isn't propagation distance but rather the computation applied AT each position (what to do with the information once it arrives).

### Implication for architecture

Increasing ODE steps or integration time is NOT the path to breaking the ceiling. Instead:
- **More FFN capacity** (wider or deeper FFN within the dynamics module)
- **Multiple dynamics modules** (different weights at different ODE steps, not weight-tied)
- **Explicit memory/scratchpad** for multi-step state tracking
- **Better task encoding** that gives the model more to work with

The T=1.0 default is already near-optimal for most tasks. T=2.0 helps context filtering specifically. A learnable T would likely converge to ~1.0-1.5.
