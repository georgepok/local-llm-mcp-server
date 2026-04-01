# Analysis: FluidGeometryLogitsProcessor (Geometric Engine v3.1)

**Source file:** `/workspace/fluid_geometry.py` (host: `/home/pokazge/models/fluid_geometry.py`)
**Confirmed running:** vLLM log line — `[GeometricEngine] v3 initialized. think_start=12, think_end=13, warmup=15 tokens, tau=15.0`
**vLLM version:** 0.13.0+faa43dbf (NV26.01 image)
**Registration:** `--logits-processors fluid_geometry:FluidGeometryLogitsProcessor`

---

## 1. What It Does

The `FluidGeometryLogitsProcessor` is a per-request self-calibrating entropy-curvature processor that modifies the token probability distribution at every generation step. It does not change model weights. It operates entirely in logit space, shaping the output distribution based on the model's own moment-to-moment uncertainty dynamics.

The core insight: the Shannon entropy of the logit distribution at each token is a real-time signal of the model's confidence. The second derivative of that entropy — called scalar curvature κ(t) — captures whether confidence is accelerating or decelerating. Both signals are exploited to drive two structural interventions.

---

## 2. Architecture

The implementation follows a four-class hierarchy:

```
FluidGeometryLogitsProcessor          (server-lifetime, stateless)
  └── GeometricRequestProcessor       (per-request, created by new_req_logits_processor())
        ├── Accumulator                (phase space tracker: entropy, curvature, derivatives)
        ├── RequestCalibration         (warmup state and calibrated thresholds)
        └── StabilityMonitor           (rolling perplexity guard)
```

### FluidGeometryLogitsProcessor

- Implements vLLM v1 `AdapterLogitsProcessor` interface
- Instantiated once at server startup
- Holds only two values: `think_start_id=12` and `think_end_id=13` (resolved from tokenizer at init)
- `new_req_logits_processor(params)` is called by vLLM for each new inference request and returns a fresh `GeometricRequestProcessor`
- `is_argmax_invariant()` returns `False` — correctly signals that it can change the selected token

### GeometricRequestProcessor

- Implements vLLM `RequestLogitsProcessor` (callable)
- Signature: `(prompt_token_ids, output_token_ids, logits) -> logits`
- Called by vLLM for every generated token, before sampling
- Owns a fresh `Accumulator`, `RequestCalibration`, and `StabilityMonitor` per request
- No shared state between requests

### Accumulator

Computes and tracks the phase space state at each token:

1. Computes exact Shannon entropy from the full logit vector (all 131,072 vocab entries)
2. Extracts top-2 probability mass and bimodality ratio for decision-point detection
3. Applies EMA smoothing to entropy (alpha=0.05, ~20-token window) before differentiation
4. Computes first derivative ΔH and second derivative Δ²H from the smoothed buffer
5. Computes scalar curvature: `κ(t) = Δ²H(t) / (1 + |ΔH(t)|)` — normalizes acceleration by speed
6. Applies separate EMA smoothing to curvature (alpha=0.15, ~7-token window)
7. Returns a `PhaseState` dataclass with all computed values

### RequestCalibration

Manages the warmup phase and stores request-specific calibration constants:

- During the first 15 tokens: collects raw (pre-EMA) entropy and |κ| samples
- After 15 tokens, computes:
  - `kappa_ref` = mean(|κ|) + σ(|κ|), floored at 0.001 — the "typical curvature" for this request
  - `H_p90` = 90th entropy percentile — think-entry threshold
  - `H_p25` = 25th entropy percentile — think-exit threshold
  - `H_mean` = mean entropy — used for entropy-regime scaling
- Provides `confidence()`: C(t) = (1 - exp(-t / 15)) × confidence_override
  - Reaches ~63% by token 15, ~95% by token 45
  - `confidence_override` is a multiplier the stability monitor can reduce (0.98 pullback per alarm)

---

## 3. Structural Laws (The Two Interventions)

### Law 1: Temperature Modulation

Applied every token after warmup:

```
T_target = T_BASE × (1 + C_eff × effective_scale × κ(t) / κ_ref)
```

Where:
- `T_BASE = 1.0`
- `C_eff` = current confidence (ramps from 0 to 1 over ~45 tokens)
- `effective_scale = T_RESPONSE_SCALE × entropy_scale × bimodal_factor`
  - `T_RESPONSE_SCALE = 0.5` (moderate base response)
  - `entropy_scale = min(1.0, H_mean / 1.0)` — reduces intervention in low-entropy regimes
  - `bimodal_factor` — see bimodal detection below
- `κ / κ_ref` is clamped to [-5.0, +5.0]

Rate limiting: temperature can change at most ±0.25 per token.
Hard clamp: T ∈ [0.7, 1.5].

Effect: when curvature is positive (entropy accelerating — model getting confused), temperature rises, widening the distribution. When curvature is negative (entropy decelerating — model getting confident), temperature falls, sharpening the distribution. The response is proportional to how unusual the current curvature is relative to this request's own baseline.

### Law 2: Think Token Bias

Applied every token after warmup, conditioned on current thinking state:

**Think entry** (when not currently in `<think>` block):
- Condition: H > H_p90 AND κ > 0 (above this request's 90th entropy percentile AND getting more confused)
- Adds logit bias to token ID 12 (`<think>`)
- Bias = C_eff × min(|κ| / κ_ref × 3.0, 15.0)
- Peak bias: up to 15 logits (makes `<think>` selection nearly certain when condition is strongly met)

**Think exit** (when currently inside `<think>` block):
- Condition: H < H_p25 AND κ < 0 (below this request's 25th entropy percentile AND getting more confident)
- Adds logit bias to token ID 13 (`</think>`)
- Same strength formula

The normalized curvature (κ / κ_ref) is critical here. Raw smoothed κ values are typically 0.001–0.01, which would produce invisible biases (~0.06 logits). Normalization by κ_ref can yield values up to ±5.0, producing biases up to 15 logits — sufficient to meaningfully influence sampling.

---

## 4. Bimodal Detection (v3.1 Addition)

At each token, the accumulator computes:
- `top2_mass` = p1 + p2 (combined probability of top two tokens)
- `bimodality` = min(p1, p2) / max(p1, p2) — approaches 1.0 when the two tokens are equally likely

If `bimodality > 0.5 AND top2_mass > 0.3`, the `bimodal_factor` in Law 1 is reduced:
```
bimodal_factor = 1.0 - 0.7 × bimodality   (up to 70% reduction)
```

This prevents the temperature intervention from distorting a genuine binary choice — cases where the model is deciding between two coherent options (e.g., pronoun selection, punctuation choice, branching logic paths). At full bimodality (1.0), the temperature response is attenuated by 70%.

---

## 5. Stability Monitor

Tracks rolling perplexity over a 50-token window using the log-probability of the selected token as a proxy. After the window fills:

- Maintains running mean and variance of rolling perplexity via EMA (rate=0.05 per step)
- If `rolling_ppl > running_mean + 2σ`: multiplies `confidence_override` by 0.98 (pullback)
- Otherwise: multiplies `confidence_override` by 1.02 (recovery), capped at 1.0

The pullback affects both structural laws simultaneously through the confidence term. The design is symmetric — the engine self-heals if it degrades generation quality, and restores intervention strength once quality recovers.

---

## 6. Trace Logging

Disabled by default. Enable by setting `FG_TRACE=1` in the container environment (requires container restart with `-e FG_TRACE=1`). Writes to `/workspace/engine_state/trace.log`, self-truncating at 10MB. Each line contains: `req_t`, `H`, `H_raw`, `dH`, `kappa`, `T_applied`, `confidence`, `confidence_override`, `kappa_ref`, `bimodality`, `top2_mass`, warmup status.

---

## 7. Confirmed Operating Parameters (from vLLM init log)

| Parameter | Value |
|---|---|
| think_start_id | 12 |
| think_end_id | 13 |
| Warmup tokens | 15 |
| Confidence τ | 15.0 |
| T_BASE | 1.0 |
| T floor / ceiling | 0.7 / 1.5 |
| Max T delta/token | 0.25 |
| H smoothing alpha | 0.05 |
| κ smoothing alpha | 0.15 |
| Perplexity window | 50 tokens |
| Stability sigma | 2.0 |

---

## 8. Assessment for Approach C (Behavioral Modification Layer)

### Strengths

**Already deployed and proven stable.** The engine has been running in the production vLLM container since the server was initialized (2026-03-10T19:15:51Z). It is not experimental infrastructure — it is the live inference path for every Nemotron generation.

**Per-request isolation with no cross-request contamination.** Each `GeometricRequestProcessor` starts with a blank `RequestCalibration`. A modification to one request's behavior cannot bleed into the next. This is critical for controlled experiments.

**Modifies behavior without touching weights.** All interventions happen in logit space. The model's checkpoint is never written. Rollback is trivial — removing the processor flag from the vLLM command reverts to unmodified behavior. No weight corruption risk.

**Automatic safety through the stability monitor.** If an experimental structural law degrades generation quality, the 2σ perplexity alarm scales down confidence, automatically reducing intervention strength. The engine self-regulates without external oversight.

**Observable signal for the model's internal state.** The processor computes Shannon entropy and curvature at every token from the full vocabulary distribution (131,072 entries). This is a real-time observation point for what the model is "feeling" — the closest approximation to internal state available without modifying the forward pass.

**Extensible.** The `StructuralLaws` class is a pure static methods class. New laws can be added without touching the accumulator or calibration logic. The `PhaseState` dataclass exposes sufficient signal for a wide range of additional interventions. Custom token biases for any token ID are possible with the same pattern as the think-token bias law.

**Hot-swappable.** Updating `fluid_geometry.py` on the host and restarting the container applies changes in ~4 minutes (3:49 to load safetensors + ~10 seconds for startup, with torch.compile cache hit). The compile cache at `/root/.cache/vllm/torch_compile_cache/14be57b90a/` persists across container restarts, eliminating the 98-second recompilation cost on subsequent restarts.

**Approach for self-modification experiments:** The model could be prompted to reason about its own current entropy/curvature dynamics (via a system prompt that explains what the engine computes), propose modifications to the structural law parameters, and those modifications could be applied by updating the processor constants — without any weight modification or container restart.

### Limitations

**Operates only on logits, not on hidden states.** The processor sees the output distribution but has no access to the 52-layer hidden state trajectory that produced it. It can shape what the model selects but cannot observe or redirect intermediate computations within the forward pass.

**Cannot install new capabilities.** If Nemotron lacks the ability to perform a task, logit-space intervention cannot create that capability. Temperature adjustments and token biases can only modulate selection from existing knowledge.

**Temperature and bias are coarse instruments.** Temperature affects the entire vocabulary distribution uniformly. Token-specific bias affects one token at a time. Neither can implement the equivalent of targeted attention reweighting or expert routing changes.

**The model has no self-awareness of the intervention.** The processor modifies the distribution after computation. Nemotron does not "know" the engine exists — there is no signal propagated back into the forward pass. Self-modification via this pathway requires prompting Nemotron to reason about the engine as an external system (which it can do, given accurate documentation), not as part of its own cognition.

**Confidence ramp limits early-request intervention.** For the first ~15 tokens, the engine is in warmup and takes no action. For tokens 15–45, confidence is 63–95%. Short responses may not receive full-strength intervention.

### Recommendation

Approach C is the correct starting point for Phase 2 behavioral experiments. The primary reason is not merely convenience — it is that this approach tests a specific and interesting hypothesis: can a model reason about logit-space intervention parameters and propose meaningful changes to a running behavioral modifier?

This is distinct from "the model modifies its weights." It is closer to "the model adjusts the parameters of an adaptive controller that governs how it generates output." That is a real and tractable form of behavioral self-modification with an automatic safety layer already in place.

Concrete Phase 2 experiment design enabled by this infrastructure:

1. Provide Nemotron with a system prompt that accurately describes the engine's structure and current parameter values
2. Present a task where the engine's current behavior is suboptimal (e.g., the think-entry threshold is calibrated to the wrong entropy level for this task type)
3. Ask Nemotron to diagnose the issue and propose new parameter values
4. Apply the proposed values by updating the processor constants
5. Measure whether the proposed values improve task performance vs. the original values

The stability monitor provides a natural safety bound: even if Nemotron proposes parameters that degrade quality, the 2σ perplexity alarm will automatically reduce intervention strength, preventing runaway degradation.

True weight modification (Approach A or B) should follow Phase 2 if Approach C demonstrates that Nemotron can reason productively about its own behavioral control systems. The vllm restart cost of ~4 minutes and the 30.5GB model checkpoint make Approach A feasible but slower to iterate on.

---

*Analysis complete. Source verified against `fluid_geometry.py` and `deployment_config.json`.*
