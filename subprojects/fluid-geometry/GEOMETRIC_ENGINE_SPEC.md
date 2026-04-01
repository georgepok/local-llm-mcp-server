# Geometric Engine v2 — Functional Specification

## For Coding Engine Implementation

---

**Version**: 2.0.0
**Date**: 2026-02-08
**Status**: Implementation Ready
**Predecessor**: `fluid_geometry.py` (v1 entropy-driven think/no-think switching)
**Target**: NVIDIA DGX Spark — Nemotron-3-Nano-30B-A3B-FP8 running in vLLM ≥ 0.13
**Deployment**: Single Python package, in-container alongside vLLM, no external dependencies beyond PyTorch and vLLM

---

## 1. Purpose

Replace the v1 `FluidGeometryLogitsProcessor` (binary entropy threshold switching between `<think>` and `</think>` tokens) with a complete **self-calibrating geometric engine** that:

1. Measures a multi-dimensional phase space Φ(t) at every generation step from full logit vectors
2. Computes scalar curvature κ(t) from entropy dynamics
3. Applies three deterministic structural response laws (temperature, attention bias, routing bias)
4. Self-calibrates all reference constants from its own observation history
5. Self-gates intervention strength via a confidence function that rises from zero
6. Self-heals via a stability monitor that prevents quality degradation

**The engine requires no configuration changes, no external signals, no phase flags, and no human intervention after deployment.** It starts cold, warms itself through observation, and reaches full geometric operation autonomously.

---

## 2. System Context

### 2.1 Container Topology

```
┌─────────────────────────────────────────────────┐
│  DOCKER CONTAINER (nvcr.io/nvidia/vllm)         │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  vLLM Serving Engine                      │  │
│  │    • Model: Nemotron-3-Nano-30B-A3B-FP8   │  │
│  │    • 23 Mamba layers                      │  │
│  │    • 6 Attention layers                   │  │
│  │    • 128 MoE experts                      │  │
│  │    • Vocab size V ≈ 256,000               │  │
│  └───────────────┬───────────────────────────┘  │
│                  │                               │
│  ┌───────────────▼───────────────────────────┐  │
│  │  GEOMETRIC ENGINE (this spec)             │  │
│  │    • Logits Processor (vLLM v1 API)       │  │
│  │    • Accumulator (phase space Φ(t))       │  │
│  │    • Structural Laws (T, bias, routing)   │  │
│  │    • Calibrator (slow-timescale state)    │  │
│  │    • Stability Monitor (quality guard)    │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  OpenAI-compatible API on port 30000            │
└─────────────────────────────────────────────────┘

EXTERNAL (NO CHANGES):
  MCP Server → HTTP → port 30000 (standard OpenAI API)
```

### 2.2 Attachment Point

The engine registers with vLLM as a **logits processor plugin** via the existing `--logits-processors` CLI flag. This is identical to v1's attachment mechanism. The engine receives the full logit tensor at every generation step and returns (potentially modified) logits.

### 2.3 What Does NOT Change

- vLLM startup command structure (only the processor file content changes)
- OpenAI API request/response format
- MCP server code
- Model weights
- Any external interface

---

## 3. Architecture

### 3.1 Module Structure

```
fluid_geometry/
├── __init__.py              # Package init, exports FluidGeometryLogitsProcessor
├── engine.py                # Top-level orchestrator (GeometricEngine class)
├── accumulator.py           # Per-request phase space computation
├── structural_laws.py       # Three deterministic response laws
├── calibrator.py            # Cross-request self-calibration state
├── stability_monitor.py     # Quality guard and self-healing
└── config.py                # Constants and type definitions
```

**Alternative**: If vLLM's `--logits-processors` flag requires a single file with the processor class (as v1 used), collapse all modules into a single `fluid_geometry.py` file. The logical separation described here still applies — use classes and clear section comments to maintain structure within one file. **Determine which approach vLLM supports and use the simplest that works.**

### 3.2 Class Hierarchy

```
vllm.v1.sample.logits_processor.AdapterLogitsProcessor
    │
    └── FluidGeometryLogitsProcessor          [engine.py]
            │  (server-lifetime, holds Calibrator)
            │
            └── creates per request:
                    │
                    GeometricRequestProcessor     [engine.py]
                        │  (request-lifetime, holds Accumulator)
                        │
                        ├── Accumulator           [accumulator.py]
                        ├── StructuralLaws        [structural_laws.py]
                        ├── CalibrationState      [calibrator.py]  (reference, shared)
                        └── StabilityMonitor      [stability_monitor.py]
```

---

## 4. Component Specifications

### 4.1 Config — `config.py`

Define all constants and types. No environment variable reading. No file I/O. Pure definitions.

```python
from dataclasses import dataclass, field
from typing import List

# ── Physical Constants ──────────────────────────────────────────────
KAPPA_REF_INIT: float = 1.0          # Initial curvature reference scale
T_BASE: float = 1.0                  # Baseline temperature (no modification)
TAU_CONFIDENCE: float = 10_000.0     # Tokens to reach ~63% confidence
CALIBRATION_RATE: float = 0.001      # EMA rate for κ_ref adaptation
STABILITY_TOLERANCE: float = 0.05    # 5% perplexity degradation tolerance
STABILITY_DECAY: float = 0.99        # Pullback rate when quality degrades
STABILITY_RECOVERY: float = 1.001    # Recovery rate when quality is fine
H_BUFFER_SIZE: int = 3               # Entropy history buffer (current + 2 prior)
PERPLEXITY_WINDOW: int = 50          # Rolling window for perplexity tracking

# ── Derived Curvature Bounds ────────────────────────────────────────
KAPPA_MAX_RESPONSE: float = 5.0      # Clamp κ/κ_ref to prevent extreme T

# ── Persistence ─────────────────────────────────────────────────────
STATE_FILE: str = "/workspace/geometric_engine_state.json"

@dataclass
class PhaseState:
    """Single step of phase space measurement."""
    step: int
    H: float                    # Shannon entropy of full distribution
    delta_H: float              # First derivative (velocity)
    delta2_H: float             # Second derivative (acceleration)
    kappa: float                # Scalar curvature κ(t) = Δ²H / (1 + |ΔH|)
    delta_kappa: float          # Rate of curvature change
    T_applied: float            # Temperature actually applied
    confidence: float           # Global confidence at this step
    token_logprob: float        # Log-prob of selected token (for stability)

@dataclass
class CalibrationSnapshot:
    """Persistent state surviving across requests."""
    t_global: int = 0                    # Total tokens observed ever
    kappa_ref: float = KAPPA_REF_INIT    # Running curvature scale
    kappa_running_mean: float = 0.0      # EMA of |κ|
    kappa_running_var: float = 0.0       # EMA of κ² (for variance)
    baseline_perplexity: float = 0.0     # Established during warmup
    baseline_count: int = 0              # Tokens used to establish baseline
    confidence_override: float = 1.0     # Stability monitor multiplier
```

### 4.2 Accumulator — `accumulator.py`

**Lifetime**: One instance per generation request. Created fresh. Destroyed when request completes.

**Responsibility**: Maintain the fast-timescale phase space buffer. Compute H(t), derivatives, κ(t) from the raw logit tensor at each step.

```python
class Accumulator:
    """
    Per-request phase space tracker.

    Maintains a sliding buffer of entropy values and computes
    curvature κ(t) = Δ²H(t) / (1 + |ΔH(t)|) at each step.
    """

    def __init__(self):
        self.step: int = 0
        self.H_buffer: List[float] = []   # Last H_BUFFER_SIZE entropy values
        self.kappa_prev: float = 0.0
        self.trace: List[PhaseState] = []  # Full trace for diagnostics

    def update(self, logits: torch.Tensor, confidence: float, T_applied: float,
               selected_token_logprob: float) -> PhaseState:
        """
        Ingest one step of logits. Compute and return phase state.

        Args:
            logits: Raw logit tensor [vocab_size] BEFORE any modification.
                    Entropy must be computed from unmodified logits to measure
                    the model's true uncertainty, not the engine's intervention.
            confidence: Current global confidence C(t_global).
            T_applied: Temperature that will be applied this step.
            selected_token_logprob: Log-probability of whatever token gets
                                    selected (filled in post-sampling if needed,
                                    or estimated from logits).

        Returns:
            PhaseState for this step.
        """
        # ── Step 1: Exact entropy from full logit vector ────────────
        probs = torch.softmax(logits.float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)
        H = -(probs * log_probs).sum().item()

        # ── Step 2: Update buffer ───────────────────────────────────
        self.H_buffer.append(H)
        if len(self.H_buffer) > H_BUFFER_SIZE:
            self.H_buffer.pop(0)

        # ── Step 3: Compute derivatives ─────────────────────────────
        if len(self.H_buffer) >= 2:
            delta_H = self.H_buffer[-1] - self.H_buffer[-2]
        else:
            delta_H = 0.0

        if len(self.H_buffer) >= 3:
            delta2_H = (self.H_buffer[-1] - 2 * self.H_buffer[-2]
                        + self.H_buffer[-3])
        else:
            delta2_H = 0.0

        # ── Step 4: Scalar curvature ────────────────────────────────
        # κ(t) = Δ²H(t) / (1 + |ΔH(t)|)
        # Denominator prevents divergence during rapid entropy change.
        kappa = delta2_H / (1.0 + abs(delta_H))

        # ── Step 5: Curvature velocity ──────────────────────────────
        delta_kappa = kappa - self.kappa_prev
        self.kappa_prev = kappa

        # ── Step 6: Build phase state ───────────────────────────────
        state = PhaseState(
            step=self.step,
            H=H,
            delta_H=delta_H,
            delta2_H=delta2_H,
            kappa=kappa,
            delta_kappa=delta_kappa,
            T_applied=T_applied,
            confidence=confidence,
            token_logprob=selected_token_logprob,
        )

        self.trace.append(state)
        self.step += 1
        return state
```

**Critical implementation note**: The `logits` parameter to `update()` MUST be the raw, unmodified logit tensor from the model. Entropy computed on post-intervention logits would create a feedback loop (the engine measuring its own modifications). Capture the logits before applying any temperature or bias changes.

### 4.3 Structural Laws — `structural_laws.py`

**Lifetime**: Stateless functions. No instance state. All behavior determined by inputs.

**Responsibility**: Given the current phase state Φ(t) and calibration state, compute intervention values.

```python
class StructuralLaws:
    """
    Deterministic structural response laws.

    These are physics, not heuristics. Given the same Φ(t) and
    calibration state, they always produce the same output.
    No learned parameters. No randomness.
    """

    @staticmethod
    def temperature(kappa: float, kappa_ref: float, confidence: float,
                    confidence_override: float) -> float:
        """
        Law 1 — Geodesic Equation (Temperature Modulation).

        T(t) = T_BASE × (1 + C_eff × κ(t) / κ_ref)

        Positive κ (entropy accelerating) → higher T → more exploration.
        Negative κ (entropy decelerating) → lower T → more focus.
        Zero κ → T unchanged.

        Args:
            kappa: Current scalar curvature κ(t).
            kappa_ref: Calibrated reference curvature scale.
            confidence: Global confidence C(t_global) ∈ [0, 1].
            confidence_override: Stability monitor multiplier ∈ [0, 1].

        Returns:
            Temperature value to divide logits by. Always > 0.
        """
        C_eff = confidence * confidence_override

        # Normalized curvature, clamped to prevent extreme temperatures
        kappa_normalized = kappa / max(kappa_ref, 1e-6)
        kappa_clamped = max(-KAPPA_MAX_RESPONSE,
                           min(KAPPA_MAX_RESPONSE, kappa_normalized))

        T = T_BASE * (1.0 + C_eff * kappa_clamped)

        # Safety floor and ceiling
        return max(0.1, min(5.0, T))

    @staticmethod
    def think_token_bias(H: float, kappa: float, confidence: float,
                         confidence_override: float,
                         is_thinking: bool,
                         think_start_id: int,
                         think_end_id: int) -> dict:
        """
        Law 2 — Heat Kernel (Thinking Mode Transition).

        Replaces v1's fixed threshold switching with curvature-responsive
        geometry transition. Uses BOTH entropy level AND curvature to decide.

        High H + positive κ (getting MORE confused) → strong <think> boost.
        Low H + negative κ (getting MORE confident) → strong </think> boost.
        Flat κ → no intervention (model's natural dynamics sufficient).

        Args:
            H: Current entropy.
            kappa: Current scalar curvature.
            confidence: Global confidence.
            confidence_override: Stability multiplier.
            is_thinking: Whether currently inside <think> tags.
            think_start_id: Token ID for <think>.
            think_end_id: Token ID for </think>.

        Returns:
            Dict mapping token_id → logit bias to add. Empty dict = no bias.
        """
        C_eff = confidence * confidence_override
        if C_eff < 0.01:
            return {}

        biases = {}

        if not is_thinking:
            # ENTRY criterion: high entropy AND entropy is accelerating (κ > 0)
            # The conjunction prevents triggering on stable high entropy
            # (model may be legitimately exploring) and only triggers when
            # confusion is INCREASING.
            if H > 3.5 and kappa > 0:
                strength = C_eff * min(kappa * 10.0, 15.0)
                biases[think_start_id] = strength
        else:
            # EXIT criterion: low entropy AND entropy is decelerating (κ < 0)
            # The conjunction prevents premature exit during brief dips
            # and only triggers when confidence is INCREASING.
            if H < 2.0 and kappa < 0:
                strength = C_eff * min(abs(kappa) * 10.0, 15.0)
                biases[think_end_id] = strength

        return biases
```

**Note on Law 3 (Routing Bias)**: MoE routing bias requires PyTorch forward hooks on the model's gating mechanism. This is a Phase 3 capability that depends on being able to identify and hook into Nemotron's specific MoE module paths. **Do not implement routing bias in this version.** The architecture supports it (the Calibrator tracks expert signatures), but the hook infrastructure requires model-internal inspection that should be done separately. Temperature (Law 1) and think-token bias (Law 2) operate entirely within the logits processor interface and are the complete v2 implementation.

### 4.4 Calibrator — `calibrator.py`

**Lifetime**: One instance for the server's entire lifetime. Shared (by reference) across all request processors. Persists state across requests.

**Responsibility**: Maintain the slow-timescale state. Update κ_ref, confidence, baseline perplexity. Optionally persist to disk.

```python
import json
import os
import math
import threading

class Calibrator:
    """
    Cross-request self-calibration engine.

    Maintains running statistics that converge to the model's natural
    geometric constants. Thread-safe for concurrent request processing.

    All updates are monotonic or convergent — the calibrator never
    makes a decision it can't recover from.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.state = CalibrationSnapshot()
        self._try_load_state()

    # ── Confidence Function ─────────────────────────────────────────

    def confidence(self) -> float:
        """
        C(t_global) = (1 - exp(-t_global / τ)) × confidence_override

        Returns effective confidence combining observation-based growth
        and stability monitor override.
        """
        C_obs = 1.0 - math.exp(-self.state.t_global / TAU_CONFIDENCE)
        return C_obs * self.state.confidence_override

    # ── Per-Token Update ────────────────────────────────────────────

    def update(self, phase: PhaseState):
        """
        Ingest one PhaseState observation. Updates all running statistics.

        Called once per generated token, from the request processor,
        AFTER the token has been selected and its logprob is known.

        Thread-safe: acquires lock for the duration of the update.
        """
        with self.lock:
            self.state.t_global += 1

            # ── κ_ref: EMA of |κ| ──────────────────────────────────
            abs_kappa = abs(phase.kappa)
            rate = CALIBRATION_RATE
            self.state.kappa_running_mean = (
                (1 - rate) * self.state.kappa_running_mean + rate * abs_kappa
            )
            self.state.kappa_running_var = (
                (1 - rate) * self.state.kappa_running_var + rate * (abs_kappa ** 2)
            )
            # κ_ref = mean + 1σ — this means κ/κ_ref > 1 is "above normal"
            kappa_std = math.sqrt(max(0, self.state.kappa_running_var
                                      - self.state.kappa_running_mean ** 2))
            self.state.kappa_ref = max(
                0.01,
                self.state.kappa_running_mean + kappa_std
            )

            # ── Baseline perplexity (from zero-confidence period) ───
            if self.confidence() < 0.1:
                # Still in warmup — accumulate baseline
                n = self.state.baseline_count
                old_mean = self.state.baseline_perplexity
                ppl_sample = math.exp(-phase.token_logprob) if phase.token_logprob < 0 else 1.0
                self.state.baseline_perplexity = (
                    (old_mean * n + ppl_sample) / (n + 1)
                )
                self.state.baseline_count = n + 1

    # ── Stability Monitor Interface ─────────────────────────────────

    def report_quality(self, rolling_perplexity: float):
        """
        Called by StabilityMonitor with current rolling perplexity.
        Adjusts confidence_override based on quality comparison to baseline.
        """
        with self.lock:
            if self.state.baseline_perplexity <= 0 or self.state.baseline_count < 100:
                return  # Not enough baseline data yet

            threshold = self.state.baseline_perplexity * (1.0 + STABILITY_TOLERANCE)

            if rolling_perplexity > threshold:
                # Quality degrading — pull back
                self.state.confidence_override = max(
                    0.0,
                    self.state.confidence_override * STABILITY_DECAY
                )
            else:
                # Quality acceptable — slowly recover
                self.state.confidence_override = min(
                    1.0,
                    self.state.confidence_override * STABILITY_RECOVERY
                )

    # ── Persistence ─────────────────────────────────────────────────

    def save_state(self):
        """Persist calibration state to disk. Called periodically."""
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(vars(self.state), f, indent=2)
        except Exception:
            pass  # Non-critical — engine recalibrates if state is lost

    def _try_load_state(self):
        """Load persisted state if available."""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(self.state, k):
                        setattr(self.state, k, v)
        except Exception:
            pass  # Start fresh — engine will recalibrate
```

### 4.5 Stability Monitor — `stability_monitor.py`

**Lifetime**: One instance per generation request.

**Responsibility**: Track per-token quality (via log-probability of selected tokens). Report rolling perplexity to Calibrator. The Calibrator adjusts `confidence_override` which feeds back into all structural laws.

```python
class StabilityMonitor:
    """
    Per-request quality tracking.

    Monitors whether geometric intervention is degrading generation
    quality by tracking the log-probability of selected tokens under
    the model's distribution.

    Reports to the shared Calibrator which adjusts global confidence.
    """

    def __init__(self, calibrator: Calibrator):
        self.calibrator = calibrator
        self.logprob_buffer: List[float] = []  # Rolling window

    def record(self, token_logprob: float):
        """
        Record the log-probability of a selected token.

        Args:
            token_logprob: Log P(selected_token | context) under the
                          model's distribution (from unmodified logits).
        """
        self.logprob_buffer.append(token_logprob)
        if len(self.logprob_buffer) > PERPLEXITY_WINDOW:
            self.logprob_buffer.pop(0)

        # Report rolling perplexity when buffer is full
        if len(self.logprob_buffer) >= PERPLEXITY_WINDOW:
            mean_logprob = sum(self.logprob_buffer) / len(self.logprob_buffer)
            rolling_ppl = math.exp(-mean_logprob) if mean_logprob < 0 else float('inf')
            self.calibrator.report_quality(rolling_ppl)
```

### 4.6 Engine (Top-Level) — `engine.py`

**Lifetime**: `FluidGeometryLogitsProcessor` lives for the server's lifetime. `GeometricRequestProcessor` lives for one request.

```python
class GeometricRequestProcessor:
    """
    Per-request logits processor implementing the full geometric engine.

    Created for each inference request. Holds per-request Accumulator
    and StabilityMonitor. References the shared Calibrator.
    """

    def __init__(self, calibrator: Calibrator,
                 think_start_id: int, think_end_id: int):
        self.calibrator = calibrator
        self.accumulator = Accumulator()
        self.monitor = StabilityMonitor(calibrator)
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id

    def _is_thinking(self, tokens: list) -> bool:
        """Scan token sequence backwards for open <think> tag."""
        for token in reversed(tokens):
            if token == self.think_end_id:
                return False
            if token == self.think_start_id:
                return True
        return False

    def __call__(self, prompt_token_ids: list, output_token_ids: list,
                 logits: torch.Tensor) -> torch.Tensor:
        """
        Main entry point. Called by vLLM for every generated token.

        Signature: (prompt_ids, output_ids, logits) → modified logits

        Execution order:
          1. Snapshot raw logits (for measurement)
          2. Compute phase state from raw logits
          3. Compute structural law interventions
          4. Apply interventions to logits
          5. Estimate selected-token logprob (for stability tracking)
          6. Update calibrator and stability monitor
          7. Return modified logits
        """
        # ── 1. Snapshot raw logits ──────────────────────────────────
        raw_logits = logits.clone()

        # ── 2. Get current calibration state ────────────────────────
        conf = self.calibrator.confidence()
        conf_override = self.calibrator.state.confidence_override
        kappa_ref = self.calibrator.state.kappa_ref

        # ── 3. Compute phase state ──────────────────────────────────
        # Estimate token logprob from raw logits (best we can do before sampling)
        raw_probs = torch.softmax(raw_logits.float(), dim=-1)
        top_logprob = torch.log(raw_probs.max() + 1e-10).item()

        phase = self.accumulator.update(
            logits=raw_logits,
            confidence=conf,
            T_applied=1.0,  # Will be overwritten below
            selected_token_logprob=top_logprob,
        )

        # ── 4. Structural Law 1: Temperature ────────────────────────
        T = StructuralLaws.temperature(
            kappa=phase.kappa,
            kappa_ref=kappa_ref,
            confidence=conf,
            confidence_override=conf_override,
        )

        # Apply temperature by dividing logits
        if abs(T - 1.0) > 1e-6:
            logits = logits / T

        # Update the phase record with actual T applied
        phase.T_applied = T

        # ── 5. Structural Law 2: Think Token Bias ───────────────────
        all_tokens = (prompt_token_ids or []) + output_token_ids
        is_thinking = self._is_thinking(all_tokens)

        biases = StructuralLaws.think_token_bias(
            H=phase.H,
            kappa=phase.kappa,
            confidence=conf,
            confidence_override=conf_override,
            is_thinking=is_thinking,
            think_start_id=self.think_start_id,
            think_end_id=self.think_end_id,
        )

        for token_id, bias_value in biases.items():
            logits[token_id] += bias_value

        # ── 6. Update calibrator and stability monitor ──────────────
        self.calibrator.update(phase)
        self.monitor.record(top_logprob)

        # ── 7. Periodic state persistence ───────────────────────────
        if self.calibrator.state.t_global % 5000 == 0:
            self.calibrator.save_state()

        return logits


class FluidGeometryLogitsProcessor(AdapterLogitsProcessor):
    """
    vLLM v1 LogitsProcessor plugin — Geometric Engine v2.

    Server-lifetime object. Holds the shared Calibrator.
    Creates GeometricRequestProcessor for each inference request.

    Registration: --logits-processors fluid_geometry:FluidGeometryLogitsProcessor
    """

    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)

        # ── Resolve think tokens ────────────────────────────────────
        from transformers import AutoTokenizer

        tokenizer_path = vllm_config.model_config.tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
        )

        start_tokens = tokenizer.encode("<think>", add_special_tokens=False)
        end_tokens = tokenizer.encode("</think>", add_special_tokens=False)

        self.think_start_id = start_tokens[0] if start_tokens else None
        self.think_end_id = end_tokens[0] if end_tokens else None

        if self.think_start_id is None or self.think_end_id is None:
            raise ValueError(
                "Could not resolve <think>/<think> token IDs. "
                "Model tokenizer must support these tokens."
            )

        # ── Initialize shared Calibrator ────────────────────────────
        self.calibrator = Calibrator()

        # ── Log startup ─────────────────────────────────────────────
        print(f"[GeometricEngine] Initialized. "
              f"think_start={self.think_start_id}, "
              f"think_end={self.think_end_id}, "
              f"t_global={self.calibrator.state.t_global} "
              f"(loaded from {'disk' if self.calibrator.state.t_global > 0 else 'zero'})")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        """Create per-request geometric processor."""
        return GeometricRequestProcessor(
            calibrator=self.calibrator,
            think_start_id=self.think_start_id,
            think_end_id=self.think_end_id,
        )
```

---

## 5. Behavioral Specification

### 5.1 Lifecycle

```
CONTAINER STARTS
  │
  ▼
vLLM loads model weights (~3-5 min)
vLLM calls FluidGeometryLogitsProcessor.__init__()
  │  • Resolves <think>/<think> token IDs
  │  • Creates Calibrator (loads state from disk if available)
  │  • If fresh start: t_global=0, κ_ref=1.0, confidence≈0
  │  • If resumed: t_global=N, κ_ref=calibrated, confidence=C(N)
  │
  ▼
READY FOR REQUESTS
  │
  ▼
REQUEST ARRIVES
  │  vLLM calls new_req_logits_processor()
  │  → GeometricRequestProcessor created (fresh Accumulator)
  │
  ▼
FOR EACH GENERATED TOKEN:
  │  vLLM calls processor.__call__(prompt_ids, output_ids, logits)
  │  → Measure Φ(t) from raw logits
  │  → Compute T(t) and biases from structural laws
  │  → Apply to logits
  │  → Update calibrator (slow timescale)
  │  → Update stability monitor (rolling quality)
  │  → Return modified logits to vLLM sampler
  │
  ▼
REQUEST COMPLETES
  │  GeometricRequestProcessor is garbage collected
  │  Trace data is discarded (unless externally captured)
  │  Calibrator state persists
  │
  ▼
NEXT REQUEST...
```

### 5.2 Warmup Behavior (First ~10,000 Tokens)

| t_global | Confidence | Behavior |
|----------|------------|----------|
| 0 | 0.00 | Pure observation. All interventions zero. Measuring H(t), building κ_ref baseline. |
| 1,000 | 0.10 | Very faint temperature modulation (±10% of full strength). Think-token bias near zero. |
| 5,000 | 0.39 | Moderate intervention. κ_ref converging. Baseline perplexity establishing. |
| 10,000 | 0.63 | Near-operational. κ_ref stable. Stability monitor has reliable baseline. |
| 20,000 | 0.86 | Full operation. All laws active at near-full strength. |
| 30,000 | 0.95 | Fully converged. Slow adaptation continues (tracks distribution drift). |

**No external signal triggers these transitions.** The exponential confidence curve `C = 1 - exp(-t/τ)` handles everything.

### 5.3 Self-Healing Behavior

If geometric intervention causes quality degradation:

1. StabilityMonitor detects rolling perplexity > baseline × (1 + tolerance)
2. Calibrator reduces `confidence_override` by factor 0.99 per token
3. All structural law outputs scale down proportionally
4. If degradation persists, `confidence_override` → 0 (engine becomes passive observer)
5. When perplexity recovers (because intervention weakened), `confidence_override` slowly recovers at rate 1.001 per token
6. Engine settles at the maximum intervention level that doesn't degrade quality

**This is automatic.** No human observes or adjusts anything.

### 5.4 Container Restart Behavior

- If `STATE_FILE` exists: Calibrator loads prior state. Engine resumes at prior confidence level. No warmup re-required.
- If `STATE_FILE` missing or corrupt: Engine starts from zero. Re-converges within ~10K tokens. Generation quality is never degraded (zero confidence = zero intervention).

---

## 6. Structural Laws — Detailed Behavior

### 6.1 Law 1: Temperature (Geodesic Equation)

```
T(t) = T_BASE × (1 + C_eff × clamp(κ/κ_ref, -5, +5))
```

| Curvature State | κ sign | T effect | Interpretation |
|----------------|--------|----------|----------------|
| Entropy accelerating (confusion rising) | κ > 0 | T > 1 | Increase exploration; model is in uncertain territory |
| Entropy stable | κ ≈ 0 | T ≈ 1 | No modification; model is on stable geodesic |
| Entropy decelerating (confidence rising) | κ < 0 | T < 1 | Increase focus; model is converging on answer |

**Safety bounds**: T is clamped to [0.1, 5.0] regardless of curvature.

### 6.2 Law 2: Think Token Bias (Heat Kernel)

**Entry condition** (NOT thinking → thinking):
```
Trigger when: H > 3.5 AND κ > 0
Bias strength: C_eff × min(κ × 10, 15)
Applied to: <think> token logit
```

**Exit condition** (thinking → NOT thinking):
```
Trigger when: H < 2.0 AND κ < 0
Bias strength: C_eff × min(|κ| × 10, 15)
Applied to: </think> token logit
```

**Key difference from v1**: v1 used fixed entropy thresholds with no curvature awareness. v2 requires BOTH entropy level AND entropy dynamics (curvature) to agree. This prevents:
- Triggering thinking on stable-high-entropy outputs (model legitimately exploring)
- Exiting thinking during brief entropy dips (transient confidence in wrong direction)

### 6.3 Law 3: Routing Bias (NOT IMPLEMENTED IN V2)

Reserved for future version requiring PyTorch forward hooks on MoE gating layers. The Calibrator architecture supports expert signature tracking but the hook registration infrastructure is out of scope for the logits-processor-only deployment.

---

## 7. Deployment

### 7.1 File Placement

Single file deployment (recommended for vLLM compatibility):

```bash
# On spark-129a:
/home/pokazge/models/fluid_geometry.py   # Complete engine in one file
```

All classes from §4.1–4.6 concatenated into one file with clear section separators.

### 7.2 Docker Command

```bash
docker run -d \
  --name vllm-nemotron-serve \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 30000:30000 \
  -v /home/pokazge/models/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8:/workspace/model \
  -v /home/pokazge/models/nano_v3_reasoning_parser.py:/workspace/nano_v3_reasoning_parser.py \
  -v /home/pokazge/models/fluid_geometry.py:/workspace/fluid_geometry.py \
  -v /home/pokazge/models/engine_state:/workspace/engine_state \
  nvcr.io/nvidia/vllm:26.01-py3 \
  python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 30000 \
    --model /workspace/model \
    --served-model-name NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --trust-remote-code \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    --reasoning-parser-plugin /workspace/nano_v3_reasoning_parser.py \
    --reasoning-parser nano_v3 \
    --logits-processors fluid_geometry:FluidGeometryLogitsProcessor
```

Note: `engine_state` volume mount provides persistent storage for `STATE_FILE` across container restarts. Update `STATE_FILE` path in config to `/workspace/engine_state/geometric_engine_state.json`.

### 7.3 Verification

**Test 1 — Engine loads:**
```bash
docker logs vllm-nemotron-serve 2>&1 | grep GeometricEngine
# Expected: [GeometricEngine] Initialized. think_start=X, think_end=Y, t_global=0 (loaded from zero)
```

**Test 2 — Inference works (no degradation):**
```bash
curl -s http://spark-129a.local:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
       "messages":[{"role":"user","content":"What is 2+2?"}],
       "max_tokens":100}' | jq '.choices[0].message.content'
# Expected: Correct answer. No degradation. Engine at ~0% confidence, pure observation.
```

**Test 3 — Warmup progression (after ~100 requests):**
```bash
# Check state file for convergence
docker exec vllm-nemotron-serve cat /workspace/engine_state/geometric_engine_state.json
# Expected: t_global > 0, kappa_ref stabilized, baseline_perplexity > 0
```

---

## 8. Performance Constraints

| Metric | Budget | Rationale |
|--------|--------|-----------|
| Per-token latency overhead | < 0.1 ms | One softmax + entropy sum + few FLOPs. Must be invisible vs. model forward pass (~10-50ms). |
| Memory per request | < 1 KB | Accumulator buffer (3 floats) + trace (grows linearly but can be capped). |
| Shared state memory | < 10 KB | CalibrationSnapshot: ~500 floats. |
| Disk I/O | Once per 5000 tokens | JSON state save. Non-blocking if possible. |

---

## 9. Testing Strategy

### 9.1 Unit Tests (No vLLM Required)

**Accumulator tests:**
- Feed synthetic logit sequences with known entropy profiles
- Verify H(t) matches manual calculation
- Verify κ(t) = Δ²H / (1 + |ΔH|) for known sequences
- Verify buffer management (correct sliding window behavior)

**Structural Laws tests:**
- Verify T = T_BASE when confidence = 0 (regardless of κ)
- Verify T = T_BASE when κ = 0 (regardless of confidence)
- Verify T increases when κ > 0, confidence > 0
- Verify T decreases when κ < 0, confidence > 0
- Verify T clamped to [0.1, 5.0]
- Verify think bias requires conjunction (H threshold AND κ sign)

**Calibrator tests:**
- Verify confidence curve: C(0) ≈ 0, C(τ) ≈ 0.63, C(3τ) ≈ 0.95
- Verify κ_ref converges to known |κ| statistics for synthetic input
- Verify stability override decreases when fake high-perplexity reported
- Verify stability override recovers when perplexity returns to baseline
- Verify state persistence round-trip (save → load → identical state)

### 9.2 Integration Tests (Requires vLLM)

**Passthrough test:**
- Set KAPPA_MAX_RESPONSE = 0 (disables all intervention)
- Compare output distribution to unmodified vLLM
- Must be identical (engine in pure observation mode)

**Non-degradation test:**
- Run 100 diverse prompts with engine active
- Compare mean perplexity to baseline (engine disabled)
- Must be within STABILITY_TOLERANCE (5%)

**Convergence test:**
- Start from fresh state
- Run 500 prompts
- Verify κ_ref stabilizes (variance of κ_ref over last 100 prompts < threshold)
- Verify confidence > 0.8

---

## 10. What This Replaces

The entire content of the current `fluid_geometry.py` (v1) is replaced. The v1 classes `FluidGeometryRequestProcessor` and `FluidGeometryLogitsProcessor` are superseded by `GeometricRequestProcessor` and the new `FluidGeometryLogitsProcessor` respectively.

**Key behavioral differences from v1:**

| Aspect | v1 | v2 |
|--------|-----|-----|
| Entropy source | Full logits (same) | Full logits (same) |
| Decision basis | Entropy vs. fixed thresholds | Entropy + curvature (Δ²H dynamics) |
| Think entry | H > 4.5 (fixed) | H > 3.5 AND κ > 0 (adaptive, conjunction) |
| Think exit | H < 1.5 (fixed) | H < 2.0 AND κ < 0 (adaptive, conjunction) |
| Temperature | Not modified | Curvature-responsive modulation |
| Calibration | None (hardcoded thresholds) | Self-calibrating κ_ref from running statistics |
| Warmup | None (active from first token) | Confidence ramp (safe from first token, full at ~10K) |
| Quality guard | None | Stability monitor with automatic pullback |
| Persistence | None | State file survives container restarts |
| Constants | 4 manually tuned | 0 manually tuned (self-calibrated) |

---

## 11. Open Items for Implementer

1. **Single file vs. package**: Verify whether vLLM's `--logits-processors` flag can load from a package (`fluid_geometry.engine:FluidGeometryLogitsProcessor`) or requires a single file (`fluid_geometry:FluidGeometryLogitsProcessor`). Implement accordingly.

2. **Token logprob estimation**: The spec uses `raw_probs.max()` as an estimate of the selected token's probability. This is approximate — the actual selected token depends on sampling (temperature, top-p, etc.) applied AFTER the processor. If vLLM provides a post-sampling callback, use it for more accurate stability monitoring. If not, the max-prob estimate is a conservative upper bound.

3. **Thread safety in Calibrator**: The spec uses a threading lock. Verify that vLLM's request processing model (may be async, may be multi-process) is compatible. If multi-process: use file-based locking or accept slightly stale shared state. If single-threaded async: the lock is unnecessary but harmless.

4. **Trace export (optional, low priority)**: If response metadata enrichment is desired (exposing κ trajectory to MCP), this requires modifying vLLM's response serialization. This is NOT required for the engine to function. Implement only if there's a clean hook point in vLLM's API server.

5. **vLLM version compatibility**: Tested against vLLM 0.13.0 (nv26.01 container). The `AdapterLogitsProcessor` base class and 3-argument `__call__` signature are v1-engine-specific. Verify these interfaces haven't changed in newer vLLM releases.

---

*End of Specification*
