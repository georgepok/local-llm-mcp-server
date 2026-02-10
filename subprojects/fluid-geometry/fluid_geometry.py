"""
Geometric Engine v3 — Per-Request Self-Calibrating Entropy-Curvature Processor for vLLM.

Implements adaptive reasoning control based on phase space dynamics:
- Measures multi-dimensional phase space Φ(t) at every generation step
- Computes scalar curvature κ(t) from entropy derivatives
- Applies structural response laws (temperature, think-token bias)
- Self-calibrates ALL reference constants per-request from a 15-token warmup
- Self-gates intervention via fast per-request confidence ramp
- Self-heals via per-request stability monitoring with symmetric pullback

No configuration required. No persistent state. Each request starts fresh,
warms up from its own observations, and applies geometry independently.

Compatible with vLLM 0.13+ v1 LogitsProcessor interface.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Dict

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Configuration & Types
# ═══════════════════════════════════════════════════════════════════════════

# Warmup
WARMUP_TOKENS: int = 15                # Tokens to observe before intervening

# Temperature Modulation
T_BASE: float = 1.0                    # Baseline temperature (no modification)
T_RESPONSE_SCALE: float = 0.5          # Moderate temperature response
T_MAX_DELTA: float = 0.25              # Max temperature change per token
T_FLOOR: float = 0.7                   # Minimum temperature
T_CEILING: float = 1.5                 # Maximum temperature

# Curvature
KAPPA_MAX_RESPONSE: float = 5.0        # Clamp κ/κ_ref to prevent extreme T
KAPPA_REF_FLOOR: float = 0.001         # Minimum κ_ref (was 0.01 in v2)

# Confidence
TAU_CONFIDENCE: float = 15.0           # Per-request confidence τ (tokens)

# Stability
STABILITY_PULLBACK: float = 0.98       # Symmetric pullback rate
STABILITY_RECOVERY: float = 1.02       # Symmetric recovery rate
PERPLEXITY_WINDOW: int = 50            # Rolling window for perplexity
STABILITY_SIGMA: float = 2.0           # Pullback at 2σ deviation

# Entropy Smoothing
H_SMOOTHING_ALPHA: float = 0.05        # EMA alpha for entropy (~20 token window)
H_BUFFER_SIZE: int = 3                 # Entropy history buffer

# Curvature Smoothing
KAPPA_SMOOTHING_ALPHA: float = 0.15    # EMA alpha for curvature smoothing

# Trace logging (disabled by default, enable with FG_TRACE=1 environment variable)
TRACE_FILE: str = "/workspace/engine_state/trace.log"
TRACE_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB max, then truncate

# Think Tokens
THINK_START_TOKEN: str = "<think>"
THINK_END_TOKEN: str = "</think>"


@dataclass
class PhaseState:
    """Single step of phase space measurement."""
    step: int
    H: float                    # Shannon entropy (EMA-smoothed, for derivatives)
    H_raw: float                # Shannon entropy (raw, for calibration)
    delta_H: float              # First derivative (velocity)
    delta2_H: float             # Second derivative (acceleration)
    kappa: float                # Scalar curvature κ(t) = Δ²H / (1 + |ΔH|)
    delta_kappa: float          # Rate of curvature change
    T_applied: float            # Temperature actually applied
    confidence: float           # Request-local confidence at this step
    token_logprob: float        # Log-prob of selected token (for stability)
    top2_mass: float            # p1 + p2 (top-2 probability concentration)
    bimodality: float           # min(p1,p2)/max(p1,p2) — 1.0 = perfectly bimodal


@dataclass
class RequestCalibration:
    """
    Per-request calibration computed from warmup observations.

    During the first WARMUP_TOKENS tokens, collects entropy and curvature
    samples. After warmup, computes request-specific reference constants
    that reflect THIS request's actual distribution, not historical averages.
    """
    warmup_complete: bool = False
    t_request: int = 0
    confidence_override: float = 1.0    # Stability monitor multiplier

    # Warmup sample buffers (cleared after warmup)
    H_samples: List[float] = field(default_factory=list)
    kappa_samples: List[float] = field(default_factory=list)

    # Computed after warmup:
    kappa_ref: float = 1.0              # mean(|κ|) + σ(|κ|) from warmup
    H_p90: float = 3.5                  # 90th percentile of H (think entry)
    H_p25: float = 2.0                  # 25th percentile of H (think exit)
    H_mean: float = 1.0                 # Mean H (for entropy scaling)

    def confidence(self) -> float:
        """
        Per-request confidence: C(t) = (1 - exp(-t / τ)) × override

        Reaches ~63% at token 15, ~95% at token 45. Much faster than v2's
        τ=10,000 which took 30K tokens for 95% confidence.
        """
        C_obs = 1.0 - math.exp(-self.t_request / TAU_CONFIDENCE)
        return C_obs * self.confidence_override

    def observe(self, H: float, abs_kappa: float) -> None:
        """
        Record one warmup observation. After WARMUP_TOKENS samples,
        compute request-specific calibration constants.
        """
        self.t_request += 1

        if self.warmup_complete:
            return

        self.H_samples.append(H)
        self.kappa_samples.append(abs_kappa)

        if self.t_request >= WARMUP_TOKENS:
            self._finalize_warmup()

    def _finalize_warmup(self) -> None:
        """Compute calibration constants from warmup observations."""
        self.warmup_complete = True

        # κ_ref: mean + 1σ of |κ| (floor at KAPPA_REF_FLOOR)
        if self.kappa_samples:
            n = len(self.kappa_samples)
            kappa_mean = sum(self.kappa_samples) / n
            kappa_var = sum((k - kappa_mean) ** 2 for k in self.kappa_samples) / max(n - 1, 1)
            kappa_std = math.sqrt(kappa_var)
            self.kappa_ref = max(KAPPA_REF_FLOOR, kappa_mean + kappa_std)

        # Entropy percentiles and mean
        if self.H_samples:
            sorted_H = sorted(self.H_samples)
            n = len(sorted_H)
            self.H_mean = sum(sorted_H) / n
            self.H_p90 = sorted_H[min(int(n * 0.9), n - 1)]
            self.H_p25 = sorted_H[min(int(n * 0.25), n - 1)]

            # Ensure thresholds have minimum separation
            if self.H_p90 - self.H_p25 < 0.1:
                mid = (self.H_p90 + self.H_p25) / 2
                self.H_p90 = mid + 0.05
                self.H_p25 = mid - 0.05

        # Free warmup buffers
        self.H_samples = []
        self.kappa_samples = []


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Accumulator
# ═══════════════════════════════════════════════════════════════════════════

class Accumulator:
    """
    Per-request phase space tracker.

    Applies EMA smoothing to both entropy and curvature before computing
    derivatives, preventing oscillation from noisy token-level measurements.
    """

    def __init__(self):
        self.step: int = 0
        self.H_raw: float = 0.0           # Raw entropy (for diagnostics)
        self.H_smooth: float = 0.0        # EMA-smoothed entropy
        self.H_smooth_buffer: List[float] = []  # Buffer of smoothed values
        self.kappa_raw: float = 0.0       # Raw curvature (before smoothing)
        self.kappa_smooth: float = 0.0    # EMA-smoothed curvature
        self.kappa_prev: float = 0.0      # Previous smoothed kappa
        self.trace: List[PhaseState] = []

    def update(
        self,
        logits: torch.Tensor,
        confidence: float,
        T_applied: float,
        selected_token_logprob: float
    ) -> PhaseState:
        """
        Ingest one step of logits. Compute and return phase state.

        Entropy is EMA-smoothed before differentiation to dampen noise.
        Curvature is also EMA-smoothed to prevent oscillation.
        """
        # Step 1: Exact entropy from full logit vector
        probs = torch.softmax(logits.float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)
        H_raw = -(probs * log_probs).sum().item()
        self.H_raw = H_raw

        # Step 1b: Distribution shape analysis (top-2 for bimodal detection)
        top2 = torch.topk(probs, k=2)
        p1 = top2.values[0].item()
        p2 = top2.values[1].item()
        top2_mass = p1 + p2
        bimodality = min(p1, p2) / max(p1, p2) if p1 > 1e-10 else 0.0

        # Step 2: Apply EMA smoothing to entropy
        if self.step == 0:
            self.H_smooth = H_raw
        else:
            self.H_smooth = (H_SMOOTHING_ALPHA * H_raw +
                            (1 - H_SMOOTHING_ALPHA) * self.H_smooth)

        # Step 3: Update smoothed buffer
        self.H_smooth_buffer.append(self.H_smooth)
        if len(self.H_smooth_buffer) > H_BUFFER_SIZE:
            self.H_smooth_buffer.pop(0)

        # Step 4: Compute derivatives from smoothed entropy
        if len(self.H_smooth_buffer) >= 2:
            delta_H = self.H_smooth_buffer[-1] - self.H_smooth_buffer[-2]
        else:
            delta_H = 0.0

        if len(self.H_smooth_buffer) >= 3:
            delta2_H = (self.H_smooth_buffer[-1] - 2 * self.H_smooth_buffer[-2]
                        + self.H_smooth_buffer[-3])
        else:
            delta2_H = 0.0

        # Step 5: Scalar curvature κ(t) = Δ²H(t) / (1 + |ΔH(t)|)
        kappa_raw = delta2_H / (1.0 + abs(delta_H))
        self.kappa_raw = kappa_raw

        # Step 6: Apply EMA smoothing to curvature
        if self.step == 0:
            self.kappa_smooth = kappa_raw
        else:
            self.kappa_smooth = (KAPPA_SMOOTHING_ALPHA * kappa_raw +
                                (1 - KAPPA_SMOOTHING_ALPHA) * self.kappa_smooth)

        # Step 7: Curvature velocity (from smoothed kappa)
        delta_kappa = self.kappa_smooth - self.kappa_prev
        self.kappa_prev = self.kappa_smooth

        # Step 8: Build phase state (smoothed values for laws, raw for calibration)
        state = PhaseState(
            step=self.step,
            H=self.H_smooth,
            H_raw=H_raw,
            delta_H=delta_H,
            delta2_H=delta2_H,
            kappa=self.kappa_smooth,
            delta_kappa=delta_kappa,
            T_applied=T_applied,
            confidence=confidence,
            token_logprob=selected_token_logprob,
            top2_mass=top2_mass,
            bimodality=bimodality,
        )

        self.trace.append(state)
        self.step += 1
        return state


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Structural Laws
# ═══════════════════════════════════════════════════════════════════════════

class StructuralLaws:
    """
    Deterministic structural response laws.

    Given the same phase state and calibration, always produce the same output.
    v3: Uses per-request adaptive thresholds instead of fixed constants.
    """

    @staticmethod
    def temperature(
        kappa: float,
        calibration: RequestCalibration,
        T_prev: float = 1.0,
        bimodality: float = 0.0,
        top2_mass: float = 0.0
    ) -> float:
        """
        Law 1 — Geodesic Equation (Temperature Modulation).

        T(t) = T_BASE × (1 + C_eff × scale × κ(t) / κ_ref)

        Uses request-local kappa_ref and H_mean for proper scaling.
        Rate-limited to prevent wild oscillations.

        v3.1: Attenuates intervention at bimodal decision points where the
        model is choosing between two strong competing options. Temperature
        distortion at these points would arbitrarily collapse a genuine choice.
        """
        C_eff = calibration.confidence()

        # Normalized curvature, clamped
        kappa_normalized = kappa / max(calibration.kappa_ref, 1e-6)
        kappa_clamped = max(-KAPPA_MAX_RESPONSE,
                           min(KAPPA_MAX_RESPONSE, kappa_normalized))

        # Scale by entropy regime (low entropy → reduce response)
        entropy_scale = min(1.0, calibration.H_mean / 1.0)

        # Bimodal attenuation: reduce intervention at genuine decision points.
        # When top-2 tokens hold >30% mass and are nearly equal (bimodality >0.5),
        # the model is choosing between two coherent options — don't distort.
        bimodal_factor = 1.0
        if bimodality > 0.5 and top2_mass > 0.3:
            bimodal_factor = 1.0 - 0.7 * bimodality  # up to 70% reduction

        effective_scale = T_RESPONSE_SCALE * entropy_scale * bimodal_factor

        T_target = T_BASE * (1.0 + C_eff * effective_scale * kappa_clamped)

        # Rate limiting: max change of T_MAX_DELTA per token
        T_delta = T_target - T_prev
        if abs(T_delta) > T_MAX_DELTA:
            T_delta = T_MAX_DELTA if T_delta > 0 else -T_MAX_DELTA
        T = T_prev + T_delta

        return max(T_FLOOR, min(T_CEILING, T))

    @staticmethod
    def think_token_bias(
        H: float,
        kappa: float,
        calibration: RequestCalibration,
        is_thinking: bool,
        think_start_id: int,
        think_end_id: int
    ) -> Dict[int, float]:
        """
        Law 2 — Heat Kernel (Thinking Mode Transition).

        v3: Uses adaptive thresholds from the request's own entropy distribution.
        - Entry: H > request's p90 AND κ > 0 (getting MORE confused)
        - Exit:  H < request's p25 AND κ < 0 (getting MORE confident)
        - Flat κ → no intervention

        Returns:
            Dict mapping token_id → logit bias to add. Empty dict = no bias.
        """
        C_eff = calibration.confidence()
        if C_eff < 0.01:
            return {}

        biases: Dict[int, float] = {}

        # v3.1: Use normalized curvature (κ/κ_ref) instead of raw κ.
        # Raw κ is ~0.001-0.01 after smoothing → bias of ~0.06 logits (invisible).
        # Normalized κ can reach ±5.0 → bias of up to 15.0 logits (meaningful).
        kappa_normalized = abs(kappa) / max(calibration.kappa_ref, 1e-6)

        if not is_thinking:
            # ENTRY criterion: entropy above request's 90th percentile AND accelerating
            if H > calibration.H_p90 and kappa > 0:
                strength = C_eff * min(kappa_normalized * 3.0, 15.0)
                biases[think_start_id] = strength
        else:
            # EXIT criterion: entropy below request's 25th percentile AND decelerating
            if H < calibration.H_p25 and kappa < 0:
                strength = C_eff * min(kappa_normalized * 3.0, 15.0)
                biases[think_end_id] = strength

        return biases


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Stability Monitor
# ═══════════════════════════════════════════════════════════════════════════

class StabilityMonitor:
    """
    Per-request quality tracking.

    Monitors whether geometric intervention is degrading generation quality
    by tracking token log-probabilities. Adjusts confidence_override on
    the request's own RequestCalibration — no global side effects.

    v3: Symmetric pullback/recovery (0.98/1.02) prevents ratchet effect.
    """

    def __init__(self, calibration: RequestCalibration):
        self.calibration = calibration
        self.logprob_buffer: List[float] = []
        self.ppl_running_mean: float = 0.0
        self.ppl_running_var: float = 0.0
        self.ppl_initialized: bool = False

    def record(self, token_logprob: float) -> None:
        """Record the log-probability of a selected token."""
        self.logprob_buffer.append(token_logprob)
        if len(self.logprob_buffer) > PERPLEXITY_WINDOW:
            self.logprob_buffer.pop(0)

        # Need enough data for meaningful statistics
        if len(self.logprob_buffer) < PERPLEXITY_WINDOW:
            return

        mean_logprob = sum(self.logprob_buffer) / len(self.logprob_buffer)
        rolling_ppl = math.exp(-mean_logprob) if mean_logprob < 0 else float('inf')

        if not self.ppl_initialized:
            self.ppl_running_mean = rolling_ppl
            self.ppl_running_var = rolling_ppl ** 2
            self.ppl_initialized = True
            return

        # Update running statistics (EMA with fast rate for per-request)
        rate = 0.05  # Faster than v2's 0.001 since per-request
        self.ppl_running_mean = (1 - rate) * self.ppl_running_mean + rate * rolling_ppl
        self.ppl_running_var = (1 - rate) * self.ppl_running_var + rate * (rolling_ppl ** 2)

        # Compute standard deviation
        ppl_variance = max(0, self.ppl_running_var - self.ppl_running_mean ** 2)
        ppl_std = math.sqrt(ppl_variance) if ppl_variance > 0 else 0.1

        # Drift detection: pullback if current exceeds mean + 2σ
        threshold = self.ppl_running_mean + STABILITY_SIGMA * ppl_std

        if rolling_ppl > threshold:
            self.calibration.confidence_override = max(
                0.0,
                self.calibration.confidence_override * STABILITY_PULLBACK
            )
        else:
            self.calibration.confidence_override = min(
                1.0,
                self.calibration.confidence_override * STABILITY_RECOVERY
            )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Engine (Top-Level)
# ═══════════════════════════════════════════════════════════════════════════

class GeometricRequestProcessor:
    """
    Per-request logits processor implementing the full geometric engine.

    Created fresh for each inference request. Holds per-request Accumulator,
    RequestCalibration, and StabilityMonitor. No shared state.
    """

    def __init__(
        self,
        think_start_id: int,
        think_end_id: int
    ):
        self.calibration = RequestCalibration()
        self.accumulator = Accumulator()
        self.monitor = StabilityMonitor(self.calibration)
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.T_prev: float = 1.0

    def _is_thinking(self, tokens: List[int]) -> bool:
        """Scan token sequence backwards for open <think> tag."""
        for token in reversed(tokens):
            if token == self.think_end_id:
                return False
            if token == self.think_start_id:
                return True
        return False

    def __call__(
        self,
        prompt_token_ids: List[int],
        output_token_ids: List[int],
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Main entry point. Called by vLLM for every generated token.

        Signature: (prompt_ids, output_ids, logits) → modified logits
        """
        # 1. Snapshot raw logits (for measurement, before any modification)
        raw_logits = logits.clone()

        # 2. Get current request-local confidence
        conf = self.calibration.confidence()

        # 3. Estimate token logprob from raw logits
        raw_probs = torch.softmax(raw_logits.float(), dim=-1)
        top_logprob = torch.log(raw_probs.max() + 1e-10).item()

        # 4. Compute phase state (from unmodified logits)
        phase = self.accumulator.update(
            logits=raw_logits,
            confidence=conf,
            T_applied=1.0,  # Will be overwritten below
            selected_token_logprob=top_logprob,
        )

        # 5. Feed RAW observations to warmup calibration (not EMA-smoothed).
        #    Raw values preserve the actual variance of entropy and curvature
        #    during warmup, preventing percentile collapse from heavy smoothing.
        self.calibration.observe(phase.H_raw, abs(self.accumulator.kappa_raw))

        # 6. Apply structural laws only after warmup
        if self.calibration.warmup_complete:
            # Law 1: Temperature (with bimodal attenuation)
            T = StructuralLaws.temperature(
                kappa=phase.kappa,
                calibration=self.calibration,
                T_prev=self.T_prev,
                bimodality=phase.bimodality,
                top2_mass=phase.top2_mass,
            )
            self.T_prev = T

            # Apply temperature by dividing logits
            if abs(T - 1.0) > 1e-6:
                logits = logits / T

            phase.T_applied = T

            # Law 2: Think Token Bias
            all_tokens = (prompt_token_ids or []) + output_token_ids
            is_thinking = self._is_thinking(all_tokens)

            biases = StructuralLaws.think_token_bias(
                H=phase.H,
                kappa=phase.kappa,
                calibration=self.calibration,
                is_thinking=is_thinking,
                think_start_id=self.think_start_id,
                think_end_id=self.think_end_id,
            )

            for token_id, bias_value in biases.items():
                logits[token_id] += bias_value

        # 7. Update stability monitor
        self.monitor.record(top_logprob)

        # 8. Diagnostic trace (DISABLED by default, enable with FG_TRACE=1)
        if os.environ.get('FG_TRACE', '0') == '1':
            trace_line = (
                f"req_t={self.calibration.t_request}"
                f" H={phase.H:.3f}"
                f" Hr={phase.H_raw:.3f}"
                f" dH={phase.delta_H:.3f}"
                f" k={phase.kappa:.4f}"
                f" T={phase.T_applied:.3f}"
                f" C={conf:.3f}"
                f" co={self.calibration.confidence_override:.3f}"
                f" kref={self.calibration.kappa_ref:.4f}"
                f" bi={phase.bimodality:.2f}"
                f" t2={phase.top2_mass:.2f}"
                f" warm={'Y' if self.calibration.warmup_complete else 'N'}"
            )
            try:
                trace_dir = os.path.dirname(TRACE_FILE)
                if trace_dir:
                    os.makedirs(trace_dir, exist_ok=True)
                if os.path.exists(TRACE_FILE):
                    if os.path.getsize(TRACE_FILE) > TRACE_MAX_SIZE:
                        with open(TRACE_FILE, 'r') as f:
                            lines = f.readlines()
                        with open(TRACE_FILE, 'w') as f:
                            f.writelines(lines[len(lines)//2:])
                with open(TRACE_FILE, 'a') as tf:
                    tf.write(trace_line + '\n')
            except Exception:
                pass

        return logits


class FluidGeometryLogitsProcessor(AdapterLogitsProcessor):
    """
    vLLM v1 LogitsProcessor plugin — Geometric Engine v3.

    Server-lifetime object. Stateless — only holds think token IDs.
    Creates a fresh GeometricRequestProcessor for each inference request.

    Registration: --logits-processors fluid_geometry:FluidGeometryLogitsProcessor
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool
    ):
        super().__init__(vllm_config, device, is_pin_memory)

        # Resolve think tokens
        from transformers import AutoTokenizer

        tokenizer_path = vllm_config.model_config.tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
        )

        start_tokens = tokenizer.encode(THINK_START_TOKEN, add_special_tokens=False)
        end_tokens = tokenizer.encode(THINK_END_TOKEN, add_special_tokens=False)

        self.think_start_id = start_tokens[0] if start_tokens else None
        self.think_end_id = end_tokens[0] if end_tokens else None

        if self.think_start_id is None or self.think_end_id is None:
            raise ValueError(
                f"Could not resolve {THINK_START_TOKEN}/{THINK_END_TOKEN} token IDs. "
                "Model tokenizer must support these tokens."
            )

        # Log startup
        print(f"[GeometricEngine] v3 initialized. "
              f"think_start={self.think_start_id}, "
              f"think_end={self.think_end_id}, "
              f"warmup={WARMUP_TOKENS} tokens, "
              f"tau={TAU_CONFIDENCE}")

    def is_argmax_invariant(self) -> bool:
        """This processor modifies logits, so it can change argmax."""
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams
    ) -> RequestLogitsProcessor:
        """Create per-request geometric processor with fresh calibration."""
        return GeometricRequestProcessor(
            think_start_id=self.think_start_id,
            think_end_id=self.think_end_id,
        )
