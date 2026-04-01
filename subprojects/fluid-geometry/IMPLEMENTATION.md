# Geometric Engine v2 — Implementation Details

## Document Info
- **Version**: 2.1.0
- **Date**: 2026-02-08
- **Type**: Implementation Documentation
- **Design Spec**: GEOMETRIC_ENGINE_SPEC.md
- **Deployed To**: spark-129a.local (NVIDIA DGX Spark)

---

## v2.1 Changes (2026-02-08)

Critical fixes applied based on VALIDATION_RESULTS.md findings. See PATCH_INSTRUCTIONS.md for rationale.

### Fix 1: Entropy Smoothing (EMA)

**Problem**: Raw entropy is spiky (many H=0 tokens interspersed with H=0.5-1.4). Taking derivatives of a spiky signal produces noise, not meaningful curvature.

**Solution**: Apply EMA smoothing to entropy BEFORE computing derivatives.

```python
# New constant
H_SMOOTHING_ALPHA = 0.1  # ~10 token window

# In Accumulator.update()
if self.step == 0:
    self.H_smooth = H_raw
else:
    self.H_smooth = (H_SMOOTHING_ALPHA * H_raw +
                    (1 - H_SMOOTHING_ALPHA) * self.H_smooth)

# Derivatives computed from H_smooth, not H_raw
```

**Effect**: Temperature now varies smoothly over 5-20 token spans instead of oscillating wildly between consecutive tokens.

### Fix 2: Drift Detection (2σ Threshold)

**Problem**: Hard 5% threshold (`baseline_perplexity × 1.05`) was too tight. The baseline was artificially low (1.29) because it used `max(softmax(logits))` during low-confidence operation.

**Solution**: Replace hard threshold with statistical drift detection.

```python
# New constants
STABILITY_SIGMA_THRESHOLD = 2.0  # Trigger at 2σ deviation

# In Calibrator
ppl_running_mean: float  # EMA of rolling perplexity
ppl_running_var: float   # EMA of perplexity² (for variance)

# In report_quality()
ppl_std = sqrt(ppl_running_var - ppl_running_mean²)
threshold = reference + 2.0 * ppl_std

if rolling_perplexity > threshold:
    # Pullback
else:
    # Recovery
```

**Effect**: Stability monitor no longer triggers false pullbacks. `confidence_override` stays at 1.0 during normal operation.

### Fix 3: Temperature Response Scaling

**Problem**: This model operates at very low entropy (0-1.4 bits). Same absolute κ produces outsized temperature response.

**Solution**: Scale temperature response by a factor and normalize by entropy regime.

```python
# New constants
T_RESPONSE_SCALE = 0.3  # Scale down response (was 1.0 implicit)

# In temperature()
entropy_scale = min(1.0, H_mean / 1.0)  # Full response when H >= 1.0
effective_scale = T_RESPONSE_SCALE * entropy_scale

T = T_BASE * (1.0 + C_eff * effective_scale * kappa_clamped)

# Tighter bounds
return max(0.7, min(1.5, T))  # Was [0.1, 5.0]
```

**Effect**: Temperature now varies 0.92-1.06 instead of 0.1-1.76. Model behavior is more stable.

### New State Fields (v2.1)

```python
@dataclass
class CalibrationSnapshot:
    # ... existing fields ...

    # v2.1 additions
    ppl_running_mean: float = 0.0   # For drift detection
    ppl_running_var: float = 0.0    # For drift detection
    H_running_mean: float = 1.0     # For entropy scaling
```

### Validation Results

After v2.1 fixes:
- `confidence_override`: 1.0 (was 0.06)
- Temperature range: 0.92-1.06 (was 0.1-1.76)
- Curvature: Sustained spans (was single-token spikes)

See VALIDATION_RESULTS_V2.1.md for full details

---

## Implementation Decisions

Key decisions made during implementation that differ from or clarify the design specification.

### File Structure: Single File vs Package

**Design spec proposed** (§3.1):
```
fluid_geometry/
├── __init__.py              # Package init
├── engine.py                # Top-level orchestrator
├── accumulator.py           # Per-request phase space
├── structural_laws.py       # Response laws
├── calibrator.py            # Self-calibration state
├── stability_monitor.py     # Quality guard
└── config.py                # Constants
```

**What was built**: Single file `fluid_geometry.py` (~575 lines)

**Why**: The design spec §3.1 states: *"If vLLM's `--logits-processors` flag requires a single file... collapse all modules into a single `fluid_geometry.py` file... use the simplest that works."*

Single-file was chosen for:
- Simpler deployment (one file to copy)
- Confirmed compatibility with v1
- Easier debugging in container

The file uses section comments to maintain logical separation:
```
Section 1: Config & Types      (lines 37-88)
Section 2: Accumulator         (lines 92-170)
Section 3: StructuralLaws      (lines 174-254)
Section 4: StabilityMonitor    (lines 258-284)
Section 5: Calibrator          (lines 288-401)
Section 6: Engine              (lines 405-575)
```

### Structural Laws: What Was Implemented

| Law | Status | Location in Code |
|-----|--------|------------------|
| Law 1: Temperature | ✓ Built | `StructuralLaws.temperature()` lines 185-214 |
| Law 2: Think Tokens | ✓ Built | `StructuralLaws.think_token_bias()` lines 216-254 |
| Law 3: Routing Bias | ✗ Not built | Per spec §4.3: "Do not implement in this version" |

### State File Path Change

| | Path |
|--|------|
| Design spec | `/workspace/geometric_engine_state.json` |
| Implementation | `/workspace/engine_state/geometric_engine_state.json` |

**Why**: Subdirectory allows separate volume mount for easier backup/inspection.

---

## What Was Built

### Classes

| Class | Purpose | Lifetime |
|-------|---------|----------|
| `PhaseState` | Dataclass holding one step's measurements | Per-token |
| `CalibrationSnapshot` | Dataclass holding persistent state | Server |
| `Accumulator` | Computes H, ΔH, Δ²H, κ from logits | Per-request |
| `StructuralLaws` | Static methods for temperature and bias | Stateless |
| `StabilityMonitor` | Tracks rolling perplexity | Per-request |
| `Calibrator` | Maintains κ_ref, confidence, baseline | Server (shared) |
| `GeometricRequestProcessor` | Per-request logits processor | Per-request |
| `FluidGeometryLogitsProcessor` | vLLM plugin entry point | Server |

### Constants (Hardcoded)

```python
# Core constants (v2.0)
KAPPA_REF_INIT = 1.0           # Initial curvature scale
T_BASE = 1.0                   # Baseline temperature
TAU_CONFIDENCE = 10_000.0      # Tokens to 63% confidence
CALIBRATION_RATE = 0.001       # EMA rate for κ_ref
STABILITY_DECAY = 0.99         # Pullback rate
STABILITY_RECOVERY = 1.001     # Recovery rate
H_BUFFER_SIZE = 3              # Entropy history window
PERPLEXITY_WINDOW = 50         # Quality tracking window
KAPPA_MAX_RESPONSE = 5.0       # Max κ/κ_ref for temperature

# v2.1 additions
H_SMOOTHING_ALPHA = 0.1        # EMA alpha for entropy (~10 token window)
STABILITY_SIGMA_THRESHOLD = 2.0  # Trigger pullback at 2σ deviation
T_RESPONSE_SCALE = 0.6         # Scale down temperature response (tuned)
```

### Execution Flow (Per Token)

In `GeometricRequestProcessor.__call__()`:

1. **Clone raw logits** — before any modification
2. **Get calibration state** — C, κ_ref, confidence_override from shared Calibrator
3. **Estimate token logprob** — `log(max(softmax(logits)))` as conservative estimate
4. **Compute phase state** — via Accumulator.update()
5. **Apply Law 1** — temperature = `T_BASE × (1 + C_eff × κ/κ_ref)`, divide logits
6. **Apply Law 2** — add bias to `<think>` or `</think>` token if conditions met
7. **Update Calibrator** — increment t_global, update κ_ref EMA
8. **Update StabilityMonitor** — record logprob, report perplexity if window full
9. **Persist state** — every 5000 tokens, write JSON to disk
10. **Return modified logits**

### Law 1: Temperature Implementation

```python
# v2.1 implementation (with scaling)
def temperature(kappa, kappa_ref, confidence, confidence_override, H_mean=1.0):
    C_eff = confidence * confidence_override
    kappa_normalized = kappa / max(kappa_ref, 1e-6)
    kappa_clamped = clamp(kappa_normalized, -5.0, +5.0)

    # v2.1: Scale by entropy regime
    entropy_scale = min(1.0, H_mean / 2.0)
    effective_scale = T_RESPONSE_SCALE * entropy_scale  # 0.6 * entropy_scale

    T = T_BASE * (1.0 + C_eff * effective_scale * kappa_clamped)
    return clamp(T, 0.7, 1.5)  # Tightened bounds (was 0.1, 5.0)
```

### Law 2: Think Token Bias Implementation

```python
def think_token_bias(H, kappa, confidence, confidence_override, is_thinking, ...):
    C_eff = confidence * confidence_override
    if C_eff < 0.01:
        return {}  # No intervention at low confidence

    if not is_thinking:
        if H > 3.5 and kappa > 0:  # Entry condition
            return {think_start_id: C_eff * min(kappa * 10.0, 15.0)}
    else:
        if H < 2.0 and kappa < 0:  # Exit condition
            return {think_end_id: C_eff * min(abs(kappa) * 10.0, 15.0)}
    return {}
```

---

## Deployment Details

### Files on spark-129a

```
/home/pokazge/models/
├── NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/  # Model weights
├── fluid_geometry.py                     # The processor
├── nano_v3_reasoning_parser.py           # Reasoning output parser
├── engine_state/                         # State persistence
│   └── geometric_engine_state.json
└── start_vllm_with_fluid.sh             # Startup script
```

### Docker Command Used

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

### Verified Behavior

**Engine initialization log:**
```
[GeometricEngine] Initialized. think_start=12, think_end=13, t_global=0 (loaded from zero)
```

**Startup time:** ~5 minutes (3 min model load + 2 min graph compilation)

---

## State Persistence

### File Format

`/workspace/engine_state/geometric_engine_state.json`:
```json
{
  "t_global": 15420,
  "kappa_ref": 0.342,
  "kappa_running_mean": 0.215,
  "kappa_running_var": 0.089,
  "baseline_perplexity": 12.7,
  "baseline_count": 1542,
  "confidence_override": 0.98
}
```

### Persistence Trigger

State is saved every 5000 tokens (line 505-506 in fluid_geometry.py):
```python
if self.calibrator.state.t_global % 5000 == 0:
    self.calibrator.save_state()
```

### Load on Startup

Calibrator attempts to load existing state in `__init__` (lines 391-401). If file missing or corrupt, starts fresh.

---

## Confidence Ramp

| t_global | Confidence C | Behavior |
|----------|--------------|----------|
| 0 | 0.00 | Pure observation, no intervention |
| 1,000 | 0.10 | Very faint intervention |
| 5,000 | 0.39 | Moderate intervention |
| 10,000 | 0.63 | Near-operational |
| 30,000 | 0.95 | Fully converged |

Formula: `C = 1 - exp(-t_global / 10000)`

---

## Stability Self-Healing

### v2.0 (had issues)
Used hard 5% threshold: `baseline × 1.05`. Failed because baseline was artificially low.

### v2.1 (current)
Uses statistical drift detection:

1. **Warmup** (C < 0.1): Collect baseline perplexity from unmodified generation
2. **Track variance**: Maintain EMA of rolling perplexity mean and variance
3. **Drift detection**: Trigger pullback only when `rolling_ppl > mean + 2σ`
4. **Pullback**: If drift detected, `confidence_override *= 0.99` per report
5. **Recovery**: If within bounds, `confidence_override *= 1.001` per report

This adapts to the actual noise level rather than an arbitrary fixed threshold. The engine now correctly self-heals without false positives.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Failed to load LogitsProcessor` | File not mounted | Check `-v` mount path |
| `Could not resolve <think>` | Wrong model | Use Nemotron or DeepSeek-R1 |
| Engine always starts at t=0 | State not persisting | Check engine_state volume mount |
| confidence_override < 0.5 | Quality degradation detected | Wait for self-recovery or delete state file |
| Temperature oscillating wildly | v2.0 code (no smoothing) | Upgrade to v2.1 |
| Engine disabled (co→0.06) | v2.0 hard threshold | Upgrade to v2.1 |

### Diagnostic Trace

Trace logging is **DISABLED by default** to avoid disk space usage.

Enable with `FG_TRACE=1` environment variable:
```bash
docker run -e FG_TRACE=1 ...
```

Trace output: `/workspace/engine_state/trace.log`
- Max size: 10MB (auto-truncates to 50% when exceeded)
- Format: `t=<token> H=<entropy> dH=<velocity> d2H=<accel> k=<curvature> T=<temp> C=<conf> co=<override>`

```
t=31685 H=0.378 dH=0.077 d2H=0.089 k=0.0830 T=1.061 C=0.958 co=1.000
```

**Healthy indicators (v2.2)**:
- T varies smoothly within 0.93-1.08 (not oscillating wildly)
- co stays near 1.0
- κ values are small and smooth (not flipping sign every token)
- No floor/ceiling hits (T=0.7 or T=1.5)

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `fluid_geometry.py` | The implementation (~650 lines) |
| `validate.py` | Local syntax/structure checker |
| `deploy.sh` | Deployment script for spark-129a |
| `README.md` | User-facing documentation |
| `IMPLEMENTATION.md` | This file — implementation details |
| `GEOMETRIC_ENGINE_SPEC.md` | Original design specification |
| `DEV_AGENT_TASKS.md` | Validation task checklist |
| `PATCH_INSTRUCTIONS.md` | v2.1 fix instructions |
| `VALIDATION_RESULTS.md` | v2.0 validation findings |
| `VALIDATION_RESULTS_V2.1.md` | v2.1 validation results |
