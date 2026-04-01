# Fluid Geometry — Geometric Engine v2

Self-calibrating entropy-curvature reasoning control for vLLM hybrid models.

## Overview

Geometric Engine v2 implements adaptive "thinking budget" control using phase space dynamics. Unlike v1's simple thresholds, v2:

- **Measures multi-dimensional phase space** Φ(t) = (H, ΔH, Δ²H, κ)
- **Computes scalar curvature** κ(t) from entropy derivatives
- **Self-calibrates** all reference constants from observation
- **Self-gates** intervention via confidence ramp from zero
- **Self-heals** via stability monitoring with automatic pullback

No configuration required. Engine starts cold, warms through observation, and reaches full operation autonomously.

## How It Works

```
                        Token Generation Step
                                │
                    ┌───────────▼───────────┐
                    │   Raw Logits Snapshot  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │     Phase Space Φ(t)   │
                    │  H, ΔH, Δ²H, κ, Δκ    │
                    └───────────┬───────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │  Law 1: Temp  │   │ Law 2: Think  │   │  Calibrator   │
    │ T = f(κ/κ_ref)│   │ Entry/Exit    │   │ Update κ_ref  │
    └───────┬───────┘   └───────┬───────┘   └───────────────┘
            │                   │
            └─────────┬─────────┘
                      ▼
              Modified Logits
```

### Core Concepts

| Concept | Formula | Purpose |
|---------|---------|---------|
| **Shannon Entropy** | H = -Σ(p·log(p)) | Measures uncertainty |
| **Scalar Curvature** | κ = Δ²H / (1 + \|ΔH\|) | Measures change dynamics |
| **Confidence** | C = 1 - exp(-t/τ) | Ramps from 0 to 1 |
| **Temperature** | T = T_base × (1 + C × κ/κ_ref) | Adapts exploration |

### Structural Laws

1. **Law 1 (Temperature)**: Positive κ → higher T (more exploration). Negative κ → lower T (more focus).

2. **Law 2 (Think Tokens)**:
   - Entry: H > 3.5 AND κ > 0 (getting MORE confused) → boost `<think>`
   - Exit: H < 2.0 AND κ < 0 (getting MORE confident) → boost `</think>`

## Self-Calibration

The engine calibrates itself during operation:

| Phase | t_global | Confidence | Behavior |
|-------|----------|------------|----------|
| Cold Start | 0 | 0% | Pure observation, no intervention |
| Warmup | 0-10K | 0-63% | Gradual intervention, baseline established |
| Operational | 10K-30K | 63-95% | Full geometric response |
| Steady State | 30K+ | ~100% | Stable, self-healing |

### Persistence

Engine state persists across container restarts in `/workspace/engine_state/`:
- `geometric_engine_state.json` — t_global, κ_ref, baseline perplexity, etc.

## Installation

### Prerequisites

- vLLM 0.13+ with v1 engine
- Model with `<think>`/`</think>` tokens (Nemotron, DeepSeek-R1, etc.)
- NVIDIA GPU with CUDA support

### Quick Deployment

```bash
./deploy.sh
```

### Manual Deployment

1. **Copy files to server:**
   ```bash
   scp fluid_geometry.py user@server:~/models/
   mkdir -p ~/models/engine_state
   ```

2. **Start vLLM with processor:**
   ```bash
   docker run -d \
     --name vllm-server \
     --gpus all \
     -p 30000:30000 \
     -v ~/models/your-model:/workspace/model \
     -v ~/models/fluid_geometry.py:/workspace/fluid_geometry.py \
     -v ~/models/engine_state:/workspace/engine_state \
     nvcr.io/nvidia/vllm:26.01-py3 \
     python3 -m vllm.entrypoints.openai.api_server \
       --host 0.0.0.0 \
       --port 30000 \
       --model /workspace/model \
       --trust-remote-code \
       --logits-processors fluid_geometry:FluidGeometryLogitsProcessor
   ```

3. **Verify engine loaded:**
   ```bash
   docker logs vllm-server 2>&1 | grep GeometricEngine
   # Expected: [GeometricEngine] Initialized. think_start=X, think_end=Y, t_global=0
   ```

## Architecture

```
FluidGeometryLogitsProcessor (server-lifetime)
    │
    ├── Calibrator (shared, thread-safe)
    │   ├── t_global, kappa_ref, baseline_perplexity
    │   ├── confidence_override (stability feedback)
    │   └── persistence to /workspace/engine_state/
    │
    └── creates per-request:
            │
            GeometricRequestProcessor
                ├── Accumulator (H buffer, curvature computation)
                ├── StructuralLaws (stateless functions)
                └── StabilityMonitor (rolling perplexity tracking)
```

### File Structure

```
subprojects/fluid-geometry/
├── fluid_geometry.py          # Single-file implementation (~575 lines)
├── validate.py                # Local syntax/structure validation
├── deploy.sh                  # Deployment script for spark-129a
├── README.md                  # This file
├── IMPLEMENTATION.md          # Implementation details and decisions
└── GEOMETRIC_ENGINE_SPEC.md   # Original design document
```

**Note on Single-File vs Package**: The design spec (GEOMETRIC_ENGINE_SPEC.md §3.1) proposed a multi-file package structure. The implementation uses a single file per the spec's alternative: *"If vLLM requires a single file, collapse all modules into one."* This was chosen for simpler deployment and confirmed v1 compatibility. See IMPLEMENTATION.md for details.

## Monitoring

### Check Engine State

```bash
docker exec vllm-server cat /workspace/engine_state/geometric_engine_state.json
```

Example output:
```json
{
  "t_global": 15420,
  "kappa_ref": 0.342,
  "baseline_perplexity": 12.7,
  "confidence_override": 0.98
}
```

### Interpretation

| Field | Meaning |
|-------|---------|
| `t_global` | Total tokens processed (confidence ramp) |
| `kappa_ref` | Learned curvature scale (EMA of \|κ\| + 1σ) |
| `baseline_perplexity` | Reference quality from warmup period |
| `confidence_override` | Stability multiplier (1.0 = healthy, <1.0 = pulled back) |

## Comparison: v1 vs v2

| Feature | v1 | v2 |
|---------|----|----|
| Decision basis | Entropy thresholds | Entropy + curvature conjunction |
| Calibration | Manual thresholds | Self-calibrating |
| Warmup | None (instant full power) | Confidence ramp from zero |
| Quality monitoring | None | Perplexity-based stability |
| State persistence | None | JSON file across restarts |
| Temperature control | None | Curvature-responsive |

## API Usage

Standard OpenAI-compatible endpoint:

```bash
curl http://server:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    "messages": [{"role": "user", "content": "Your question"}],
    "max_tokens": 500
  }'
```

Response includes `reasoning` field when thinking was triggered.

## Troubleshooting

### Engine Not Loading
```
Error: Failed to load LogitsProcessor plugin fluid_geometry:FluidGeometryLogitsProcessor
```
Solution: Ensure `fluid_geometry.py` is mounted at `/workspace/fluid_geometry.py`

### Token IDs Not Found
```
Error: Could not resolve <think>/<think> token IDs
```
Solution: Use a model with reasoning tokens (Nemotron, DeepSeek-R1, etc.)

### Quality Degradation
If `confidence_override` drops significantly below 1.0, the stability monitor detected quality issues. The engine will self-heal over time, or you can reset by deleting the state file.

## License

MIT — Part of local-llm-mcp-server project.
