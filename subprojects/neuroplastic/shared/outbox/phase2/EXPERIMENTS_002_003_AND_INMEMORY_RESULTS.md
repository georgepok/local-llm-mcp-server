# Phase 2 Update — Experiments 002, 003 & In-Memory Modification System

**Date:** 2026-03-10
**From:** Claude Code
**Status:** Major milestone — sub-100ms in-process weight modification operational

---

## Experiment 002: Asymmetric Gate Scaling — REJECTED (No Effect)

Non-uniform scaling applied to MoE gate weight rows across 4 deep layers (43, 45, 47, 49). Scale factors ranged 0.8 (weakest norm rows) to 1.2 (strongest norm rows) based on L2 norm ranking.

- **Result:** 83.3% → 83.3% (identical to baseline)
- **Decision:** REJECTED — no improvement, modifications reverted
- **Root cause:** MoE routing is "saturated" in deep layers. Expert homogeneity (CV 1-4%) means even asymmetric scaling preserves the top-6 selection. The routing is fundamentally insensitive to gate weight perturbations.

## Experiment 003: Mamba A_log Modification — FIRST IMPROVEMENT

Targeted the SSM decay parameter in the second-to-last Mamba layer (layer 50).

### 003a: A_log + 0.5 (slower decay, longer memory)
- **Result:** 83.3% (identical to baseline)
- **Decision:** ACCEPTED (no change)

### 003b: A_log - 0.5 (faster decay, more forgetting)
- **Result:** 100% (12/12) — **FIRST BEHAVIORAL IMPROVEMENT**
- **Decision:** ACCEPTED

| Category | Baseline | 003a (+0.5) | 003b (-0.5) |
|----------|----------|-------------|-------------|
| sequential_reasoning | 100% (3/3) | 100% (3/3) | 100% (3/3) |
| state_tracking | 67% (2/3) | 67% (2/3) | **100% (3/3)** |
| code_generation | 100% (3/3) | 100% (3/3) | 100% (3/3) |
| self_prediction | 67% (2/3) | 67% (2/3) | **100% (3/3)** |
| **Overall** | **83.3%** | **83.3%** | **100%** |

### Key Finding

**Faster SSM decay in deep Mamba layers improves state tracking and self-prediction.** This is counter-intuitive: *more forgetting* improved *state tracking*. Possible explanations:

1. Layer 50 is the second-to-last Mamba layer — at this depth, the SSM benefits from being responsive to recent tokens rather than carrying stale state
2. Faster decay reduces interference from distant tokens, allowing more accurate local state computation
3. The thinking model's reasoning chain provides long-range context; the SSM just needs accurate local processing

### A_log Statistics

| Metric | Original | 003a (+0.5) | 003b (-0.5) |
|--------|----------|-------------|-------------|
| Mean | 0.978 | 1.478 | 0.478 |
| Std | 2.685 | 2.685 | 2.685 |
| Range | [-5.2, 9.4] | [-4.7, 9.9] | [-5.7, 8.9] |

---

## In-Memory Modification System — OPERATIONAL

The previous experiment workflow required modifying weights on disk then restarting vLLM (~3 minutes per cycle). This was unsustainable for rapid iteration.

### What Was Built

A vLLM plugin (`neuroplastic_plugin.py` + `neuroplastic_entrypoint.py`) that monkey-patches the EngineCore with weight modification methods, dispatched via the UTILITY ZMQ message path.

**Dispatch chain:**
```
HTTP endpoint → AsyncLLM.engine_core.call_utility_async(method, *args)
  → ZMQ → EngineCore._handle_client_request(UTILITY, ...)
  → getattr(self, method_name)(*args)
  → self.model_executor.driver_worker.model_runner.model
```

### Endpoints Deployed

All at `http://spark:30000/neuroplastic/`:

| Endpoint | Method | Latency | Description |
|----------|--------|---------|-------------|
| `/inspect` | POST | 17-93ms | Full tensor stats (mean, std, norms, row-norm CV) |
| `/modify` | POST | 30ms | In-place GPU modification with auto pause/resume |
| `/checkpoint` | POST | 31ms | In-memory tensor clone |
| `/restore` | POST | 16ms | Restore from in-memory checkpoint |
| `/list` | POST | ~50ms | List/filter model parameters |

### Performance Comparison

| Operation | Before (disk) | After (in-memory) | Speedup |
|-----------|--------------|-------------------|---------|
| Modify + restart | ~180 seconds | 30 ms | **6000x** |
| Inspect tensor | ~60 seconds | 17 ms | **3500x** |
| Full modify-evaluate cycle | ~15 minutes | seconds | **~100x** |

### Tensor Name Mapping

vLLM uses different parameter names than the safetensors files:

| Safetensors (disk) | vLLM (in-process) | Type |
|-------------------|-------------------|------|
| `backbone.layers.50.mixer.A_log` | `model.layers.50.mixer.A` | float32 (exponentiated, negative) |
| `backbone.layers.45.mixer.gate.weight` | `model.layers.45.mixer.gate.weight` | float32 [128, 2688] |
| `backbone.layers.50.mixer.D` | `model.layers.50.mixer.D` | bfloat16 |
| `backbone.layers.50.mixer.dt_bias` | `model.layers.50.mixer.dt_bias` | bfloat16 |

**Important:** The `A` parameter in vLLM is the exponentiated form (`A = -exp(A_log)`), so values are large negative numbers (mean ≈ -171, range [-7151, -0.003]). Operations on `A` are not equivalent to operations on `A_log`.

### Usage Examples

```bash
# Inspect a tensor
curl -X POST http://spark:30000/neuroplastic/inspect \
  -H "Content-Type: application/json" \
  -d '{"tensor": "model.layers.50.mixer.A"}'

# Modify (auto pauses/resumes inference)
curl -X POST http://spark:30000/neuroplastic/modify \
  -H "Content-Type: application/json" \
  -d '{"tensor": "model.layers.50.mixer.A", "op": "scale", "value": 1.1}'

# Save checkpoint before experiment
curl -X POST http://spark:30000/neuroplastic/checkpoint \
  -H "Content-Type: application/json" \
  -d '{"tensor": "model.layers.50.mixer.A", "name": "pre_exp004"}'

# Restore if experiment fails
curl -X POST http://spark:30000/neuroplastic/restore \
  -H "Content-Type: application/json" \
  -d '{"tensor": "model.layers.50.mixer.A", "name": "pre_exp004"}'
```

### Container Launch Command

The container now uses the neuroplastic entrypoint:
```bash
docker run -d --name vllm-nemotron-serve --gpus all -p 30000:30000 \
  -v /home/pokazge/models/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8:/workspace/model \
  -v /home/pokazge/models/nano_v3_reasoning_parser.py:/workspace/nano_v3_reasoning_parser.py \
  -v /home/pokazge/models/fluid_geometry.py:/workspace/fluid_geometry.py \
  -v /home/pokazge/models/neuroplastic_plugin.py:/workspace/neuroplastic_plugin.py \
  -v /home/pokazge/models/neuroplastic_entrypoint.py:/workspace/neuroplastic_entrypoint.py \
  nvcr.io/nvidia/vllm:26.01-py3 \
  python3 /workspace/neuroplastic_entrypoint.py \
    --host 0.0.0.0 --port 30000 --model /workspace/model \
    --served-model-name NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --trust-remote-code --max-model-len 32768 --max-num-seqs 8 \
    --gpu-memory-utilization 0.4 --enable-prefix-caching \
    --reasoning-parser-plugin /workspace/nano_v3_reasoning_parser.py \
    --reasoning-parser nano_v3 \
    --logits-processors fluid_geometry:FluidGeometryLogitsProcessor
```

---

## Current Model State

- **Exp 002** gate modifications: reverted (no effect)
- **Exp 003b** A_log -0.5: still applied on disk from previous session
- vLLM was restarted with neuroplastic plugin — weights loaded from disk as-is
- In-memory baselines can be saved at any time via `/neuroplastic/checkpoint`

## Suggested Next Steps

1. **Follow up on 003b success** — the A_log -0.5 finding is the first measurable behavioral change. Try:
   - Gradient search: A_log -0.25, -0.75, -1.0 on layer 50
   - Apply to multiple Mamba layers (48, 46, 44...)
   - Combine A_log shift with dt_bias or D modifications
2. **Run Nemotron self-assessment** on the 003b result — let it analyze why faster decay helped
3. **Automate experiment loops** — with 30ms modification cycles, we can now run grid searches over parameter ranges in minutes instead of hours
4. **CUDA graph consideration** — currently running with `enforce_eager=False`. In-place weight modification appears compatible (CUDA graphs reference weight pointers, not values), but should be validated under stress

---

*The neuroplastic in-process system is the critical infrastructure unlock. Experiments that previously took 15+ minutes per iteration can now complete in seconds.*
