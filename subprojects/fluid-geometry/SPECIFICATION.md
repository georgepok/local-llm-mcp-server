# Fluid Geometry LogitsProcessor - Technical Specification

## Document Info
- **Version**: 1.0.0
- **Date**: 2026-02-04
- **Status**: Implemented and Deployed
- **Target**: spark-129a.local (NVIDIA DGX Spark)

---

## 1. Objective

Implement a dynamic control loop that modulates reasoning depth ("Thinking Budget") in real-time using Shannon Entropy as a proxy for model uncertainty. The system triggers reasoning tokens only when the probability distribution indicates confusion.

## 2. Theoretical Foundation

### 2.1 Geometry Metaphor

- **Curved Geometry (Flow Mode)**: Sequential token generation via Mamba layers. Efficient for confident, low-entropy predictions.
- **Flat Geometry (Thinking Mode)**: Deliberative reasoning via Attention layers. Activated when entropy indicates uncertainty.

### 2.2 Shannon Entropy

Entropy measures uncertainty in the probability distribution:

```
H(X) = -Σ P(xᵢ) · log(P(xᵢ))
```

- **Low entropy** (H < 1.5): Model is confident → stay in flow mode
- **High entropy** (H > 4.5): Model is confused → switch to thinking mode

### 2.3 Hysteretic Control Loop

The system uses hysteresis to prevent oscillation:

```
State: FLOW                           State: THINKING
  │                                       │
  │ H > HIGH_THRESHOLD                    │ H < LOW_THRESHOLD
  │ ─────────────────►                    │ ─────────────────►
  │   Boost <think>                       │   Boost </think>
  │                                       │
  ◄────────────────────────────────────────
            Middle zone: no action
```

## 3. Implementation Details

### 3.1 Class Hierarchy

```
vllm.v1.sample.logits_processor.LogitsProcessor (ABC)
    │
    └── vllm.v1.sample.logits_processor.AdapterLogitsProcessor
            │
            └── FluidGeometryLogitsProcessor
                    │
                    └── FluidGeometryRequestProcessor (per-request)
```

### 3.2 Key Methods

#### FluidGeometryLogitsProcessor.__init__
```python
def __init__(self, vllm_config, device, is_pin_memory):
    # Load tokenizer from model path
    # Resolve <think> and </think> token IDs
    # Initialize parent AdapterLogitsProcessor
```

#### FluidGeometryRequestProcessor.__call__
```python
def __call__(self, prompt_token_ids, output_token_ids, logits):
    # 1. Determine if currently in thinking mode (scan for open <think>)
    # 2. Calculate entropy of current logit distribution
    # 3. Apply geometry switching:
    #    - If FLOW and H > HIGH: boost <think> token
    #    - If THINKING and H < LOW: boost </think> token
    # 4. Return modified logits
```

### 3.3 Configuration Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `HIGH_ENTROPY_THRESHOLD` | 4.5 | Typical "confused" entropy for 30B+ models |
| `LOW_ENTROPY_THRESHOLD` | 1.5 | Typical "confident" entropy |
| `GEOMETRY_BIAS` | 15.0 | Soft nudge (not hard switch) |
| `THINK_START_TOKEN` | `<think>` | Standard reasoning token |
| `THINK_END_TOKEN` | `</think>` | Standard reasoning close |

### 3.4 Entropy Calculation

```python
def _calculate_entropy(self, logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-9)  # Avoid log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.item()
```

### 3.5 Thinking State Detection

```python
def _is_thinking(self, tokens: list[int]) -> bool:
    # Scan backwards for efficiency
    for token in reversed(tokens):
        if token == self.think_end_id:
            return False  # Closed
        if token == self.think_start_id:
            return True   # Open
    return False
```

## 4. vLLM Integration

### 4.1 Plugin Loading

vLLM loads the processor via fully-qualified class name (FQCN):

```
--logits-processors fluid_geometry:FluidGeometryLogitsProcessor
```

This triggers:
1. `importlib.import_module("fluid_geometry")`
2. `getattr(module, "FluidGeometryLogitsProcessor")`
3. Validation that class is subclass of `LogitsProcessor`

### 4.2 Lifecycle

```
Server Start
    │
    ▼
build_logitsprocs() called
    │
    ▼
FluidGeometryLogitsProcessor.__init__(vllm_config, device, is_pin_memory)
    │
    ▼
For each request:
    │
    ├── new_req_logits_processor(params) → FluidGeometryRequestProcessor
    │
    └── On each token:
            │
            └── processor.__call__(prompt_ids, output_ids, logits)
```

### 4.3 Required vLLM Version

- **Minimum**: vLLM 0.13.0
- **Tested**: vLLM 0.13.0+faa43dbf.nv26.01 (NVIDIA container)

## 5. Deployment

### 5.1 File Locations (spark-129a)

```
/home/pokazge/models/
├── NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/  # Model weights
├── fluid_geometry.py                     # This processor
├── nano_v3_reasoning_parser.py           # Reasoning output parser
└── start_vllm_with_fluid.sh             # Startup script
```

### 5.2 Docker Configuration

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

### 5.3 Startup Time

- Model loading: ~3 minutes (9 safetensor shards)
- Graph compilation: ~2 minutes
- Total: ~5 minutes until API ready

## 6. Verification

### 6.1 Simple Query (Should NOT trigger thinking)

```bash
curl -s http://spark-129a.local:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
       "messages": [{"role": "user", "content": "What is 2+2?"}],
       "max_tokens": 100}'
```

**Expected**: Direct answer without `<think>` tags, `reasoning` field empty.

### 6.2 Complex Query (Should trigger thinking)

```bash
curl -s http://spark-129a.local:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
       "messages": [{"role": "user", "content": "Explain the paradox where a circle is seen as a spiral."}],
       "max_tokens": 500}'
```

**Expected**: Response with populated `reasoning` and `reasoning_content` fields.

## 7. Troubleshooting

### 7.1 Processor Not Loading

**Error**: `Failed to load LogitsProcessor plugin fluid_geometry:FluidGeometryLogitsProcessor`

**Solution**: Ensure file is mounted at `/workspace/fluid_geometry.py` in container.

### 7.2 Token IDs Not Found

**Error**: `Could not resolve token IDs for <think> and </think>`

**Solution**: Model tokenizer doesn't have these tokens. Use a model with reasoning tokens (Nemotron, DeepSeek-R1, etc.).

### 7.3 Too Much/Little Thinking

**Symptom**: Model thinks on everything or never thinks.

**Solution**: Adjust thresholds in `fluid_geometry.py`:
```python
HIGH_ENTROPY_THRESHOLD = 4.5  # Raise to reduce thinking
LOW_ENTROPY_THRESHOLD = 1.5   # Raise to shorten thinking
```

## 8. Future Improvements

1. **Per-request configuration**: Allow clients to specify thresholds via API
2. **Entropy logging**: Add metrics endpoint for monitoring entropy distribution
3. **Adaptive thresholds**: Learn optimal thresholds from feedback
4. **Multi-token lookahead**: Consider entropy trends, not just current step

## 9. References

- Original specification: User-provided FluidLogitsProcessor design document
- vLLM LogitsProcessor interface: `vllm/v1/sample/logits_processor/interface.py`
- Shannon entropy: Information Theory fundamentals
- Nemotron Nano architecture: NVIDIA hybrid Mamba-Attention model

---

## Appendix A: Complete Source Code

See `fluid_geometry.py` in this directory.

## Appendix B: Test Results

### Test 1: Simple Query
- **Input**: "What is 2+2?"
- **Entropy observed**: < 1.5 (low)
- **Thinking triggered**: No
- **Output**: "The answer to **2 + 2** is **4**."

### Test 2: Complex Query
- **Input**: "Explain the paradox where a circle is seen as a spiral."
- **Entropy observed**: > 4.5 (high)
- **Thinking triggered**: Yes
- **Output**: Extended reasoning exploring multiple interpretations
