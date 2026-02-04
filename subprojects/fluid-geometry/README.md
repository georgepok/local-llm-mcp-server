# Fluid Geometry LogitsProcessor

Entropy-driven dynamic reasoning control for vLLM hybrid models.

## Overview

FluidGeometry implements an adaptive "thinking budget" that monitors Shannon entropy during token generation. Instead of fixed reasoning (always on or always off), the model dynamically switches between:

- **Flow Mode** (Curved/Mamba): Direct sequential generation for confident predictions
- **Thinking Mode** (Flat/Attention): Deliberative reasoning when uncertainty is detected

## How It Works

```
                    ┌─────────────────┐
                    │  Token Logits   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Calculate Entropy│
                    │ H = -Σ(p·log(p))│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       H > HIGH_THRESHOLD    │    H < LOW_THRESHOLD
       (Confused)            │    (Confident)
              │              │              │
              ▼              ▼              ▼
        Boost <think>    No Change    Boost </think>
        (Enter Thinking)             (Exit Thinking)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HIGH_ENTROPY_THRESHOLD` | 4.5 | Entropy level to trigger thinking |
| `LOW_ENTROPY_THRESHOLD` | 1.5 | Entropy level to collapse thinking |
| `GEOMETRY_BIAS` | 15.0 | Logit boost magnitude (soft nudge) |
| `THINK_START_TOKEN` | `<think>` | Token to enter reasoning mode |
| `THINK_END_TOKEN` | `</think>` | Token to exit reasoning mode |

### Tuning Guidelines

- **More thinking**: Lower `HIGH_ENTROPY_THRESHOLD` (e.g., 3.0)
- **Less thinking**: Raise `HIGH_ENTROPY_THRESHOLD` (e.g., 6.0)
- **Longer thinking**: Lower `LOW_ENTROPY_THRESHOLD` (e.g., 0.8)
- **Shorter thinking**: Raise `LOW_ENTROPY_THRESHOLD` (e.g., 2.5)
- **Stronger switching**: Increase `GEOMETRY_BIAS` (e.g., 50.0)
- **Softer nudges**: Decrease `GEOMETRY_BIAS` (e.g., 5.0)

## Installation

### Prerequisites

- vLLM 0.13+ with v1 engine
- Model with `<think>`/`</think>` tokens in vocabulary (e.g., Nemotron, DeepSeek-R1)

### Deployment Steps

1. **Copy processor to server:**
   ```bash
   scp fluid_geometry.py user@server:~/models/
   ```

2. **Start vLLM with processor:**
   ```bash
   docker run -d \
     --name vllm-server \
     --gpus all \
     -p 30000:30000 \
     -v ~/models/your-model:/workspace/model \
     -v ~/models/fluid_geometry.py:/workspace/fluid_geometry.py \
     nvcr.io/nvidia/vllm:26.01-py3 \
     python3 -m vllm.entrypoints.openai.api_server \
       --host 0.0.0.0 \
       --port 30000 \
       --model /workspace/model \
       --trust-remote-code \
       --logits-processors fluid_geometry:FluidGeometryLogitsProcessor
   ```

3. **Verify:**
   ```bash
   curl http://localhost:30000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "your-model", "messages": [{"role": "user", "content": "Test"}]}'
   ```

## API Behavior

### Request Format (Standard OpenAI)
```json
{
  "model": "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
  "messages": [{"role": "user", "content": "Your question"}],
  "max_tokens": 500,
  "temperature": 0.7
}
```

### Response Format
```json
{
  "choices": [{
    "message": {
      "content": "Final answer",
      "reasoning": "Thinking process (if triggered)",
      "reasoning_content": "Same as reasoning"
    }
  }]
}
```

## Observed Behavior

| Query Type | Entropy | Behavior |
|------------|---------|----------|
| Simple ("What is 2+2?") | Low | Direct answer, no thinking |
| Ambiguous | High | Triggers thinking, explores options |
| Multi-step reasoning | Variable | May pulse in/out of thinking |
| Novel/unusual | High | Extended deliberation |

## Files

- `fluid_geometry.py` - Main processor implementation
- `README.md` - This specification
- `deploy.sh` - Deployment script for spark-129a

## Architecture

```
FluidGeometryLogitsProcessor (vLLM v1 interface)
    └── AdapterLogitsProcessor (base class)
            └── FluidGeometryRequestProcessor (per-request logic)
                    ├── _calculate_entropy()
                    ├── _is_thinking()
                    └── __call__() → modified logits
```

## License

MIT - Part of local-llm-mcp-server project.
