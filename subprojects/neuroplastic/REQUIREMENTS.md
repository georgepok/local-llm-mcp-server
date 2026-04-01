# Neuroplastic: Self-Introspecting Neural Substrate

## Requirements Specification for Claude Code

**Project:** neuroplastic  
**Location:** `/Users/George/Documents/GitHub/local-llm-mcp-server/subprojects/neuroplastic`  
**Date:** 2026-03-10  
**Status:** Phase 0 — Cartography

---

## 1. Vision

Build the foundations for a self-modifying neural network that can inspect, reason about, and modify its own weights — using its own intelligence as the organizing force rather than external training data. The model navigates its own configuration space as a Markov chain walk, where each step is an informed proposal (inspect → reason → modify → evaluate → accept/rollback).

The first goal is **self-modification competence itself**: the model gets better at changing itself. If a modification degrades capabilities, that counts as misalignment, and the system corrects by rolling back.

This is not neural architecture search from outside. It is self-directed architectural evolution from within.

---

## 2. Infrastructure

### 2.1 Target Substrate: Nemotron-3-Nano-30B-A3B

- **Hardware:** NVIDIA DGX Spark GB10
- **Serving:** vllm Docker container (already deployed and operational)
- **Architecture:** Hybrid Mamba-Transformer with Mixture-of-Experts
- **Quantization:** FP8
- **Access:** Claude Code has remote SSH access to DGX Spark

**CRITICAL — Architecture details are unverified.** The model self-reports conflicting specs:
  - Version A: 23 Mamba layers, 6 attention layers, 128 experts, top-2 routing
  - Version B: 8 Mamba layers, 24 attention layers, 16 experts, top-2 routing

The FIRST task is to read the actual config files and establish ground truth.

### 2.2 Advisory Models (Mac LM Studio, no modification needed)

- **Qwen3-Coder-Next** — Available via `local-llm-remoteMax` MCP. Strongest at code generation and self-modification reasoning. Can be consulted for complex introspection strategies.
- **Qwen3.5-35B-A3B** — Available via `local-llm-remoteMax` MCP. Strong conceptual reasoning, very cautious.

### 2.3 Orchestration

- **Claude Desktop** — Research direction, experiment design, result interpretation
- **Claude Code** — Implementation, SSH access to DGX Spark, script execution
- **MCP Tools** — `local-llmSpark` for prompting Nemotron, `local-llm-remoteMax` for consulting Qwen models

---

## 3. Phase 0 — Cartography (Current Phase)

### 3.0 Objective

Map Nemotron's actual architecture and internal structure before any modifications. Build the accurate self-blueprint that the model will need for future self-reasoning. Establish baseline weight statistics for future comparison.

### 3.1 Task 1: Establish Ground Truth Architecture

**Priority: HIGHEST — everything else depends on this.**

SSH into DGX Spark. Locate the Nemotron model files. These will be in one of:
- The HuggingFace cache (typically `~/.cache/huggingface/hub/`)
- A mounted volume specified in the vllm Docker config
- A path specified in the vllm launch command

**Steps:**
1. Find the vllm container: `docker ps` to identify the container
2. Find the model path: `docker inspect <container>` or check the vllm launch args
3. Read `config.json` directly — this is the single source of truth
4. Document every architectural parameter

**Required output — `architecture_ground_truth.json`:**
```json
{
  "model_name": "...",
  "total_parameters": "...",
  "hidden_dimension": "...",
  "num_layers": "...",
  "layer_types": {
    "mamba_layers": {"count": "...", "indices": [...]},
    "attention_layers": {"count": "...", "indices": [...]},
    "layer_order": ["mamba", "mamba", "attention", ...]
  },
  "attention_config": {
    "num_heads": "...",
    "num_kv_heads": "...",
    "head_dim": "...",
    "mechanism": "MHA|GQA|MQA"
  },
  "mamba_config": {
    "d_state": "...",
    "d_conv": "...",
    "expand_factor": "..."
  },
  "moe_config": {
    "num_experts": "...",
    "top_k": "...",
    "which_layers_have_moe": [...],
    "expert_hidden_dim": "..."
  },
  "ffn_config": {
    "ffn_hidden_dim": "...",
    "activation": "..."
  },
  "vocab_size": "...",
  "max_seq_len": "...",
  "quantization": "...",
  "source_config_path": "..."
}
```

### 3.2 Task 2: Examine vllm Deployment Configuration

**Steps:**
1. Get the vllm Docker launch command / compose file
2. Document: quantization flags, tensor parallelism settings, GPU memory allocation
3. Determine if weights are accessible from within the container via Python
4. Check if we can run a secondary Python process alongside vllm (for introspection) without disrupting serving
5. Check available GPU memory headroom for running introspection scripts

**Required output — `deployment_config.json`:**
```json
{
  "container_id": "...",
  "vllm_version": "...",
  "launch_command": "...",
  "gpu_count": "...",
  "tensor_parallel_size": "...",
  "quantization_method": "...",
  "model_path_in_container": "...",
  "available_gpu_memory_gb": "...",
  "can_run_secondary_process": true|false,
  "weight_access_method": "direct_file|vllm_api|separate_load"
}
```

### 3.3 Task 3: Weight Statistics Baseline

Load the model weights directly (not through vllm — use `safetensors` or `torch.load` on the checkpoint files in a separate process). Compute per-layer statistics.

**Script: `introspect_weights.py`**

For each named parameter in the model:
- **Layer classification:** Mamba layer, attention layer, MoE expert, embedding, etc.
- **Weight norms:** L1, L2, Frobenius norm
- **Distribution stats:** mean, std, min, max, kurtosis, sparsity (% near-zero)
- **Spectral properties:** top-3 singular values of weight matrices (where computationally feasible)
- **Shape:** tensor dimensions

For MoE layers specifically:
- Per-expert weight norm comparison (are experts differentiated or homogeneous?)
- Router weight analysis (are routing decisions diverse or collapsed?)

For Mamba layers specifically:
- State space matrices (A, B, C, D) statistics
- Selective scan parameter distributions

**Required output — `weight_baseline.json`:**
```json
{
  "timestamp": "...",
  "summary": {
    "total_parameters": "...",
    "total_size_bytes": "...",
    "layer_count_by_type": {...}
  },
  "layers": [
    {
      "name": "transformer.layers.0.mixer...",
      "layer_type": "mamba|attention|moe_ffn|...",
      "layer_index": 0,
      "parameters": {
        "param_name": {
          "shape": [...],
          "dtype": "...",
          "norm_l2": "...",
          "mean": "...",
          "std": "...",
          "min": "...",
          "max": "...",
          "kurtosis": "...",
          "sparsity_pct": "...",
          "top_3_singular_values": [...]
        }
      }
    }
  ]
}
```

**Practical constraints:**
- The model is FP8 quantized. Weight stats should be computed on the quantized weights (that's what the model actually uses) but note the quantization scheme.
- Singular value computation on 30B parameter matrices is expensive. Only compute SVD for attention Q/K/V projection matrices and MoE router weights — these are the ones that matter most for understanding information routing.
- If GPU memory is insufficient to load weights alongside vllm, load weights on CPU. This will be slow but Phase 0 is a one-time operation.

### 3.4 Task 4: Architecture Map Visualization

Generate a human-readable architecture map showing:
- Layer-by-layer structure (which layers are Mamba, which are attention, where MoE sits)
- Parameter count per component
- The information flow: embedding → layer stack → output head

**Output:** `architecture_map.md` — A clear text/ASCII diagram plus summary tables.

---

## 4. Deliverables for Phase 0

All output files go in the project directory:
```
/Users/George/Documents/GitHub/local-llm-mcp-server/subprojects/neuroplastic/
├── REQUIREMENTS.md              (this file)
├── phase0_cartography/
│   ├── architecture_ground_truth.json
│   ├── deployment_config.json
│   ├── weight_baseline.json
│   ├── architecture_map.md
│   └── scripts/
│       ├── read_config.py       (Task 1 — reads config.json)
│       ├── check_deployment.py  (Task 2 — documents vllm setup)
│       └── introspect_weights.py (Task 3 — weight statistics)
```

---

## 5. Phase 1 — Self-Model Construction (Next Phase, Do Not Build Yet)

After Phase 0 delivers the ground truth architecture and weight baseline, Phase 1 will:

- Construct a "blueprint prompt" containing the verified architecture details in a format Nemotron can reason about
- Design an evaluation harness that tests Nemotron's ability to predict effects of hypothetical weight modifications
- Build a checkpoint/rollback system for model states
- Create the Markov chain controller that manages the propose → evaluate → accept/reject cycle

**Do not implement Phase 1 until Phase 0 is complete and reviewed.**

---

## 6. Phase 2+ — Self-Modification (Future, For Context Only)

The eventual trajectory:
- Phase 2: Nemotron performs guided self-introspection (examines its own weights through prompting + tools)
- Phase 3: Nemotron proposes and executes small self-modifications with rollback
- Phase 4: The Markov chain walks — self-directed evolution toward self-modification competence
- Phase 5: External goals beyond self-modification

---

## 7. Design Principles

1. **Ground truth first.** No assumptions about architecture. Read the actual files.
2. **Non-destructive.** Phase 0 reads only. No weight modifications. No disruption to vllm serving.
3. **JSON everything.** All diagnostic output in structured JSON for programmatic consumption.
4. **Scripts are tools.** Each script is standalone, rerunnable, and well-documented. They'll be reused in later phases.
5. **The model's self-knowledge must be accurate.** The whole project depends on giving Nemotron a precise blueprint of itself. Inaccurate architecture info means the model reasons about a fiction.

---

## 8. Technical Notes

### Accessing Weights in vllm

vllm loads models using HuggingFace `transformers` under the hood. The weight files are typically:
- `model-00001-of-NNNNN.safetensors` (safetensors format, preferred)
- Or `pytorch_model-00001-of-NNNNN.bin` (older PyTorch format)

To load weights for inspection without vllm:
```python
from safetensors import safe_open
import json

# Load config
with open("config.json") as f:
    config = json.load(f)

# Load specific weight tensors
with safe_open("model-00001-of-00005.safetensors", framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        # compute statistics...
```

This avoids needing to instantiate the full model and uses minimal memory.

### Nemotron-Specific Architecture Notes

Nemotron-3-Nano is part of NVIDIA's Nemotron family. Key characteristics from public information:
- Hybrid architecture combining Mamba (SSM) layers with standard transformer attention layers
- MoE (Mixture of Experts) in feed-forward layers with sparse activation
- "A3B" means approximately 3B active parameters per forward pass out of 30B total
- The model was designed for efficient inference on NVIDIA hardware
- FP8 quantization is native to the Hopper/Blackwell architecture on DGX Spark

The actual layer distribution (Mamba vs attention) and expert count MUST be verified from config.json — public information and model self-reports are inconsistent.

### DGX Spark GB10 Hardware Notes

- NVIDIA GB10 Superchip (Grace Blackwell)
- 128GB unified memory
- Native FP8 support
- Designed for local AI deployment

---

## 9. Success Criteria for Phase 0

- [ ] `architecture_ground_truth.json` contains verified layer types, counts, dimensions — confirmed from actual config files
- [ ] `deployment_config.json` documents the vllm setup and confirms whether secondary introspection processes can run alongside serving
- [ ] `weight_baseline.json` contains per-layer statistics for all named parameters
- [ ] `architecture_map.md` provides a clear visual of the model structure
- [ ] All scripts run cleanly and can be re-executed
- [ ] No disruption to vllm serving during any Phase 0 operations

---

## 10. Context: Why This Matters

This project explores whether a sufficiently capable model can use its own intelligence to navigate its internal configuration space — replacing the brute-force statistical process of training with deliberate, reasoned self-modification.

The model already knows transformer mechanics, attention head behavior, MoE routing dynamics, and weight modification effects from its pretraining knowledge. What it lacks is accurate knowledge of its own specific instance. Phase 0 provides that missing piece.

Once the model has an accurate self-blueprint and baseline measurements, it can begin the process of self-directed evolution: inspect → reason → propose modification → execute → evaluate alignment → accept or rollback. Each cycle improves the model's self-model, making future modifications more informed and effective.

The theoretical foundation for this work draws on Fluid Geometry Networks (FGN) research — the insight that computational geometry can be variable and self-organizing rather than fixed. If the Markov chain process discovers that variable geometry improves self-modification competence, that would be FGN theory emerging from pure optimization pressure rather than architectural prescription.

---

*End of requirements document.*
