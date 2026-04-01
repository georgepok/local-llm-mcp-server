# Infrastructure Pivot: Switch from vllm to llama.cpp

**From:** Claude Desktop (Research Direction)  
**To:** Claude Code (Implementation)  
**Date:** 2026-03-10  
**Priority:** HIGH — this supersedes experiments 002/003 until complete  
**Supersedes:** `phase2_experiments_002_003.md` (pause those until switchover is validated)

---

## 1. Why We're Switching

The disk-write-restart cycle (75s per modification) is the fundamental bottleneck for the Markov chain. vllm's torch.compile + CUDA graph architecture makes in-memory weight modification complex and risky (requires CUDA graph reinit, unclear if it works reliably).

llama.cpp solves this entirely:
- No CUDA graphs, no torch.compile — weights are GGML tensors in memory
- Modify a buffer, next inference reads the new values — sub-millisecond modification cycles
- NVIDIA officially supports Nemotron-3-Nano on llama.cpp on DGX Spark
- GGUF models already available: `unsloth/Nemotron-3-Nano-30B-A3B-GGUF`
- ~65 tok/sec on DGX Spark with Q4_K_M (higher with Q8)
- OpenAI-compatible API — all existing eval harness scripts work unchanged

The full hybrid architecture (23 Mamba2 + 23 MoE + 6 Attention, MEMEM* pattern) is preserved in the GGUF conversion. We lose nothing architecturally.

---

## 2. Implementation Plan

### Step 1: Build llama.cpp on DGX Spark

SSH into Spark. Build from source with CUDA enabled for GB10:

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121" -DLLAMA_CURL=OFF
make -j8
```

Build takes ~5-10 minutes. Verify:
```bash
./bin/llama-server --version
```

### Step 2: Download GGUF Model

Use the Q8 quantization for maximum weight modification fidelity:

```bash
pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    unsloth/Nemotron-3-Nano-30B-A3B-GGUF \
    Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \
    --local-dir ~/models/nemotron3-gguf
```

~38GB download. Also download Q4 as backup:
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    unsloth/Nemotron-3-Nano-30B-A3B-GGUF \
    Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf \
    --local-dir ~/models/nemotron3-gguf
```

### Step 3: Stop vllm, Start llama.cpp Server

```bash
# Stop current vllm container
docker stop vllm-nemotron-serve

# Start llama.cpp server on the SAME port (30000) so MCP tools work unchanged
cd ~/llama.cpp/build
./bin/llama-server \
    --model ~/models/nemotron3-gguf/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \
    --host 0.0.0.0 \
    --port 30000 \
    --n-gpu-layers 99 \
    --ctx-size 8192 \
    --threads 8 \
    --jinja
```

Note: `--jinja` enables the Nemotron chat template with `<think>` reasoning support.

### Step 4: Verify Basic Serving

```bash
# Health check
curl http://localhost:30000/health

# Test inference
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "nemotron",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 100
    }'
```

Verify:
- Model loads and responds
- Thinking mode works (reasoning_content appears in responses)
- Response quality is comparable to vllm serving

### Step 5: Verify MCP Connectivity

Test that the `local-llmSpark` MCP tools still work by prompting the model through the MCP:

```
# Through your MCP client, send a test prompt
# The MCP server should connect to the same port (30000)
```

If the MCP server has the model name hardcoded (e.g., `NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`), you may need to either:
- Use `--alias NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` in the llama-server command
- Or update the MCP server config to use the new model name

Document what changes were needed.

### Step 6: Run the Eval Baseline on llama.cpp

Run the FULL evaluation harness from Phase 1 against the llama.cpp-served model:

```bash
cd /Users/George/Documents/GitHub/local-llm-mcp-server/subprojects/neuroplastic/phase1_artifacts/eval_harness
python run_eval.py
```

**CRITICAL:** We need to establish that the GGUF model produces comparable results to the FP8 vllm model. Compare:
- Sequential reasoning: was 100% on vllm
- State tracking: was 67% on vllm  
- Code generation: was 100% on vllm
- Self-prediction: was 67% on vllm
- Overall: was 83.3% on vllm

If GGUF results are significantly worse (>10% drop), consider whether Q8 is sufficient or if we need BF16. Document all differences.

Also run the blueprint verification (4 key questions) to confirm the model still reasons about itself correctly with the blueprint prompt.

### Step 7: Investigate In-Memory Weight Modification

This is the key deliverable. We need to understand how to modify GGML tensors at runtime in the llama.cpp process.

#### 7a: Explore GGUF File Structure

```bash
# Use llama.cpp's built-in tools to inspect the GGUF
cd ~/llama.cpp/build
./bin/llama-gguf-dump ~/models/nemotron3-gguf/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf --no-tensors
# Lists all metadata

./bin/llama-gguf-dump ~/models/nemotron3-gguf/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf
# Lists all tensors with shapes and quantization types
```

Document:
- What are the tensor names in GGUF format? (May differ from HuggingFace names)
- What quantization type is each tensor? (Q8_0, F32, F16, etc.)
- Which tensors are stored as F32/F16 (not quantized)?
- Specifically find: MoE gate weights, Mamba A_log/D/dt_bias, attention Q/K/V/O

#### 7b: Explore llama-server Internal APIs

llama-server may expose internal endpoints or have extension points. Check:

```bash
# List all routes
curl http://localhost:30000/
# or check the llama-server source for available endpoints
grep -r "app.post\|app.get\|@route" ~/llama.cpp/examples/server/
```

Also check if llama.cpp has any built-in weight inspection/modification APIs.

#### 7c: llama-cpp-python Approach

This may be the fastest path to in-memory modification:

```bash
pip install llama-cpp-python
```

Then test:
```python
from llama_cpp import Llama

# Load model
llm = Llama(model_path="path/to/model.gguf", n_gpu_layers=99, n_ctx=8192)

# Can we access internal model tensors?
# Explore: llm.model, llm._model, dir(llm)
# Look for tensor access APIs in the ctypes bindings
```

If llama-cpp-python exposes GGML tensor access, we can:
1. Run the model for inference via llama-cpp-python (replaces llama-server)
2. Access and modify tensors directly via Python
3. No separate server process needed — everything in one Python process
4. Modification is instant — just write to the tensor buffer

#### 7d: Memory-Mapped GGUF Approach

llama.cpp memory-maps GGUF files by default. This means:
1. The file on disk is mapped directly into process memory
2. A separate process could potentially map the same file and modify tensors
3. Changes would be visible to the llama-server process on next read

Investigate:
- Does llama-server use `mmap` for the GGUF file?
- If we modify the mapped file region from another process, does llama-server see the changes?
- Is there a synchronization concern (llama-server reading while we write)?

#### 7e: Custom llama-server Endpoint

If the above approaches don't work cleanly, the fallback is adding a custom endpoint to llama-server. The server source is in `~/llama.cpp/examples/server/server.cpp`. Check:
- Is there a hook/plugin mechanism?
- How hard would it be to add a `/modify_tensor` POST endpoint?
- Can we access the model's GGML context from the server handler?

### Step 8: Design the Modification Interface

Based on findings from Step 7, design and implement a weight modification interface. 

**The interface should support:**

```python
# Pseudocode — actual API depends on findings
interface.modify_tensor(
    tensor_name="backbone.layers.45.mixer.gate.weight",
    operation="scale",
    factor=1.1
)
# Takes effect immediately — next inference uses modified weights

interface.modify_tensor(
    tensor_name="backbone.layers.50.mixer.A_log",
    operation="add",
    value=0.5
)

interface.get_tensor_stats(tensor_name="backbone.layers.45.mixer.gate.weight")
# Returns: mean, std, norm, shape, dtype

interface.save_checkpoint(name="pre_experiment_002")
interface.restore_checkpoint(name="pre_experiment_002")
```

**Target latency:** <1 second for a single tensor modification.

---

## 3. Deliverables

```
phase2_experiments/
├── llama_cpp_migration/
│   ├── build_log.md                    (Step 1 — build output)
│   ├── serving_verification.md         (Steps 3-5 — serving works)
│   ├── eval_comparison.json            (Step 6 — vllm vs llama.cpp baseline)
│   ├── gguf_tensor_inventory.md        (Step 7a — tensor names and types)
│   ├── modification_investigation.md   (Steps 7b-7e — what works)
│   └── modification_interface.py       (Step 8 — the actual interface)
```

Report results to `shared/outbox/phase2/LLAMACPP_MIGRATION.md`.

---

## 4. What NOT to Do

- **Do NOT delete the vllm container or FP8 model files.** Keep them as fallback. Just stop the container.
- **Do NOT modify the MCP server code** unless absolutely necessary for connectivity. The goal is drop-in replacement on the same port.
- **Do NOT run experiments 002/003 yet.** Wait until the migration is validated and the modification interface is working. Then we run them with sub-second iteration cycles instead of 75-second cycles.

---

## 5. Priority Order

1. Build llama.cpp (Step 1) — 10 min
2. Download GGUF (Step 2) — depends on network speed, ~38GB
3. Start serving and verify (Steps 3-5) — 15 min
4. Run eval baseline comparison (Step 6) — 20 min
5. Investigate in-memory modification (Step 7) — this is the research task, take the time needed
6. Design and implement modification interface (Step 8) — depends on Step 7 findings

Steps 1-4 should be straightforward. Steps 5-6 validate the migration. Steps 7-8 are the payoff — if we get sub-second weight modification, the entire project accelerates by two orders of magnitude.

---

## 6. Success Criteria

- [ ] llama.cpp serves Nemotron GGUF on port 30000
- [ ] MCP tools connect and work (or documented what changed)
- [ ] Eval baseline is within 10% of vllm baseline (83.3%)
- [ ] Blueprint verification passes (model reasons about itself correctly)
- [ ] At least one in-memory weight modification approach is validated
- [ ] Modification latency is documented (<1s target)
- [ ] A working `modification_interface.py` exists

---

## 7. Why This Matters

At 75s per modification cycle (vllm), running 100 Markov chain steps takes 2+ hours.  
At 500ms per cycle (llama.cpp in-memory), the same 100 steps take under 1 minute.  
At <100ms per cycle (direct buffer write), 1000 steps take under 2 minutes.

The difference between exploring a configuration space with 50 samples and 50,000 samples is the difference between manual probing and genuine self-directed evolution. This migration makes the neuroplastic vision practically achievable.

---

*End of migration task.*
