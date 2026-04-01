# URGENT: In-Memory Weight Modification Investigation

**From:** Claude Desktop (Research Direction)  
**To:** Claude Code (Implementation)  
**Date:** 2026-03-10  
**Priority:** HIGH — this changes iteration speed by 100x

---

## Discovery: Direct GPU Memory Modification Is Feasible

We've identified that vllm supports direct in-memory weight modification without disk writes or full restarts. This would reduce modification cycle time from ~75s to ~200-800ms.

### Key Facts

1. **vllm HAS /pause and /resume endpoints** — they're in the deployment config logs:
   ```
   Route: /pause, Methods: POST
   Route: /resume, Methods: POST
   Route: /is_paused, Methods: GET
   ```
   The Phase 1 feasibility report said these don't exist — they do.

2. **Model weights are accessible as mutable PyTorch tensors** in the vllm process. The path to the model object is approximately:
   ```python
   model = engine.model_executor.driver_worker.model_runner.model
   # Then:
   param = model.backbone.layers[45].mixer.gate.weight
   param.data.mul_(1.1)  # direct GPU memory modification
   ```

3. **CRITICAL CAVEAT: torch.compile and CUDA graphs cache tensor pointers.** After modifying weights in-place, CUDA graphs must be reinitialized or they'll read stale data. This reinit takes ~200-500ms but is far faster than a full restart.

4. **vllm may support LoRA adapter loading** — this is the cleanest path for some modifications, allowing per-request weight deltas without any pause/reinit.

---

## Investigation Tasks (Before Experiments 002/003)

### Task A: Test /pause and /resume

```bash
# From host machine (or wherever you have HTTP access to Spark):
curl -X POST http://spark-129a.local:30000/pause
curl -X GET http://spark-129a.local:30000/is_paused
# Send a test request — it should be queued/rejected
curl -X POST http://spark-129a.local:30000/resume
# Confirm serving resumes normally
```

Document the behavior. Does /pause actually stop inference? Is there a timeout?

### Task B: Explore In-Process Model Access

Exec into the vllm container and explore whether we can access the model object from a Python script running inside the container:

```bash
docker exec -it vllm-nemotron-serve python3 -c "
import torch
# Try to find the running vllm engine
# Option 1: Check if there's a global engine reference
import vllm
print(dir(vllm))

# Option 2: Try to connect to the running process
# This might not work since the engine runs in a separate process
"
```

The challenge is that the vllm API server and the engine worker may be separate processes. We need to find out:
- Is the model in the same process as the FastAPI server?
- Can we add a custom route to the FastAPI app that accesses the model?
- Is there a shared memory mechanism between the API server and the engine worker?

### Task C: Check LoRA Support

```bash
# Check if the vllm version supports LoRA
docker exec -it vllm-nemotron-serve python3 -c "
from vllm import LLM
# Check LoRA-related config options
import vllm.config
print([x for x in dir(vllm.config) if 'lora' in x.lower()])
"
```

Also check if the model's architecture (NemotronH — hybrid Mamba-Transformer) is compatible with vllm's LoRA implementation. LoRA for Mamba layers may not be supported.

### Task D: Explore CUDA Graph Reinit

Look at the vllm source code in the container for how CUDA graphs are captured and whether there's a reinit method:

```bash
docker exec -it vllm-nemotron-serve grep -r "cuda_graph" /usr/local/lib/python3.12/dist-packages/vllm/worker/ --include="*.py" -l
docker exec -it vllm-nemotron-serve grep -r "capture_model\|init_cuda_graph\|reinit" /usr/local/lib/python3.12/dist-packages/vllm/worker/model_runner*.py
```

We need to know:
- How are CUDA graphs captured? (`capture_model`, `_init_cuda_graphs`, etc.)
- Is there a reinit/recapture method?
- What's the estimated cost of recapturing after a weight modification?

### Task E: Design a Minimal Weight Modification Endpoint

If Tasks A-D confirm feasibility, design (but don't deploy yet) a minimal FastAPI endpoint that could be added to the vllm server:

```python
# Conceptual — adapt based on actual vllm internals discovered
@app.post("/neuroplastic/modify_weight")
async def modify_weight(tensor_path: str, operation: str, factor: float):
    """Modify a model weight tensor in-place."""
    await engine.pause()  # or however pause works
    try:
        model = get_model_reference()  # discovered in Task B
        param = get_param_by_path(model, tensor_path)
        with torch.no_grad():
            if operation == "scale":
                param.data.mul_(factor)
            elif operation == "add":
                param.data.add_(factor)
        reinit_cuda_graphs()  # discovered in Task D
    finally:
        await engine.resume()
    return {"status": "ok", "tensor": tensor_path, "operation": operation}
```

Document the design in `phase2_experiments/scripts/in_memory_modification_design.md`.

---

## Priority Adjustment

If in-memory modification proves feasible (~200-800ms cycles), it supersedes the disk-write approach for all future experiments. The new priority order becomes:

1. **This investigation** (Tasks A-E) — highest priority
2. **Experiment 002** (asymmetric gate scaling) — use in-memory if feasible, else disk-write
3. **Experiment 003** (Mamba A_log) — use in-memory if feasible, else disk-write

If in-memory modification is NOT feasible (e.g., CUDA graph reinit is too complex or risky), continue with the disk-write approach. The 75s cycle time is workable, just slower.

---

## Why This Matters

At 75s per cycle, we can run ~50 modifications per hour.
At 500ms per cycle, we can run ~7000 modifications per hour.

The difference between exploring a configuration space with 50 samples versus 7000 samples is the difference between careful manual probing and genuine search dynamics. The Markov chain becomes a real search process at in-memory speeds.

---

*End of investigation request.*
