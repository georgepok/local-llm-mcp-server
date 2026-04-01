# In-Memory Weight Modification — Investigation Results & Design

**Date:** 2026-03-10

## Investigation Results

### Task A: /pause and /resume — WORKS
```
POST /pause  → {"status": "paused"}
GET  /is_paused → {"is_paused": true}
POST /resume → {"status": "resumed"}
```
These endpoints work perfectly. Pausing stops inference, resuming restores it.

### Task B: In-Process Model Access — BLOCKED BY MULTIPROCESSING

vLLM 0.13.0 (V1 engine) runs the EngineCore in a **separate process** communicating via ZMQ sockets. The API server (FastAPI) and the model executor (GPU worker) are in different processes.

- **PID 1:** API server (python3 -m vllm.entrypoints.openai.api_server)
- **Engine core:** spawned as subprocess via `AsyncMPClient.make_async_mp_client()`
- **Communication:** ZMQ PUSH/PULL sockets (not shared memory for control)

This means: you cannot `docker exec` a Python script that accesses the model's GPU tensors. The model lives in the engine core subprocess.

### Task C: LoRA Support — EXISTS BUT UNTESTED

`LoRAConfig` exists in vllm. The engine core has `add_lora`, `remove_lora`, `list_loras`, `pin_lora` methods. Whether NemotronH (hybrid Mamba-Transformer) is LoRA-compatible is unknown. This could be explored later.

### Task D: CUDA Graph Infrastructure

CUDA graphs are captured via `CUDAGraphWrapper` in `vllm/compilation/cuda_graph.py`. The model runner has a `capture_model()` method that captures graphs for specific batch sizes. After weight modification, CUDA graphs would need recapturing since they store tensor pointers.

`reload_weights()` exists on `GPUModelRunner` and calls `model_loader.load_weights()` — it reloads from disk, but does NOT explicitly recapture CUDA graphs. This suggests that either:
1. The reload path handles graph invalidation internally, or
2. Weights are accessed by pointer, so in-place modification (via `reload_weights` which copies data into existing tensors) preserves graph validity.

### Key Discovery: `call_utility` + `collective_rpc`

EngineCore has a utility dispatch mechanism that calls arbitrary methods by name:
```python
# In core_client.py:
def call_utility(self, method: str, *args) -> Any:
    self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))

# In core.py - dispatches to:
method = getattr(self, method_name)
result = method(*args)
```

EngineCore also has:
```python
def collective_rpc(self, method, timeout, args, kwargs):
    return self.model_executor.collective_rpc(method, timeout, args, kwargs)
```

This chain: `call_utility("collective_rpc", "reload_weights")` could trigger weight reload from within the engine core process. However, this requires the API server to be able to call `call_utility`, which requires access to the `engine_core` client object.

## Feasible Architecture

### Option 1: Pause → Modify Disk → Reload via Utility → Resume (Estimated: 5-30s)

1. `POST /pause` — stop inference
2. Modify safetensors on disk (current approach, ~3-12s)
3. Trigger `reload_weights` through the utility mechanism:
   - Add a custom HTTP endpoint that calls `engine_core.call_utility_async("collective_rpc", "reload_weights")`
   - This requires monkey-patching the API server or adding a vllm plugin
4. `POST /resume` — restart inference

This avoids full restart (~4 min) but still requires disk I/O.

### Option 2: Custom Endpoint with Direct GPU Modification (Estimated: 200-800ms)

Add a FastAPI endpoint that sends a utility command to the engine core, which then modifies tensors in-place on GPU:

```python
# Added to EngineCore (or via monkey-patch):
def modify_weight_inplace(self, tensor_path: str, operation: str, factor: float):
    """Modify a weight tensor in-place on GPU."""
    worker = self.model_executor.driver_worker
    model = worker.model_runner.model

    # Navigate to tensor
    parts = tensor_path.split(".")
    param = model
    for part in parts:
        if part.isdigit():
            param = param[int(part)]
        else:
            param = getattr(param, part)

    # Modify in-place
    with torch.no_grad():
        if operation == "scale":
            param.data.mul_(factor)
        elif operation == "add":
            param.data.add_(factor)

    return {"status": "ok", "tensor": tensor_path}
```

Then expose via HTTP:
```python
@app.post("/neuroplastic/modify_weight")
async def modify_weight(request: ModifyWeightRequest):
    await app.state.engine_client.call_utility_async(
        "modify_weight_inplace",
        request.tensor_path, request.operation, request.factor
    )
    return {"status": "ok"}
```

### Option 3: Reload From Disk Without Restart (Estimated: 30-60s)

Simplest approach: modify on disk, then restart the container. Current approach takes ~75s.

A middle ground: modify on disk, then use `reload_weights` (if we can trigger it). The reload skips tokenizer, CUDA initialization, KV cache allocation, etc.

## Recommendation

**For immediate use (experiments 002-003):** Continue with disk modification + container restart (75s). This works and is proven.

**For future speedup:** Implement Option 2 by:
1. Creating a small Python script that monkey-patches the vllm API server at startup
2. Mount it into the container and load it via `--logits-processors` or `--additional-config`
3. This gives us a `/neuroplastic/modify_weight` HTTP endpoint for ~200ms modifications

**The 100x speedup is feasible** but requires:
- A custom vllm plugin/middleware (~50 lines of code)
- Testing that CUDA graph cache handles in-place weight modification correctly
- Confirming that `driver_worker.model_runner.model` attribute path works for NemotronH

## Open Questions

1. Does `reload_weights()` handle CUDA graph invalidation automatically?
2. Does NemotronH support LoRA in vllm?
3. What is the `driver_worker` attribute path for the V1 engine with single-GPU?
4. Can we use vllm's `--additional-config` or plugin system to add custom endpoints?
