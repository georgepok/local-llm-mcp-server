# Checkpoint/Rollback Feasibility Report

**Date:** 2026-03-10
**Investigator:** Claude Code

## Investigation Results

### 1. Can vLLM hot-swap individual weight tensors without full restart?

**Partially yes.** vLLM 0.13.0 has `reload_weights()` on both `GPUModelRunner` and `Worker`:

```python
def reload_weights(self) -> None:
    model_loader = get_model_loader(self.load_config)
    model_loader.load_weights(self.get_model(), model_config=self.model_config)
```

This reloads ALL weights from the model files on disk. It does NOT support individual tensor swap — it's a full reload. However, this means:
- If we modify specific safetensors files on disk, then call `reload_weights`, the model picks up the changes
- This is faster than a full restart (~4 min) because it skips KV cache allocation, torch.compile, and CUDA graph capture

**Limitation:** `reload_weights` is NOT exposed via the API. There is no HTTP endpoint or engine command to trigger it. It exists only as an internal method. To use it, we would need either:
- A custom API endpoint (modify vLLM source or add a route)
- A side-channel: exec into the container and trigger it via Python

### 2. How long does a vLLM restart take?

From logs:
- Model loading: **243 seconds** (~4 min)
- KV cache allocation: ~7 seconds
- torch.compile + CUDA graph capture: ~100 seconds
- **Total cold restart: ~6 minutes**

A `reload_weights` call would skip everything except model loading, so estimated **~4 minutes**.

### 3. Is there a vLLM API for pausing/resuming inference?

**No.** vLLM v1 engine has `ABORT` and `ADD` request types but no pause/resume. The engine processes requests continuously. However:
- With `max_num_seqs=8`, during quiet periods (no active requests) the model is idle
- We can effectively "pause" by not sending requests during modification

### 4. Model file access

- Safetensors files are **writable** from inside the container (bind mount, RW)
- Model path: `/workspace/model` (container) = `/home/pokazge/models/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` (host)
- 10 shards, 32.68 GB total
- **2.4 TB free disk space** — enough for ~70 full model backups

## Approach Assessment

### Approach A: Offline Modification (RECOMMENDED)

**Flow:** backup tensors → modify safetensors on disk → restart vLLM → evaluate → rollback if needed

**Pros:**
- Simplest and most reliable
- Full isolation: model serves with either original or modified weights, never in between
- Easy rollback: copy backup files back, restart
- Can modify any tensor(s) across any shard

**Cons:**
- 6-minute restart per modification cycle (cold) or ~4 min (reload_weights if we add a hook)
- No A/B comparison without two instances

**Optimization:** We can create a small script that modifies specific tensors in-place within a safetensors file, avoiding the need to copy the full 32GB. The `safetensors` library supports this.

### Approach B: Side-loading / Hot-swap

**Flow:** modify safetensors on disk → trigger `reload_weights` via docker exec → evaluate

**Pros:**
- Faster than full restart (skips compile/KV cache)
- No service interruption beyond the reload time

**Cons:**
- `reload_weights` is not API-exposed — requires docker exec side-channel
- Reloads ALL weights, not just the modified ones
- torch.compile'd graphs may have cached weight references — unclear if reload properly invalidates them
- Risk of inconsistent state during reload

**Verdict:** Feasible but needs testing. Worth investigating as a faster iteration path after Approach A is proven.

### Approach C: Logits Processor Augmentation (FASTEST FOR BEHAVIORAL EXPERIMENTS)

**Flow:** modify fluid_geometry.py → restart container (or hot-reload if we add reloader)

**Pros:**
- Zero weight modification risk
- Hot-swappable via file mount (just update the .py file)
- Can implement arbitrary logit-space interventions
- Already deployed and stable
- Fastest iteration cycle

**Cons:**
- Not true self-modification (weights unchanged)
- Can only modulate output distribution, not internal representations
- Limited to what's expressible as logit bias/temperature changes
- Cannot add new knowledge or capabilities

**Verdict:** Best for Phase 2 initial experiments. Test whether Nemotron can reason about and propose logits processor modifications before moving to weight modification.

## Recommended Phased Approach

1. **Phase 2a:** Approach C — behavioral modification via logits processor. Nemotron proposes parameter changes to the geometric engine. Fast iteration, zero risk.
2. **Phase 2b:** Approach A — offline weight modification of safe targets (expert weights, with CV 1-4% homogeneity as baseline). Backup → modify → restart → evaluate → rollback. 6-min cycles.
3. **Phase 2c:** Approach B — hot-swap via `reload_weights` for faster iteration once Approach A patterns are proven.

## Disk Space Budget

| Item | Size |
|---|---|
| Full model backup | 32.7 GB |
| Per-tensor backup (single expert) | ~7 MB |
| Available disk | 2,400 GB |
| Max full backups | ~70 |
