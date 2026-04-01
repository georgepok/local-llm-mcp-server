# FGN v3 Phase 1a — Results

## Training

- **Config**: small.yaml (d=256, 6 layers, 8 heads, 3 scales, seq_len=512)
- **Parameters**: 30,825,904
- **Data**: WikiText-103 (226,007 sequences of 512 tokens)
- **Hardware**: DGX Spark (GB10 / Grace Blackwell), 128GB unified memory
- **Container**: nvcr.io/nvidia/vllm:26.01-py3
- **Optimizer**: AdamW (lr=3e-4 base, 3e-5 metric/diffusion, weight_decay=0.1)
- **torch.compile**: mode="default" (~30 min compile on Grace ARM, then 14K tok/s)

### Loss Trajectory (10K steps)

| Step | Total Loss | CE Loss |
|------|-----------|---------|
| 0 | 10.857 | 10.867 |
| 1000 | 6.121 | 6.131 |
| 2000 | 5.987 | 5.997 |
| 3000 | 5.218 | 5.228 |
| 4000 | 5.096 | 5.106 |
| 5000 | 5.127 | 5.136 |
| 6000 | 5.279 | 5.288 |
| 7000 | 4.887 | 4.897 |
| 8000 | 5.086 | 5.096 |
| 9000 | 4.872 | 4.882 |
| 10000 | 4.623 | 4.633 |

Curvature regularization loss ~0.0001 throughout (metric near identity).
Scale entropy loss ~-0.01 (all 3 scales balanced).

### Checkpoints

Saved in `output/checkpoints/` on spark-129a:
- `step_2000.pt`, `step_4000.pt`, `step_6000.pt`, `step_8000.pt`
- `best.pt` (step 8000, loss 4.77)
- `final.pt` (step 10000, loss 4.62)

## Validation (Synthetic Copy-Pattern)

5/5 checks passed. Run on GPU, 1000 steps, d=64, 2 layers.

| Check | Result | Threshold |
|-------|--------|-----------|
| Loss convergence | -0.0087 | < 0.5 |
| Metric CV (std/mean) | 0.0118 | > 0.01 |
| Curvature at SEP vs elsewhere | 3.08x | > 1.2x |
| Min scale weight | 0.329 | > 0.1 |
| Grad norm ratio (max/min layer) | 1.29 | < 100 |

## Holonomy Test

**PASS** — Phase 1a identity transport confirmed.

```
Holonomy norms (n=100 triplets):
  Mean:  0.000000e+00
  Std:   0.000000e+00
  Max:   0.000000e+00
  Min:   0.000000e+00
```

Phase 1a transport is exactly identity. Any non-zero holonomy would indicate a bug.

## OVL Separability Test

**FAIL** — curvature distributions not yet task-discriminating.

| Layer | OVL | Target |
|-------|-----|--------|
| 0 | 0.955 | < 0.3 |
| 1 | 0.971 | < 0.3 |
| 2 | 0.974 | < 0.3 |
| 3 | 0.975 | < 0.3 |
| 4 | 0.976 | < 0.3 |
| 5 | 0.969 | < 0.3 |

Expected at this stage. The model has only trained 10K steps on WikiText with a
small architecture. Task-discriminating curvature requires the model to have
learned distinct internal representations for different task types, which needs
longer training and potentially a larger model.

## Issues Found and Fixed During Development

### Critical: Metric Partition Across Heads (Issue 1)
Original code split the d-dimensional metric into per-head d_head slices, giving
each head a different metric. This violated the shared-metric contract. Fixed by
averaging across head groups to produce a single d_head-dimensional metric shared
by all heads.

### Scale Entropy Used Stale Hidden States (Issue 3)
Scale entropy was computed from the final layer's output h for all layers. Fixed
by caching scale weights during each layer's forward pass and computing entropy
from cached values.

### Unused curvature_eta Config (Issue 4)
Config had `curvature_eta` that was never used (eta is derived from lambda and
correlation_length_init). Removed to prevent confusion.

### No Dropout (Issue 6)
Added dropout (default 0.1) to attention output and FFN for regularization.

### CurvatureEngine conv1d Backward Crash on GB10
cuDNN on GB10 failed on conv1d backward with reflect padding. Replaced with
direct tensor slicing (g[:, 2:] - g[:, :-2] for first difference, etc.). This
is clearer code anyway since the stencils are trivially simple.

### torch.compile
- `mode="max-autotune"` takes >30 min on Grace ARM and the kernel benchmarking
  sometimes fails. Switched to `mode="default"`.
- `MetricNetwork` weight init increased from std=0.01 to std=0.05 to allow
  metric to develop position-dependent structure through Softplus activation.

## DGX Spark / GB10 Notes

- torch.compile works with `mode="default"` but takes ~30 min to compile
  (ARM cores are slower at code generation than x86)
- cuDNN lacks backward support for 1D convolutions with reflect padding
- `nvidia-smi` memory queries return N/A (unified memory architecture)
- seq_len=512 with batch=8 fits in memory; seq_len=2048 OOMs during
  the N^2 attention distance matrix materialization
- Throughput: ~14K tok/s after compile warmup (batch=8, seq=512, d=256)
