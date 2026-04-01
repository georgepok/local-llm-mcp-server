# FGN v3 — Fluid Geometry Network with Riemannian Attention

Phase 1a: Resolution Thesis — diagonal metric, curvature engine, multi-scale heat kernel attention.

## Architecture

Standard Transformer attention (`softmax(QK^T/sqrt(d)) V`) is replaced with diffusion on a learned Riemannian manifold:

1. **MetricNetwork**: Maps hidden states to positive-definite diagonal metric tensors (shared across heads)
2. **CurvatureEngine**: Computes differentiable scalar curvature from the metric field via 1D convolutions
3. **HeatKernelAttention**: Multi-scale heat kernel attention using geodesic distances weighted by the metric
4. **CurvatureRegularization**: Learnable correlation length regularization on curvature variance and smoothness

Each token lives in a local tangent space defined by the learned metric. "Resolution" and "causality" become properties of manifold curvature.

## Quick Start

```bash
# Validate on synthetic copy-pattern (CPU, no GPU needed)
python scripts/validate_minimal.py

# Train on WikiText-103 (GPU)
python scripts/train.py --config configs/small.yaml --batch_size 8 --max_steps 10000

# Run OVL separability test
python scripts/ovl_test.py --checkpoint output/checkpoints/best.pt --output ovl.png

# Export for vLLM serving
python scripts/convert_to_vllm.py --checkpoint output/checkpoints/best.pt --output /tmp/fgn-model
```

## Target Environment

- **Hardware**: NVIDIA DGX Spark (GB10 / Grace Blackwell), 128GB unified memory
- **Container**: `nvcr.io/nvidia/vllm:26.01-py3`
- **Server**: spark-129a.local:30000

## Project Structure

```
fgn/
├── config.py       # FGNConfig dataclass
├── metric.py       # MetricNetwork (diagonal → low-rank)
├── curvature.py    # CurvatureEngine (1D tensor trace)
├── attention.py    # HeatKernelAttention (multi-scale)
├── transport.py    # Phase 2 stubs (identity in Phase 1)
├── layer.py        # FGNTransformerLayer
├── model.py        # FGNModel (full LM)
└── losses.py       # Curvature regularization

scripts/
├── train.py              # Training (single-GPU)
├── validate_minimal.py   # Synthetic copy-pattern validation
├── ovl_test.py           # OVL separability diagnostic
├── holonomy_test.py      # ε noise floor measurement
└── convert_to_vllm.py   # Export for vLLM serving

configs/
├── small.yaml      # d=256, 6 layers, 8 heads
└── medium.yaml     # d=512, 12 layers, 8 heads
```

## Key Design Decisions

- **Diagonal metric shared across heads** — prevents projection absorption
- **Softplus activation** (not exp) — bounded gradients, no explosion
- **Log-space heat kernels** — numerically stable via softmax(log_K)
- **Per-scale normalization before mixing** — prevents sharpest scale from dominating
- **Scale entropy regularization** — prevents scale collapse
- **0.1x learning rate for metric/diffusion times** — slow geometric evolution
- **No detach()** — full gradient flow from metric to all three loss terms

## Phase Roadmap

- **Phase 1a** (this): Diagonal metric, curvature engine, heat kernel attention
- **Phase 1b**: Low-rank metric upgrade (diag + LL^T)
- **Phase 2**: Parallel transport, holonomy detection, causal structure from geometry
