# Latent Oracle — Amortized Test-Time Training via HyperNet

Distills oracle model (Qwen3.5-9B) knowledge into task-specific weight deltas for LiquidARC, replacing expensive per-task TTT with a single forward pass.

## Architecture

1. **Oracle embedding**: Qwen3.5-9B processes each ARC task, producing a fixed-size task descriptor
2. **HyperNet**: Low-rank (rank-8) network predicts W_o weight deltas from oracle embedding
3. **Application**: Delta applied to LiquidARC's W_o before inference — task-specific routing in one shot

~226K additional params (4.6% of 5M base model).

## Setup

Requires two containers on DGX Spark:
- `oracle-train`: Qwen model for embedding extraction
- `fgn-train`: LiquidARC for training with HyperNet

```bash
# Precompute oracle embeddings for all ARC tasks
python scripts/precompute.py

# Train HyperNet
python scripts/train_hypernet.py --config configs/hypernet.yaml

# Deploy
./deploy.sh
```

**IMPORTANT**: Set `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` in oracle-train container before any torch.compile run.

## Results

See `EXPERIMENT_REPORT_HYPERNET.md` for detailed results.
