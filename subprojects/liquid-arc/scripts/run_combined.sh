#!/bin/bash
# Combined universality probe — all 4 non-spatial domains simultaneously
set -euo pipefail

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/liquid-arc

CKPT=/workspace/liquid-arc/output_30m/checkpoints/step_10000.pt

echo "=== COMBINED: all 4 non-spatial domains from 5M checkpoint ==="
python scripts/train.py \
    --config configs/universality_combined.yaml \
    --data_dir /workspace/fgn-v3/data/arc-repo/data \
    --output_dir output_universality/combined_transfer \
    --domain combined \
    --resume "$CKPT" \
    --max_steps 12000 \
    --log_every 50 \
    --eval_every 500 \
    --save_every 5000 \
    --batch_size 16

echo "=== COMBINED COMPLETE ==="
