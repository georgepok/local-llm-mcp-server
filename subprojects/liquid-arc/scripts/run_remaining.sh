#!/bin/bash
# Graph rerun (output=coloring row only) + combined all-domain
set -euo pipefail

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/liquid-arc

CKPT=/workspace/liquid-arc/output_30m/checkpoints/step_10000.pt
CFG=configs/universality_probe.yaml
DATA=/workspace/fgn-v3/data/arc-repo/data
OUT=output_universality

echo "=== TRANSFER: graph (fixed - output=coloring row only) ==="
python scripts/train.py --config $CFG --data_dir $DATA \
    --output_dir $OUT/graph_transfer_v2 \
    --domain graph --resume $CKPT \
    --max_steps 11500 --log_every 50 --eval_every 500 \
    --save_every 5000 --batch_size 16

echo "=== COMBINED: all 4 non-spatial domains ==="
python scripts/train.py \
    --config configs/universality_combined.yaml \
    --data_dir $DATA \
    --output_dir $OUT/combined_transfer \
    --domain combined \
    --resume $CKPT \
    --max_steps 12000 --log_every 50 --eval_every 500 \
    --save_every 5000 --batch_size 16

echo "=== ALL COMPLETE ==="
