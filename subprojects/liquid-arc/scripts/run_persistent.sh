#!/bin/bash
# Persistent State Experiment — persistent vs non-persistent on sequential episodes
set -euo pipefail

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/liquid-arc

CKPT=output_30m/checkpoints/step_10000.pt
CFG=configs/agentic_persistent.yaml
OUT=output_persistent

echo "=== CONDITION A: PERSISTENT (alpha=0.7) ==="
python scripts/train_sequential.py \
    --config $CFG --resume $CKPT \
    --output_dir $OUT/with_state \
    --persist_alpha 0.7 \
    --max_steps 3000 --batch_size 4 \
    --log_every 50 --eval_every 500 --save_every 1000

echo "=== CONDITION B: NON-PERSISTENT (alpha=1.0) ==="
python scripts/train_sequential.py \
    --config $CFG --resume $CKPT \
    --output_dir $OUT/no_state \
    --persist_alpha 1.0 \
    --max_steps 3000 --batch_size 4 \
    --log_every 50 --eval_every 500 --save_every 1000

echo "=== BOTH CONDITIONS COMPLETE ==="
