#!/bin/bash
# Integration Time Sweep — T values on combined agentic tasks
set -euo pipefail

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/liquid-arc

CKPT=output_30m/checkpoints/step_10000.pt
CFG=configs/universality_probe.yaml
DATA=/workspace/fgn-v3/data/arc-repo/data
OUT=output_T_sweep

for T in 0.5 0.75 1.0 1.5 2.0 3.0; do
    echo "=== T=${T} (dt=$(echo "scale=4; $T/16" | bc)) ==="
    python scripts/train.py \
        --config $CFG --data_dir $DATA \
        --output_dir $OUT/T_${T} \
        --domain agentic \
        --resume $CKPT \
        --integration_time $T \
        --max_steps 12000 --log_every 50 --eval_every 500 \
        --save_every 5000 --batch_size 4 --grad_accum_steps 4
done

echo "=== T SWEEP COMPLETE ==="
