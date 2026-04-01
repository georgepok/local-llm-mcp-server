#!/bin/bash
# Agentic State Controller — 3 single-domain transfer runs + combined
set -euo pipefail

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/liquid-arc

CKPT=output_30m/checkpoints/step_10000.pt
CFG=configs/universality_probe.yaml
DATA=/workspace/fgn-v3/data/arc-repo/data
OUT=output_agentic

for domain in stateful context dependency; do
    echo "=== TRANSFER: $domain ==="
    python scripts/train.py --config $CFG --data_dir $DATA \
        --output_dir $OUT/${domain}_transfer \
        --domain $domain --resume $CKPT \
        --max_steps 11500 --log_every 50 --eval_every 250 \
        --save_every 5000 --batch_size 16
done

echo "=== COMBINED: agentic (3 domains interleaved) ==="
python scripts/train.py --config $CFG --data_dir $DATA \
    --output_dir $OUT/agentic_combined \
    --domain agentic --resume $CKPT \
    --max_steps 12000 --log_every 50 --eval_every 500 \
    --save_every 5000 --batch_size 4 --grad_accum_steps 4

echo "=== ALL AGENTIC COMPLETE ==="
