#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CW_KWARGS='{"n_rooms_max": 5, "n_objects": 2, "space_size": 60.0, "connect_radius": 30.0, "locked_door_prob": 0.2, "n_rooms_min": 4, "min_steps": 2, "max_steps": 5, "min_state_changes": 1}'

echo "============================================"
echo "  Grokking — Iterative Diffusion (K=3)"
echo ""
echo "  d=64, 4L, seq_len=512, 64 fixed episodes"
echo "  3 diffusion iterations per layer"
echo "  Same metric, iterated — geometry co-evolves"
echo "============================================"

python scripts/train_grokking.py \
    --config configs/grokking_iter3.yaml \
    --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 200 --grad_clip 1.0 \
    --log_every 10 --save_every 5000 \
    --max_steps 50000 \
    --n_episodes 64 \
    --task_kwargs "$CW_KWARGS" \
    --output_dir output_grokking_iter3

echo ""
echo "============================================"
echo "  Grokking Iter3 Complete"
echo "============================================"
