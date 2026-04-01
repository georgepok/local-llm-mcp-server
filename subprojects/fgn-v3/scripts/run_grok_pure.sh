#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Diverse environments — force generalization
CW_KWARGS='{"n_rooms_min": 5, "n_rooms_max": 25, "space_size": 150.0, "connect_radius": 40.0, "n_objects": 4, "min_steps": 3, "max_steps": 12, "min_state_changes": 1}'

echo "============================================"
echo "  Pure Grokking — No Structural Energy"
echo "  Just CE + weight decay (0.1) for 100K steps"
echo "  Let the model discover geometry on its own"
echo "============================================"

python scripts/train_resonant.py \
    --config configs/resonant_6l.yaml \
    --task CW --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 1000 --grad_clip 1.0 --log_every 500 \
    --save_every 10000 --max_steps 100000 \
    --lambda_struct 0.0 \
    --task_kwargs "$CW_KWARGS" \
    --output_dir output_grok_pure_100k

echo ""
echo "============================================"
echo "  Fidelity Trajectory"
echo "============================================"

for CKPT in step_10000 step_20000 step_30000 step_40000 step_50000 step_60000 step_70000 step_80000 step_90000 final; do
    CKPT_PATH="output_grok_pure_100k/checkpoints/${CKPT}.pt"
    if [ -f "$CKPT_PATH" ]; then
        echo ""
        echo "--- rho @ ${CKPT} ---"
        python scripts/diagnose_geometry_fidelity.py \
            --config configs/resonant_6l.yaml \
            --checkpoint "$CKPT_PATH" \
            --n_episodes 50
    fi
done

echo ""
echo "============================================"
echo "  Pure Grokking Run Complete"
echo "============================================"
