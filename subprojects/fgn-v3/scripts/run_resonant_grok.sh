#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CW_KWARGS='{"n_rooms_min": 5, "n_rooms_max": 25, "space_size": 150.0, "connect_radius": 40.0, "n_objects": 4, "min_steps": 3, "max_steps": 12, "min_state_changes": 1}'

echo "============================================"
echo "  Grokking Run — 50K steps"
echo "  Looking for emergent spatial structure"
echo "  Checkpoints every 5K for fidelity tracking"
echo "============================================"

python scripts/train_resonant.py \
    --config configs/resonant_6l.yaml \
    --task CW --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 1000 --grad_clip 1.0 --log_every 500 \
    --save_every 5000 --max_steps 50000 \
    --lambda_struct 0.1 \
    --task_kwargs "$CW_KWARGS" \
    --output_dir output_grok_50k

echo ""
echo "============================================"
echo "  Fidelity Trajectory (tracking grokking)"
echo "============================================"

for CKPT in step_5000 step_10000 step_15000 step_20000 step_25000 step_30000 step_35000 step_40000 step_45000 final; do
    CKPT_PATH="output_grok_50k/checkpoints/${CKPT}.pt"
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
echo "  Grokking Run Complete"
echo "============================================"
