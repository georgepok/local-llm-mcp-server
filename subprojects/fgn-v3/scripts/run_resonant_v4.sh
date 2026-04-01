#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CW_KWARGS='{"n_rooms_min": 10, "n_rooms_max": 15, "space_size": 100.0, "connect_radius": 30.0, "n_objects": 4, "min_steps": 4, "max_steps": 10, "min_state_changes": 1}'

echo "============================================"
echo "  Experiment A-v4 — Projection Head (d_proj=32)"
echo "  Tests whether graph distance is linearly"
echo "  accessible in h (bypasses diagonal metric)"
echo "============================================"

echo ""
echo ">>> Training FluidNet lambda_struct=0.1, d_proj=32..."
python scripts/train_resonant.py \
    --config configs/resonant_6l.yaml \
    --task CW --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 500 --grad_clip 1.0 --log_every 100 \
    --save_every 1000 --max_steps 3000 \
    --lambda_struct 0.1 \
    --task_kwargs "$CW_KWARGS" \
    --output_dir output_resonant_v4_proj32

echo ""
echo "============================================"
echo "  Geometry Fidelity at Each Checkpoint"
echo "============================================"

for CKPT in step_1000 step_2000 final; do
    CKPT_PATH="output_resonant_v4_proj32/checkpoints/${CKPT}.pt"
    if [ -f "$CKPT_PATH" ]; then
        echo ""
        echo "--- Fidelity @ ${CKPT} ---"
        python scripts/diagnose_geometry_fidelity.py \
            --config configs/resonant_6l.yaml \
            --checkpoint "$CKPT_PATH" \
            --n_episodes 50
    else
        echo "  Checkpoint $CKPT_PATH not found, skipping"
    fi
done

echo ""
echo "============================================"
echo "  Experiment A-v4 Complete"
echo "============================================"
