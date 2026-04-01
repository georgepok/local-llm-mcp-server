#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Base task kwargs (n_rooms overridden by complexity schedule)
# locked_door_prob=0.3 — 30% of edges randomly locked per episode
CW_KWARGS='{"n_rooms_min": 5, "n_rooms_max": 5, "space_size": 150.0, "connect_radius": 40.0, "locked_door_prob": 0.3, "n_objects": 4, "min_steps": 3, "max_steps": 12, "min_state_changes": 1}'

# Progressive: 5 rooms → 5-15 → 5-25
SCHEDULE="0:5:5,20000:5:15,40000:5:25"

echo "============================================"
echo "  Progressive Complexity + Locked Doors"
echo "  Schedule: 5 rooms (0-20K), 5-15 (20-40K), 5-25 (40K+)"
echo "  30% edges locked per episode (breaks position=connectivity heuristic)"
echo "  No structural energy — does task difficulty prevent deflation?"
echo "============================================"

python scripts/train_resonant.py \
    --config configs/resonant_6l.yaml \
    --task CW --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 1000 --grad_clip 1.0 --log_every 500 \
    --save_every 10000 --max_steps 100000 \
    --lambda_struct 0.0 \
    --task_kwargs "$CW_KWARGS" \
    --complexity_schedule "$SCHEDULE" \
    --output_dir output_progressive_100k

echo ""
echo "============================================"
echo "  Fidelity Trajectory"
echo "============================================"

for CKPT in step_10000 step_20000 step_30000 step_40000 step_50000 step_60000 step_70000 step_80000 step_90000 final; do
    CKPT_PATH="output_progressive_100k/checkpoints/${CKPT}.pt"
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
echo "  Progressive Run Complete"
echo "============================================"
