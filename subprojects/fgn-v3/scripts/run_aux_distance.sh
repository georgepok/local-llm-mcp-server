#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resume from progressive run step_50K (W-peak state, fully trained on 5-25 rooms + locked doors)
RESUME="output_progressive_100k/checkpoints/step_50000.pt"

# Diverse environments matching progressive phase 3
CW_KWARGS='{"n_rooms_min": 5, "n_rooms_max": 25, "space_size": 150.0, "connect_radius": 40.0, "locked_door_prob": 0.3, "n_objects": 4, "min_steps": 3, "max_steps": 12, "min_state_changes": 1}'

echo "============================================"
echo "  Aux Distance + Curvature Floor"
echo "  Resume from progressive step_50K"
echo "  aux_distance: 10 hop classes, weight=1.0"
echo "  kappa_floor: 10.0, mu=0.1"
echo "  weight_decay: 0.01 (10x lower)"
echo "============================================"

# Phase A: Probe — freeze main model, train only aux head (1K steps)
# Phase B: Induce — unfreeze all, low LR (4K steps)
# Combined in one run with warmup handling the transition

python scripts/train_resonant.py \
    --config configs/resonant_6l.yaml \
    --task CW --batch_size 4 --lr 1e-4 --weight_decay 0.01 \
    --warmup_steps 200 --grad_clip 1.0 --log_every 100 \
    --save_every 1000 --max_steps 5000 \
    --lambda_struct 0.0 \
    --task_kwargs "$CW_KWARGS" \
    --resume_checkpoint "$RESUME" \
    --aux_distance_weight 1.0 \
    --aux_distance_max_hops 10 \
    --kappa_floor 10.0 \
    --kappa_floor_mu 0.1 \
    --output_dir output_aux_distance_v1

echo ""
echo "============================================"
echo "  Fidelity Trajectory"
echo "============================================"

for CKPT in step_1000 step_2000 step_3000 step_4000 final; do
    CKPT_PATH="output_aux_distance_v1/checkpoints/${CKPT}.pt"
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
echo "  Aux Distance Run Complete"
echo "============================================"
