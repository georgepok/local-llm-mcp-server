#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET="data/cw_fixed_2000.pt"

# Task kwargs for dataset generation (diverse, locked doors)
CW_KWARGS='{"n_rooms_min": 5, "n_rooms_max": 25, "space_size": 150.0, "connect_radius": 40.0, "locked_door_prob": 0.3, "n_objects": 4, "min_steps": 3, "max_steps": 12, "min_state_changes": 1}'

echo "============================================"
echo "  Fixed-Dataset Grokking"
echo "  2000 episodes, 200K steps (~400 epochs)"
echo "  weight_decay=0.1, no structural energy"
echo "  Train: fixed set | Eval: fresh episodes"
echo "============================================"

# Step 1: Generate fixed dataset (if not exists)
if [ ! -f "$DATASET" ]; then
    echo ""
    echo "--- Generating fixed dataset ---"
    mkdir -p data
    python scripts/generate_fixed_dataset.py \
        --n_episodes 2000 \
        --seq_len 1024 \
        --task_kwargs "$CW_KWARGS" \
        --output "$DATASET"
    echo ""
fi

# Step 2: Train from scratch on fixed dataset
echo ""
echo "--- Training (200K steps) ---"
python scripts/train_resonant.py \
    --config configs/resonant_6l.yaml \
    --task CW --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 1000 --grad_clip 1.0 --log_every 500 \
    --save_every 10000 --max_steps 200000 \
    --lambda_struct 0.0 \
    --task_kwargs "$CW_KWARGS" \
    --fixed_dataset "$DATASET" \
    --output_dir output_grok_fixed

# Step 3: Fidelity on fresh episodes at each checkpoint
echo ""
echo "============================================"
echo "  Fidelity Trajectory (fresh episodes)"
echo "============================================"

for CKPT in step_10000 step_20000 step_30000 step_40000 step_50000 \
            step_60000 step_70000 step_80000 step_90000 step_100000 \
            step_120000 step_140000 step_160000 step_180000 final; do
    CKPT_PATH="output_grok_fixed/checkpoints/${CKPT}.pt"
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
echo "  Fixed-Dataset Grokking Complete"
echo "============================================"
