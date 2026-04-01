#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# World parameters for MasterWorld initialization
CW_KWARGS='{"n_rooms_max": 8, "n_objects": 3, "space_size": 80.0, "connect_radius": 35.0, "locked_door_prob": 0.2, "n_rooms_min": 4, "min_steps": 2, "max_steps": 6, "min_state_changes": 1}'

echo "============================================"
echo "  Phase 3v2: Criticality Run — Starved Model"
echo "  Edge of Chaos — Forcing Geometric Plasticity"
echo ""
echo "  1. Starved model: ~1.2M params (d=64, 4L)"
echo "  2. MasterWorld: 20 rooms, catastrophic mutate 20% every 500 steps"
echo "  3. MetricMonitor: shrink-perturb bottom 5% dims"
echo "  4. DynamicWeightDecay: EMA-smoothed, 0.01-0.30 range"
echo "  5. Plasticity telemetry: Vg, t½, perturbation count"
echo "============================================"

python scripts/train_criticality.py \
    --config configs/criticality_starved.yaml \
    --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 500 --grad_clip 1.0 --log_every 200 \
    --save_every 5000 --max_steps 50000 \
    --lambda_struct 0.0 \
    --task_kwargs "$CW_KWARGS" \
    --mutate_every 500 \
    --catastrophic_fraction 0.2 \
    --master_seed 42 \
    --crystal_percentile 0.05 \
    --perturb_alpha 0.9 \
    --perturb_sigma 0.01 \
    --perturb_every 2000 \
    --output_dir output_criticality_v2

echo ""
echo "============================================"
echo "  Fidelity Trajectory"
echo "============================================"

for CKPT in step_5000 step_10000 step_15000 step_20000 step_25000 \
            step_30000 step_35000 step_40000 step_45000 final; do
    CKPT_PATH="output_criticality_v2/checkpoints/${CKPT}.pt"
    if [ -f "$CKPT_PATH" ]; then
        echo ""
        echo "--- rho @ ${CKPT} ---"
        python scripts/diagnose_geometry_fidelity.py \
            --config configs/criticality_starved.yaml \
            --checkpoint "$CKPT_PATH" \
            --n_episodes 50
    fi
done

echo ""
echo "============================================"
echo "  Criticality v2 Run Complete"
echo "============================================"
