#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1

CW_KWARGS='{"n_rooms_max": 5, "n_objects": 2, "space_size": 60.0, "connect_radius": 30.0, "locked_door_prob": 0.2, "n_rooms_min": 4, "min_steps": 2, "max_steps": 5, "min_state_changes": 1}'

echo "============================================"
echo "  Generalization Test — Iter3 (K=3)"
echo "============================================"

python scripts/test_generalization.py \
    --config configs/grokking_iter3.yaml \
    --checkpoint_dir output_grokking_iter3/checkpoints \
    --checkpoints step_5000.pt step_10000.pt step_15000.pt \
    --n_episodes 128 \
    --seed 99999 \
    --task_kwargs "$CW_KWARGS"

echo ""
echo "============================================"
echo "  Generalization Test — Baseline (K=1)"
echo "============================================"

python scripts/test_generalization.py \
    --config configs/criticality_starved.yaml \
    --checkpoint_dir output_grokking/checkpoints \
    --checkpoints step_5000.pt step_10000.pt step_15000.pt step_20000.pt final.pt \
    --n_episodes 128 \
    --seed 99999 \
    --task_kwargs "$CW_KWARGS"
