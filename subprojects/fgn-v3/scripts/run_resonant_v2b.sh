#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CW_KWARGS='{"n_rooms_min": 10, "n_rooms_max": 15, "space_size": 100.0, "connect_radius": 30.0, "n_objects": 4, "min_steps": 4, "max_steps": 10, "min_state_changes": 1}'

echo "============================================"
echo "  Experiment A-v2b — Last-Layer Energy"
echo "============================================"

echo ""
echo ">>> Training FluidNet lambda_struct=0.1 (last-layer energy)..."
python scripts/train_resonant.py \
    --config configs/resonant_6l.yaml \
    --task CW --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 1000 --grad_clip 1.0 --log_every 100 \
    --save_every 5000 --max_steps 10000 \
    --lambda_struct 0.1 \
    --task_kwargs "$CW_KWARGS" \
    --output_dir output_resonant_v2b_lambda0.1

echo ""
echo ">>> Eval lambda=0.1 (last-layer)..."
python scripts/eval_resonant.py \
    --config configs/resonant_6l.yaml \
    --checkpoint output_resonant_v2b_lambda0.1/checkpoints/final.pt \
    --task CW --task_kwargs "$CW_KWARGS" \
    --n_batches 50 --batch_size 8

echo ""
echo ">>> Geometry Fidelity Diagnostic..."
python scripts/diagnose_geometry_fidelity.py \
    --config configs/resonant_6l.yaml \
    --checkpoint output_resonant_v2b_lambda0.1/checkpoints/final.pt \
    --n_episodes 100 \
    --baseline_checkpoint output_resonant_v2_lambda0.0/checkpoints/final.pt

echo ""
echo "============================================"
echo "  Experiment A-v2b Complete"
echo "============================================"
