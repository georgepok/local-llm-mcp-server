#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CW_KWARGS='{"n_rooms_min": 10, "n_rooms_max": 15, "space_size": 100.0, "connect_radius": 30.0, "n_objects": 4, "min_steps": 4, "max_steps": 10, "min_state_changes": 1}'
EVAL_ARGS="--n_batches 50 --batch_size 8"
STEPS=10000

echo "============================================"
echo "  Experiment A-v2 — Graph-Distance Energy"
echo "  Does structural energy align metric to"
echo "  actual world geometry (not token order)?"
echo "============================================"

# Train lambda=0.0 (baseline)
echo ""
echo ">>> Training FluidNet lambda_struct=0.0 (baseline)..."
python scripts/train_resonant.py \
    --config configs/resonant_6l.yaml \
    --task CW --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 1000 --grad_clip 1.0 --log_every 100 \
    --save_every 5000 --max_steps $STEPS \
    --lambda_struct 0.0 \
    --task_kwargs "$CW_KWARGS" \
    --output_dir output_resonant_v2_lambda0.0

# Train lambda=0.1
echo ""
echo ">>> Training FluidNet lambda_struct=0.1..."
python scripts/train_resonant.py \
    --config configs/resonant_6l.yaml \
    --task CW --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 1000 --grad_clip 1.0 --log_every 100 \
    --save_every 5000 --max_steps $STEPS \
    --lambda_struct 0.1 \
    --task_kwargs "$CW_KWARGS" \
    --output_dir output_resonant_v2_lambda0.1

echo ""
echo "============================================"
echo "  Evaluation — Both lambda values"
echo "============================================"

for LAMBDA in 0.0 0.1; do
    echo ""
    echo "--- lambda_struct=${LAMBDA} ---"
    python scripts/eval_resonant.py \
        --config configs/resonant_6l.yaml \
        --checkpoint output_resonant_v2_lambda${LAMBDA}/checkpoints/final.pt \
        --task CW --task_kwargs "$CW_KWARGS" \
        $EVAL_ARGS
done

echo ""
echo "============================================"
echo "  Geometry Fidelity Diagnostic"
echo "============================================"

echo ""
echo "--- lambda=0.1 (should show rho > 0.3) ---"
python scripts/diagnose_geometry_fidelity.py \
    --config configs/resonant_6l.yaml \
    --checkpoint output_resonant_v2_lambda0.1/checkpoints/final.pt \
    --n_episodes 100 \
    --baseline_checkpoint output_resonant_v2_lambda0.0/checkpoints/final.pt

echo ""
echo "============================================"
echo "  Experiment A-v2 Complete"
echo "============================================"
