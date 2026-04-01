#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_KWARGS='{"n_rooms_min": 10, "n_rooms_max": 15, "space_size": 100.0, "connect_radius": 30.0, "n_objects": 4, "min_steps": 4, "max_steps": 10, "min_state_changes": 1}'
EVAL_ARGS="--n_batches 50 --batch_size 8"

echo "============================================"
echo "  FluidNet v1 — 3-Way Comparison"
echo "============================================"

echo ""
echo ">>> Training FluidNet-6L (10K steps)..."
python scripts/train_fluid.py \
    --config configs/fluid_6l.yaml \
    --task CW --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 1000 --grad_clip 1.0 --log_every 100 \
    --save_every 5000 --max_steps 10000 \
    --task_kwargs "$TASK_KWARGS" \
    --output_dir output_fluid_6l

echo ""
echo ">>> Training Flat-6L (10K steps)..."
python scripts/train_fluid.py \
    --config configs/fluid_flat.yaml \
    --task CW --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 1000 --grad_clip 1.0 --log_every 100 \
    --save_every 5000 --max_steps 10000 \
    --task_kwargs "$TASK_KWARGS" \
    --output_dir output_fluid_flat

echo ""
echo ">>> Training v6-metric-6L (10K steps)..."
python scripts/train_fluid.py \
    --config configs/fluid_v6metric.yaml \
    --task CW --batch_size 4 --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 1000 --grad_clip 1.0 --log_every 100 \
    --save_every 5000 --max_steps 10000 \
    --task_kwargs "$TASK_KWARGS" \
    --output_dir output_fluid_v6metric

echo ""
echo "============================================"
echo "  Evaluation — 3 Models"
echo "============================================"

echo ""
echo ">>> Evaluating FluidNet-6L..."
python scripts/eval_fluid_gridworld.py \
    --config configs/fluid_6l.yaml \
    --checkpoint output_fluid_6l/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo ">>> Evaluating Flat-6L..."
python scripts/eval_fluid_gridworld.py \
    --config configs/fluid_flat.yaml \
    --checkpoint output_fluid_flat/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo ">>> Evaluating v6-metric-6L..."
python scripts/eval_fluid_gridworld.py \
    --config configs/fluid_v6metric.yaml \
    --checkpoint output_fluid_v6metric/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo "============================================"
echo "  FluidNet v1 Experiment Complete"
echo "============================================"
