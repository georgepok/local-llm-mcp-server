#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_KWARGS='{"n_rooms_min": 8, "n_rooms_max": 12, "n_objects": 6, "min_steps": 8, "max_steps": 15, "min_state_changes": 2, "randomize_topology": true}'
TRAIN_ARGS="--task W --batch_size 4 --lr 3e-4 --weight_decay 0.1 --warmup_steps 1000 --grad_clip 1.0 --log_every 100 --save_every 5000"
EVAL_ARGS="--n_batches 50 --batch_size 8"

echo "============================================"
echo "  FGN v7 Sandwich Experiment"
echo "============================================"

echo ""
echo ">>> [1/3] Training v7-sandwich (15K steps)..."
rm -rf output_v7_sandwich
python scripts/train_v7.py \
    --config configs/v7_sandwich.yaml \
    $TRAIN_ARGS \
    --max_steps 15000 \
    --task_kwargs "$TASK_KWARGS" \
    --output_dir output_v7_sandwich

echo ""
echo ">>> [2/3] Training v6-metric baseline (10K steps)..."
rm -rf output_v7_v6baseline
python scripts/train_v7.py \
    --config configs/v6_hier_metric.yaml \
    $TRAIN_ARGS \
    --max_steps 10000 \
    --task_kwargs "$TASK_KWARGS" \
    --output_dir output_v7_v6baseline

echo ""
echo ">>> [3/3] Training flat-8 baseline (15K steps)..."
rm -rf output_v7_flat8
python scripts/train_v7.py \
    --config configs/v7_flat8.yaml \
    $TRAIN_ARGS \
    --max_steps 15000 \
    --task_kwargs "$TASK_KWARGS" \
    --output_dir output_v7_flat8

echo ""
echo "============================================"
echo "  Evaluation"
echo "============================================"

echo ""
echo ">>> Evaluating v7-sandwich..."
python scripts/eval_v7_gridworld.py \
    --config configs/v7_sandwich.yaml \
    --checkpoint output_v7_sandwich/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo ">>> Evaluating v6-metric..."
python scripts/eval_v7_gridworld.py \
    --config configs/v6_hier_metric.yaml \
    --checkpoint output_v7_v6baseline/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo ">>> Evaluating flat-8..."
python scripts/eval_v7_gridworld.py \
    --config configs/v7_flat8.yaml \
    --checkpoint output_v7_flat8/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo "============================================"
echo "  v7 Experiment Complete"
echo "============================================"
