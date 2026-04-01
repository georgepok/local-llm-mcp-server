#!/bin/bash
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_KWARGS='{"n_rooms_min": 8, "n_rooms_max": 12, "n_objects": 6, "min_steps": 8, "max_steps": 15, "min_state_changes": 2, "randomize_topology": true}'
TRAIN_ARGS="--task W --batch_size 4 --lr 3e-4 --weight_decay 0.1 --warmup_steps 1000 --grad_clip 1.0 --log_every 100 --save_every 5000 --max_steps 10000"

echo "============================================"
echo "  FGN v6 Grid World Experiment"
echo "============================================"

echo ""
echo ">>> [1/3] Training Flat Baseline..."
rm -rf output_v6_flat
python scripts/train_v6.py \
    --config configs/v6_flat.yaml \
    $TRAIN_ARGS \
    --task_kwargs "$TASK_KWARGS" \
    --output_dir output_v6_flat

echo ""
echo ">>> [2/3] Training Hierarchical-Flat..."
rm -rf output_v6_hier_flat
python scripts/train_v6.py \
    --config configs/v6_hier_flat.yaml \
    $TRAIN_ARGS \
    --task_kwargs "$TASK_KWARGS" \
    --output_dir output_v6_hier_flat

echo ""
echo ">>> [3/3] Training Hierarchical-Metric..."
rm -rf output_v6_hier_metric
python scripts/train_v6.py \
    --config configs/v6_hier_metric.yaml \
    $TRAIN_ARGS \
    --task_kwargs "$TASK_KWARGS" \
    --output_dir output_v6_hier_metric

echo ""
echo "============================================"
echo "  Evaluation — v6 Grid World"
echo "============================================"

EVAL_ARGS="--n_batches 50 --batch_size 8"

echo ""
echo ">>> Evaluating Flat Baseline..."
python scripts/eval_v6_gridworld.py \
    --config configs/v6_flat.yaml \
    --checkpoint output_v6_flat/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo ">>> Evaluating Hierarchical-Flat..."
python scripts/eval_v6_gridworld.py \
    --config configs/v6_hier_flat.yaml \
    --checkpoint output_v6_hier_flat/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo ">>> Evaluating Hierarchical-Metric..."
python scripts/eval_v6_gridworld.py \
    --config configs/v6_hier_metric.yaml \
    --checkpoint output_v6_hier_metric/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo "============================================"
echo "  v6 Experiment Complete"
echo "============================================"
