#!/bin/bash
# FGN v5 — 3-way Hierarchical Gridworld Experiment
#
# Three configurations:
#   1. Flat baseline
#   2. Hierarchical with flat metric (g=1, learned log_t + learned threshold)
#   3. Hierarchical with learned metric (full v5)
#
# Harder gridworld: 8-12 rooms randomized, 6 objects, 8-15 step plans, world desc prefix.

set -e

cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1

TASK_KWARGS='{"n_rooms_min": 8, "n_rooms_max": 12, "n_objects": 6, "min_steps": 8, "max_steps": 15, "min_state_changes": 2, "randomize_topology": true}'
COMMON_ARGS="--task W --batch_size 8 --lr 3e-4 --weight_decay 0.1 --warmup_steps 1000 --grad_clip 1.0 --log_every 100 --save_every 10000"

export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "  FGN v5 — Hierarchical Gridworld Experiment"
echo "  World: 8-12 rooms, 6 objects, 8-15 steps"
echo "  Eval: ID through 16-18rm, 25-30 step stress"
echo "============================================"

# --- 1. Flat baseline ---
echo ""
echo ">>> [1/3] Training Flat Baseline..."
python scripts/train_v5.py \
    --config configs/v5_gridworld_flat.yaml \
    $COMMON_ARGS \
    --task_kwargs "$TASK_KWARGS" \
    --max_steps 50000 \
    --output_dir output_v5_flat

# --- 2. Hierarchical-flat (g=1, learned log_t + threshold) ---
echo ""
echo ">>> [2/3] Training Hierarchical-Flat..."
python scripts/train_v5.py \
    --config configs/v5_gridworld_hierarchical_flat.yaml \
    $COMMON_ARGS \
    --task_kwargs "$TASK_KWARGS" \
    --max_steps 50000 \
    --output_dir output_v5_hier_flat

# --- 3. Hierarchical-metric (learned metric + log_t + threshold) ---
echo ""
echo ">>> [3/3] Training Hierarchical-Metric..."
python scripts/train_v5.py \
    --config configs/v5_gridworld_hierarchical_metric.yaml \
    $COMMON_ARGS \
    --task_kwargs "$TASK_KWARGS" \
    --max_steps 50000 \
    --output_dir output_v5_hier_metric

# --- Evaluation ---
echo ""
echo "============================================"
echo "  Evaluation — v5 Grid World"
echo "============================================"

EVAL_ARGS="--n_batches 50 --batch_size 8"

echo ""
echo ">>> Evaluating Flat Baseline..."
python scripts/eval_v5_gridworld.py \
    --config configs/v5_gridworld_flat.yaml \
    --checkpoint output_v5_flat/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo ">>> Evaluating Hierarchical-Flat..."
python scripts/eval_v5_gridworld.py \
    --config configs/v5_gridworld_hierarchical_flat.yaml \
    --checkpoint output_v5_hier_flat/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo ">>> Evaluating Hierarchical-Metric..."
python scripts/eval_v5_gridworld.py \
    --config configs/v5_gridworld_hierarchical_metric.yaml \
    --checkpoint output_v5_hier_metric/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo "============================================"
echo "  v5 Experiment Complete"
echo "============================================"
