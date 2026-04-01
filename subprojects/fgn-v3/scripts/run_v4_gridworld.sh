#!/bin/bash
# FGN v4 — 3-way Grid World Navigation Experiment
#
# Three configurations to test metric hypothesis on interactive environment:
#   1. Flat baseline: dot-product attention only
#   2. GeoRoute-flat: GeoRoute with flat metric (g=1), learned log_t per layer
#   3. GeoRoute-metric: GeoRoute with learned metric + learned log_t per layer
#
# Grid world: 5 rooms, 4 objects, 4-7 step plans, ≥1 state change.
# Eval: ID (5 rooms, 4-7 steps) through far-OOD (8 rooms, 16-20 steps).

set -e

cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1

TASK_KWARGS='{"n_rooms": 5, "n_objects": 4, "min_steps": 4, "max_steps": 7, "min_state_changes": 1}'
COMMON_ARGS="--task W --batch_size 8 --lr 3e-4 --weight_decay 0.1 --warmup_steps 500 --grad_clip 1.0 --log_every 100 --save_every 5000"

export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================"
echo "  FGN v4 — 3-way Grid World Navigation Experiment"
echo "  World: 5 rooms, 4 objects, 4-7 steps, ≥1 state change"
echo "  Eval: ID through 8 rooms, 16-20 step OOD"
echo "================================================================"

# --- 1. Flat baseline ---
echo ""
echo ">>> [1/3] Training Flat Baseline..."
python scripts/train_v4.py \
    --config configs/v4_gridworld_flat.yaml \
    $COMMON_ARGS \
    --task_kwargs "$TASK_KWARGS" \
    --max_steps 30000 \
    --output_dir output_gridworld_flat

# --- 2. GeoRoute-flat (g=1, learned log_t) ---
echo ""
echo ">>> [2/3] Training GeoRoute-Flat (g=1, learned log_t)..."
python scripts/train_v4.py \
    --config configs/v4_gridworld_georoute_flat.yaml \
    $COMMON_ARGS \
    --task_kwargs "$TASK_KWARGS" \
    --max_steps 31000 \
    --output_dir output_gridworld_georoute_flat

# --- 3. GeoRoute-metric (learned metric + learned log_t) ---
echo ""
echo ">>> [3/3] Training GeoRoute-Metric (learned metric + learned log_t)..."
python scripts/train_v4.py \
    --config configs/v4_gridworld_georoute_metric.yaml \
    $COMMON_ARGS \
    --task_kwargs "$TASK_KWARGS" \
    --max_steps 31000 \
    --output_dir output_gridworld_georoute_metric

# --- Evaluation ---
echo ""
echo "================================================================"
echo "  Evaluation — Grid World Length Generalization"
echo "================================================================"

EVAL_ARGS="--n_batches 50 --batch_size 8"

echo ""
echo ">>> Evaluating Flat Baseline..."
python scripts/eval_gridworld.py \
    --config configs/v4_gridworld_flat.yaml \
    --checkpoint output_gridworld_flat/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo ">>> Evaluating GeoRoute-Flat..."
python scripts/eval_gridworld.py \
    --config configs/v4_gridworld_georoute_flat.yaml \
    --checkpoint output_gridworld_georoute_flat/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo ">>> Evaluating GeoRoute-Metric..."
python scripts/eval_gridworld.py \
    --config configs/v4_gridworld_georoute_metric.yaml \
    --checkpoint output_gridworld_georoute_metric/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo "================================================================"
echo "  3-way Grid World Experiment Complete"
echo "================================================================"
