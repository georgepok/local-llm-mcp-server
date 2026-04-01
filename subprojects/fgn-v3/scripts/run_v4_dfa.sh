#!/bin/bash
# FGN v4 — 3-way Random DFA Experiment
#
# Three configurations to isolate metric contribution:
#   1. Flat baseline: dot-product attention only
#   2. GeoRoute-flat: GeoRoute with flat metric (g=1), learned log_t per layer
#   3. GeoRoute-metric: GeoRoute with learned metric + learned log_t per layer
#
# DFA: K=512 states, A=16 symbols, train 20-50 steps, eval 20-150 steps.

set -e

cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1

TASK_KWARGS='{"n_states": 512, "n_symbols": 16, "min_steps": 20, "max_steps": 50}'
COMMON_ARGS="--task R --batch_size 8 --lr 3e-4 --weight_decay 0.1 --warmup_steps 500 --grad_clip 1.0 --log_every 100 --save_every 5000"

export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================"
echo "  FGN v4 — 3-way Random DFA Experiment"
echo "  DFA: K=512 states, A=16 symbols"
echo "  Train: 20-50 steps, Eval: 20-150 steps"
echo "================================================================"

# --- 1. Flat baseline ---
echo ""
echo ">>> [1/3] Training Flat Baseline..."
python scripts/train_v4.py \
    --config configs/v4_dfa_flat.yaml \
    $COMMON_ARGS \
    --task_kwargs "$TASK_KWARGS" \
    --max_steps 30000 \
    --output_dir output_dfa_flat

# --- 2. GeoRoute-flat (g=1, learned log_t) ---
echo ""
echo ">>> [2/3] Training GeoRoute-Flat (g=1, learned log_t)..."
python scripts/train_v4.py \
    --config configs/v4_dfa_georoute_flat.yaml \
    $COMMON_ARGS \
    --task_kwargs "$TASK_KWARGS" \
    --max_steps 31000 \
    --output_dir output_dfa_georoute_flat

# --- 3. GeoRoute-metric (learned metric + learned log_t) ---
echo ""
echo ">>> [3/3] Training GeoRoute-Metric (learned metric + learned log_t)..."
python scripts/train_v4.py \
    --config configs/v4_dfa_georoute_metric.yaml \
    $COMMON_ARGS \
    --task_kwargs "$TASK_KWARGS" \
    --max_steps 31000 \
    --output_dir output_dfa_georoute_metric

# --- Evaluation ---
echo ""
echo "================================================================"
echo "  Evaluation — Random DFA Length Generalization"
echo "================================================================"

EVAL_ARGS="--n_batches 50 --batch_size 8 --n_states 512 --n_symbols 16"

echo ""
echo ">>> Evaluating Flat Baseline..."
python scripts/eval_dfa.py \
    --config configs/v4_dfa_flat.yaml \
    --checkpoint output_dfa_flat/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo ">>> Evaluating GeoRoute-Flat..."
python scripts/eval_dfa.py \
    --config configs/v4_dfa_georoute_flat.yaml \
    --checkpoint output_dfa_georoute_flat/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo ">>> Evaluating GeoRoute-Metric..."
python scripts/eval_dfa.py \
    --config configs/v4_dfa_georoute_metric.yaml \
    --checkpoint output_dfa_georoute_metric/checkpoints/final.pt \
    $EVAL_ARGS

echo ""
echo "================================================================"
echo "  3-way DFA Experiment Complete"
echo "================================================================"
