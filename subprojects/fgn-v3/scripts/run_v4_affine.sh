#!/bin/bash
# FGN v4 — Affine Group Length Generalization Experiment
#
# Primary success criterion: v4 FGN must not degrade on far-OOD conditions
# (fix the v3 failure where FGN dropped to 81.9% at 120 ops)
#
# Training: 10-20 ops, sup_every=20 (final answer only)
# Evaluation: 10, 25, 50, 75, 100, 120 ops
#
# Models:
#   1. FGN v4 (GeoRoute + StandardAttention + learned gate)
#   2. Flat baseline (standard transformer)
set -e

cd /workspace/fgn-v3

export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASK_KWARGS='{"min_ops": 10, "max_ops": 20, "sup_every": 20}'
COMMON="--task H --lr 3e-4 --warmup_steps 500 --max_steps 31000 --log_every 100 --save_every 5000 --batch_size 8"

echo "============================================================"
echo "FGN v4 — Affine Group Length Generalization"
echo "Train on 10-20 ops, eval on 10-120 ops"
echo "CUDA_MEMORY_FRACTION=$CUDA_MEMORY_FRACTION"
echo "Started: $(date)"
echo "============================================================"

# --- Train Flat Baseline ---
echo ""
echo "[$(date)] Training FLAT baseline..."
python -u scripts/train_v4.py \
    --config configs/v4_affine_flat.yaml \
    $COMMON \
    --task_kwargs "$TASK_KWARGS" \
    --output_dir output_v4_affine_flat \
    2>&1 | tee v4_affine_flat.log

# --- Train FGN v4 ---
echo ""
echo "[$(date)] Training FGN v4 (Phase 0: 1000 steps, Phase 1: 30000 steps)..."
python -u scripts/train_v4.py \
    --config configs/v4_affine_fgn.yaml \
    $COMMON \
    --task_kwargs "$TASK_KWARGS" \
    --output_dir output_v4_affine_fgn \
    2>&1 | tee v4_affine_fgn.log

# --- Evaluate ---
echo ""
echo "[$(date)] === Flat Baseline Evaluation ==="
python -u scripts/eval_affine.py \
    --config configs/v4_affine_flat.yaml \
    --checkpoint output_v4_affine_flat/checkpoints/final.pt \
    --n_batches 100 --batch_size 8 \
    2>&1 | tee eval_v4_affine_flat.log

echo ""
echo "[$(date)] === FGN v4 Evaluation ==="
python -u scripts/eval_affine.py \
    --config configs/v4_affine_fgn.yaml \
    --checkpoint output_v4_affine_fgn/checkpoints/final.pt \
    --n_batches 100 --batch_size 8 \
    2>&1 | tee eval_v4_affine_fgn.log

echo ""
echo "============================================================"
echo "FGN v4 affine experiment complete: $(date)"
echo "============================================================"
