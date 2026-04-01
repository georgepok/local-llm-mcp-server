#!/bin/bash
# FGN v3 — Affine Group Length Generalization Experiment
#
# Question: Can the model learn the composition algorithm on short chains
# and generalize to chains it has never seen?
#
# Training: 10-20 ops, sup_every matches n_ops (final answer only)
# Evaluation: 10, 25, 50, 75, 100, 120 ops (all final answer only)
#
# Why this tests transformers: The flat model may learn a shortcut for
# 10-20 ops that doesn't generalize. FGN's metric may provide geometric
# inductive bias that enables length generalization.
set -e

cd /workspace/fgn-v3

export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--stage 1 --tasks H --lr 3e-4 --warmup_steps 500 --max_steps 30000 --log_every 100 --save_every 5000"

echo "============================================================"
echo "Affine Group — Length Generalization Experiment"
echo "Train on 10-20 ops, eval on 10-120 ops"
echo "CUDA_MEMORY_FRACTION=$CUDA_MEMORY_FRACTION"
echo "Started: $(date)"
echo "============================================================"

# --- Training Phase ---
echo ""
echo "[$(date)] Training FLAT on 10-20 ops (final answer only)..."
python -u scripts/train_multitask.py \
    --config configs/affine_flat.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"min_ops": 10, "max_ops": 20, "sup_every": 20}' \
    --output_dir output_affine_lengthgen_flat \
    2>&1 | tee affine_lengthgen_flat.log

echo ""
echo "[$(date)] Training FGN on 10-20 ops (final answer only)..."
python -u scripts/train_multitask.py \
    --config configs/affine_fgn.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"min_ops": 10, "max_ops": 20, "sup_every": 20}' \
    --output_dir output_affine_lengthgen_fgn \
    2>&1 | tee affine_lengthgen_fgn.log

# --- Evaluation Phase ---
echo ""
echo "[$(date)] === Flat Model Evaluation ==="
python -u scripts/eval_affine.py \
    --config configs/affine_flat.yaml \
    --checkpoint output_affine_lengthgen_flat/stage1_taskH/checkpoints/final.pt \
    --n_batches 100 --batch_size 8 \
    2>&1 | tee eval_affine_lengthgen_flat.log

echo ""
echo "[$(date)] === FGN Model Evaluation ==="
python -u scripts/eval_affine.py \
    --config configs/affine_fgn.yaml \
    --checkpoint output_affine_lengthgen_fgn/stage1_taskH/checkpoints/final.pt \
    --n_batches 100 --batch_size 8 \
    2>&1 | tee eval_affine_lengthgen_fgn.log

# --- Also train an "oracle" on full range for comparison ---
echo ""
echo "[$(date)] Training FLAT oracle on 10-100 ops..."
python -u scripts/train_multitask.py \
    --config configs/affine_flat.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"min_ops": 10, "max_ops": 100, "sup_every": 100}' \
    --output_dir output_affine_oracle_flat \
    2>&1 | tee affine_oracle_flat.log

echo ""
echo "[$(date)] === Oracle Flat Evaluation ==="
python -u scripts/eval_affine.py \
    --config configs/affine_flat.yaml \
    --checkpoint output_affine_oracle_flat/stage1_taskH/checkpoints/final.pt \
    --n_batches 100 --batch_size 8 \
    2>&1 | tee eval_affine_oracle_flat.log

echo ""
echo "============================================================"
echo "Length generalization experiment complete: $(date)"
echo "============================================================"
