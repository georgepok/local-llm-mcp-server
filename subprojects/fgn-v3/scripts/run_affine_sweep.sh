#!/bin/bash
# FGN v3 — Affine Group (Z₉₇) Composition Sweep
# Tests flat vs FGN at increasing supervision sparsity.
# Hardest condition first (sup_every=100, final answer only).
# Sequential execution for DGX Spark unified memory safety.
set -e

cd /workspace/fgn-v3

export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--stage 1 --tasks H --lr 3e-4 --warmup_steps 500 --max_steps 30000 --log_every 100 --save_every 5000"

echo "============================================================"
echo "Affine Group Aff(Z₉₇) — Supervision Sparsity Sweep"
echo "CUDA_MEMORY_FRACTION=$CUDA_MEMORY_FRACTION"
echo "Started: $(date)"
echo "============================================================"

# --- Condition 1: sup_every=100 (hardest — 100 ops, final answer only) ---
echo ""
echo "[$(date)] === 100 ops, sup_every=100 (final only) ==="

echo "[$(date)] Training FLAT..."
python -u scripts/train_multitask.py \
    --config configs/affine_flat.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 100, "min_ops": 100, "max_ops": 100}' \
    --output_dir output_affine_flat_sup100 \
    2>&1 | tee affine_flat_sup100.log

echo "[$(date)] Training FGN..."
python -u scripts/train_multitask.py \
    --config configs/affine_fgn.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 100, "min_ops": 100, "max_ops": 100}' \
    --output_dir output_affine_fgn_sup100 \
    2>&1 | tee affine_fgn_sup100.log

echo "[$(date)] Evaluating..."
echo "--- Flat ---"
python -u scripts/eval_affine.py \
    --config configs/affine_flat.yaml \
    --checkpoint output_affine_flat_sup100/stage1_taskH/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_affine_flat_sup100.log
echo "--- FGN ---"
python -u scripts/eval_affine.py \
    --config configs/affine_fgn.yaml \
    --checkpoint output_affine_fgn_sup100/stage1_taskH/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_affine_fgn_sup100.log

# --- Condition 2: sup_every=50 ---
echo ""
echo "[$(date)] === 100 ops, sup_every=50 ==="

echo "[$(date)] Training FLAT..."
python -u scripts/train_multitask.py \
    --config configs/affine_flat.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 50, "min_ops": 100, "max_ops": 100}' \
    --output_dir output_affine_flat_sup50 \
    2>&1 | tee affine_flat_sup50.log

echo "[$(date)] Training FGN..."
python -u scripts/train_multitask.py \
    --config configs/affine_fgn.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 50, "min_ops": 100, "max_ops": 100}' \
    --output_dir output_affine_fgn_sup50 \
    2>&1 | tee affine_fgn_sup50.log

echo "[$(date)] Evaluating..."
echo "--- Flat ---"
python -u scripts/eval_affine.py \
    --config configs/affine_flat.yaml \
    --checkpoint output_affine_flat_sup50/stage1_taskH/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_affine_flat_sup50.log
echo "--- FGN ---"
python -u scripts/eval_affine.py \
    --config configs/affine_fgn.yaml \
    --checkpoint output_affine_fgn_sup50/stage1_taskH/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_affine_fgn_sup50.log

# --- Condition 3: sup_every=25 ---
echo ""
echo "[$(date)] === 100 ops, sup_every=25 ==="

echo "[$(date)] Training FLAT..."
python -u scripts/train_multitask.py \
    --config configs/affine_flat.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 25, "min_ops": 100, "max_ops": 100}' \
    --output_dir output_affine_flat_sup25 \
    2>&1 | tee affine_flat_sup25.log

echo "[$(date)] Training FGN..."
python -u scripts/train_multitask.py \
    --config configs/affine_fgn.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 25, "min_ops": 100, "max_ops": 100}' \
    --output_dir output_affine_fgn_sup25 \
    2>&1 | tee affine_fgn_sup25.log

echo "[$(date)] Evaluating..."
echo "--- Flat ---"
python -u scripts/eval_affine.py \
    --config configs/affine_flat.yaml \
    --checkpoint output_affine_flat_sup25/stage1_taskH/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_affine_flat_sup25.log
echo "--- FGN ---"
python -u scripts/eval_affine.py \
    --config configs/affine_fgn.yaml \
    --checkpoint output_affine_fgn_sup25/stage1_taskH/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_affine_fgn_sup25.log

# --- Condition 4: sup_every=10 (easiest) ---
echo ""
echo "[$(date)] === 100 ops, sup_every=10 ==="

echo "[$(date)] Training FLAT..."
python -u scripts/train_multitask.py \
    --config configs/affine_flat.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 10, "min_ops": 100, "max_ops": 100}' \
    --output_dir output_affine_flat_sup10 \
    2>&1 | tee affine_flat_sup10.log

echo "[$(date)] Training FGN..."
python -u scripts/train_multitask.py \
    --config configs/affine_fgn.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 10, "min_ops": 100, "max_ops": 100}' \
    --output_dir output_affine_fgn_sup10 \
    2>&1 | tee affine_fgn_sup10.log

echo "[$(date)] Evaluating..."
echo "--- Flat ---"
python -u scripts/eval_affine.py \
    --config configs/affine_flat.yaml \
    --checkpoint output_affine_flat_sup10/stage1_taskH/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_affine_flat_sup10.log
echo "--- FGN ---"
python -u scripts/eval_affine.py \
    --config configs/affine_fgn.yaml \
    --checkpoint output_affine_fgn_sup10/stage1_taskH/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_affine_fgn_sup10.log

echo ""
echo "============================================================"
echo "Affine sweep complete: $(date)"
echo "============================================================"
