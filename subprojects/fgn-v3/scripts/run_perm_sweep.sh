#!/bin/bash
# FGN v3 — S₅ Permutation Composition Sweep
# Tests flat vs FGN at increasing supervision sparsity.
# Hardest condition first (sup_every=50) — stop early if flat breaks.
# Sequential execution for DGX Spark unified memory safety.
set -e

cd /workspace/fgn-v3

export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--stage 1 --tasks G --lr 3e-4 --warmup_steps 500 --max_steps 30000 --log_every 100 --save_every 5000"

echo "============================================================"
echo "S₅ Permutation Composition — Supervision Sparsity Sweep"
echo "CUDA_MEMORY_FRACTION=$CUDA_MEMORY_FRACTION"
echo "Started: $(date)"
echo "============================================================"

# --- Condition 1: sup_every=50 (hardest — final answer only) ---
echo ""
echo "[$(date)] === sup_every=50 (50 compositions, final only) ==="

echo "[$(date)] Training FLAT (sup_every=50)..."
python -u scripts/train_multitask.py \
    --config configs/perm_flat.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 50, "min_perms": 50, "max_perms": 50}' \
    --output_dir output_perm_flat_sup50 \
    2>&1 | tee perm_flat_sup50.log

echo "[$(date)] Training FGN (sup_every=50)..."
python -u scripts/train_multitask.py \
    --config configs/perm_fgn.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 50, "min_perms": 50, "max_perms": 50}' \
    --output_dir output_perm_fgn_sup50 \
    2>&1 | tee perm_fgn_sup50.log

echo "[$(date)] Evaluating sup_every=50..."
echo "--- Flat ---"
python -u scripts/eval_perm.py \
    --config configs/perm_flat.yaml \
    --checkpoint output_perm_flat_sup50/stage1_taskG/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_perm_flat_sup50.log
echo "--- FGN ---"
python -u scripts/eval_perm.py \
    --config configs/perm_fgn.yaml \
    --checkpoint output_perm_fgn_sup50/stage1_taskG/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_perm_fgn_sup50.log

# --- Condition 2: sup_every=25 ---
echo ""
echo "[$(date)] === sup_every=25 (25 compositions per checkpoint) ==="

echo "[$(date)] Training FLAT (sup_every=25)..."
python -u scripts/train_multitask.py \
    --config configs/perm_flat.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 25, "min_perms": 50, "max_perms": 50}' \
    --output_dir output_perm_flat_sup25 \
    2>&1 | tee perm_flat_sup25.log

echo "[$(date)] Training FGN (sup_every=25)..."
python -u scripts/train_multitask.py \
    --config configs/perm_fgn.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 25, "min_perms": 50, "max_perms": 50}' \
    --output_dir output_perm_fgn_sup25 \
    2>&1 | tee perm_fgn_sup25.log

echo "[$(date)] Evaluating sup_every=25..."
echo "--- Flat ---"
python -u scripts/eval_perm.py \
    --config configs/perm_flat.yaml \
    --checkpoint output_perm_flat_sup25/stage1_taskG/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_perm_flat_sup25.log
echo "--- FGN ---"
python -u scripts/eval_perm.py \
    --config configs/perm_fgn.yaml \
    --checkpoint output_perm_fgn_sup25/stage1_taskG/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_perm_fgn_sup25.log

# --- Condition 3: sup_every=10 ---
echo ""
echo "[$(date)] === sup_every=10 (10 compositions per checkpoint) ==="

echo "[$(date)] Training FLAT (sup_every=10)..."
python -u scripts/train_multitask.py \
    --config configs/perm_flat.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 10, "min_perms": 50, "max_perms": 50}' \
    --output_dir output_perm_flat_sup10 \
    2>&1 | tee perm_flat_sup10.log

echo "[$(date)] Training FGN (sup_every=10)..."
python -u scripts/train_multitask.py \
    --config configs/perm_fgn.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 10, "min_perms": 50, "max_perms": 50}' \
    --output_dir output_perm_fgn_sup10 \
    2>&1 | tee perm_fgn_sup10.log

echo "[$(date)] Evaluating sup_every=10..."
echo "--- Flat ---"
python -u scripts/eval_perm.py \
    --config configs/perm_flat.yaml \
    --checkpoint output_perm_flat_sup10/stage1_taskG/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_perm_flat_sup10.log
echo "--- FGN ---"
python -u scripts/eval_perm.py \
    --config configs/perm_fgn.yaml \
    --checkpoint output_perm_fgn_sup10/stage1_taskG/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_perm_fgn_sup10.log

# --- Condition 4: sup_every=5 (easiest — baseline) ---
echo ""
echo "[$(date)] === sup_every=5 (5 compositions per checkpoint — easy baseline) ==="

echo "[$(date)] Training FLAT (sup_every=5)..."
python -u scripts/train_multitask.py \
    --config configs/perm_flat.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 5, "min_perms": 50, "max_perms": 50}' \
    --output_dir output_perm_flat_sup5 \
    2>&1 | tee perm_flat_sup5.log

echo "[$(date)] Training FGN (sup_every=5)..."
python -u scripts/train_multitask.py \
    --config configs/perm_fgn.yaml \
    $COMMON --batch_size 8 \
    --task_kwargs '{"sup_every": 5, "min_perms": 50, "max_perms": 50}' \
    --output_dir output_perm_fgn_sup5 \
    2>&1 | tee perm_fgn_sup5.log

echo "[$(date)] Evaluating sup_every=5..."
echo "--- Flat ---"
python -u scripts/eval_perm.py \
    --config configs/perm_flat.yaml \
    --checkpoint output_perm_flat_sup5/stage1_taskG/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_perm_flat_sup5.log
echo "--- FGN ---"
python -u scripts/eval_perm.py \
    --config configs/perm_fgn.yaml \
    --checkpoint output_perm_fgn_sup5/stage1_taskG/checkpoints/final.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_perm_fgn_sup5.log

echo ""
echo "============================================================"
echo "Permutation sweep complete: $(date)"
echo "============================================================"
