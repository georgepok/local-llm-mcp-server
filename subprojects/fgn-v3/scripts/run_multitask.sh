#!/bin/bash
# Phase 1b: Multi-task training protocol
# Run on fgn-train container (DGX Spark)
set -e

cd /workspace/fgn-v3

FGN_CFG=configs/multitask_fgn.yaml
FLAT_CFG=configs/multitask_flat.yaml
FGN_OUT=output_multitask_fgn
FLAT_OUT=output_multitask_flat
RESUME=output/checkpoints/post_curriculum.pt
BS=8

mkdir -p $FGN_OUT $FLAT_OUT

echo "=========================================="
echo "PHASE 1b: Multi-Task Geometric Advantage"
echo "=========================================="

# ===== STAGE 1: Single-task baselines =====
echo ""
echo "=========================================="
echo "STAGE 1: Single-task baselines (2000 steps each)"
echo "=========================================="

echo "--- FGN model ---"
python -u scripts/train_multitask.py \
  --config $FGN_CFG \
  --stage 1 \
  --tasks A,B,C,D \
  --resume $RESUME \
  --max_steps 2000 \
  --batch_size $BS \
  --log_every 50 \
  --save_every 500 \
  --output_dir $FGN_OUT \
  2>&1 | tee $FGN_OUT/stage1.log

echo "--- Flat model ---"
python -u scripts/train_multitask.py \
  --config $FLAT_CFG \
  --stage 1 \
  --tasks A,B,C,D \
  --max_steps 2000 \
  --batch_size $BS \
  --log_every 50 \
  --save_every 500 \
  --output_dir $FLAT_OUT \
  2>&1 | tee $FLAT_OUT/stage1.log

# ===== STAGE 2: Mixed training =====
echo ""
echo "=========================================="
echo "STAGE 2: Mixed training (8000 steps)"
echo "=========================================="

echo "--- FGN model ---"
python -u scripts/train_multitask.py \
  --config $FGN_CFG \
  --stage 2 \
  --tasks A,B,C,D \
  --resume $RESUME \
  --max_steps 8000 \
  --batch_size $BS \
  --log_every 50 \
  --save_every 500 \
  --output_dir $FGN_OUT \
  2>&1 | tee $FGN_OUT/stage2.log

echo "--- Flat model ---"
python -u scripts/train_multitask.py \
  --config $FLAT_CFG \
  --stage 2 \
  --tasks A,B,C,D \
  --max_steps 8000 \
  --batch_size $BS \
  --log_every 50 \
  --save_every 500 \
  --output_dir $FLAT_OUT \
  2>&1 | tee $FLAT_OUT/stage2.log

# ===== STAGE 3: Task-switching speed =====
echo ""
echo "=========================================="
echo "STAGE 3: Task-switching speed (1000 steps each)"
echo "=========================================="

echo "--- FGN model ---"
python -u scripts/train_multitask.py \
  --config $FGN_CFG \
  --stage 3 \
  --tasks A,B,C,D \
  --max_steps 1000 \
  --batch_size $BS \
  --log_every 20 \
  --save_every 200 \
  --output_dir $FGN_OUT \
  2>&1 | tee $FGN_OUT/stage3.log

echo "--- Flat model ---"
python -u scripts/train_multitask.py \
  --config $FLAT_CFG \
  --stage 3 \
  --tasks A,B,C,D \
  --max_steps 1000 \
  --batch_size $BS \
  --log_every 20 \
  --save_every 200 \
  --output_dir $FLAT_OUT \
  2>&1 | tee $FLAT_OUT/stage3.log

# ===== EVALUATION =====
echo ""
echo "=========================================="
echo "EVALUATION"
echo "=========================================="

echo "--- FGN Stage 2 final ---"
python -u scripts/eval_multitask.py \
  --config $FGN_CFG \
  --checkpoint $FGN_OUT/stage2_mixed/checkpoints/final.pt \
  --tasks A,B,C,D \
  2>&1 | tee $FGN_OUT/eval_stage2.log

echo "--- Flat Stage 2 final ---"
python -u scripts/eval_multitask.py \
  --config $FLAT_CFG \
  --checkpoint $FLAT_OUT/stage2_mixed/checkpoints/final.pt \
  --tasks A,B,C,D \
  2>&1 | tee $FLAT_OUT/eval_stage2.log

echo ""
echo "ALL STAGES COMPLETE"
echo "Results in: $FGN_OUT/ and $FLAT_OUT/"
