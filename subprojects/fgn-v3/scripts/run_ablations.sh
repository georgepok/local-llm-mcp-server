#!/bin/bash
# Phase 1a.2 Ablation experiments
# Run both sequentially on single GPU, log to bind-mounted volume
set -e

cd /workspace/fgn-v3
mkdir -p output_ablation_no_smooth output_ablation_fast_metric

echo "=========================================="
echo "ABLATION A: No smoothness penalty (lambda=0)"
echo "=========================================="
python -u scripts/train.py \
  --config configs/phase1a2_no_smooth.yaml \
  --resume output/checkpoints/post_curriculum.pt \
  --max_steps 5000 \
  --batch_size 4 \
  --log_every 50 \
  --save_every 500 \
  --output_dir output_ablation_no_smooth \
  2>&1 | tee output_ablation_no_smooth/train.log

echo ""
echo "=========================================="
echo "ABLATION B: Fast metric LR (1x instead of 0.1x)"
echo "=========================================="
python -u scripts/train.py \
  --config configs/phase1a2_fast_metric.yaml \
  --resume output/checkpoints/post_curriculum.pt \
  --max_steps 5000 \
  --batch_size 4 \
  --log_every 50 \
  --save_every 500 \
  --output_dir output_ablation_fast_metric \
  2>&1 | tee output_ablation_fast_metric/train.log

echo ""
echo "BOTH ABLATIONS COMPLETE"
