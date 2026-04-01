#!/bin/bash
# FGN v3 Phase 1b v2 — Compound Reasoning Task training
# Both models from random init, 50K steps each.
set -e

cd /workspace/fgn-v3

BS=8
LR=3e-4
WARMUP=500
STEPS=50000
LOG=100
SAVE=5000

echo "============================================="
echo "CRT Phase 1b v2 — Compound Reasoning Task"
echo "============================================="
echo "Batch size: $BS"
echo "Steps: $STEPS"
echo "Date: $(date)"
echo ""

# ---- FGN Model ----
echo "===== FGN MODEL (random init) ====="
python scripts/train_multitask.py \
    --config configs/crt_fgn.yaml \
    --stage 1 --tasks E \
    --batch_size $BS --lr $LR \
    --warmup_steps $WARMUP --max_steps $STEPS \
    --output_dir output_crt_fgn \
    --log_every $LOG --save_every $SAVE \
    2>&1 | tee output_crt_fgn_train.log

echo ""
echo "===== FGN EVALUATION ====="
python scripts/eval_crt.py \
    --config configs/crt_fgn.yaml \
    --checkpoint output_crt_fgn/stage1_taskE/checkpoints/final.pt \
    --n_batches 200 --batch_size 8 \
    2>&1 | tee output_crt_fgn_eval.log

# ---- Flat Model ----
echo ""
echo "===== FLAT MODEL (random init) ====="
python scripts/train_multitask.py \
    --config configs/crt_flat.yaml \
    --stage 1 --tasks E \
    --batch_size $BS --lr $LR \
    --warmup_steps $WARMUP --max_steps $STEPS \
    --output_dir output_crt_flat \
    --log_every $LOG --save_every $SAVE \
    2>&1 | tee output_crt_flat_train.log

echo ""
echo "===== FLAT EVALUATION ====="
python scripts/eval_crt.py \
    --config configs/crt_flat.yaml \
    --checkpoint output_crt_flat/stage1_taskE/checkpoints/final.pt \
    --n_batches 200 --batch_size 8 \
    2>&1 | tee output_crt_flat_eval.log

echo ""
echo "============================================="
echo "CRT training complete!"
echo "Date: $(date)"
echo "============================================="
