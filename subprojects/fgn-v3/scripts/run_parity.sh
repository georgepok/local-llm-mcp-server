#!/bin/bash
# FGN v3 Phase 2 — Parity Task Training + OOD Evaluation
# Both models from random init, parameter-matched.
set -e

cd /workspace/fgn-v3

BS=16
LR=3e-4
WARMUP=500
STEPS=50000
LOG=100
SAVE=5000

echo "============================================="
echo "Phase 2 — Parity (Transformer-Breaking Task)"
echo "============================================="
echo "d_model=128, n_layers=4, n_heads=4"
echo "Task: binary parity, length 40, p(1)=0.5"
echo "Batch size: $BS"
echo "Steps: $STEPS"
echo "Date: $(date)"
echo ""

# ---- FGN Model ----
echo "===== FGN MODEL (random init) ====="
python scripts/train_multitask.py \
    --config configs/phase2_fgn.yaml \
    --stage 1 --tasks P \
    --batch_size $BS --lr $LR \
    --warmup_steps $WARMUP --max_steps $STEPS \
    --output_dir output_parity_fgn \
    --log_every $LOG --save_every $SAVE \
    2>&1 | tee output_parity_fgn_train.log

echo ""
echo "===== FGN EVALUATION ====="
python scripts/eval_parity.py \
    --config configs/phase2_fgn.yaml \
    --checkpoint output_parity_fgn/stage1_taskP/checkpoints/final.pt \
    --n_batches 200 --batch_size 16 \
    2>&1 | tee output_parity_fgn_eval.log

# ---- Flat Model ----
echo ""
echo "===== FLAT MODEL (random init, param-matched) ====="
python scripts/train_multitask.py \
    --config configs/phase2_flat.yaml \
    --stage 1 --tasks P \
    --batch_size $BS --lr $LR \
    --warmup_steps $WARMUP --max_steps $STEPS \
    --output_dir output_parity_flat \
    --log_every $LOG --save_every $SAVE \
    2>&1 | tee output_parity_flat_train.log

echo ""
echo "===== FLAT EVALUATION ====="
python scripts/eval_parity.py \
    --config configs/phase2_flat.yaml \
    --checkpoint output_parity_flat/stage1_taskP/checkpoints/final.pt \
    --n_batches 200 --batch_size 16 \
    2>&1 | tee output_parity_flat_eval.log

echo ""
echo "============================================="
echo "Phase 2 Parity complete!"
echo "Date: $(date)"
echo "============================================="
