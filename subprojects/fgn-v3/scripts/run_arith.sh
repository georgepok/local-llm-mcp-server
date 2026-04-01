#!/bin/bash
# FGN v3 Phase 1b v3 — Arithmetic Chain Training (Small Models)
# d=128, 4L, 4H — capacity-constrained regime
# Task F: variable-depth mod-100 arithmetic, depth 4-8, 3 programs, shuffled
set -e

cd /workspace/fgn-v3

BS=16
LR=3e-4
WARMUP=500
STEPS=50000
LOG=100
SAVE=5000

echo "============================================="
echo "Phase 1b v3 — Arithmetic Chains (Small Models)"
echo "============================================="
echo "d_model=128, n_layers=4, n_heads=4"
echo "Batch size: $BS"
echo "Steps: $STEPS"
echo "Date: $(date)"
echo ""

# ---- Probe run (5K steps) to check difficulty ----
echo "===== PROBE: FGN (5K steps) ====="
python scripts/train_multitask.py \
    --config configs/arith_small_fgn.yaml \
    --stage 1 --tasks F \
    --batch_size $BS --lr $LR \
    --warmup_steps 200 --max_steps 5000 \
    --output_dir output_arith_probe_fgn \
    --log_every 100 --save_every 5000 \
    2>&1 | tee output_arith_probe_fgn.log

echo ""
echo "===== PROBE: FLAT (5K steps) ====="
python scripts/train_multitask.py \
    --config configs/arith_small_flat.yaml \
    --stage 1 --tasks F \
    --batch_size $BS --lr $LR \
    --warmup_steps 200 --max_steps 5000 \
    --output_dir output_arith_probe_flat \
    --log_every 100 --save_every 5000 \
    2>&1 | tee output_arith_probe_flat.log

echo ""
echo "============================================="
echo "Probe complete — check logs for loss curves"
echo "If CE > 0.5 at step 5000: task is hard enough"
echo "If CE < 0.01: still too easy, try d=64"
echo "============================================="
echo ""

# ---- Full run (only if probe shows task is hard) ----
# Uncomment after verifying probe results:

# echo "===== FULL: FGN (50K steps) ====="
# python scripts/train_multitask.py \
#     --config configs/arith_small_fgn.yaml \
#     --stage 1 --tasks F \
#     --batch_size $BS --lr $LR \
#     --warmup_steps $WARMUP --max_steps $STEPS \
#     --output_dir output_arith_fgn \
#     --log_every $LOG --save_every $SAVE \
#     2>&1 | tee output_arith_fgn_train.log
#
# echo ""
# echo "===== FULL: FLAT (50K steps) ====="
# python scripts/train_multitask.py \
#     --config configs/arith_small_flat.yaml \
#     --stage 1 --tasks F \
#     --batch_size $BS --lr $LR \
#     --warmup_steps $WARMUP --max_steps $STEPS \
#     --output_dir output_arith_flat \
#     --log_every $LOG --save_every $SAVE \
#     2>&1 | tee output_arith_flat_train.log

echo ""
echo "============================================="
echo "Done!"
echo "Date: $(date)"
echo "============================================="
