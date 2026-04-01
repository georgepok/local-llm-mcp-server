#!/bin/bash
# FGN v3 Phase 2 — Long-Sequence State Tracking (Sequential, Small Model)
# Runs flat first, then FGN, to avoid unified memory exhaustion on DGX Spark.
# Model halved (d=64, 3L, 2H) and torch.compile disabled to fit seq_len=5120.
# Memory fraction capped at 85% to leave headroom for OS on unified memory.
set -e

cd /workspace/fgn-v3

# Unified memory safety: cap PyTorch's GPU pool at 85% of physical RAM
# and use expandable segments to avoid up-front reservation.
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS="--stage 1 --tasks S --lr 3e-4 --warmup_steps 500 --max_steps 50000 --log_every 100 --save_every 5000"

echo "============================================================"
echo "Phase 2 Long-Seq State Tracking — Sequential (Small Model)"
echo "CUDA_MEMORY_FRACTION=$CUDA_MEMORY_FRACTION"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "Started: $(date)"
echo "============================================================"

# --- Phase 1: Flat baseline ---
echo ""
echo "[$(date)] Starting FLAT training (bs=8, d=64, 3L, seq_len=5120)..."
python -u scripts/train_multitask.py \
    --config configs/phase2_long_small_flat.yaml \
    $COMMON_ARGS \
    --batch_size 8 \
    --output_dir output_state_long_flat \
    2>&1 | tee state_long_flat.log

echo "[$(date)] FLAT training complete."

# --- Phase 2: FGN ---
echo ""
echo "[$(date)] Starting FGN training (bs=2, d=64, 3L, seq_len=5120)..."
python -u scripts/train_multitask.py \
    --config configs/phase2_long_small_fgn.yaml \
    $COMMON_ARGS \
    --batch_size 2 \
    --output_dir output_state_long_fgn \
    2>&1 | tee state_long_fgn.log

echo "[$(date)] FGN training complete."

# --- Phase 3: Evaluation ---
echo ""
echo "[$(date)] Running OOD evaluation — Flat..."
python -u scripts/eval_state.py \
    --config configs/phase2_long_small_flat.yaml \
    --checkpoint output_state_long_flat/stage1_taskS/checkpoints/step_50000.pt \
    --n_batches 50 --batch_size 8 \
    2>&1 | tee eval_state_long_flat.log

echo ""
echo "[$(date)] Running OOD evaluation — FGN..."
python -u scripts/eval_state.py \
    --config configs/phase2_long_small_fgn.yaml \
    --checkpoint output_state_long_fgn/stage1_taskS/checkpoints/step_50000.pt \
    --n_batches 50 --batch_size 2 \
    2>&1 | tee eval_state_long_fgn.log

echo ""
echo "============================================================"
echo "All done: $(date)"
echo "============================================================"
