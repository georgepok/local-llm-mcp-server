#!/bin/bash
# Universality Probe — run all 8 experiments (4 domains × 2 conditions)
#
# Usage:
#   bash scripts/run_universality_probe.sh <5M_CHECKPOINT_PATH>
#
# Example:
#   bash scripts/run_universality_probe.sh /workspace/liquid-arc/output_30m/checkpoint_10000.pt
#
# Runs sequentially on DGX Spark (unified memory, must cap at 85%)

set -euo pipefail

CHECKPOINT="${1:?Usage: $0 <5M_CHECKPOINT_PATH>}"
CONFIG="configs/universality_probe.yaml"
DATA_DIR="/workspace/fgn-v3/data/arc-repo/data"
BASE_OUT="output_universality"

export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "  UNIVERSALITY PROBE — Geometric Substrate"
echo "  Checkpoint: ${CHECKPOINT}"
echo "============================================"
echo ""

TRANSFER_STEPS=11500   # resume from 10000, run 1500 additional
BASELINE_STEPS=2000    # from scratch, enough to see dynamics

for domain in sorting logic pattern graph; do
    echo "================================================"
    echo "  TRANSFER: ${domain} (from 5M checkpoint)"
    echo "================================================"
    python scripts/train.py \
        --config "${CONFIG}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${BASE_OUT}/${domain}_transfer" \
        --domain "${domain}" \
        --resume "${CHECKPOINT}" \
        --max_steps ${TRANSFER_STEPS} \
        --log_every 50 \
        --eval_every 500 \
        --save_every 5000 \
        --batch_size 16

    echo ""
    echo "================================================"
    echo "  BASELINE: ${domain} (from scratch)"
    echo "================================================"
    python scripts/train.py \
        --config "${CONFIG}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${BASE_OUT}/${domain}_baseline" \
        --domain "${domain}" \
        --max_steps ${BASELINE_STEPS} \
        --log_every 50 \
        --eval_every 500 \
        --save_every 5000 \
        --batch_size 16

    echo ""
done

echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Results in: ${BASE_OUT}/"
echo "============================================"
