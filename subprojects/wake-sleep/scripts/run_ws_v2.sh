#!/bin/bash
# Wake-Sleep V2 training runner.
#
# Usage (inside container):
#   bash scripts/run_ws_v2.sh smoke    # 200-step sanity check
#   bash scripts/run_ws_v2.sh full     # Full 50K training
#   bash scripts/run_ws_v2.sh full --ode_checkpoint /path/to/ode.pt  # With warm-start

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=/workspace/wake-sleep:/workspace/liquid-arc:/workspace/fgn-v3
export CUDA_MEMORY_FRACTION=0.85

MODE="${1:-smoke}"
shift || true  # consume mode arg, pass rest through

DATA_DIR="${DATA_DIR:-/workspace/fgn-v3/data/arc}"
CONFIG="configs/wake_sleep_v2.yaml"

case "$MODE" in
  smoke)
    echo "=== Wake-Sleep V2 Smoke Test (200 steps) ==="
    echo "$(date): Starting..."
    python3 -u scripts/train_ws.py \
      --config "$CONFIG" \
      --data_dir "$DATA_DIR" \
      --output_dir output_ws_v2_smoke \
      --batch_size 2 \
      --max_steps 200 \
      --log_every 10 \
      --save_every 200 \
      --eval_every_cycles 999 \
      --eval_n_tasks 5 \
      "$@" \
      2>&1 | tee output_ws_v2_smoke/train.log
    echo "$(date): Smoke test complete"
    ;;

  full)
    echo "=== Wake-Sleep V2 Full Training (50K steps) ==="
    echo "$(date): Starting..."
    python3 -u scripts/train_ws.py \
      --config "$CONFIG" \
      --data_dir "$DATA_DIR" \
      --output_dir output_ws_v2 \
      --batch_size 8 \
      --max_steps 50000 \
      --log_every 50 \
      --save_every 2500 \
      --eval_every_cycles 2 \
      --eval_n_tasks 50 \
      "$@" \
      2>&1 | tee output_ws_v2/train.log
    echo "$(date): Full training complete"
    ;;

  *)
    echo "Usage: bash scripts/run_ws_v2.sh [smoke|full] [extra args...]"
    echo ""
    echo "Extra args (passed to train_ws.py):"
    echo "  --ode_checkpoint PATH   Warm-start from pre-trained ODE"
    echo "  --resume PATH           Resume from WS V2 checkpoint"
    echo "  --grad_clip FLOAT       Gradient clipping (default 1.0)"
    exit 1
    ;;
esac
