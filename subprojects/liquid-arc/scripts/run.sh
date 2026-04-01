#!/bin/bash
# LiquidARC training runner.
#
# Usage:
#   bash scripts/run.sh liquid   # Full 100K training (LiquidARC)
#   bash scripts/run.sh flat     # Flat transformer baseline
#   bash scripts/run.sh smoke    # 100-step sanity check

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-smoke}"
DATA_DIR="${DATA_DIR:-/workspace/fgn-v3/data/arc}"
CONFIG="configs/liquid_arc.yaml"

case "$MODE" in
  liquid)
    echo "=== LiquidARC full training (100K steps) ==="
    python scripts/train.py \
      --config "$CONFIG" \
      --data_dir "$DATA_DIR" \
      --output_dir output_liquid \
      --batch_size 16 \
      --lr 3e-4 \
      --geo_lr_mult 1.0 \
      --max_steps 100000 \
      --warmup_steps 1000 \
      --log_every 50 \
      --eval_every 1000 \
      --eval_batches 20 \
      --save_every 5000
    ;;

  flat)
    echo "=== Flat baseline training (100K steps) ==="
    # Override model_type in config via a temp file
    FLAT_CONFIG=$(mktemp /tmp/flat_config_XXXXX.yaml)
    python -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['model_type'] = 'flat'
with open('$FLAT_CONFIG', 'w') as f:
    yaml.dump(cfg, f)
"
    python scripts/train.py \
      --config "$FLAT_CONFIG" \
      --data_dir "$DATA_DIR" \
      --output_dir output_flat \
      --batch_size 16 \
      --lr 3e-4 \
      --max_steps 100000 \
      --warmup_steps 1000 \
      --log_every 50 \
      --eval_every 1000 \
      --eval_batches 20 \
      --save_every 5000
    rm -f "$FLAT_CONFIG"
    ;;

  smoke)
    echo "=== Smoke test (100 steps) ==="
    python scripts/train.py \
      --config "$CONFIG" \
      --data_dir "$DATA_DIR" \
      --output_dir output_smoke \
      --batch_size 2 \
      --lr 3e-4 \
      --max_steps 100 \
      --warmup_steps 10 \
      --log_every 10 \
      --eval_every 50 \
      --eval_batches 5 \
      --save_every 100
    ;;

  *)
    echo "Usage: bash scripts/run.sh [liquid|flat|smoke]"
    exit 1
    ;;
esac
