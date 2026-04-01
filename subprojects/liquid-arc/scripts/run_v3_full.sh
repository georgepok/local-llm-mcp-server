#!/bin/bash
# V3 Full Pipeline: wait for hypernet training → run eval sweep
#
# Usage (inside container):
#   bash scripts/run_v3_full.sh

set -euo pipefail
cd /workspace/liquid-arc
export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3

HYPERNET_CKPT="output_hypernet/checkpoints/final.pt"

echo "=== V3 Full Pipeline ==="
echo "$(date): Waiting for hypernet training to complete..."

# Poll for the final checkpoint (train_hypernet.py creates it last)
while [ ! -f "$HYPERNET_CKPT" ]; do
    # Check if training process is still alive
    if ! pgrep -f "train_hypernet.py" > /dev/null 2>&1; then
        echo "$(date): ERROR — train_hypernet.py process died before creating final checkpoint"
        echo "Last 20 lines of training log:"
        tail -20 train_hypernet.log 2>/dev/null
        exit 1
    fi
    echo "$(date): Training still running... ($(tail -1 train_hypernet.log 2>/dev/null))"
    sleep 60
done

echo "$(date): Hypernet training complete! Final checkpoint: $HYPERNET_CKPT"
echo "Last 5 lines of training log:"
tail -5 train_hypernet.log

echo ""
echo "=== Starting Full Eval Sweep ==="
echo "$(date): V2 + V3a + Amortized across all checkpoints"

python3 -u scripts/eval_v3_sweep.py \
    --checkpoint_dir output_ttt_v2/checkpoints \
    --data_dir /workspace/fgn-v3/data/arc \
    --hypernet_checkpoint "$HYPERNET_CKPT" \
    --output eval_v3_results.json

echo ""
echo "$(date): Full pipeline complete"
echo "Results: eval_v3_results.json"
