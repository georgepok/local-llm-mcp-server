#!/bin/bash
# Visualize trained Anymal policy with live rendering
set -e
export TERM=xterm
export PYTHONPATH=/home/pokazge/liquid-arc:${PYTHONPATH:-}
export WARP_CACHE_PATH=/home/pokazge/.cache/warp_kernels
export DISPLAY=:0

cd /home/pokazge/IsaacLab

echo "=== Anymal Visualization (live render + video save) ==="
./isaaclab.sh -p /home/pokazge/liquid-arc/scripts/record_anymal.py \
    --checkpoint /home/pokazge/liquid-arc/output_isaac/anymal_run1.pt \
    --num_envs 4 \
    --n_steps 500 \
    --no-headless
