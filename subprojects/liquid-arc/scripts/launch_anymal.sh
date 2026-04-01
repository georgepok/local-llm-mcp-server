#!/bin/bash
# Anymal-C locomotion with post-transition LiquidARC
set -e
export TERM=xterm
export PYTHONPATH=/home/pokazge/liquid-arc:${PYTHONPATH:-}
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
# Use default Warp cache at ~/.cache/warp/ (per Isaac Lab issue #4813)

cd /home/pokazge/IsaacLab

echo "=== ANYMAL-C FLAT: Post-transition LiquidARC (UNFROZEN dynamics) ==="
./isaaclab.sh -p /home/pokazge/liquid-arc/scripts/train_isaac.py \
    --task Isaac-Velocity-Flat-Anymal-C-Direct-v0 \
    --checkpoint /home/pokazge/liquid-arc/output_30m/checkpoints/step_10000.pt \
    --unfreeze_dynamics \
    --headless \
    --num_envs 1024 \
    --total_steps 5000000 \
    --rollout_length 24 \
    --n_epochs 5 \
    --minibatch_size 4096 \
    --log_every 5

echo "=== ANYMAL COMPLETE ==="
