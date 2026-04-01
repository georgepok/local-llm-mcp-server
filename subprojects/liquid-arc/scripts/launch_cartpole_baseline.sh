#!/bin/bash
# Cartpole with pre-transition checkpoint (no geometric structure)
set -e
export TERM=xterm
export PYTHONPATH=/home/pokazge/liquid-arc:${PYTHONPATH:-}

cd /home/pokazge/IsaacLab

echo "=== PRE-TRANSITION (step 2500, CV~0.1) ==="
./isaaclab.sh -p /home/pokazge/liquid-arc/scripts/train_isaac.py \
    --task Isaac-Cartpole-Direct-v0 \
    --checkpoint /home/pokazge/liquid-arc/output_30m/checkpoints/step_2500.pt \
    --headless \
    --num_envs 1024 \
    --total_steps 500000 \
    --rollout_length 32 \
    --log_every 1

echo "=== BASELINE COMPLETE ==="
