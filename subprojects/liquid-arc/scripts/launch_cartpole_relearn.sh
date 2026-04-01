#!/bin/bash
# Cartpole re-learning test: train from ARC checkpoint, save, then re-train from saved
set -e
export TERM=xterm
export PYTHONPATH=/home/pokazge/liquid-arc:${PYTHONPATH:-}

cd /home/pokazge/IsaacLab

echo "=== RUN 1: From ARC checkpoint (cold start) ==="
./isaaclab.sh -p /home/pokazge/liquid-arc/scripts/train_isaac.py \
    --task Isaac-Cartpole-Direct-v0 \
    --checkpoint /home/pokazge/liquid-arc/output_30m/checkpoints/step_10000.pt \
    --headless \
    --num_envs 1024 \
    --total_steps 500000 \
    --rollout_length 32 \
    --log_every 1

echo ""
echo "=== RUN 2: From Cartpole checkpoint (warm start) ==="
./isaaclab.sh -p /home/pokazge/liquid-arc/scripts/train_isaac.py \
    --task Isaac-Cartpole-Direct-v0 \
    --checkpoint /home/pokazge/liquid-arc/output_30m/checkpoints/step_10000.pt \
    --resume_isaac /home/pokazge/liquid-arc/output_isaac/cartpole_final.pt \
    --headless \
    --num_envs 1024 \
    --total_steps 500000 \
    --rollout_length 32 \
    --log_every 1

echo "=== BOTH RUNS COMPLETE ==="
