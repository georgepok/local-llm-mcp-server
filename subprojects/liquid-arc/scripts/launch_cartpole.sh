#!/bin/bash
# Launch LiquidARC Cartpole training via Isaac Lab's python environment
set -e
export TERM=xterm
export PYTHONPATH=/home/pokazge/liquid-arc:${PYTHONPATH:-}

cd /home/pokazge/IsaacLab
./isaaclab.sh -p /home/pokazge/liquid-arc/scripts/train_isaac.py \
    --task Isaac-Cartpole-Direct-v0 \
    --checkpoint /home/pokazge/liquid-arc/output_30m/checkpoints/step_10000.pt \
    --headless \
    --num_envs 1024 \
    --total_steps 500000 \
    --rollout_length 32 \
    --n_epochs 4 \
    --log_every 5
