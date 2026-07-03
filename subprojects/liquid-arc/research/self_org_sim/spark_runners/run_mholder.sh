#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/liquid-arc:/home/pokazge/Isaac-GR00T
cd /home/pokazge/liquid-arc/research/self_org_sim
python train_manifold_holder.py
