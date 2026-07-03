#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/liquid-arc:/home/pokazge/Isaac-GR00T
export HF_HOME=/home/pokazge/hf_cache; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/pokazge/liquid-arc/research/self_org_sim
python train_manifold_actuator3.py
