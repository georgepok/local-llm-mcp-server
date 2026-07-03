#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### HABITAT SUBSTRATE CYCLE (always-on field + adaptive slots; consequence-distill; held-out template) #####"
env SEED=0 N_TRAIN=40 N_TEST=18 K=12 SLOW_K=6 D_S=768 FIELD_LAYERS=48,56 EPS=0.10 FIELD_EPOCHS=40 MAXNEW=24 python -u habitat_substrate.py
echo "=== HAB_SUBSTRATE_DONE ==="
