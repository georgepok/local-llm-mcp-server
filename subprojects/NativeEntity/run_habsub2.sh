#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### HABITAT SUBSTRATE 2 (continuous margin viability; in-dist + held-out; EPS=0.20) #####"
env SEED=0 N_TRAIN=40 N_INDIST=16 N_TEST=18 K=12 SLOW_K=6 D_S=768 FIELD_LAYERS=40,48,56 EPS=0.20 FIELD_EPOCHS=50 MAXNEW=24 python -u habitat_substrate2.py
echo "=== HAB_SUB2_DONE ==="
