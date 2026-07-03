#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P12_RELATIONAL_ADAPTIVE (8-name unseen-pair relational; adaptive-gate slots + always-on field) #####"
env SEED=0 ARM=adaptive K=12 SLOW_K=6 D_S=768 FIELD_LAYERS=48,56 EPS=0.10 EPOCHS=150 FIELD_EPOCHS=40 FIELD_LR=5e-4 GEN_MAXNEW=8 python -u native_p12_rel.py
echo "=== P12_REL_FULL_DONE ==="
