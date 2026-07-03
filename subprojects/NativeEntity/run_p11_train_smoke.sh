#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P11 MODE=train SMOKE (always-on field-actuated relational; deep field layers, bounded backprop) #####"
env PHASE=11 SEED=0 MODE=train FIELD_LAYERS=48,56 EPS=0.10 EPOCHS=120 FIELD_EPOCHS=20 FIELD_LR=3e-4 GEN_MAXNEW=8 python -u native_entity.py
echo "=== P11_TRAIN_SMOKE_DONE ==="
