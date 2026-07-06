#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P11 MODE=train FULL (balanced match/nonmatch; always-on field relational) #####"
env PHASE=11 SEED=0 MODE=train FIELD_LAYERS=48,56 EPS=0.10 EPOCHS=150 FIELD_EPOCHS=40 FIELD_LR=5e-4 BALANCE=1 GEN_MAXNEW=8 python -u native_entity.py
echo "=== P11_TRAIN_FULL_DONE ==="
