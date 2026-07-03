#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P9 SMOKE (N_CONV=8, code-path) #####"
env PHASE=9 SEED=5 N_CONV=8 EPOCHS=30 MAXNEW=16 RECOLLECT=1 python -u native_entity.py
echo "=== P9_SMOKE_DONE ==="
