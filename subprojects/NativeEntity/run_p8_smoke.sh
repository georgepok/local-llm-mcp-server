#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P8 SMOKE (N_PER=1, code-path validation) #####"
env PHASE=8 SEED=7 N_PER=1 EPOCHS=20 V14_EPOCHS=10 GEN_MAXNEW=30 RECOLLECT=1 python -u native_entity.py
echo "=== P8_SMOKE_DONE ==="
