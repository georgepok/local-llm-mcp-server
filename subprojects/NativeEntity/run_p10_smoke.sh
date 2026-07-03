#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P10 SMOKE (forward-only collection, small) #####"
env PHASE=10 SEED=9 N_CONV=24 N_VAULT=6 EPOCHS=40 HEAD_EPOCHS=60 BILINEAR=1 RECOLLECT=1 python -u native_entity.py
echo "=== P10_SMOKE_DONE ==="
