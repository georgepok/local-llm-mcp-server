#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P10 CACHE-TEST (post-hoc preservation check on existing 480 cache) #####"
env PHASE=10 SEED=0 N_CONV=480 N_VAULT=6 NTOK=24 EPOCHS=120 HEAD_EPOCHS=200 BILINEAR=1 RECOLLECT=0 python -u native_entity.py
echo "=== P10_FULL_DONE ==="
