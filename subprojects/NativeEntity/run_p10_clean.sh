#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P10 CLEAN (multi-depth aux, 6-prefix, unique cache, N_CONV=240) #####"
env PHASE=10 SEED=0 CACHE_TAG=_clean1 N_CONV=240 N_VAULT=6 NTOK=24 MAXNEW=14 EPOCHS=150 HEAD_EPOCHS=400 BILINEAR=1 RECOLLECT=1 python -u native_entity.py
echo "=== P10_FULL_DONE ==="
