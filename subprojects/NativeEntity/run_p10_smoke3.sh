#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P10 SMOKE3 (generation prefix -> response hidden; check preservation) #####"
env PHASE=10 SEED=9 N_CONV=48 N_VAULT=6 NTOK=24 MAXNEW=14 EPOCHS=120 HEAD_EPOCHS=150 BILINEAR=1 RECOLLECT=1 python -u native_entity.py
echo "=== P10_SMOKE_DONE ==="
