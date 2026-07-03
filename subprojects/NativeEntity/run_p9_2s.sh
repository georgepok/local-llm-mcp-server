#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P9 TWO_STAGE (cached): preserve-then-decide; bilinear match on FROZEN preserved slots #####"
env PHASE=9 SEED=0 N_CONV=48 EPOCHS=200 HEAD_EPOCHS=400 TWO_STAGE=1 BILINEAR=1 MAXNEW=16 RECOLLECT=0 python -u native_entity.py
echo "=== P9_FULL_DONE ==="
