#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P8 FULL (unseen-value split, N_PER=3) #####"
env PHASE=8 SEED=0 N_PER=3 EPOCHS=150 V14_EPOCHS=40 GEN_MAXNEW=40 RECOLLECT=1 python -u native_entity.py
echo "=== P8_FULL_DONE ==="
