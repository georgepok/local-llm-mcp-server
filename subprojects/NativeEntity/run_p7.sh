#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P7 V1.4 slot-cross-attn CONTINUOUS actuator #####"
env PHASE=7 SEED=0 EPOCHS=150 V14_EPOCHS=40 V14_INJ=52 GEN_MAXNEW=40 N_CONV=32 RECOLLECT=0 python -u native_entity.py
echo "=== P7_FULL_DONE ==="
