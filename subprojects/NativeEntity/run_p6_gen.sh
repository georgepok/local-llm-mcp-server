#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P6 GEN (re-inject slot-carried value, ask, check output) #####"
env PHASE=6 SEED=0 GEN6=1 ABLATE=none EPOCHS=150 GEN_EPOCHS=150 GEN_MAXNEW=80 N_CONV=32 MAXNEW=20 RECOLLECT=0 python -u native_entity.py
echo "=== P6_GEN_DONE ==="
