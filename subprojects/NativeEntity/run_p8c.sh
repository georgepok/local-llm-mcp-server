#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P8c COPY/POINTER actuator (reuses even cache, 12 seen/4 unseen) #####"
env PHASE=8 ACT=copy SEED=0 N_PER=3 EPOCHS=150 V14_EPOCHS=60 GEN_MAXNEW=40 RECOLLECT=0 python -u native_entity.py
echo "=== P8c_FULL_DONE ==="
