#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P5 full (V1.1) #####"; env PHASE=5 SEED=0 EPOCHS=40 MAXNEW=24 ARMS=trained,reset,frozen,base GROUPS=HELDOUT,TRAIN NDEP=8 python -u native_entity.py
echo "=== P5_FULL_DONE ==="
