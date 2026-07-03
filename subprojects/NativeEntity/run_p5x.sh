#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P5 CROSSWORLD (V1.2) #####"; env PHASE=5 SEED=0 P5_DEPLOY_ONLY=1 CROSSWORLD=1 MAXNEW=24 ARMS=trained,reset,frozen,base GROUPS=TRAIN,HELDOUT NDEP=8 python -u native_entity.py
echo "=== P5X_DONE ==="
