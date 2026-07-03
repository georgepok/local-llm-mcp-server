#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P5 smoke (V1.1) #####"; env PHASE=5 SEED=0 EPOCHS=6 MAXNEW=24 ARMS=trained,base GROUPS=HELDOUT NDEP=2 python -u native_entity.py
echo "=== P5_SMOKE_DONE ==="
