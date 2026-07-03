#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### MECH smoke #####"; env PHASE=5 MECH=1 SEED=0 MAXNEW=24 python -u native_entity.py
echo "=== MECH_SMOKE_DONE ==="
