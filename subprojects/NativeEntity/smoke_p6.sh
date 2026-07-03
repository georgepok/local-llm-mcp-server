#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P6 smoke #####"; env PHASE=6 SEED=0 N_CONV=8 EPOCHS=80 MAXNEW=20 RECOLLECT=1 python -u native_entity.py
echo "=== P6_SMOKE_DONE ==="
