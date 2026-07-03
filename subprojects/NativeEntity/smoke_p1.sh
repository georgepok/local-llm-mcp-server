#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P1 collect+train smoke #####"; env PHASE=1 SEED=0 N_CONV=12 T_TURNS=10 EPOCHS=80 RECOLLECT=1 python -u native_entity.py
echo "=== P1V2_SMOKE_DONE ==="
