#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P2 smoke #####"; env PHASE=2 SEED=0 N_CONV=6 T_TURNS=8 MAXNEW=28 python -u native_entity.py
echo "=== P2_SMOKE_DONE ==="
