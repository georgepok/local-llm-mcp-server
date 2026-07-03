#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P0 baselines #####";        env SMOKE=1 PHASE=0 SEED=0 python -u native_entity.py
echo "##### P1 slot auto-continuity #####"; env SMOKE=1 PHASE=1 SEED=0 python -u native_entity.py
echo "=== NATIVE_SMOKE_DONE ==="
