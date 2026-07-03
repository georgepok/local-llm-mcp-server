#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### WORLD_POP mode=${WPMODE} families=${FAMILIES} SEED=${SEED:-0} #####"
env SEED=${SEED:-0} python -u world_pop.py
echo "=== WP_SCRIPT_DONE ==="
