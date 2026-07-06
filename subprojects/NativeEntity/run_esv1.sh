#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### ENTITY_SUBSTRATE_V1 (closed-loop characterization, no action signal) #####"
env SEED=0 python -u entity_substrate.py
echo "=== ESV1_SCRIPT_DONE ==="
