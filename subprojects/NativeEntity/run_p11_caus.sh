#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P11 CAUSALITY (always-on field; does varying S change the trajectory?) #####"
env PHASE=11 SEED=0 MODE=causality FIELD_LAYERS=40,48,56 EPS=0.10 python -u native_entity.py
echo "=== P11_DONE ==="
