#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### HABITAT_CF mode=${CFMODE} #####"
env SEED=0 python -u habitat_cf.py
echo "=== HAB_CF_SCRIPT_DONE ==="
