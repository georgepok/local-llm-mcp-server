#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### HABITAT_RCF mode=${CFMODE} #####"
env SEED=0 python -u habitat_rcf.py
echo "=== HAB_RCF_SCRIPT_DONE ==="
