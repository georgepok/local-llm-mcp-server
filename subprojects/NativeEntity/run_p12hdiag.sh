#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P12 HIDDEN-INFO LOCALIZER (raw response-hidden 8-rule separability) #####"
env SEED=0 python -u native_p12_hiddendiag.py
echo "=== P12_HIDDENDIAG_DONE ==="
