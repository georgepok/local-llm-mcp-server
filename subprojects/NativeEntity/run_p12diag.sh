#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P12 DIAG (capacity vs collection: post-hoc recall at NR=2,4,8 on P12 cache) #####"
env SEED=0 python -u native_p12_diag.py
echo "=== P12_DIAG_DONE ==="
