#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P12_CAPACITY_BUMP_DIAG (5 capacity variants x NR=2,4,8 on cached P12) #####"
env SEED=0 EPOCHS=120 RCLF_EP=150 python -u native_p12_capdiag.py
echo "=== P12_CAPDIAG_DONE ==="
