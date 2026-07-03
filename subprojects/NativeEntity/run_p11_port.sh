#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P11 PRESERVATION-PORT SMOKE (phase9 full-episode preservation, post-hoc rclf) #####"
env PHASE=11 SEED=0 MODE=preserve_port EPOCHS=150 python -u native_entity.py
echo "=== P11_PORT_DONE ==="
