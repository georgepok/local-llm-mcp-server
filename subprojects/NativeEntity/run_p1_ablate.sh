#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P1 main (persistent+trained) #####"; env PHASE=1 SEED=0 EPOCHS=120 python -u native_entity.py
echo "##### P1 reset (no persistence) #####";     env PHASE=1 SEED=0 EPOCHS=120 ABLATE=reset python -u native_entity.py
echo "##### P1 frozen (no training) #####";       env PHASE=1 SEED=0 ABLATE=frozen python -u native_entity.py
echo "=== P1_ABLATE_DONE ==="
