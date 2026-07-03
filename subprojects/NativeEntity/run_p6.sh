#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P6 main (collect+train) #####"; env PHASE=6 SEED=0 N_CONV=32 EPOCHS=150 MAXNEW=20 RECOLLECT=1 python -u native_entity.py
echo "##### P6 reset #####";  env PHASE=6 SEED=0 EPOCHS=150 ABLATE=reset python -u native_entity.py
echo "##### P6 frozen #####"; env PHASE=6 SEED=0 ABLATE=frozen python -u native_entity.py
echo "=== P6_FULL_DONE ==="
