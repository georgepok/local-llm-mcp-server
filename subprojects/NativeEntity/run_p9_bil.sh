#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P9 BILINEAR (cached; relational match head + aux preservation) #####"
env PHASE=9 SEED=0 N_CONV=48 EPOCHS=200 AUX_W=0.5 BILINEAR=1 MAXNEW=16 RECOLLECT=0 python -u native_entity.py
echo "=== P9_FULL_DONE ==="
