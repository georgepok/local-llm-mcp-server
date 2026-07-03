#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P8b BIG_VOCAB compositional-decoding test (40 values, 30 seen/10 unseen) #####"
env PHASE=8 BIG_VOCAB=1 SEED=0 BIG_M=2 N_SEEN=30 EPOCHS=150 V14_EPOCHS=60 MAXNEW=24 GEN_MAXNEW=40 RECOLLECT=1 python -u native_entity.py
echo "=== P8b_FULL_DONE ==="
