#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P8d CONTENT-CROSS actuator (position-dependent, readout-keyed; reuses even cache) #####"
env PHASE=8 ACT=copy ACTKIND=content SEED=0 N_PER=3 EPOCHS=150 V14_EPOCHS=80 GEN_MAXNEW=40 RECOLLECT=0 python -u native_entity.py
echo "=== P8d_FULL_DONE ==="
