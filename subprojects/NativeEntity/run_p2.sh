#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P2 full (N_CONV=24) #####"; env PHASE=2 SEED=0 N_CONV=24 T_TURNS=10 MAXNEW=28 BETA_SIGMA=0.4 python -u native_entity.py
echo "=== P2_FULL_DONE ==="
