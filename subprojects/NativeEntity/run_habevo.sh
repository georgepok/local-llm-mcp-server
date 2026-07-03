#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### HABITAT ECOLOGY VALIDATION (oracle-good vs base vs always-bad viability) #####"
env SEED=0 N_WORLDS=24 MAXNEW=24 python -u habitat_evo.py
echo "=== HABITAT_VALID_DONE ==="
