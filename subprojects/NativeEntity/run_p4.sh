#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
# decisive cells first: HELDOUT (rule A/D) then TRAIN reference; slot-load-bearing arms + base/static
echo "##### P4a HELDOUT #####"; env PHASE=4 SEED=0 T_TURNS=8 MAXNEW=24 NDEP=9 ARMS=trained,reset,frozen,static,base GROUPS=HELDOUT python -u native_entity.py
echo "##### P4a TRAIN #####";   env PHASE=4 SEED=0 T_TURNS=8 MAXNEW=24 NDEP=9 ARMS=trained,reset,frozen,static,base GROUPS=TRAIN python -u native_entity.py
echo "=== P4A_DONE ==="
