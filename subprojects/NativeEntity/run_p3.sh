#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P3 deploy TRAIN worlds (in-dist) #####"; env PHASE=2 SEED=0 DEPLOY_ONLY=1 N_CONV=12 T_TURNS=10 MAXNEW=28 WORLDS=lighthouse,spacecraft,archive python -u native_entity.py
echo "##### P3 deploy HELD-OUT worlds (transfer) #####"; env PHASE=2 SEED=0 DEPLOY_ONLY=1 N_CONV=12 T_TURNS=10 MAXNEW=28 WORLDS=legal,patient,codebase python -u native_entity.py
echo "=== P3_DONE ==="
