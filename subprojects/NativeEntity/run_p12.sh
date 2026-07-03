#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P12 UNSEEN-PAIR GENERALIZATION (8 names, held-out pairs, always-on field) #####"
env SEED=0 N_CONV=80 N_VAULT=4 FIELD_LAYERS=48,56 EPS=0.10 EPOCHS=150 FIELD_EPOCHS=40 FIELD_LR=5e-4 MAXNEW=14 GEN_MAXNEW=8 RECOLLECT=1 python -u native_p12.py
echo "=== P12_FULL_DONE ==="
