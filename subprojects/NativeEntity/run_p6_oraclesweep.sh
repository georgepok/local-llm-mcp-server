#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P6 ORACLE-SWEEP (can soft-latent injection drive verbatim recall?) #####"
env PHASE=6 SEED=0 GEN_ORACLE_SWEEP=1 python -u native_entity.py
echo "=== P6_GEN_DONE ==="
