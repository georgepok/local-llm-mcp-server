#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P12_ADAPTIVE_GATE_SLOT (input-dependent write gate vs fixed-gate, cached) #####"
env SEED=0 K=12 SLOW_K=6 D_S=768 EPOCHS=150 CONTRASTIVE=0 python -u native_p12_adapt.py
echo "=== P12_ADAPT_DONE ==="
