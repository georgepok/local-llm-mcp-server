#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/NativeEntity
echo "##### P12_SLOT_GRAPH_CONTRASTIVE (slot-self-attn + supcon + curriculum, cached) #####"
env SEED=0 K=12 SLOW_K=6 D_S=768 TAU=0.1 CW=1.0 STAGE_EP=80 BS=16 python -u native_p12_graph.py
echo "=== P12_GRAPH_DONE ==="
