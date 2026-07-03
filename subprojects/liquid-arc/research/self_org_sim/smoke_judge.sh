#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
SMOKE=1 SEED=0 FLAT=0 READMODE=gnn WRITE=stake ALIGN=1.0 GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=8 JUDGE=1 GOAL="keep returning to a lighthouse keeper alone for years" python -u organism3.py
