#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
G="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
B="SMOKE=0 LIFE=30 KCONT=3 GAIN0=1.0 FLAT=0 READMODE=gnn WRITE=off READT=1.0 ALIGN=1.0 GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=0 JUDGE=1 PHASE=1 PHASEF=1"
echo "##### SMOKE v1_cjudge (brief-factual F) #####"; env $B SEED=0 LORA=goal VIABILITY=cjudge    GOAL="$G" python -u organism3.py
echo "##### SMOKE v2collect (brief-factual F) #####"; env $B SEED=0 LORA=goal VIABILITY=v2collect GOAL="$G" python -u organism3.py
echo "=== SMOKE_V21B_DONE ==="
