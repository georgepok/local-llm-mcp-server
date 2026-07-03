#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
GOALTEXT="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
# RM WR AL GATE  per arm
for ARM in llmonly gnn_nogate gnn_oracle gnn_learned point_learned; do
  case $ARM in
    llmonly)       RM=gnn;   WR=off;     AL=0;   GT=none ;;
    gnn_nogate)    RM=gnn;   WR=natural; AL=1.0; GT=none ;;
    gnn_oracle)    RM=gnn;   WR=natural; AL=1.0; GT=oracle ;;
    gnn_learned)   RM=gnn;   WR=natural; AL=1.0; GT=learned ;;
    point_learned) RM=point; WR=natural; AL=1.0; GT=learned ;;
  esac
  echo "##### ARM: $ARM #####"
  SMOKE=0 LIFE=110 GAIN0=1.0 SEED=0 FLAT=0 READMODE=$RM WRITE=$WR READT=1.0 ALIGN=$AL \
    GATE=$GT SLOW_DIM=32 DISTRACT=8 GOAL="$GOALTEXT" python -u organism3.py
done
echo FIVE_ARM_DONE
