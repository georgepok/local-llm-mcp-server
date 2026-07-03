#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
GOALTEXT="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
for ARM in llmonly point gnn; do
  case $ARM in
    llmonly) RM=point; WR=off; AL=0 ;;
    point)   RM=point; WR=natural; AL=1.0 ;;
    gnn)     RM=gnn;   WR=natural; AL=1.0 ;;
  esac
  echo "##### ARM: $ARM #####"
  SMOKE=0 LIFE=120 GAIN0=1.0 SEED=0 FLAT=0 READMODE=$RM WRITE=$WR READT=1.0 ALIGN=$AL DISTRACT=8 GOAL="$GOALTEXT" python -u organism3.py
done
echo THREE_ARM_DONE
