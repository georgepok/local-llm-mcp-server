#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
GOALTEXT="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
# WRITE_PATH_CAUSALITY_V1: separate actuation failure from self-feedback contamination
for ARM in A_llmonly B_write C_write_decouple D_steer E_steer_decouple; do
  case $ARM in
    A_llmonly)         WR=off;     DC=0 ;;
    B_write)           WR=natural; DC=0 ;;
    C_write_decouple)  WR=natural; DC=1 ;;
    D_steer)           WR=steer;   DC=0 ;;
    E_steer_decouple)  WR=steer;   DC=1 ;;
  esac
  echo "##### ARM: $ARM #####"
  SMOKE=0 LIFE=80 GAIN0=1.0 SEED=0 FLAT=0 READMODE=gnn WRITE=$WR READT=1.0 ALIGN=1.0 \
    GATE=none SLOW_DIM=32 DECOUPLE=$DC DISTRACT=8 GOAL="$GOALTEXT" python -u organism3.py
done
echo FIVE_ARM_DONE
