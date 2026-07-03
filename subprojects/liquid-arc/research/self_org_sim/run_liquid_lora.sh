#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
GOALTEXT="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
# Q2: can the Liquid CONTROL the actuator? static always-on (alpha=1) vs liquid-scalar alpha_t from h_t
for SD in 0 1 2; do
  for ARM in lora_static lora_liquid; do
    case $ARM in
      lora_static) LR=goal ;;     # alpha fixed = 1.0 (always on)
      lora_liquid) LR=liquid ;;   # alpha_t = clipped scalar from Liquid h_t, REINFORCE on judge - cost
    esac
    echo "##### SEED $SD ARM: $ARM #####"
    SMOKE=0 LIFE=55 GAIN0=1.0 SEED=$SD FLAT=0 READMODE=gnn WRITE=off READT=1.0 ALIGN=1.0 \
      GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=8 JUDGE=1 LORA=$LR ALPHA_COST=0.1 GOAL="$GOALTEXT" python -u organism3.py
  done
done
echo LIQUID_LORA_DONE
