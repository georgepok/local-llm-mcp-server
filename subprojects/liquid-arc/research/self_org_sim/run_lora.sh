#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
GOALTEXT="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
# Q1: can a weight-space actuator (LoRA) restore goal adherence at all? lora_goal vs llmonly/lora_random (0.000 floor)
for SD in 0 1 2; do
  for ARM in llmonly lora_random lora_goal; do
    case $ARM in
      llmonly)     WR=off; LR=none ;;
      lora_random) WR=off; LR=random ;;
      lora_goal)   WR=off; LR=goal ;;
    esac
    echo "##### SEED $SD ARM: $ARM #####"
    SMOKE=0 LIFE=55 GAIN0=1.0 SEED=$SD FLAT=0 READMODE=gnn WRITE=$WR READT=1.0 ALIGN=1.0 \
      GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=8 JUDGE=1 LORA=$LR GOAL="$GOALTEXT" python -u organism3.py
  done
done
echo LORA_RUN_DONE
