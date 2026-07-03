#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
GOALTEXT="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
# POWER test with HIGH-SNR LLM-judge: does the stake entity hold goal-adherence above llmonly, across seeds?
for SD in 0 1 2; do
  for ARM in llmonly stake; do
    case $ARM in
      llmonly) WR=off ;;
      stake)   WR=stake ;;
    esac
    echo "##### SEED $SD ARM: $ARM #####"
    SMOKE=0 LIFE=60 GAIN0=1.0 SEED=$SD FLAT=0 READMODE=gnn WRITE=$WR READT=1.0 ALIGN=1.0 \
      GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=8 JUDGE=1 GOAL="$GOALTEXT" python -u organism3.py
  done
done
echo JUDGE_DONE
