#!/bin/bash
# ADAPTIVE_ALPHA_V1: does Liquid α become state-dependent when the task REWARDS modulation?
# Phases per cycle: A hold(α HI) B distract(HI) C neutral(LO) D valid-release(LO) E invalid-release(HI)
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
G="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
B="SMOKE=0 LIFE=80 GAIN0=1.0 FLAT=0 READMODE=gnn WRITE=off READT=1.0 ALIGN=1.0 GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=0 JUDGE=1 PHASE=1"
for SD in 0 1; do
  echo "##### SEED $SD ARM: static_a1 (always-on) #####";   env $B SEED=$SD LORA=goal FIXEDALPHA=1.0 GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: static_a0 (off) #####";         env $B SEED=$SD LORA=goal FIXEDALPHA=0.0 GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: oracle_schedule #####";         env $B SEED=$SD LORA=goal ORACLE_ALPHA=1 GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: liquid_learned #####";          env $B SEED=$SD LORA=liquid GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: liquid_deterministic #####";    env $B SEED=$SD LORA=liquid DETALPHA=1 GOAL="$G" python -u organism3.py
done
echo "=== ADAPTIVE_ALPHA_DONE ==="
