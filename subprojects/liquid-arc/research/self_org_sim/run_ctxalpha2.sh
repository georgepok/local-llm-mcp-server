#!/bin/bash
# CTXALPHA2: committing (nonlinear) α-head (AHID=128). CTXALPHA gave the right ORDERING (corr +0.72) but the LINEAR head
# hedged to a compressed global α~0.72 -> under-held HOLD (judge 0.5 vs oracle 0.79). Test whether a nonlinear head COMMITS
# to the extremes (HOLD α->1.0, REL α->0.1), recovering HOLD while keeping release. sup arms decisive; rl arms = learnability.
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
G="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
B="SMOKE=0 LIFE=80 GAIN0=1.0 FLAT=0 READMODE=gnn WRITE=off READT=1.0 ALIGN=1.0 GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=0 JUDGE=1 PHASE=1 CTXALPHA=1 AHID=128"
for SD in 0 1; do
  echo "##### SEED $SD ARM: ctx2_sup_train #####";    env $B SEED=$SD LORA=liquid SUPALPHA=1            GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: ctx2_sup_det_eval #####"; env $B SEED=$SD LORA=liquid SUPALPHA=1 DETALPHA=1 GOAL="$G" python -u organism3.py
done
for SD in 0 1; do
  echo "##### SEED $SD ARM: ctx2_rl_learned #####";   env $B SEED=$SD LORA=liquid                      GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: ctx2_rl_det_eval #####";  env $B SEED=$SD LORA=liquid DETALPHA=1           GOAL="$G" python -u organism3.py
done
echo "=== CTXALPHA2_DONE ==="
