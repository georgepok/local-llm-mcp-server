#!/bin/bash
# CTXALPHA: condition α on [h ; embed(INCOMING user turn)]. Tests whether the on-switch collapse was a PERCEPTION gap
# (phase signal lives in the context the belief doesn't read) rather than an actuator/representation limit.
#   sup arms (MSE->oracle): decisive identifiability test. corr->+1 & hold/release separation => context-conditioning is the fix.
#   rl arms (REINFORCE):    can it be LEARNED once the signal is visible?
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
G="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
B="SMOKE=0 LIFE=80 GAIN0=1.0 FLAT=0 READMODE=gnn WRITE=off READT=1.0 ALIGN=1.0 GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=0 JUDGE=1 PHASE=1 CTXALPHA=1"
for SD in 0 1; do
  echo "##### SEED $SD ARM: ctx_sup_train #####";    env $B SEED=$SD LORA=liquid SUPALPHA=1            GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: ctx_sup_det_eval #####"; env $B SEED=$SD LORA=liquid SUPALPHA=1 DETALPHA=1 GOAL="$G" python -u organism3.py
done
for SD in 0 1; do
  echo "##### SEED $SD ARM: ctx_rl_learned #####";   env $B SEED=$SD LORA=liquid                      GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: ctx_rl_det_eval #####";  env $B SEED=$SD LORA=liquid DETALPHA=1           GOAL="$G" python -u organism3.py
done
echo "=== CTXALPHA_DONE ==="
