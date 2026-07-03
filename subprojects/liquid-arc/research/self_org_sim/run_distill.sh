#!/bin/bash
# DISTILL: remove the ~75-step online budget limit. Collect (ctx,h, oracle-α) UNDER oracle behavior (states phase-separated),
# fit the committing ctx-head OFFLINE to convergence (4000 steps), then DEPLOY it deterministically.
# Decides: can a WELL-TRAINED context head reproduce the oracle's α swing (HOLD->1.0 / REL->0.1) & hold+release behavior?
#   DISTILL_FIT line shows offline α HOLD vs REL (fit ceiling). deploy_eval shows the deployed phase metrics vs oracle.
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
G="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
B="SMOKE=0 LIFE=80 GAIN0=1.0 FLAT=0 READMODE=gnn WRITE=off READT=1.0 ALIGN=1.0 GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=0 JUDGE=1 PHASE=1 CTXALPHA=1 AHID=128"
for SD in 0 1; do
  echo "##### SEED $SD ARM: distill_collect_fit #####"; env $B SEED=$SD LORA=liquid DISTILL=1  GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: distill_deploy_eval #####"; env $B SEED=$SD LORA=liquid DETALPHA=1 GOAL="$G" python -u organism3.py
done
echo "=== DISTILL_DONE ==="
