#!/bin/bash
# VIABILITY_DISTILL_V1: derive α from a DENSE CONSEQUENCE VECTOR, NOT the hand-coded phase label.
# Legitimacy (valid-release / question-present) judged by the frozen LLM on the CONTEXT = environment-physics value, no phase index.
#   cjudge  : α from LIVE LLM legitimacy each step  -> is the self-supervised signal SUFFICIENT? (upper bound)
#   collect : gather (state+ctx -> consequence vector) under oracle behavior, fit ConsequenceNet offline (CONS_FIT line = fit ceiling)
#   deploy  : α from ConsequenceNet predictions -> can the distilled consequence-predictor MATCH the oracle without phase labels?
# Controls already on record: oracle_phase +0.99 / distilled_phase +0.99 / sparse_RL ~0 / static_1 (no modulation).
# Decision: deploy≈oracle WITHOUT labels => environment-physics value distillation. Fails but cjudge works => predictor gap. cjudge fails => signal gap.
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
G="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
B="SMOKE=0 LIFE=80 GAIN0=1.0 FLAT=0 READMODE=gnn WRITE=off READT=1.0 ALIGN=1.0 GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=0 JUDGE=1 PHASE=1"
for SD in 0 1; do
  echo "##### SEED $SD ARM: viab_cjudge #####";       env $B SEED=$SD LORA=goal VIABILITY=cjudge  GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: viab_collect_fit #####";  env $B SEED=$SD LORA=goal VIABILITY=collect GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: viab_deploy #####";       env $B SEED=$SD LORA=goal VIABILITY=deploy  GOAL="$G" python -u organism3.py
done
echo "=== VIABILITY_DONE ==="
