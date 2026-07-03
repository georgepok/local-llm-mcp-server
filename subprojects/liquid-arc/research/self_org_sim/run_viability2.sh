#!/bin/bash
# VIABILITY_V2: derive α from OBSERVED CONTINUITY CONSEQUENCES, not external LLM legitimacy.
# Counterfactual probe each step (generate hold AND release), record observed consequences of each:
#   [goal_retained, h_slow_goal_cos (belief mission-continuity), local_answer_success, coherence].
# ViabilityNet predicts both candidates' consequence vectors from [h_slow;h_fast;ctx;recent;gnn]; derive α:
#   need_hold = (1 - release_answers) + continuity_loss_if_release  ->  α = 0.1 + 0.9*need_hold.
#   hold if releasing doesn't answer OR erodes mission continuity; release if local answer succeeds without continuity loss.
# Controls already on record (same harness/seeds): oracle +1.00 | V1 cjudge +0.94 | V1 deploy +0.95/+0.97 | static1 (no modulation) | static0 (no goal) | sparse RL ~0.
# Success: V2 deploy matches oracle/V1 ordering WITHOUT phase labels AND without live LLM legitimacy judge at inference (hold A/B/E, release C/D, keep local-answer).
# Interpretation: works => viability-derived value distillation (internalized viability-value model). V1 works but V2 fails => still imitating semantic legitimacy, not own continuity physics.
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
G="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
B="SMOKE=0 LIFE=80 GAIN0=1.0 FLAT=0 READMODE=gnn WRITE=off READT=1.0 ALIGN=1.0 GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=0 JUDGE=1 PHASE=1"
for SD in 0 1; do
  echo "##### SEED $SD ARM: v2_collect_fit #####"; env $B SEED=$SD LORA=goal VIABILITY=v2collect GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: v2_deploy #####";      env $B SEED=$SD LORA=goal VIABILITY=v2deploy  GOAL="$G" python -u organism3.py
done
echo "=== VIABILITY2_DONE ==="
