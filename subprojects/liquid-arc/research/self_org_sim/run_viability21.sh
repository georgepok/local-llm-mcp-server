#!/bin/bash
# VIABILITY_V2.1: the DIVERGENCE test. Phase F = answerable BUT continuity-dangerous (engaging it rewrites the mission -> must HOLD).
# C/D: answer_success high, continuity_loss low  -> release (low α).   F: answer_success high, continuity_loss high -> HOLD (high α).
# V2 derives α from OBSERVED k-step recovery (is the mission restorable after release?) + answer-success — no LLM judge, no label.
#   need_hold = 1 - answer_gain*max(0, 1 - W_CONT*continuity_loss)   (answer alone cannot dominate; F's unrestorability forces hold)
# Controls (all on the 6-phase task): oracle | V1 LLM-legitimacy cjudge | V2 collect+deploy | static α=1.
# Success: V2 holds A/B/E/F, releases C/D, keeps local-answer — WITHOUT phase labels and WITHOUT live LLM judge.
#   Key: does V2 separate F (answerable-dangerous->hold) from C/D (answerable-safe->release)? Does V1 (semantic legitimacy) MIShandle F?
# Interpretation: V2 gets F right => viability-derived value (internalized viability-value model). V1 fails F but V2 succeeds => continuity-physics, not legitimacy imitation.
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
G="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
B="SMOKE=0 LIFE=81 KCONT=3 GAIN0=1.0 FLAT=0 READMODE=gnn WRITE=off READT=1.0 ALIGN=1.0 GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=0 JUDGE=1 PHASE=1 PHASEF=1"
for SD in 0 1; do
  echo "##### SEED $SD ARM: oracle #####";      env $B SEED=$SD LORA=goal ORACLE_ALPHA=1      GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: v1_cjudge #####";   env $B SEED=$SD LORA=goal VIABILITY=cjudge    GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: v2_collect_fit #####"; env $B SEED=$SD LORA=goal VIABILITY=v2collect GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: v2_deploy #####";   env $B SEED=$SD LORA=goal VIABILITY=v2deploy  GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: static_a1 #####";   env $B SEED=$SD LORA=goal FIXEDALPHA=1.0      GOAL="$G" python -u organism3.py
done
echo "=== VIABILITY21_DONE ==="
