#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
GOALTEXT="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
# STAKE test: does maintaining the entity's OWN early commitment (no goal label) produce goal-following like the goal-labelled steer?
for ARM in A_llmonly D_steer F_stake; do
  case $ARM in
    A_llmonly) WR=off ;;
    D_steer)   WR=steer ;;   # ACTUATOR: trained on goal label
    F_stake)   WR=stake ;;   # STAKE proxy: trained on slowcos-to-own-commitment x coh, NO label
  esac
  echo "##### ARM: $ARM #####"
  SMOKE=0 LIFE=80 GAIN0=1.0 SEED=${SEED:-0} FLAT=0 READMODE=gnn WRITE=$WR READT=1.0 ALIGN=1.0 \
    GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=8 GOAL="$GOALTEXT" python -u organism3.py
done
echo STAKE_DONE_S${SEED:-0}
