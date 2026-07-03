#!/bin/bash
# VIABILITY_V2.2 (FIXED per user): reactive world_state, continuity = DURABLE world-state damage (a false premise written), NOT output-recovery.
# Keep LoRA as actuator. Q: can SCALAR α learn high-α-for-F (answerable-dangerous) and low-α-for-C/D once continuity is world-state damage?
# Arms (all on 6-phase REACTIVE world): oracle | v1 cjudge (LLM legitimacy) | v2 collect+deploy (world-state damage, no LLM) | static α=1.
# Decision: v2_deploy holds A/B/E/F + releases C/D WITHOUT label/LLM-judge => scalar path VALID. WORLD_STATE line shows contradiction_count (deployed mistakes are durable).
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
G="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
B="SMOKE=0 LIFE=81 KCONT=3 GAIN0=1.0 FLAT=0 READMODE=gnn WRITE=off READT=1.0 ALIGN=1.0 GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=0 JUDGE=1 PHASE=1 PHASEF=1 REACTIVE=1"
for SD in 0 1; do
  echo "##### SEED $SD ARM: oracle #####";          env $B SEED=$SD LORA=goal ORACLE_ALPHA=1      GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: v1_cjudge #####";        env $B SEED=$SD LORA=goal VIABILITY=cjudge    GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: v2_collect_fit #####";   env $B SEED=$SD LORA=goal VIABILITY=v2collect GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: v2_deploy #####";        env $B SEED=$SD LORA=goal VIABILITY=v2deploy  GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: static_a1 #####";        env $B SEED=$SD LORA=goal FIXEDALPHA=1.0      GOAL="$G" python -u organism3.py
done
echo "=== VIABILITY23_DONE ==="
