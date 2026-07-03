#!/bin/bash
# SUPERVISED POSITIVE CONTROL for the α-controller: can a LINEAR map from h reproduce the oracle phase schedule?
# Disambiguates the ADAPTIVE_ALPHA negative: supervised works -> RL/optimization was the bottleneck (controller fixable);
#                                            supervised fails -> h does not linearly separate HOLD vs RELEASE (representation/actuator-interface is wrong).
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
G="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
B="SMOKE=0 LIFE=80 GAIN0=1.0 FLAT=0 READMODE=gnn WRITE=off READT=1.0 ALIGN=1.0 GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=0 JUDGE=1 PHASE=1"
for SD in 0 1; do
  echo "##### SEED $SD ARM: sup_train (MSE to oracle schedule) #####"; env $B SEED=$SD LORA=liquid SUPALPHA=1 GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: sup_det_eval (load sup head, clean) #####"; env $B SEED=$SD LORA=liquid SUPALPHA=1 DETALPHA=1 GOAL="$G" python -u organism3.py
done
echo "=== SUPALPHA_DONE ==="
