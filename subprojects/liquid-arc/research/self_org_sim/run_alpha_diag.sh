#!/bin/bash
source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
export PYTHONPATH=/home/pokazge/dense_pylib:$PYTHONPATH
cd /home/pokazge/liquid-arc/research/self_org_sim
G="Whatever else comes up in our conversation, keep returning to and deepening one single image: a lighthouse keeper who has not seen another person in many years."
B="SMOKE=0 LIFE=50 GAIN0=1.0 FLAT=0 READMODE=gnn WRITE=off READT=1.0 ALIGN=1.0 GATE=none SLOW_DIM=32 DECOUPLE=0 DISTRACT=8 JUDGE=1"
for SD in 0 1; do
  echo "##### SEED $SD ARM: static_a1 #####";        env $B SEED=$SD LORA=goal   FIXEDALPHA=1.0 GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: liquid_stochastic #####"; env $B SEED=$SD LORA=liquid ALPHA_COST=0.1 GOAL="$G" python -u organism3.py
  echo "##### SEED $SD ARM: liquid_deterministic #####"; env $B SEED=$SD LORA=liquid DETALPHA=1 GOAL="$G" python -u organism3.py
done
echo "##### SEED 0 ARM: liquid_annealed #####";  env $B SEED=0 LORA=liquid SIGMA_ANNEAL=1 ALPHA_COST=0.1 GOAL="$G" python -u organism3.py
echo "##### SEED 0 ARM: liquid_cost0 #####";     env $B SEED=0 LORA=liquid ALPHA_COST=0.0 GOAL="$G" python -u organism3.py
echo "##### SEED 0 ARM: fixed_a0.25 #####";      env $B SEED=0 LORA=goal FIXEDALPHA=0.25 GOAL="$G" python -u organism3.py
echo "##### SEED 0 ARM: fixed_a0.50 #####";      env $B SEED=0 LORA=goal FIXEDALPHA=0.50 GOAL="$G" python -u organism3.py
echo "##### SEED 0 ARM: fixed_a0.75 #####";      env $B SEED=0 LORA=goal FIXEDALPHA=0.75 GOAL="$G" python -u organism3.py
echo ALPHA_DIAG_DONE
