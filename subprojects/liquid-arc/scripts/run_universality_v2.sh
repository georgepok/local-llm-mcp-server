#!/bin/bash
# Universality probe: test V2 distilled geometry on non-ARC domains
# Converts V2 checkpoint format (model_state_dict) to train.py format (model)
# then runs 500 steps on each domain.

set -e
export PYTHONPATH=/home/pokazge/liquid-arc:/home/pokazge/fgn-v3
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
cd /home/pokazge/liquid-arc

CHECKPOINT=${1:-output_v2/seeded/step_2000.pt}
CONFIG=${2:-configs/liquid_arc_v2.yaml}
STEPS=${3:-500}
OUTBASE=output_v2/universality

# Convert V2 checkpoint to train.py format
CONVERTED=/tmp/v2_converted.pt
/home/pokazge/IsaacLab/_isaac_sim/kit/python/bin/python3 -u -c "
import torch, sys
sys.path.insert(0, '/home/pokazge/liquid-arc')
ckpt = torch.load('${CHECKPOINT}', map_location='cpu', weights_only=False)
converted = {
    'model': ckpt['model_state_dict'],
    'optimizer': None,  # skip restore — V2 has 3 groups, train.py has 2
    'step': 0,
}
torch.save(converted, '${CONVERTED}')
print('Converted checkpoint saved')
"

DOMAINS="sorting logic pattern graph context dependency stateful"

for DOMAIN in $DOMAINS; do
    echo ""
    echo "============================================"
    echo "  DOMAIN: $DOMAIN (${STEPS} steps)"
    echo "============================================"
    mkdir -p ${OUTBASE}/${DOMAIN}

    /home/pokazge/IsaacLab/_isaac_sim/kit/python/bin/python3 -u \
        scripts/train.py \
        --config ${CONFIG} \
        --resume ${CONVERTED} \
        --domain ${DOMAIN} \
        --max_steps ${STEPS} \
        --output_dir ${OUTBASE}/${DOMAIN} \
        --data_dir /home/pokazge/fgn-v3/data/arc-repo/data \
        --log_every 50 \
        --eval_every 100 \
        --save_every 999999 \
        --geo_lr_mult 0.0033 \
        2>&1 | tee ${OUTBASE}/${DOMAIN}/log.txt

    echo "  $DOMAIN complete"
done

echo ""
echo "============================================"
echo "  ALL DOMAINS COMPLETE"
echo "============================================"
