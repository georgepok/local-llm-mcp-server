#!/bin/bash
# v11 memory-augmented eval. Two modes:
#   smoke - libero_10 only, 3 rollouts × 10 tasks, ~30 min
#   full  - all 4 suites × 3 rollouts × 10 tasks, ~2h
#
# Runs sequentially per suite to avoid GPU memory contention with GR00T server.
# Assumes:
#   - /home/pokazge/datasets/memory_bank_v11.npz already built
#   - GR00T servers running on port 5555 (libero_10), 5556 (spatial), 5558 (goal), etc.
#     (matching the per-suite ports the prior eval used)

set -euo pipefail

MODE="${1:-smoke}"
CKPT="/tmp/distill_v10_goal/step_008000.pt"
BANK="/home/pokazge/datasets/memory_bank_v11.npz"
LOG_DIR=/tmp
TIMESTAMP=$(date +%H%M%S)

cd /home/pokazge/liquid-arc/research/self_org_sim
source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate

# Per-suite ports matching prior eval setup
declare -A PORTS=(
  ["libero_10"]=5555
  ["libero_spatial"]=5556
  ["libero_object"]=5557
  ["libero_goal"]=5558
)

run_suite() {
  local suite="$1"
  local port="${PORTS[$suite]}"
  local log="${LOG_DIR}/eval_v11_${suite}_${TIMESTAMP}.log"
  echo ""
  echo "=== v11 RETRIEVAL on ${suite} (port ${port}) ==="
  echo "  log: ${log}"
  python rollout_libero_s1s2_retrieval.py \
    --student_ckpt "${CKPT}" \
    --memory_bank "${BANK}" \
    --task_suite "${suite}" \
    --rollouts_per_task 3 \
    --port "${port}" \
    --query_bank \
    --query_channel depth \
    --depth_indices 0,1,2,3 \
    --groot_freq 2 \
    --max_steps 720 \
    --exec_horizon 8 \
    --adaptive \
    --adaptive_lr 1e-4 \
    --demo_replay_n 4 \
    --retrieve_top_k 3 \
    --retrieve_alpha 0.5 \
    --retrieve_adaptive \
    --retrieve_filter_suite \
    --out_json "${LOG_DIR}/eval_v11_${suite}_${TIMESTAMP}.json" \
    2>&1 | tee "${log}"
}

case "${MODE}" in
  smoke)
    run_suite libero_10
    ;;
  full)
    for suite in libero_object libero_goal libero_spatial libero_10; do
      run_suite "${suite}"
    done
    ;;
  *)
    echo "usage: $0 {smoke|full}" >&2
    exit 1
    ;;
esac

echo ""
echo "=== v11 eval complete ==="
