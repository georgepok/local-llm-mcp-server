#!/bin/bash
# v11 GPU-server eval. Liquid + DINOv2 retrieval + adaptive SGD run on GPU
# via liquid_server.py. Client (this script) runs in the LIBERO sim venv.
#
# Assumes:
#   - liquid_server.py already running at tcp://localhost:7777 with
#     --memory_bank loaded and --adaptive --demo_replay_n 4 set
#   - per-suite GR00T servers running at 5555/5556/5557/5558
#
# Order: object → goal → spatial → libero_10 (matches prior eval order).

set -euo pipefail

MODE="${1:-full}"
LIQUID_ADDR="tcp://localhost:7777"
LOG_DIR=/tmp
TIMESTAMP=$(date +%H%M%S)

declare -A PORTS=(
  ["libero_10"]=5555
  ["libero_spatial"]=5556
  ["libero_object"]=5557
  ["libero_goal"]=5558
)

cd /home/pokazge/liquid-arc/research/self_org_sim
source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate

run_suite() {
  local suite="$1"
  local port="${PORTS[$suite]}"
  local log="${LOG_DIR}/eval_v11gpu_${suite}_${TIMESTAMP}.log"
  echo ""
  echo "=== v11 GPU on ${suite} (groot port ${port}) ==="
  echo "  log: ${log}"
  # Tell liquid_server to switch its retrieval filter to this suite
  python -c "
import zmq, pickle, sys
ctx = zmq.Context.instance()
s = ctx.socket(zmq.REQ); s.setsockopt(zmq.RCVTIMEO, 60000); s.connect('${LIQUID_ADDR}')
s.send(pickle.dumps({'cmd': 'set_retrieval_filter', 'suite': '${suite}'}))
resp = pickle.loads(s.recv())
print(f'[client] set_retrieval_filter -> {resp}', flush=True)
s.close()
"
  python rollout_libero_v11_client.py \
    --liquid_addr "${LIQUID_ADDR}" \
    --groot_port "${port}" \
    --task_suite "${suite}" \
    --rollouts_per_task 3 \
    --max_steps 720 \
    --exec_horizon 8 \
    --infer_steps 10 \
    --groot_freq 2 \
    --depth_indices 0,1,2,3 \
    --adaptive \
    --demo_replay_n 4 \
    --use_goal_img \
    --out_json "${LOG_DIR}/eval_v11gpu_${suite}_${TIMESTAMP}.json" \
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
echo "=== v11 GPU eval complete ==="
