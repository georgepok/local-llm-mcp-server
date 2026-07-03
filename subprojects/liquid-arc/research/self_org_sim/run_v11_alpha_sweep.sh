#!/bin/bash
# v11 alpha sweep: restart liquid_server at different alpha values, run full eval each.
# Total ~30min × N alphas.

set -euo pipefail

ALPHAS="${1:-0.3 0.2 0.4}"
LOG_DIR=/tmp

declare -A PORTS=(
  ["libero_10"]=5555
  ["libero_spatial"]=5556
  ["libero_object"]=5557
  ["libero_goal"]=5558
)

cd /home/pokazge/liquid-arc/research/self_org_sim

start_server() {
  local alpha="$1"
  # Kill any old liquid_server
  ps aux | grep liquid_server | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
  sleep 2
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
  nohup python liquid_server.py \
    --student_ckpt /tmp/distill_v10_goal/step_008000.pt \
    --port 7777 \
    --memory_bank /home/pokazge/datasets/memory_bank_v11.npz \
    --adaptive --adaptive_lr 1e-4 \
    --demo_replay_n 4 \
    --demo_replay_suites libero_10 \
    --retrieve_top_k 3 \
    --retrieve_alpha "${alpha}" \
    --retrieve_adaptive \
    --retrieve_filter_suite_default libero_10 \
    > /tmp/liquid_server_alpha${alpha}.log 2>&1 &
  echo "Liquid server PID=$!"
  # Wait for ready
  while ! grep -q 'ready' /tmp/liquid_server_alpha${alpha}.log 2>/dev/null; do
    sleep 3
  done
  echo "  [server ready alpha=${alpha}]"
  deactivate || true
}

run_suite() {
  local suite="$1"
  local alpha="$2"
  local port="${PORTS[$suite]}"
  local log="${LOG_DIR}/eval_v11gpu_a${alpha}_${suite}.log"
  echo ""
  echo "=== v11 GPU alpha=${alpha} on ${suite} ==="
  python -c "
import zmq, pickle
ctx = zmq.Context.instance()
s = ctx.socket(zmq.REQ); s.setsockopt(zmq.RCVTIMEO, 60000); s.connect('tcp://localhost:7777')
s.send(pickle.dumps({'cmd': 'set_retrieval_filter', 'suite': '${suite}'}))
print(pickle.loads(s.recv()), flush=True); s.close()
"
  python rollout_libero_v11_client.py \
    --liquid_addr tcp://localhost:7777 \
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
    --out_json "${LOG_DIR}/eval_v11gpu_a${alpha}_${suite}.json" \
    2>&1 | tee "${log}"
}

for alpha in $ALPHAS; do
  start_server "${alpha}"
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  for suite in libero_object libero_goal libero_spatial libero_10; do
    run_suite "${suite}" "${alpha}"
  done
  deactivate || true
done

echo ""
echo "=== alpha sweep complete ==="
for alpha in $ALPHAS; do
  echo "--- alpha=${alpha} ---"
  for suite in libero_object libero_goal libero_spatial libero_10; do
    log="${LOG_DIR}/eval_v11gpu_a${alpha}_${suite}.log"
    echo -n "  ${suite}: "; grep 'OVERALL:' "${log}" | tail -1
  done
done
