#!/bin/bash
# Launch one groot_server per LIBERO suite, each with its suite-specific
# checkpoint, on a unique port. Lets eval pick the right teacher per suite.
#
# Ports:
#   5555 -> libero_10        (default; existing convention)
#   5556 -> libero_spatial
#   5557 -> libero_object
#   5558 -> libero_goal
#
# Each server uses ~6GB GPU memory + ~20GB unified RAM. 4 servers ≈ 24GB GPU,
# fits comfortably in GB10's 128GB unified memory.
#
# Usage:
#   bash launch_groot_servers.sh            # start all 4
#   bash launch_groot_servers.sh stop       # stop all 4
#   bash launch_groot_servers.sh status     # check status

set -e

CKPT_BASE=/home/pokazge/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO
LOG_DIR=/tmp/groot_servers
mkdir -p "$LOG_DIR"

declare -A SUITE_PORT=(
    [libero_10]=5555
    [libero_spatial]=5556
    [libero_object]=5557
    [libero_goal]=5558
)

case "${1:-start}" in
    start)
        for suite in libero_10 libero_spatial libero_object libero_goal; do
            port=${SUITE_PORT[$suite]}
            ckpt=$CKPT_BASE/$suite
            if ss -ltn | grep -q ":$port "; then
                echo "[skip] $suite already listening on $port"
                continue
            fi
            echo "[start] $suite on port $port"
            nohup bash -c "
                source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
                source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
                export HF_HOME=/home/pokazge/hf_cache
                export HF_TOKEN=\$(cat /home/pokazge/.cache/huggingface/token)
                cd /home/pokazge/liquid-arc/research/self_org_sim
                python groot_server.py --teacher_path $ckpt --port $port --action_horizon 16
            " > "$LOG_DIR/${suite}_${port}.log" 2>&1 < /dev/null &
            disown
        done
        echo "[wait] giving servers ~60s to load..."
        sleep 60
        for suite in libero_10 libero_spatial libero_object libero_goal; do
            port=${SUITE_PORT[$suite]}
            if ss -ltn | grep -q ":$port "; then
                echo "[ready] $suite on port $port"
            else
                echo "[FAIL] $suite did NOT come up on port $port; check $LOG_DIR/${suite}_${port}.log"
            fi
        done
        ;;
    stop)
        for suite in libero_10 libero_spatial libero_object libero_goal; do
            port=${SUITE_PORT[$suite]}
            pids=$(ps -ef | grep groot_server | grep "port $port" | grep -v grep | awk '{print $2}')
            if [[ -n "$pids" ]]; then
                echo "[stop] $suite (port $port) pids=$pids"
                kill $pids
            fi
        done
        ;;
    status)
        for suite in libero_10 libero_spatial libero_object libero_goal; do
            port=${SUITE_PORT[$suite]}
            if ss -ltn | grep -q ":$port "; then
                echo "[up]   $suite on port $port"
            else
                echo "[down] $suite (would be port $port)"
            fi
        done
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 1
        ;;
esac
