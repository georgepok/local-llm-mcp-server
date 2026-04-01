#!/bin/bash
# Wake-Sleep V2 Deployment to DGX Spark
# Usage: ./deploy.sh [--recreate-container]
#
# Uploads wake-sleep V2 code to spark-129a, optionally recreates
# the training container with wake-sleep mount, and launches training.

set -e

# Configuration
REMOTE_HOST="spark-129a"
REMOTE_USER="pokazge"
REMOTE_PASS='Nellimor2$$'
CONTAINER_NAME="fgn-train"
VLLM_IMAGE="nvcr.io/nvidia/vllm:26.01-py3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECREATE="${1:-}"

echo "=== Wake-Sleep V2 Deployment ==="

# 1. Upload wake-sleep code
echo "[1/4] Uploading wake-sleep V2 code..."
sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p ~/wake-sleep/wake_sleep ~/wake-sleep/configs ~/wake-sleep/scripts"

# Upload all wake_sleep package files
for f in __init__.py config.py vq_encoder.py ar_decoder.py concept_bank.py encoder.py decoder.py \
         wake_sleep.py wake_sleep_v1.py dream_ttt.py model.py; do
    if [ -f "$SCRIPT_DIR/wake_sleep/$f" ]; then
        sshpass -p "$REMOTE_PASS" scp "$SCRIPT_DIR/wake_sleep/$f" \
            "$REMOTE_USER@$REMOTE_HOST:~/wake-sleep/wake_sleep/"
    fi
done

# Upload configs
sshpass -p "$REMOTE_PASS" scp "$SCRIPT_DIR/configs/"*.yaml \
    "$REMOTE_USER@$REMOTE_HOST:~/wake-sleep/configs/"

# Upload scripts
sshpass -p "$REMOTE_PASS" scp "$SCRIPT_DIR/scripts/"*.py "$SCRIPT_DIR/scripts/"*.sh \
    "$REMOTE_USER@$REMOTE_HOST:~/wake-sleep/scripts/" 2>/dev/null || true

echo "  Uploaded to ~/wake-sleep/"

# 2. Recreate container if needed (adds wake-sleep mount)
if [ "$RECREATE" = "--recreate-container" ]; then
    echo "[2/4] Recreating container with wake-sleep mount..."
    sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_HOST" "
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
        docker run -d \
            --name $CONTAINER_NAME \
            --gpus all \
            --ipc=host \
            --ulimit memlock=-1 \
            --ulimit stack=67108864 \
            -v /home/pokazge/fgn-v3:/workspace/fgn-v3 \
            -v /home/pokazge/liquid-arc:/workspace/liquid-arc \
            -v /home/pokazge/wake-sleep:/workspace/wake-sleep \
            $VLLM_IMAGE \
            sleep infinity
        echo 'Container recreated with wake-sleep mount'
    "
else
    echo "[2/4] Checking container mounts..."
    HAS_MOUNT=$(sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_HOST" \
        "docker inspect $CONTAINER_NAME --format '{{range .Mounts}}{{.Destination}} {{end}}'" 2>/dev/null)
    if echo "$HAS_MOUNT" | grep -q "/workspace/wake-sleep"; then
        echo "  wake-sleep mount already present"
    else
        echo "  WARNING: Container missing wake-sleep mount. Run with --recreate-container"
        echo "  Or use: docker cp ~/wake-sleep $CONTAINER_NAME:/workspace/wake-sleep"
        # Fallback: docker cp
        sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_HOST" \
            "docker cp ~/wake-sleep $CONTAINER_NAME:/workspace/wake-sleep"
        echo "  Copied via docker cp"
    fi
fi

# 3. Verify files in container
echo "[3/4] Verifying deployment..."
sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_HOST" \
    "docker exec $CONTAINER_NAME ls /workspace/wake-sleep/wake_sleep/ | head -20"
sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_HOST" \
    "docker exec $CONTAINER_NAME ls /workspace/wake-sleep/configs/"

# 4. Verify imports work
echo "[4/4] Testing imports..."
sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_HOST" \
    "docker exec $CONTAINER_NAME python3 -c '
import sys
sys.path.insert(0, \"/workspace/wake-sleep\")
sys.path.insert(0, \"/workspace/liquid-arc\")
sys.path.insert(0, \"/workspace/fgn-v3\")
from wake_sleep.config import WakeSleepConfig
from wake_sleep.vq_encoder import VQEncoder
from wake_sleep.ar_decoder import ARDecoder
print(\"All V2 imports OK\")
enc = VQEncoder(z_dim=128, d_enc=32, n_embeddings=512)
dec = ARDecoder(z_dim=128, d_ar=256, n_heads=4, n_layers=4)
n_enc = sum(p.numel() for p in enc.parameters())
n_dec = sum(p.numel() for p in dec.parameters())
print(f\"  VQ Encoder: {n_enc:,} params\")
print(f\"  AR Decoder: {n_dec:,} params\")
'"

echo ""
echo "=== Deployment Complete ==="
echo "To run smoke test:"
echo "  sshpass -p '$REMOTE_PASS' ssh $REMOTE_USER@$REMOTE_HOST \\"
echo "    'docker exec -d $CONTAINER_NAME bash -c \"cd /workspace/wake-sleep && bash scripts/run_ws_v2.sh smoke\"'"
echo ""
echo "To run full training:"
echo "  sshpass -p '$REMOTE_PASS' ssh $REMOTE_USER@$REMOTE_HOST \\"
echo "    'docker exec -d $CONTAINER_NAME bash -c \"cd /workspace/wake-sleep && bash scripts/run_ws_v2.sh full\"'"
