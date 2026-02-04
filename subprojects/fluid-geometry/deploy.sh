#!/bin/bash
# Fluid Geometry Deployment Script for spark-129a
# Usage: ./deploy.sh

set -e

# Configuration
REMOTE_HOST="spark-129a"
REMOTE_USER="pokazge"
REMOTE_PASS='Nellimor2$$'
MODEL_PATH="/home/pokazge/models/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
CONTAINER_NAME="vllm-nemotron-serve"
PORT="30000"
VLLM_IMAGE="nvcr.io/nvidia/vllm:26.01-py3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Fluid Geometry Deployment ==="

# 1. Upload processor
echo "[1/3] Uploading fluid_geometry.py..."
sshpass -p "$REMOTE_PASS" scp "$SCRIPT_DIR/fluid_geometry.py" \
    "$REMOTE_USER@$REMOTE_HOST:~/models/"

# 2. Create remote startup script
echo "[2/3] Creating startup script on remote..."
sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_HOST" "cat > ~/start_vllm_with_fluid.sh << 'SCRIPT'
#!/bin/bash
# Stop existing container
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Start vLLM with Fluid Geometry LogitsProcessor
docker run -d \\
  --name $CONTAINER_NAME \\
  --gpus all \\
  --ipc=host \\
  --ulimit memlock=-1 \\
  --ulimit stack=67108864 \\
  -p $PORT:$PORT \\
  -v $MODEL_PATH:/workspace/model \\
  -v /home/pokazge/models/nano_v3_reasoning_parser.py:/workspace/nano_v3_reasoning_parser.py \\
  -v /home/pokazge/models/fluid_geometry.py:/workspace/fluid_geometry.py \\
  $VLLM_IMAGE \\
  python3 -m vllm.entrypoints.openai.api_server \\
    --host 0.0.0.0 \\
    --port $PORT \\
    --model /workspace/model \\
    --served-model-name NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \\
    --trust-remote-code \\
    --max-model-len 32768 \\
    --max-num-seqs 8 \\
    --enable-prefix-caching \\
    --reasoning-parser-plugin /workspace/nano_v3_reasoning_parser.py \\
    --reasoning-parser nano_v3 \\
    --logits-processors fluid_geometry:FluidGeometryLogitsProcessor

echo 'Container started. Model loading takes ~5 minutes...'
SCRIPT
chmod +x ~/start_vllm_with_fluid.sh"

# 3. Start the container
echo "[3/3] Starting vLLM container..."
sshpass -p "$REMOTE_PASS" ssh "$REMOTE_USER@$REMOTE_HOST" "~/start_vllm_with_fluid.sh"

echo ""
echo "=== Deployment Complete ==="
echo "Server: http://$REMOTE_HOST:$PORT"
echo "Model loading takes ~5 minutes. Check status with:"
echo "  ssh $REMOTE_USER@$REMOTE_HOST 'docker logs -f $CONTAINER_NAME'"
echo ""
echo "Test with:"
echo "  curl http://$REMOTE_HOST:$PORT/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"NVIDIA-Nemotron-3-Nano-30B-A3B-FP8\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
