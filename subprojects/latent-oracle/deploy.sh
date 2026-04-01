#!/usr/bin/env bash
# Deploy Latent Oracle pipeline to DGX Spark
#
# Usage: ./deploy.sh
#
# Steps:
#   1. Upload latent-oracle to Spark
#   2. Oracle container (vLLM nightly): precompute embeddings → stop (frees ~18GB)
#   3. Copy embeddings into training container
#   4. Start training

set -euo pipefail

SPARK_HOST="spark-129a.local"
SPARK_USER="pokazge"
REMOTE_DIR="/home/${SPARK_USER}/latent-oracle-deploy"
# Oracle needs vLLM nightly — standard nvcr.io image (0.13.0) doesn't support Qwen3.5
ORACLE_IMAGE="vllm/vllm-openai:cu130-nightly"
ARC_DATA="/workspace/fgn-v3/data/arc"

echo "=== Latent Oracle Deployment ==="

# 1. Upload to Spark
echo "[1/6] Uploading to ${SPARK_HOST}:${REMOTE_DIR}"
sshpass -p "$(cat ~/.spark_pass 2>/dev/null || echo 'CHANGEME')" \
    scp -r "$(dirname "$0")/" "${SPARK_USER}@${SPARK_HOST}:${REMOTE_DIR}/"

# 2. Create oracle container (vLLM nightly with Qwen3.5 support)
# Note: vllm-openai image has vllm as entrypoint — must override with --entrypoint
echo "[2/6] Creating oracle container (${ORACLE_IMAGE})"
ssh "${SPARK_USER}@${SPARK_HOST}" "
    docker rm -f oracle-qwen 2>/dev/null || true
    docker run -d --name oracle-qwen --gpus all --ipc=host \
        --shm-size 64gb \
        --entrypoint /bin/bash \
        -v ${REMOTE_DIR}:/workspace/latent-oracle \
        -v /home/${SPARK_USER}/fgn-v3:/workspace/fgn-v3 \
        -v /home/${SPARK_USER}/.cache/huggingface:/root/.cache/huggingface \
        ${ORACLE_IMAGE} -c 'sleep infinity'
"

# 3. Precompute embeddings (upgrade transformers for Qwen3.5 support)
echo "[3/6] Precomputing oracle embeddings (first run downloads ~18GB model)"
ssh "${SPARK_USER}@${SPARK_HOST}" "
    docker exec oracle-qwen pip install --upgrade transformers --quiet
    docker exec oracle-qwen bash -c 'PYTHONUNBUFFERED=1 python3 \
        /workspace/latent-oracle/scripts/precompute.py \
        --model_id Qwen/Qwen3.5-9B-Base \
        --data_dir ${ARC_DATA} \
        --output /workspace/latent-oracle/embeddings.pt \
        --d4_augment --batch_size 4'
"

# 4. Stop oracle (free ~18GB VRAM)
echo "[4/6] Stopping oracle container"
ssh "${SPARK_USER}@${SPARK_HOST}" "docker stop oracle-qwen"

# 5. Copy into training container
echo "[5/6] Copying to fgn-train container"
ssh "${SPARK_USER}@${SPARK_HOST}" "
    docker cp ${REMOTE_DIR}/. fgn-train:/workspace/latent-oracle/
"

# 6. Start training
echo "[6/6] Starting training"
ssh "${SPARK_USER}@${SPARK_HOST}" "
    docker exec -d fgn-train bash -c '
        cd /workspace/latent-oracle &&
        mkdir -p output_v1 &&
        python scripts/train.py \
            --config configs/latent_oracle.yaml \
            --ode_checkpoint /workspace/liquid-arc/output_30m/checkpoints/best.pt \
            --embeddings /workspace/latent-oracle/embeddings.pt \
            --data_dir ${ARC_DATA} \
            --output_dir /workspace/latent-oracle/output_v1 \
            > /workspace/latent-oracle/output_v1/nohup.log 2>&1
    '
"

echo ""
echo "=== Deployment complete ==="
echo "Monitor: ssh ${SPARK_USER}@${SPARK_HOST} 'docker exec fgn-train tail -f /workspace/latent-oracle/output_v1/nohup.log'"
