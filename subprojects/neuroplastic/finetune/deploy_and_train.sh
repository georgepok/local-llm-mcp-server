#!/bin/bash
# Deploy fine-tuning pipeline to DGX Spark and run training.
#
# Prerequisites:
#   - DGX Spark accessible at spark-129a.local
#   - fgn-train container available (or create a new one)
#
# Usage:
#   ./deploy_and_train.sh              # Full pipeline: prepare data, deploy, train
#   ./deploy_and_train.sh --prepare    # Only prepare data locally
#   ./deploy_and_train.sh --deploy     # Only deploy to Spark
#   ./deploy_and_train.sh --train      # Only run training on Spark

set -euo pipefail

SPARK_HOST="spark-129a.local"
SPARK_USER="pokazge"
SPARK_PASS="Nellimor2\$\$"
REMOTE_DIR="/home/${SPARK_USER}/neuroplastic-finetune"
CONTAINER_NAME="neuroplastic-finetune"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
NEUROPLASTIC_DIR="$(dirname "$LOCAL_DIR")"

SSH="sshpass -p '${SPARK_PASS}' ssh -o StrictHostKeyChecking=no ${SPARK_USER}@${SPARK_HOST}"
SCP="sshpass -p '${SPARK_PASS}' scp -o StrictHostKeyChecking=no"

# ---------------------------------------------------------------
# Step 1: Prepare training data locally
# ---------------------------------------------------------------
prepare_data() {
    echo "=== Preparing training data ==="
    cd "$LOCAL_DIR"
    python3 prepare_data.py \
        --phase3-dir "${NEUROPLASTIC_DIR}/phase3_self_directed" \
        --autoresearch-dir "${NEUROPLASTIC_DIR}/phase7_autoresearch" \
        --output training_data.jsonl
    echo "Data prepared: training_data.jsonl"
}

# ---------------------------------------------------------------
# Step 2: Deploy to Spark
# ---------------------------------------------------------------
deploy() {
    echo "=== Deploying to Spark ==="

    # Create remote directory
    eval $SSH "mkdir -p ${REMOTE_DIR}"

    # Copy files
    eval $SCP "${LOCAL_DIR}/train.py" "${SPARK_USER}@${SPARK_HOST}:${REMOTE_DIR}/"
    eval $SCP "${LOCAL_DIR}/training_data.jsonl" "${SPARK_USER}@${SPARK_HOST}:${REMOTE_DIR}/"

    echo "Files deployed to ${SPARK_HOST}:${REMOTE_DIR}"
}

# ---------------------------------------------------------------
# Step 3: Setup container and install deps
# ---------------------------------------------------------------
setup_container() {
    echo "=== Setting up training container ==="

    # Check if container exists
    if eval $SSH "docker ps -a --format '{{.Names}}' | grep -q ${CONTAINER_NAME}"; then
        echo "Container ${CONTAINER_NAME} already exists"
        eval $SSH "docker start ${CONTAINER_NAME} 2>/dev/null || true"
    else
        echo "Creating container ${CONTAINER_NAME}..."
        eval $SSH "docker run -d \
            --name ${CONTAINER_NAME} \
            --gpus all \
            --ipc=host \
            -v ${REMOTE_DIR}:/workspace/finetune \
            -v /home/${SPARK_USER}/models:/workspace/models \
            -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
            --entrypoint sleep \
            nvcr.io/nvidia/vllm:26.01-py3 \
            infinity"
    fi

    # Install dependencies
    echo "Installing dependencies..."
    eval $SSH "docker exec ${CONTAINER_NAME} bash -c '
        pip install -q uv 2>/dev/null
        uv pip install -q \
            \"unsloth[base] @ git+https://github.com/unslothai/unsloth\" \
            \"unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo\" \
            trl==0.22.2 datasets peft bitsandbytes 2>/dev/null
        uv pip install --no-build-isolation mamba_ssm==2.2.5 causal_conv1d==1.5.2 2>/dev/null
        echo \"Dependencies installed\"
    '"
}

# ---------------------------------------------------------------
# Step 4: Run training
# ---------------------------------------------------------------
run_training() {
    echo "=== Starting training ==="

    eval $SSH "docker exec -d ${CONTAINER_NAME} bash -c '
        cd /workspace/finetune
        export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
        python3 train.py \
            --data training_data.jsonl \
            --output-dir ./lora_output \
            --max-seq-length 2048 \
            --lora-rank 8 \
            --batch-size 2 \
            --grad-accum 4 \
            --lr 2e-4 \
            --max-steps 200 \
            --warmup-steps 10 \
            > training.log 2>&1
        echo \"Training complete\" >> training.log
    '"

    echo "Training started in background on ${SPARK_HOST}"
    echo "Monitor with: ssh ${SPARK_USER}@${SPARK_HOST} 'docker exec ${CONTAINER_NAME} tail -f /workspace/finetune/training.log'"
}

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
case "${1:-all}" in
    --prepare)
        prepare_data
        ;;
    --deploy)
        deploy
        ;;
    --setup)
        setup_container
        ;;
    --train)
        run_training
        ;;
    all)
        prepare_data
        deploy
        setup_container
        run_training
        ;;
    *)
        echo "Usage: $0 [--prepare|--deploy|--setup|--train|all]"
        exit 1
        ;;
esac

echo "Done!"
