#!/bin/bash
# Run all five Geometric Navigator experiments end-to-end.
# Execute on DGX Spark inside any container that has:
#   - PyTorch + liquid-arc at PYTHONPATH
#   - Access to the graph-engine checkpoint
#   - vLLM serving the extraction model (default Nemotron-3-Nano on :30000)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/liquid-arc}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/output_graph_engine_final/checkpoints/step_500.pt}"
DATA_DIR="${REPO_ROOT}/data/navigator"
OUT_DIR="${REPO_ROOT}/shared/outbox"
VLLM_URL="${VLLM_URL:-http://localhost:30000/v1}"
MODEL="${MODEL:-NVIDIA-Nemotron-3-Nano-30B-A3B-FP8}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"

mkdir -p "${OUT_DIR}"

echo "=== Navigator Experiment 1 — extraction validation ==="
python -m liquid_arc.scripts.nav_exp1_extraction \
    --testset "${DATA_DIR}/extraction_testset.jsonl" \
    --vllm_url "${VLLM_URL}" \
    --model "${MODEL}" \
    --out_json "${OUT_DIR}/nav_exp1_extraction.json" || true

echo "=== Navigator Experiment 2 — h_state accumulation ==="
python -m liquid_arc.scripts.nav_exp2_accumulation \
    --checkpoint "${CHECKPOINT}" \
    --interactions "${DATA_DIR}/supply_chain_interactions.jsonl" \
    --state_path "${REPO_ROOT}/navigator_state_exp2.json" \
    --out_json "${OUT_DIR}/nav_exp2_accumulation.json" || true

echo "=== Navigator Experiment 3 — context relevance ==="
python -m liquid_arc.scripts.nav_exp3_context \
    --checkpoint "${CHECKPOINT}" \
    --interactions "${DATA_DIR}/supply_chain_interactions.jsonl" \
    --state_path "${REPO_ROOT}/navigator_state_exp3.json" \
    --out_json "${OUT_DIR}/nav_exp3_context.json" || true

echo "=== Navigator Experiment 4 — end-to-end reasoning ==="
python -m liquid_arc.scripts.nav_exp4_reasoning \
    --checkpoint "${CHECKPOINT}" \
    --suite "${DATA_DIR}/reasoning_suite.jsonl" \
    --vllm_url "${VLLM_URL}" \
    --model "${MODEL}" \
    --out_json "${OUT_DIR}/nav_exp4_reasoning.json" || true

echo "=== Navigator Experiment 5 — cross-domain transfer ==="
python -m liquid_arc.scripts.nav_exp5_transfer \
    --checkpoint "${CHECKPOINT}" \
    --supply_chain "${DATA_DIR}/supply_chain_interactions.jsonl" \
    --ecology "${DATA_DIR}/ecology_isomorphic.jsonl" \
    --vllm_url "${VLLM_URL}" \
    --out_json "${OUT_DIR}/nav_exp5_transfer.json" || true

echo "=== all experiments done. outputs in ${OUT_DIR} ==="
