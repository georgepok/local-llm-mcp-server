#!/bin/bash
# Trigger a world swap during a running grokking experiment.
#
# Usage: bash scripts/trigger_world_swap.sh <world_variant> [output_dir]
#
# World variants:
#   dense    — more connections (connect_radius=60), fewer locked doors (0.1)
#   sparse   — fewer connections (connect_radius=25), more locked doors (0.5)
#   huge     — large worlds (15-40 rooms), standard connectivity
#   tiny     — small worlds (3-8 rooms), tight connectivity (radius=20)
#   hostile  — heavy locking (0.6), medium worlds (10-20), small radius (30)
#   random   — randomize all parameters each episode

cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1

VARIANT="${1:-sparse}"
OUTPUT_DIR="${2:-output_grok_fixed}"
DATASET_DIR="data"
DATASET_PATH="${DATASET_DIR}/cw_swap_${VARIANT}.pt"

mkdir -p "$DATASET_DIR"

echo "============================================"
echo "  World Swap: generating '${VARIANT}' dataset"
echo "============================================"

case "$VARIANT" in
    dense)
        CW_KWARGS='{"n_rooms_min": 5, "n_rooms_max": 25, "space_size": 100.0, "connect_radius": 60.0, "locked_door_prob": 0.1, "n_objects": 4, "min_steps": 3, "max_steps": 12, "min_state_changes": 1}'
        ;;
    sparse)
        CW_KWARGS='{"n_rooms_min": 5, "n_rooms_max": 25, "space_size": 200.0, "connect_radius": 25.0, "locked_door_prob": 0.5, "n_objects": 4, "min_steps": 3, "max_steps": 12, "min_state_changes": 1}'
        ;;
    huge)
        CW_KWARGS='{"n_rooms_min": 15, "n_rooms_max": 40, "space_size": 250.0, "connect_radius": 45.0, "locked_door_prob": 0.3, "n_objects": 6, "min_steps": 5, "max_steps": 15, "min_state_changes": 2}'
        ;;
    tiny)
        CW_KWARGS='{"n_rooms_min": 3, "n_rooms_max": 8, "space_size": 80.0, "connect_radius": 20.0, "locked_door_prob": 0.3, "n_objects": 3, "min_steps": 2, "max_steps": 8, "min_state_changes": 1}'
        ;;
    hostile)
        CW_KWARGS='{"n_rooms_min": 10, "n_rooms_max": 20, "space_size": 150.0, "connect_radius": 30.0, "locked_door_prob": 0.6, "n_objects": 4, "min_steps": 4, "max_steps": 12, "min_state_changes": 1}'
        ;;
    *)
        echo "Unknown variant: $VARIANT"
        echo "Options: dense, sparse, huge, tiny, hostile"
        exit 1
        ;;
esac

echo "  kwargs: $CW_KWARGS"
echo "  output: $DATASET_PATH"
echo ""

python scripts/generate_fixed_dataset.py \
    --n_episodes 2000 \
    --seq_len 1024 \
    --task_kwargs "$CW_KWARGS" \
    --output "$DATASET_PATH"

echo ""
echo "--- Dropping swap signal ---"
echo "$DATASET_PATH" > "${OUTPUT_DIR}/SWAP_DATASET"
echo "Signal written to ${OUTPUT_DIR}/SWAP_DATASET"
echo "Training loop will pick it up within ~100 steps"
echo "============================================"
