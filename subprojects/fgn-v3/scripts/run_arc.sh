#!/bin/bash
# ARC-AGI experiment for FluidNet vs flat transformer.
# Run inside fgn-train container on Spark.
#
# Usage:
#   bash scripts/run_arc.sh [flat|fluid|fluid-struct|fluid-iter3]
#
# First time: download ARC data:
#   git clone https://github.com/fchollet/ARC-AGI.git data/arc-repo
#   ln -s data/arc-repo/data data/arc

set -euo pipefail
cd /workspace/fgn-v3
export PYTHONUNBUFFERED=1
export CUDA_MEMORY_FRACTION=0.85
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE="${1:-fluid}"

# Download ARC data if not present
if [ ! -d "data/arc/training" ]; then
    echo "Downloading ARC-AGI data..."
    mkdir -p data
    if [ ! -d "data/arc-repo" ]; then
        git clone --depth 1 https://github.com/fchollet/ARC-AGI.git data/arc-repo
    fi
    ln -sf "$(pwd)/data/arc-repo/data" data/arc
    echo "ARC data ready: $(ls data/arc/training | wc -l) training tasks, $(ls data/arc/evaluation | wc -l) eval tasks"
fi

COMMON_ARGS="--data_dir data/arc --lr 3e-4 --weight_decay 0.1 \
    --warmup_steps 500 --grad_clip 1.0 --log_every 50 --eval_every 500 \
    --eval_batches 20 --save_every 2000 --max_steps 10000"

case "$MODE" in
    flat)
        echo "=== ARC Flat Baseline ==="
        python scripts/train_arc.py \
            --config configs/arc_flat.yaml \
            --output_dir output_arc_flat \
            --batch_size 16 \
            $COMMON_ARGS
        python scripts/eval_arc.py \
            --config configs/arc_flat.yaml \
            --checkpoint output_arc_flat/checkpoints/final.pt \
            --data_dir data/arc --n_batches 50 --batch_size 8
        ;;
    fluid)
        echo "=== ARC FluidNet (task-only geometry) ==="
        python scripts/train_arc.py \
            --config configs/arc_fluid.yaml \
            --output_dir output_arc_fluid \
            --batch_size 4 \
            $COMMON_ARGS
        python scripts/eval_arc.py \
            --config configs/arc_fluid.yaml \
            --checkpoint output_arc_fluid/checkpoints/final.pt \
            --data_dir data/arc --n_batches 50 --batch_size 4
        ;;
    fluid-struct)
        echo "=== ARC FluidNet + Structural Energy ==="
        python scripts/train_arc.py \
            --config configs/arc_fluid_struct.yaml \
            --output_dir output_arc_fluid_struct \
            --batch_size 4 \
            $COMMON_ARGS
        python scripts/eval_arc.py \
            --config configs/arc_fluid_struct.yaml \
            --checkpoint output_arc_fluid_struct/checkpoints/final.pt \
            --data_dir data/arc --n_batches 50 --batch_size 4
        ;;
    fluid-iter3)
        echo "=== ARC FluidNet + Iterative Diffusion (K=3) ==="
        python scripts/train_arc.py \
            --config configs/arc_fluid_iter3.yaml \
            --output_dir output_arc_fluid_iter3 \
            --batch_size 4 \
            $COMMON_ARGS
        python scripts/eval_arc.py \
            --config configs/arc_fluid_iter3.yaml \
            --checkpoint output_arc_fluid_iter3/checkpoints/final.pt \
            --data_dir data/arc --n_batches 50 --batch_size 4
        ;;
    fluid-grok)
        echo "=== ARC FluidNet — Grok Pure (d=256, no penalties, 100K steps) ==="
        python scripts/train_arc.py \
            --config configs/arc_fluid_grok.yaml \
            --output_dir output_arc_fluid_grok \
            --batch_size 4 \
            --data_dir data/arc --lr 3e-4 --weight_decay 0.1 \
            --warmup_steps 1000 --grad_clip 1.0 --log_every 500 \
            --eval_every 2000 --eval_batches 20 --save_every 10000 \
            --max_steps 100000
        python scripts/eval_arc.py \
            --config configs/arc_fluid_grok.yaml \
            --checkpoint output_arc_fluid_grok/checkpoints/final.pt \
            --data_dir data/arc --n_batches 50 --batch_size 4
        ;;
    fluid-grok-30m)
        echo "=== ARC FluidNet — 30M Grok (d=512, 10L, 100K steps) ==="
        python scripts/train_arc.py \
            --config configs/arc_fluid_grok_30m.yaml \
            --output_dir output_arc_fluid_grok_30m \
            --batch_size 2 \
            --data_dir data/arc --lr 1e-4 --weight_decay 0.1 \
            --warmup_steps 2000 --grad_clip 1.0 --log_every 500 \
            --eval_every 2000 --eval_batches 10 --save_every 10000 \
            --max_steps 100000
        python scripts/eval_arc.py \
            --config configs/arc_fluid_grok_30m.yaml \
            --checkpoint output_arc_fluid_grok_30m/checkpoints/final.pt \
            --data_dir data/arc --n_batches 50 --batch_size 2
        ;;
    sandwich-30m)
        echo "=== ARC Sandwich — 30M (3 geo + 4 attn + 3 geo, 100K steps) ==="
        python scripts/train_arc.py \
            --config configs/arc_sandwich_30m.yaml \
            --output_dir output_arc_sandwich_30m \
            --batch_size 2 \
            --data_dir data/arc --lr 1e-4 --weight_decay 0.1 \
            --warmup_steps 2000 --grad_clip 1.0 --log_every 500 \
            --eval_every 2000 --eval_batches 10 --save_every 10000 \
            --max_steps 100000
        python scripts/eval_arc.py \
            --config configs/arc_sandwich_30m.yaml \
            --checkpoint output_arc_sandwich_30m/checkpoints/final.pt \
            --data_dir data/arc --n_batches 50 --batch_size 2
        ;;
    sandwich-deep)
        echo "=== ARC Sandwich Deep — d=256, 3+16+3 layers, 100K steps ==="
        python scripts/train_arc.py \
            --config configs/arc_sandwich_deep.yaml \
            --output_dir output_arc_sandwich_deep \
            --batch_size 4 \
            --data_dir data/arc --lr 3e-4 --weight_decay 0.1 \
            --warmup_steps 1000 --grad_clip 1.0 --log_every 500 \
            --eval_every 2000 --eval_batches 20 --save_every 10000 \
            --max_steps 100000
        python scripts/eval_arc.py \
            --config configs/arc_sandwich_deep.yaml \
            --checkpoint output_arc_sandwich_deep/checkpoints/final.pt \
            --data_dir data/arc --n_batches 50 --batch_size 4
        ;;
    sandwich-loop)
        echo "=== ARC Sandwich Loop — 2 attn layers × 8 iters, 100K steps ==="
        python scripts/train_arc.py \
            --config configs/arc_sandwich_loop.yaml \
            --output_dir output_arc_sandwich_loop \
            --batch_size 4 \
            --data_dir data/arc --lr 3e-4 --weight_decay 0.1 \
            --warmup_steps 1000 --grad_clip 1.0 --log_every 500 \
            --eval_every 2000 --eval_batches 20 --save_every 10000 \
            --max_steps 100000
        python scripts/eval_arc.py \
            --config configs/arc_sandwich_loop.yaml \
            --checkpoint output_arc_sandwich_loop/checkpoints/final.pt \
            --data_dir data/arc --n_batches 50 --batch_size 4
        ;;
    sandwich-learn)
        echo "=== ARC Sandwich Learn — transform-weighted + deep supervision, 100K steps ==="
        python scripts/train_arc.py \
            --config configs/arc_sandwich_learn.yaml \
            --output_dir output_arc_sandwich_learn \
            --batch_size 4 \
            --data_dir data/arc --lr 3e-4 --weight_decay 0.1 \
            --warmup_steps 1000 --grad_clip 1.0 --log_every 500 \
            --eval_every 2000 --eval_batches 20 --save_every 10000 \
            --max_steps 100000
        python scripts/eval_arc.py \
            --config configs/arc_sandwich_learn.yaml \
            --checkpoint output_arc_sandwich_learn/checkpoints/final.pt \
            --data_dir data/arc --n_batches 50 --batch_size 4
        ;;
    sandwich-refine)
        echo "=== ARC Sandwich Refine — recursive self-refinement (4 iters), 100K steps ==="
        python scripts/train_arc.py \
            --config configs/arc_sandwich_refine.yaml \
            --output_dir output_arc_sandwich_refine \
            --batch_size 4 \
            --data_dir data/arc --lr 3e-4 --weight_decay 0.1 \
            --warmup_steps 1000 --grad_clip 1.0 --log_every 500 \
            --eval_every 2000 --eval_batches 20 --save_every 10000 \
            --max_steps 100000
        python scripts/eval_arc.py \
            --config configs/arc_sandwich_refine.yaml \
            --checkpoint output_arc_sandwich_refine/checkpoints/final.pt \
            --data_dir data/arc --n_batches 50 --batch_size 4
        ;;
    sandwich-liquid)
        echo "=== ARC Sandwich Liquid — continuous-time LTC ODE, 100K steps ==="
        python scripts/train_arc.py \
            --config configs/arc_sandwich_liquid.yaml \
            --output_dir output_arc_sandwich_liquid \
            --batch_size 2 \
            --data_dir data/arc --lr 3e-4 --weight_decay 0.1 \
            --warmup_steps 1000 --grad_clip 1.0 --log_every 50 \
            --eval_every 2000 --eval_batches 20 --save_every 10000 \
            --max_steps 100000
        ;;
    sandwich-liquid-smoke)
        echo "=== ARC Sandwich Liquid — 100-step smoke test ==="
        python scripts/train_arc.py \
            --config configs/arc_sandwich_liquid.yaml \
            --output_dir output_arc_sandwich_liquid_smoke \
            --batch_size 2 \
            --data_dir data/arc --lr 3e-4 --weight_decay 0.1 \
            --warmup_steps 20 --grad_clip 1.0 --log_every 10 \
            --eval_every 50 --eval_batches 5 --save_every 100 \
            --max_steps 100
        ;;
    eval-ttt)
        CKPT="${2:-output_arc_sandwich_refine/checkpoints/best.pt}"
        echo "=== ARC TTT Evaluation — checkpoint: $CKPT ==="
        python scripts/eval_arc_ttt.py \
            --config configs/arc_sandwich_refine.yaml \
            --checkpoint "$CKPT" \
            --data_dir data/arc \
            --ttt_steps 2000 --ttt_lr 1e-3 \
            --n_tta_augments 32 --max_seq_len 2048
        ;;
    eval-ttt-quick)
        CKPT="${2:-output_arc_sandwich_refine/checkpoints/best.pt}"
        echo "=== ARC TTT Quick Eval (10 tasks) — checkpoint: $CKPT ==="
        python scripts/eval_arc_ttt.py \
            --config configs/arc_sandwich_refine.yaml \
            --checkpoint "$CKPT" \
            --data_dir data/arc \
            --ttt_steps 2000 --ttt_lr 1e-3 \
            --n_tta_augments 32 --max_seq_len 2048 --max_tasks 10
        ;;
    all)
        echo "=== Running all ARC ablations sequentially ==="
        for m in flat fluid fluid-struct fluid-iter3; do
            echo ""
            echo "=========================================="
            echo "  Starting: $m"
            echo "=========================================="
            bash scripts/run_arc.sh "$m"
        done
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [flat|fluid|fluid-struct|fluid-iter3|all]"
        exit 1
        ;;
esac

echo "Done: $MODE"
