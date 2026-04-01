"""LiquidARC Test-Time Training evaluation.

Load a checkpoint and evaluate on ARC-AGI eval split with and without TTT.
Compares adapted vs baseline accuracy. Supports both gradient-based TTT
and amortized TTT via hypernetwork.

Usage:
    # Gradient-based TTT (V2/V3a/V3b/V3ab):
    python scripts/eval_ttt.py \
        --checkpoint output_geo_v2/checkpoints/best.pt \
        --config configs/liquid_arc_ttt_v3ab.yaml \
        --data_dir data/arc \
        --ttt_steps 100 --ttt_lr 1e-3

    # Amortized TTT via hypernetwork (V3c):
    python scripts/eval_ttt.py \
        --checkpoint output_geo_v2/checkpoints/best.pt \
        --hypernet_checkpoint output_hypernet/checkpoints/final.pt \
        --config configs/liquid_arc_ttt_v3c.yaml \
        --data_dir data/arc
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import create_model, LiquidARCModel
from liquid_arc.ttt import evaluate_ttt, evaluate_ttt_amortized, evaluate_baseline


def main():
    parser = argparse.ArgumentParser(description="LiquidARC TTT Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (overrides checkpoint config)")
    parser.add_argument("--data_dir", type=str, default="data/arc",
                        help="Path to ARC data directory")
    parser.add_argument("--n_tasks", type=int, default=None,
                        help="Limit number of eval tasks (default: all)")
    parser.add_argument("--ttt_steps", type=int, default=30)
    parser.add_argument("--ttt_lr", type=float, default=1e-3)
    parser.add_argument("--ttt_curvature_lambda", type=float, default=0.01)
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-task results")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline (no-TTT) evaluation")
    parser.add_argument("--hypernet_checkpoint", type=str, default=None,
                        help="HyperNetwork checkpoint for amortized TTT (V3c)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    if isinstance(config, dict):
        config = LiquidARCConfig(**config)

    # Override config from YAML if provided
    if args.config:
        config = LiquidARCConfig.from_yaml(args.config)

    # Apply TTT config overrides
    config.ttt_steps = args.ttt_steps
    config.ttt_lr = args.ttt_lr
    config.ttt_curvature_lambda = args.ttt_curvature_lambda

    # Load model (NO torch.compile — deepcopy needs raw modules)
    config.use_torch_compile = False
    model = create_model(config, device)

    # Strip torch.compile _orig_mod. prefix from state dict keys if present
    state_dict = ckpt["model"]
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("_orig_mod.", "")] = v
    model.load_state_dict(cleaned)
    step = ckpt.get("step", "?")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint from step {step}")
    print(f"Model: {config.model_type}, params: {n_params:,}")

    if not isinstance(model, LiquidARCModel):
        print("ERROR: TTT requires LiquidARCModel (not flat baseline)")
        sys.exit(1)

    # Load hypernetwork if provided (V3c amortized TTT)
    hypernet = None
    if args.hypernet_checkpoint:
        from liquid_arc.hypernet import HyperNetwork
        hypernet_ckpt = torch.load(args.hypernet_checkpoint, map_location=device,
                                    weights_only=False)
        hypernet = HyperNetwork(config, model).to(device)
        hypernet.load_state_dict(hypernet_ckpt["hypernet"])
        hypernet.eval()
        n_hyper_params = sum(p.numel() for p in hypernet.parameters())
        print(f"Loaded HyperNetwork: {n_hyper_params:,} params")

    if hypernet is not None:
        print(f"\nTTT mode: AMORTIZED (hypernetwork)")
    else:
        print(f"\nTTT config: steps={args.ttt_steps}, lr={args.ttt_lr}, "
              f"curv_lambda={args.ttt_curvature_lambda}")
        if config.ttt_unfreeze_ffn:
            print(f"  V3a: FFN[-1] unfreeze enabled")
        if config.ttt_d4_augment:
            print(f"  V3b: D4 augmentation enabled")

    # Baseline evaluation (no TTT)
    if not args.skip_baseline:
        print(f"\n{'='*60}")
        print("BASELINE (no adaptation)")
        print(f"{'='*60}")
        t0 = time.time()
        bl_cell, bl_xform = evaluate_baseline(
            model, args.data_dir, config, device, n_tasks=args.n_tasks)
        bl_time = time.time() - t0
        print(f"  Time: {bl_time:.1f}s")
    else:
        bl_cell, bl_xform = 0.0, 0.0

    # TTT evaluation
    print(f"\n{'='*60}")
    if hypernet is not None:
        print("AMORTIZED TTT (hypernetwork)")
    else:
        print("TTT (test-time training)")
    print(f"{'='*60}")
    t0 = time.time()

    if hypernet is not None:
        # Amortized TTT via hypernetwork
        ttt_cell, ttt_xform = evaluate_ttt_amortized(
            model, args.data_dir, config, device, hypernet,
            n_tasks=args.n_tasks, verbose=args.verbose,
        )
    else:
        # Standard gradient-based TTT
        ttt_cell, ttt_xform = evaluate_ttt(
            model, args.data_dir, config, device,
            n_tasks=args.n_tasks, verbose=args.verbose,
            ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr,
        )
    ttt_time = time.time() - t0
    print(f"  Time: {ttt_time:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    if not args.skip_baseline:
        print(f"  Baseline  — cell: {bl_cell:.4f}, xform: {bl_xform:.4f}")
    mode = "Amortized" if hypernet is not None else "TTT"
    print(f"  {mode:10s} — cell: {ttt_cell:.4f}, xform: {ttt_xform:.4f}")
    if not args.skip_baseline:
        cell_delta = ttt_cell - bl_cell
        xform_delta = ttt_xform - bl_xform
        print(f"  Delta     — cell: {cell_delta:+.4f}, xform: {xform_delta:+.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
