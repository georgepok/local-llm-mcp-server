"""V3 TTT sweep: evaluate V2 / V3a / Amortized across all checkpoints.

Runs gradient-based TTT (V2, V3a) and amortized TTT (hypernetwork) on every
checkpoint, saves results to JSON and prints a comparison table.

Usage (inside container):
    python scripts/eval_v3_sweep.py \
        --checkpoint_dir output_ttt_v2/checkpoints \
        --data_dir /workspace/fgn-v3/data/arc \
        --hypernet_checkpoint output_hypernet/checkpoints/final.pt \
        --output eval_v3_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if FGN_ROOT not in sys.path:
    sys.path.insert(0, FGN_ROOT)

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import create_model, LiquidARCModel
from liquid_arc.ttt import evaluate_ttt, evaluate_ttt_amortized, evaluate_baseline


# Gradient-based configs to sweep (D4 abandoned — hurts TTT)
GRADIENT_CONFIGS = {
    "V2":  "configs/liquid_arc_ttt_v2.yaml",
    "V3a": "configs/liquid_arc_ttt_v3a.yaml",
}

TTT_STEPS = 100
TTT_LR = 1e-3


def load_model(checkpoint_path, device):
    """Load model from checkpoint, return (model, config, step)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    if isinstance(config, dict):
        config = LiquidARCConfig(**config)
    config.use_torch_compile = False

    model = create_model(config, device)
    state_dict = ckpt["model"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    step = ckpt.get("step", "?")
    return model, config, step


def run_gradient_eval(model, config_path, data_dir, device, n_tasks=None):
    """Run gradient-based TTT eval. Returns (cell_acc, xform_acc, elapsed)."""
    config = LiquidARCConfig.from_yaml(config_path)
    config.use_torch_compile = False
    config.ttt_steps = TTT_STEPS
    config.ttt_lr = TTT_LR
    config.ttt_curvature_lambda = 0.01

    t0 = time.time()
    cell_acc, xform_acc = evaluate_ttt(
        model, data_dir, config, device,
        n_tasks=n_tasks,
        ttt_steps=TTT_STEPS,
        ttt_lr=TTT_LR,
    )
    elapsed = time.time() - t0
    return cell_acc, xform_acc, elapsed


def run_amortized_eval(model, hypernet, data_dir, config, device, n_tasks=None):
    """Run amortized TTT eval via hypernetwork. Returns (cell_acc, xform_acc, elapsed)."""
    t0 = time.time()
    cell_acc, xform_acc = evaluate_ttt_amortized(
        model, data_dir, config, device, hypernet,
        n_tasks=n_tasks,
    )
    elapsed = time.time() - t0
    return cell_acc, xform_acc, elapsed


def main():
    parser = argparse.ArgumentParser(description="V3 TTT sweep across checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval_v3_results.json")
    parser.add_argument("--hypernet_checkpoint", type=str, default=None,
                        help="HyperNetwork checkpoint for amortized TTT eval")
    parser.add_argument("--n_tasks", type=int, default=None,
                        help="Limit eval tasks (default: all)")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated checkpoint names to eval (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.checkpoint_dir)

    # Discover checkpoints
    if args.checkpoints:
        ckpt_names = [c.strip() for c in args.checkpoints.split(",")]
    else:
        ckpt_names = sorted([f.name for f in ckpt_dir.glob("*.pt")])

    # Sort: step_5000, step_10000, ..., best, final
    def sort_key(name):
        if name.startswith("step_"):
            return (0, int(name.replace("step_", "").replace(".pt", "")))
        if name == "best.pt":
            return (1, 0)
        if name == "final.pt":
            return (2, 0)
        return (3, 0)
    ckpt_names.sort(key=sort_key)

    # Build config name list for display
    all_config_names = list(GRADIENT_CONFIGS.keys())

    # Load hypernetwork if provided
    hypernet = None
    if args.hypernet_checkpoint:
        from liquid_arc.hypernet import HyperNetwork
        hypernet_ckpt = torch.load(args.hypernet_checkpoint, map_location=device,
                                    weights_only=False)
        # Use config from hypernet checkpoint (has correct hypernet_include_ffn etc.)
        hp_config = hypernet_ckpt.get("config", {})
        if isinstance(hp_config, dict):
            hp_config = LiquidARCConfig(**{k: v for k, v in hp_config.items()
                                           if k in LiquidARCConfig.__dataclass_fields__})
        else:
            hp_config = hp_config
        hp_config.use_torch_compile = False
        # Need a model to infer architecture — load first checkpoint temporarily
        tmp_model, _, _ = load_model(str(ckpt_dir / ckpt_names[0]), device)
        hypernet = HyperNetwork(hp_config, tmp_model).to(device)
        hypernet.load_state_dict(hypernet_ckpt["hypernet"])
        hypernet.eval()
        n_hp = sum(p.numel() for p in hypernet.parameters())
        print(f"HyperNetwork loaded: {n_hp:,} params (include_ffn={hp_config.hypernet_include_ffn})")
        all_config_names.append("Amort")
        del tmp_model
        torch.cuda.empty_cache()

    print(f"Device: {device}")
    print(f"Checkpoints ({len(ckpt_names)}): {ckpt_names}")
    print(f"Configs: {all_config_names}")
    print(f"TTT steps: {TTT_STEPS}, LR: {TTT_LR}")
    n_str = str(args.n_tasks) if args.n_tasks else "all (400)"
    print(f"Tasks: {n_str}")
    print()

    all_results = {}
    sweep_start = time.time()

    for ci, ckpt_name in enumerate(ckpt_names):
        ckpt_path = str(ckpt_dir / ckpt_name)
        print(f"\n{'#'*70}")
        print(f"# Checkpoint {ci+1}/{len(ckpt_names)}: {ckpt_name}")
        print(f"{'#'*70}")

        model, base_config, step = load_model(ckpt_path, device)
        if not isinstance(model, LiquidARCModel):
            print(f"  SKIP: not LiquidARCModel")
            continue

        print(f"  Step: {step}, params: {sum(p.numel() for p in model.parameters()):,}")
        ckpt_results = {"step": step, "checkpoint": ckpt_name}

        # Baseline (no TTT) — same for all configs, run once
        print(f"\n  --- Baseline (no TTT) ---")
        t0 = time.time()
        bl_cell, bl_xform = evaluate_baseline(
            model, args.data_dir, base_config, device, n_tasks=args.n_tasks)
        bl_time = time.time() - t0
        ckpt_results["baseline"] = {
            "cell_acc": bl_cell, "xform_acc": bl_xform, "time": bl_time
        }

        # Gradient-based TTT with each config
        for config_name, config_path in GRADIENT_CONFIGS.items():
            print(f"\n  --- {config_name} TTT ---")
            try:
                cell_acc, xform_acc, elapsed = run_gradient_eval(
                    model, config_path, args.data_dir, device,
                    n_tasks=args.n_tasks,
                )
                ckpt_results[config_name] = {
                    "cell_acc": cell_acc, "xform_acc": xform_acc, "time": elapsed
                }
                delta = xform_acc - bl_xform
                print(f"  {config_name}: xform={xform_acc:.4f} (delta={delta:+.4f}), "
                      f"{elapsed:.0f}s")
            except Exception as e:
                print(f"  {config_name}: ERROR — {e}")
                import traceback; traceback.print_exc()
                ckpt_results[config_name] = {"error": str(e)}

        # Amortized TTT via hypernetwork
        if hypernet is not None:
            print(f"\n  --- Amortized TTT (hypernetwork) ---")
            try:
                cell_acc, xform_acc, elapsed = run_amortized_eval(
                    model, hypernet, args.data_dir, base_config, device,
                    n_tasks=args.n_tasks,
                )
                ckpt_results["Amort"] = {
                    "cell_acc": cell_acc, "xform_acc": xform_acc, "time": elapsed
                }
                delta = xform_acc - bl_xform
                print(f"  Amort: xform={xform_acc:.4f} (delta={delta:+.4f}), "
                      f"{elapsed:.0f}s")
            except Exception as e:
                print(f"  Amort: ERROR — {e}")
                import traceback; traceback.print_exc()
                ckpt_results["Amort"] = {"error": str(e)}

        all_results[ckpt_name] = ckpt_results

        # Save incrementally
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)

        # Print checkpoint summary
        print(f"\n  Summary for {ckpt_name} (step {step}):")
        print(f"    {'Config':<8} {'Cell':>8} {'Xform':>8} {'Delta':>8} {'Time':>6}")
        print(f"    {'---':<8} {'---':>8} {'---':>8} {'---':>8} {'---':>6}")
        print(f"    {'BL':<8} {bl_cell:>8.4f} {bl_xform:>8.4f} {'—':>8} {bl_time:>5.0f}s")
        for cn in all_config_names:
            r = ckpt_results.get(cn, {})
            if "error" in r:
                print(f"    {cn:<8} {'ERROR':>8}")
            else:
                d = r.get("xform_acc", 0) - bl_xform
                print(f"    {cn:<8} {r.get('cell_acc',0):>8.4f} "
                      f"{r.get('xform_acc',0):>8.4f} {d:>+8.4f} "
                      f"{r.get('time',0):>5.0f}s")

        # Free model
        del model
        torch.cuda.empty_cache()

    total_time = time.time() - sweep_start

    # Final summary table
    print(f"\n\n{'='*80}")
    print(f"FULL SWEEP RESULTS ({total_time/3600:.1f}h total)")
    print(f"{'='*80}")
    header = f"{'Checkpoint':<18} {'Step':>6} {'BL xf':>8} "
    header += "".join(f"{cn+' xf':>10}" for cn in all_config_names)
    print(header)
    print("-" * 80)

    for ckpt_name in ckpt_names:
        r = all_results.get(ckpt_name, {})
        step = r.get("step", "?")
        bl_xf = r.get("baseline", {}).get("xform_acc", 0)
        line = f"{ckpt_name:<18} {str(step):>6} {bl_xf:>8.4f} "
        for cn in all_config_names:
            cr = r.get(cn, {})
            if "error" in cr:
                line += f"{'ERR':>10}"
            else:
                line += f"{cr.get('xform_acc', 0):>10.4f}"
        print(line)

    print(f"{'='*80}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
