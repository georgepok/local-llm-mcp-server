#!/usr/bin/env python3
"""Experiment 003: Mamba A_log Modification (Curvature Adjustment)

Modifies A_log in a Mamba SSM layer to change per-head state decay rate.
A_log controls how quickly the SSM forgets past tokens:
  - More positive A_log → slower decay → longer memory (003a: +0.5)
  - More negative A_log → faster decay → more responsive (003b: -0.5)

Target: backbone.layers.50.mixer.A_log (BF16, shape [64])
"""

import argparse
import json
import os
import sys
import time

try:
    import torch  # type: ignore[import-not-found]
    from safetensors import safe_open  # type: ignore[import-not-found]
    from safetensors.torch import save_file  # type: ignore[import-not-found]
except ImportError:
    print("ERROR: torch and safetensors required. Run inside the vLLM container.")
    sys.exit(1)

DEFAULT_LAYER = 50
DEFAULT_TENSOR = "backbone.layers.{layer}.mixer.A_log"


def load_weight_map(model_path: str) -> dict:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        return json.load(f)["weight_map"]


def modify_alog(
    model_path: str,
    backup_dir: str,
    layer: int,
    shift: float,
    dry_run: bool = False,
    restore: bool = False,
) -> dict:
    tensor_key = DEFAULT_TENSOR.format(layer=layer)
    weight_map = load_weight_map(model_path)

    if tensor_key not in weight_map:
        print(f"ERROR: {tensor_key} not found in weight map")
        # List available A_log tensors
        alog_keys = sorted(k for k in weight_map if "A_log" in k)
        print(f"Available A_log tensors: {alog_keys}")
        sys.exit(1)

    shard_name = weight_map[tensor_key]
    shard_path = os.path.join(model_path, shard_name)
    backup_path = os.path.join(backup_dir, shard_name)

    # Load all tensors from shard
    all_tensors = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            all_tensors[k] = f.get_tensor(k)

    if restore:
        print("=== RESTORING from backup ===")
        if not os.path.exists(backup_path):
            print(f"  ERROR: backup not found: {backup_path}")
            sys.exit(1)
        import shutil
        shutil.copy2(backup_path, shard_path)
        print(f"  Restored {shard_name} from backup")
        print("Restart vLLM to load original weights.")
        return {"action": "restored"}

    t = all_tensors[tensor_key]
    t_float = t.float()

    direction = "increase_memory" if shift > 0 else "increase_forgetting"
    sub_exp = "003a" if shift > 0 else "003b"

    print("=" * 60)
    print(f"Experiment {sub_exp}: Mamba A_log Modification ({direction})")
    print("=" * 60)
    print(f"Tensor: {tensor_key}")
    print(f"Shard: {shard_name}")
    print(f"Shape: {t.shape}")
    print(f"Dtype: {t.dtype}")
    print(f"Shift: {shift:+.1f}")
    print(f"Dry run: {dry_run}")
    print()

    # Before stats
    before = {
        "mean": t_float.mean().item(),
        "std": t_float.std().item(),
        "min": t_float.min().item(),
        "max": t_float.max().item(),
        "median": t_float.median().item(),
    }
    print(f"Before: mean={before['mean']:.4f}, std={before['std']:.4f}, "
          f"range=[{before['min']:.4f}, {before['max']:.4f}]")

    # Apply shift
    t_modified = t_float + shift

    # After stats
    after = {
        "mean": t_modified.mean().item(),
        "std": t_modified.std().item(),
        "min": t_modified.min().item(),
        "max": t_modified.max().item(),
        "median": t_modified.median().item(),
    }
    print(f"After:  mean={after['mean']:.4f}, std={after['std']:.4f}, "
          f"range=[{after['min']:.4f}, {after['max']:.4f}]")

    result = {
        "sub_experiment": sub_exp,
        "direction": direction,
        "tensor_key": tensor_key,
        "shard": shard_name,
        "layer": layer,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "shift": shift,
        "before": before,
        "after": after,
        "dry_run": dry_run,
    }

    if not dry_run:
        # Backup
        os.makedirs(backup_dir, exist_ok=True)
        if not os.path.exists(backup_path):
            print(f"\nBacking up {shard_name}...")
            save_file(all_tensors, backup_path)
        else:
            print(f"\nBackup already exists: {backup_path}")

        # Convert back to original dtype and write
        all_tensors[tensor_key] = t_modified.to(t.dtype)
        print(f"Writing {shard_name}...")
        save_file(all_tensors, shard_path)
        print("Written successfully.")
        result["written"] = True
    else:
        result["written"] = False

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 003: Mamba A_log Modification")
    parser.add_argument("--model-path", required=True, help="Path to model directory")
    parser.add_argument("--backup-dir", default=None, help="Backup directory")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER, help=f"Mamba layer (default: {DEFAULT_LAYER})")
    parser.add_argument("--shift", type=float, required=True, help="A_log shift (+0.5 for 003a, -0.5 for 003b)")
    parser.add_argument("--dry-run", action="store_true", help="Compute stats without modifying")
    parser.add_argument("--restore", action="store_true", help="Restore from backup")
    args = parser.parse_args()

    backup_dir = args.backup_dir or os.path.join(os.path.dirname(args.model_path), "exp003_backup")

    t0 = time.time()
    result = modify_alog(
        model_path=args.model_path,
        backup_dir=backup_dir,
        layer=args.layer,
        shift=args.shift,
        dry_run=args.dry_run,
        restore=args.restore,
    )
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    result["elapsed_seconds"] = elapsed
    print(json.dumps(result, indent=2))

    if not args.dry_run and not args.restore:
        print(f"\nRestart vLLM to load modified weights.")
        print(f"To restore: python3 {__file__} --model-path {args.model_path} --layer {args.layer} --shift 0 --restore")


if __name__ == "__main__":
    main()
