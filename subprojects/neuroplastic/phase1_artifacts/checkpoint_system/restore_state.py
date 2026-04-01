"""
restore_state.py — Restore weight tensors from a checkpoint.

Two modes:
  1. Full restore: copies all safetensors shards back to model directory
  2. Selective restore: writes specific tensors back into the appropriate shards

Usage:
    # Full restore
    python restore_state.py --model-path /workspace/model --checkpoint /workspace/checkpoints/baseline --full

    # Selective restore (from a selective checkpoint)
    python restore_state.py --model-path /workspace/model --checkpoint /workspace/checkpoints/exp001

    # After restore, restart vLLM to pick up changes (from host):
    # docker restart vllm-nemotron-serve
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def restore_full(model_path: Path, checkpoint_dir: Path) -> None:
    """Copy all safetensors shards from checkpoint back to model directory."""
    shards = sorted(checkpoint_dir.glob("model-*.safetensors"))
    if not shards:
        print(f"ERROR: No safetensors files in {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Full restore: {len(shards)} shards from {checkpoint_dir}")
    for shard in shards:
        dest = model_path / shard.name
        print(f"  Restoring {shard.name}...")
        shutil.copy2(shard, dest)

    # Restore index if present
    index_src = checkpoint_dir / "model.safetensors.index.json"
    if index_src.exists():
        shutil.copy2(index_src, model_path / "model.safetensors.index.json")

    print("Full restore complete. Restart vLLM to load restored weights.")


def restore_selective(model_path: Path, checkpoint_dir: Path) -> None:
    """Restore specific tensors from a selective checkpoint into model shards."""
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
        from safetensors.torch import save_file  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]  # noqa: F811
        _ = torch  # needed for safetensors framework="pt"
    except ImportError:
        print("ERROR: safetensors/torch not installed", file=sys.stderr)
        sys.exit(1)

    checkpoint_file = checkpoint_dir / "checkpoint.safetensors"
    meta_file = checkpoint_dir / "checkpoint_meta.json"

    if not checkpoint_file.exists():
        print(f"ERROR: No checkpoint.safetensors in {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    # Load checkpoint metadata
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"Restoring checkpoint from {meta.get('timestamp', 'unknown')}")
        print(f"  Tensors: {meta.get('tensor_count', '?')}")
    else:
        print("WARNING: No checkpoint_meta.json found, proceeding anyway")

    # Load weight map to find which shard each key belongs to
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f).get("weight_map", {})

    # Load checkpoint tensors
    checkpoint_tensors: dict = {}
    with safe_open(str(checkpoint_file), framework="pt") as f:
        for key in f.keys():
            checkpoint_tensors[key] = f.get_tensor(key)

    # Group by shard
    shard_keys: dict[str, list[str]] = {}
    for key in checkpoint_tensors:
        if key not in weight_map:
            print(f"  WARNING: key '{key}' not in current weight map, skipping")
            continue
        shard = weight_map[key]
        if shard not in shard_keys:
            shard_keys[shard] = []
        shard_keys[shard].append(key)

    # For each affected shard: load all tensors, replace the checkpoint ones, re-save
    for shard_name, keys_to_restore in shard_keys.items():
        shard_path = model_path / shard_name
        print(f"  Patching {shard_name} ({len(keys_to_restore)} tensors)...")

        # Load all tensors from current shard
        all_tensors: dict = {}
        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

        # Replace with checkpoint tensors
        for key in keys_to_restore:
            if key in all_tensors:
                old_shape = all_tensors[key].shape
                new_shape = checkpoint_tensors[key].shape
                if old_shape != new_shape:
                    print(f"    WARNING: shape mismatch for {key}: "
                          f"{old_shape} vs {new_shape}, skipping")
                    continue
                all_tensors[key] = checkpoint_tensors[key]
                print(f"    Restored: {key}")
            else:
                print(f"    WARNING: {key} not found in shard, skipping")

        # Re-save the shard
        save_file(all_tensors, str(shard_path))

    restored_count = sum(len(v) for v in shard_keys.values())
    print(f"\nRestored {restored_count} tensors across {len(shard_keys)} shards.")
    print("Restart vLLM to load restored weights.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore model weights from checkpoint")
    parser.add_argument("--model-path", type=str, default="/workspace/model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint directory")
    parser.add_argument("--full", action="store_true",
                        help="Full restore (copy all shards)")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    checkpoint_dir = Path(args.checkpoint)

    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    if args.full:
        restore_full(model_path, checkpoint_dir)
    else:
        restore_selective(model_path, checkpoint_dir)


if __name__ == "__main__":
    main()
