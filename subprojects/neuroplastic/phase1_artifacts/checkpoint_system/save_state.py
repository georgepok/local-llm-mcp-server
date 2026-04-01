"""
save_state.py — Save specific weight tensors for checkpoint/rollback.

Supports two modes:
  1. Full backup: copies all safetensors shards (32GB, ~60s)
  2. Selective backup: saves only specified tensor keys (fast, minimal disk)

Usage:
    # Full backup
    python save_state.py --model-path /workspace/model --output /workspace/checkpoints/baseline

    # Selective backup (specific tensors)
    python save_state.py --model-path /workspace/model --output /workspace/checkpoints/exp001 \
        --keys "backbone.layers.42.mixer.v_proj.weight" "backbone.layers.42.mixer.o_proj.weight"

    # Backup all tensors for a specific layer
    python save_state.py --model-path /workspace/model --output /workspace/checkpoints/exp002 \
        --layers 42
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def get_weight_map(model_path: Path) -> dict:
    """Load the weight map from safetensors index."""
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"ERROR: No safetensors index at {index_path}", file=sys.stderr)
        sys.exit(1)
    with open(index_path) as f:
        return json.load(f).get("weight_map", {})


def save_full(model_path: Path, output_dir: Path) -> None:
    """Copy all safetensors shards to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(model_path.glob("model-*.safetensors"))
    print(f"Full backup: {len(shards)} shards")

    for shard in shards:
        dest = output_dir / shard.name
        print(f"  Copying {shard.name} ({shard.stat().st_size / 1e9:.2f} GB)...")
        shutil.copy2(shard, dest)

    # Also copy the index
    index_src = model_path / "model.safetensors.index.json"
    if index_src.exists():
        shutil.copy2(index_src, output_dir / "model.safetensors.index.json")

    print(f"Full backup complete: {output_dir}")


def save_selective(model_path: Path, output_dir: Path, keys: list[str]) -> None:
    """Save only specified tensor keys using safetensors."""
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
        from safetensors.torch import save_file  # type: ignore[import-not-found]
    except ImportError:
        print("ERROR: safetensors not installed", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    weight_map = get_weight_map(model_path)

    # Group keys by shard
    shard_keys: dict[str, list[str]] = {}
    for key in keys:
        if key not in weight_map:
            print(f"  WARNING: key '{key}' not found in weight map, skipping")
            continue
        shard = weight_map[key]
        if shard not in shard_keys:
            shard_keys[shard] = []
        shard_keys[shard].append(key)

    # Extract tensors shard by shard
    tensors: dict = {}
    for shard_name, shard_key_list in shard_keys.items():
        shard_path = model_path / shard_name
        print(f"  Reading {len(shard_key_list)} tensors from {shard_name}...")
        with safe_open(str(shard_path), framework="pt") as f:
            for key in shard_key_list:
                tensors[key] = f.get_tensor(key)

    if not tensors:
        print("ERROR: No tensors to save", file=sys.stderr)
        sys.exit(1)

    # Save as single safetensors file
    output_file = output_dir / "checkpoint.safetensors"
    save_file(tensors, str(output_file))

    # Save metadata
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_path": str(model_path),
        "keys": keys,
        "tensor_count": len(tensors),
        "total_bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
    }
    with open(output_dir / "checkpoint_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Selective backup: {len(tensors)} tensors → {output_file}")
    print(f"  Total size: {meta['total_bytes'] / 1e6:.2f} MB")


def keys_for_layers(model_path: Path, layer_indices: list[int]) -> list[str]:
    """Get all weight keys belonging to specified layer indices."""
    weight_map = get_weight_map(model_path)
    keys = []
    for key in weight_map:
        for idx in layer_indices:
            prefix = f"backbone.layers.{idx}."
            if key.startswith(prefix):
                # Skip scale tensors — they're derived from weights
                if not key.endswith((".weight_scale", ".input_scale")):
                    keys.append(key)
                break
    return sorted(keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Save model weight checkpoint")
    parser.add_argument("--model-path", type=str, default="/workspace/model")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for checkpoint")
    parser.add_argument("--keys", nargs="+", type=str, default=None,
                        help="Specific tensor keys to save")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Save all tensors for these layer indices")
    parser.add_argument("--full", action="store_true",
                        help="Full backup of all shards")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_dir = Path(args.output)

    if args.full:
        save_full(model_path, output_dir)
    elif args.layers is not None:
        keys = keys_for_layers(model_path, args.layers)
        print(f"Layer {args.layers} → {len(keys)} tensor keys")
        save_selective(model_path, output_dir, keys)
    elif args.keys is not None:
        save_selective(model_path, output_dir, args.keys)
    else:
        print("ERROR: Specify --full, --keys, or --layers", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
