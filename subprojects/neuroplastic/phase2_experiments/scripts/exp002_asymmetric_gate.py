#!/usr/bin/env python3
"""Experiment 002: Asymmetric Gate Row Scaling

Applies non-uniform scaling to MoE gate weight rows across 4 deep layers (43, 45, 47, 49).
Each row is scaled by a factor based on its L2 norm rank:
  - Rank 0 (weakest norm): scaled by 0.8
  - Rank 127 (strongest norm): scaled by 1.2
  - Linear interpolation between

This amplifies existing directional differences between expert routing vectors,
creating the asymmetry Nemotron identified as needed to break expert homogeneity.
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

TARGET_LAYERS = [43, 45, 47, 49]
SCALE_MIN = 0.8
SCALE_MAX = 1.2


def load_weight_map(model_path: str) -> dict:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        return json.load(f)["weight_map"]


def get_gate_key(layer: int) -> str:
    return f"backbone.layers.{layer}.mixer.gate.weight"


def apply_asymmetric_scaling(
    model_path: str,
    backup_dir: str,
    dry_run: bool = False,
    restore: bool = False,
) -> dict:
    weight_map = load_weight_map(model_path)
    results = {}

    # Group gate keys by shard
    shard_tensors: dict[str, list[str]] = {}
    for layer in TARGET_LAYERS:
        key = get_gate_key(layer)
        shard = weight_map[key]
        shard_tensors.setdefault(shard, []).append(key)

    if restore:
        print("=== RESTORING from backup ===")
        for shard_name in shard_tensors:
            backup_path = os.path.join(backup_dir, shard_name)
            shard_path = os.path.join(model_path, shard_name)
            if not os.path.exists(backup_path):
                print(f"  ERROR: backup not found: {backup_path}")
                continue
            # Load backup shard and overwrite
            all_tensors = {}
            with safe_open(backup_path, framework="pt", device="cpu") as f:
                backup_keys = list(f.keys())
            # Copy entire backup shard over
            import shutil
            shutil.copy2(backup_path, shard_path)
            print(f"  Restored {shard_name} from backup ({len(backup_keys)} tensors)")
        print("Restore complete. Restart vLLM to load original weights.")
        return {"action": "restored"}

    print("=" * 60)
    print("Experiment 002: Asymmetric Gate Row Scaling")
    print("=" * 60)
    print(f"Target layers: {TARGET_LAYERS}")
    print(f"Scale range: {SCALE_MIN} to {SCALE_MAX}")
    print(f"Dry run: {dry_run}")
    print()

    for shard_name, gate_keys in shard_tensors.items():
        shard_path = os.path.join(model_path, shard_name)
        backup_path = os.path.join(backup_dir, shard_name)

        # Load all tensors from shard
        all_tensors = {}
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                all_tensors[k] = f.get_tensor(k)

        # Backup before modification
        if not dry_run:
            os.makedirs(backup_dir, exist_ok=True)
            if not os.path.exists(backup_path):
                print(f"Backing up {shard_name}...")
                save_file(all_tensors, backup_path)
            else:
                print(f"Backup already exists: {backup_path}")

        # Modify each gate in this shard
        for gate_key in gate_keys:
            t = all_tensors[gate_key].clone()
            layer = int(gate_key.split(".")[2])

            # Compute row norms
            row_norms = t.norm(dim=1)  # [128]
            n_experts = t.shape[0]

            # Rank by norm (0 = weakest, n-1 = strongest)
            ranks = row_norms.argsort().argsort().float()

            # Scale factors: linear from SCALE_MIN to SCALE_MAX
            scale_factors = SCALE_MIN + (SCALE_MAX - SCALE_MIN) * (ranks / (n_experts - 1))

            # Before stats
            cv_before = (row_norms.std() / row_norms.mean() * 100).item()
            cos_before = _mean_cosine_sim(t)

            # Apply per-row scaling
            t_modified = t * scale_factors.unsqueeze(1)

            # After stats
            row_norms_after = t_modified.norm(dim=1)
            cv_after = (row_norms_after.std() / row_norms_after.mean() * 100).item()
            cos_after = _mean_cosine_sim(t_modified)

            result = {
                "layer": layer,
                "gate_key": gate_key,
                "shard": shard_name,
                "n_experts": n_experts,
                "before": {
                    "row_norm_mean": row_norms.mean().item(),
                    "row_norm_std": row_norms.std().item(),
                    "row_norm_cv_pct": cv_before,
                    "mean_cosine_sim": cos_before,
                    "tensor_norm": t.norm().item(),
                },
                "after": {
                    "row_norm_mean": row_norms_after.mean().item(),
                    "row_norm_std": row_norms_after.std().item(),
                    "row_norm_cv_pct": cv_after,
                    "mean_cosine_sim": cos_after,
                    "tensor_norm": t_modified.norm().item(),
                },
                "scale_factors": {
                    "min": scale_factors.min().item(),
                    "max": scale_factors.max().item(),
                    "mean": scale_factors.mean().item(),
                },
            }
            results[gate_key] = result

            print(f"Layer {layer}: CV {cv_before:.1f}% → {cv_after:.1f}%, "
                  f"cos_sim {cos_before:.4f} → {cos_after:.4f}")

            if not dry_run:
                all_tensors[gate_key] = t_modified

        # Write modified shard
        if not dry_run:
            print(f"Writing {shard_name}...")
            save_file(all_tensors, shard_path)
            print(f"  Written successfully.")

    return results


def _mean_cosine_sim(t: "torch.Tensor") -> float:
    """Mean pairwise cosine similarity between rows."""
    normed = t / t.norm(dim=1, keepdim=True)
    cos_sim = torch.mm(normed, normed.t())
    n = t.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=t.device)
    return cos_sim[mask].mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 002: Asymmetric Gate Row Scaling")
    parser.add_argument("--model-path", required=True, help="Path to model directory")
    parser.add_argument("--backup-dir", default=None, help="Backup directory (default: <model-path>/../exp002_backup)")
    parser.add_argument("--dry-run", action="store_true", help="Compute stats without modifying")
    parser.add_argument("--restore", action="store_true", help="Restore from backup")
    args = parser.parse_args()

    backup_dir = args.backup_dir or os.path.join(os.path.dirname(args.model_path), "exp002_backup")

    t0 = time.time()
    results = apply_asymmetric_scaling(
        model_path=args.model_path,
        backup_dir=backup_dir,
        dry_run=args.dry_run,
        restore=args.restore,
    )
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"{'DRY RUN — no changes written' if args.dry_run else 'MODIFICATION COMPLETE'}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 60)
    print(json.dumps(results, indent=2))

    if not args.dry_run and not args.restore:
        print("\nRestart vLLM to load modified weights.")
        print(f"To restore: python3 {__file__} --model-path {args.model_path} --backup-dir {backup_dir} --restore")


if __name__ == "__main__":
    main()
