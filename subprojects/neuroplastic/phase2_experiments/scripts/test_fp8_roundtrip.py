"""
test_fp8_roundtrip.py — Validate the FP8 modification pipeline on a non-critical tensor.

Tests that dequantize → modify → requantize → write-back is safe by exercising it on a
single expert weight deep in the network (backbone.layers.45.mixer.experts.127.up_proj),
computing roundtrip error statistics, and optionally restoring from backup.

This is the safety validation step that MUST pass before any meaningful weight surgery.

FP8 Format Notes:
  - Weights are torch.float8_e4m3fn with values in [-448, 448]
  - Each quantized tensor has a companion {name}.weight_scale (float32 scalar)
  - Dequant: w_real = w_fp8.float() * weight_scale
  - Requant: scale = abs_max / 448.0; w_fp8 = (w_real / scale).clamp(-448, 448).to(fp8)
  - input_scale is for activation quantization — DO NOT MODIFY

Usage:
    python test_fp8_roundtrip.py
    python test_fp8_roundtrip.py --model-path /workspace/model
    python test_fp8_roundtrip.py --model-path /workspace/model --dry-run
    python test_fp8_roundtrip.py --model-path /workspace/model --restore
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Non-critical target: last expert in a deep MoE layer. Expert 127 (0-indexed)
# in layer 45 is the last routed expert — likely lower-traffic, safe for testing.
TARGET_TENSOR = "backbone.layers.45.mixer.experts.127.up_proj.weight"
TARGET_SCALE  = "backbone.layers.45.mixer.experts.127.up_proj.weight_scale"

# Tiny modification: 0.1% multiplicative perturbation — below perceptual threshold
MODIFICATION_FACTOR = 1.001

# FP8 e4m3fn max representable value
FP8_MAX = 448.0


# ---------------------------------------------------------------------------
# FP8 utilities
# ---------------------------------------------------------------------------

def dequantize_fp8(w_fp8, scale):
    """
    Dequantize an FP8 tensor to float32.

    Args:
        w_fp8: torch.Tensor of dtype torch.float8_e4m3fn
        scale: torch.Tensor scalar float32 (per-tensor scale factor)

    Returns:
        torch.Tensor float32 with actual weight values
    """
    import torch
    return w_fp8.to(torch.float32) * scale.to(torch.float32)


def requantize_fp8(w_real):
    """
    Requantize a float32 tensor back to FP8 with a fresh per-tensor scale.

    The scale is computed as abs_max / FP8_MAX, which is the standard
    per-tensor quantization approach. Returns both the quantized tensor
    and the new scale so the caller can update the weight_scale entry.

    Args:
        w_real: torch.Tensor float32

    Returns:
        (w_fp8, new_scale) where w_fp8 is torch.float8_e4m3fn and
        new_scale is a torch.Tensor scalar float32
    """
    import torch
    abs_max = w_real.abs().max()
    # Guard against all-zero tensors
    if abs_max < 1e-12:
        new_scale = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
    else:
        new_scale = (abs_max / FP8_MAX).to(torch.float32)
    w_scaled = (w_real / new_scale).clamp(-FP8_MAX, FP8_MAX)
    w_fp8 = w_scaled.to(torch.float8_e4m3fn)
    return w_fp8, new_scale


def compute_roundtrip_error(w_original_f32, w_roundtrip_f32):
    """
    Compute error statistics between original and roundtripped float32 tensors.

    Args:
        w_original_f32: float32 tensor before modification
        w_roundtrip_f32: float32 tensor after modify → requantize → dequantize

    Returns:
        dict with max_abs_error, mean_abs_error, relative_error_pct, shape, numel
    """
    import torch
    with torch.no_grad():
        diff = (w_roundtrip_f32 - w_original_f32).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        orig_norm = w_original_f32.abs().mean().item()
        rel_err_pct = (mean_err / orig_norm * 100.0) if orig_norm > 1e-12 else float("inf")
    return {
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "relative_error_pct": round(rel_err_pct, 6),
        "shape": list(w_original_f32.shape),
        "numel": w_original_f32.numel(),
    }


# ---------------------------------------------------------------------------
# Weight map helpers
# ---------------------------------------------------------------------------

def load_weight_map(model_path: Path) -> dict:
    """
    Load the weight map from model.safetensors.index.json.

    Returns:
        dict mapping tensor_key -> shard_filename (relative to model_path)
    """
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"ERROR: Index file not found: {index_path}", file=sys.stderr)
        sys.exit(1)
    with open(index_path) as f:
        index = json.load(f)
    return index.get("weight_map", {})


def find_shard(weight_map: dict, tensor_key: str) -> str:
    """
    Return the shard filename (relative) for a given tensor key.

    Raises SystemExit if the key is not in the weight map.
    """
    if tensor_key not in weight_map:
        print(f"ERROR: Tensor key not found in weight map: {tensor_key}", file=sys.stderr)
        print(f"  Available keys matching 'experts.127': "
              f"{[k for k in weight_map if 'experts.127' in k][:5]}", file=sys.stderr)
        sys.exit(1)
    return weight_map[tensor_key]


# ---------------------------------------------------------------------------
# Shard read / write
# ---------------------------------------------------------------------------

def load_shard_tensors(shard_path: Path, keys: list) -> dict:
    """
    Load specific tensors from a safetensors shard.

    Args:
        shard_path: absolute path to the .safetensors file
        keys: list of tensor keys to load

    Returns:
        dict mapping key -> torch.Tensor (on CPU, original dtype)
    """
    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: safetensors not installed.", file=sys.stderr)
        sys.exit(1)

    tensors = {}
    with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
        available = set(sf.keys())
        for key in keys:
            if key not in available:
                print(f"ERROR: Key {key!r} not found in shard {shard_path.name}", file=sys.stderr)
                sys.exit(1)
            tensors[key] = sf.get_tensor(key)
    return tensors


def rewrite_shard(shard_path: Path, modifications: dict) -> None:
    """
    Rewrite a safetensors shard with specific tensors replaced.

    Loads all tensors from the shard, applies the modifications dict
    (key -> new tensor), then saves back to the same path.

    Args:
        shard_path: path to the shard file (will be overwritten)
        modifications: dict of key -> torch.Tensor to replace in the shard
    """
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        print("ERROR: safetensors not installed.", file=sys.stderr)
        sys.exit(1)

    # Load ALL tensors from the shard first
    all_tensors = {}
    with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
        for key in sf.keys():
            all_tensors[key] = sf.get_tensor(key)

    # Apply modifications
    for key, tensor in modifications.items():
        if key not in all_tensors:
            print(f"WARNING: Modification key {key!r} not found in shard — skipping",
                  file=sys.stderr)
            continue
        if tensor.shape != all_tensors[key].shape:
            print(f"ERROR: Shape mismatch for {key!r}: "
                  f"original {all_tensors[key].shape} vs new {tensor.shape}", file=sys.stderr)
            sys.exit(1)
        all_tensors[key] = tensor

    # Write back
    save_file(all_tensors, str(shard_path))


# ---------------------------------------------------------------------------
# Backup / restore
# ---------------------------------------------------------------------------

def make_backup(shard_path: Path) -> Path:
    """
    Copy shard to {shard_path}.bak before any modification.

    If the .bak file already exists, it is NOT overwritten — that preserves
    the original pre-any-experiment state.

    Returns:
        Path to the backup file
    """
    bak_path = shard_path.with_suffix(shard_path.suffix + ".bak")
    if bak_path.exists():
        print(f"  Backup already exists (preserving original): {bak_path.name}")
    else:
        shutil.copy2(str(shard_path), str(bak_path))
        print(f"  Backup created: {bak_path.name}")
    return bak_path


def restore_from_backup(shard_path: Path) -> None:
    """
    Restore shard from its .bak file, then delete the backup.

    Raises SystemExit if the backup does not exist.
    """
    bak_path = shard_path.with_suffix(shard_path.suffix + ".bak")
    if not bak_path.exists():
        print(f"ERROR: No backup found at {bak_path}", file=sys.stderr)
        print("  Nothing to restore.", file=sys.stderr)
        sys.exit(1)
    shutil.copy2(str(bak_path), str(shard_path))
    bak_path.unlink()
    print(f"  Restored {shard_path.name} from backup and removed .bak file.")


# ---------------------------------------------------------------------------
# Main roundtrip test
# ---------------------------------------------------------------------------

def run_roundtrip_test(model_path: Path, dry_run: bool) -> dict:
    """
    Execute the full FP8 roundtrip validation pipeline.

    Steps:
        1. Read weight map to find shard
        2. Load tensor + weight_scale
        3. Dequantize to float32
        4. Apply tiny modification (× 1.001)
        5. Requantize back to FP8
        6. Compute roundtrip error vs original
        7. If not dry_run: backup shard, write modified tensors

    Returns:
        dict with all statistics and metadata (JSON-serializable)
    """
    import torch

    t_start = time.time()

    print(f"Target tensor: {TARGET_TENSOR}")
    print(f"Model path:    {model_path}")
    print(f"Dry run:       {dry_run}")
    print()

    # Step 1: find shard
    weight_map = load_weight_map(model_path)
    shard_name = find_shard(weight_map, TARGET_TENSOR)
    # weight_scale may be in same or different shard
    scale_shard_name = find_shard(weight_map, TARGET_SCALE)
    shard_path = model_path / shard_name
    scale_shard_path = model_path / scale_shard_name

    print(f"Weight shard:  {shard_name}")
    print(f"Scale shard:   {scale_shard_name}")
    print()

    if not shard_path.exists():
        print(f"ERROR: Shard file not found: {shard_path}", file=sys.stderr)
        sys.exit(1)

    # Step 2: load tensor and scale
    print("Loading tensor and weight_scale...")
    weight_tensors = load_shard_tensors(shard_path, [TARGET_TENSOR])
    w_fp8 = weight_tensors[TARGET_TENSOR]

    if shard_name == scale_shard_name:
        scale_tensors = load_shard_tensors(shard_path, [TARGET_SCALE])
    else:
        scale_tensors = load_shard_tensors(scale_shard_path, [TARGET_SCALE])
    scale = scale_tensors[TARGET_SCALE]

    print(f"  Weight shape:  {list(w_fp8.shape)}")
    print(f"  Weight dtype:  {w_fp8.dtype}")
    print(f"  Scale value:   {scale.item():.8f}")
    print(f"  Numel:         {w_fp8.numel():,}")
    print()

    # Step 3: dequantize
    print("Dequantizing to float32...")
    w_real = dequantize_fp8(w_fp8, scale)

    original_stats = {
        "mean": w_real.mean().item(),
        "std": w_real.std().item(),
        "abs_max": w_real.abs().max().item(),
        "l2_norm": w_real.pow(2).sum().sqrt().item(),
        "scale_original": scale.item(),
    }
    print(f"  mean={original_stats['mean']:.6f}  std={original_stats['std']:.6f}  "
          f"abs_max={original_stats['abs_max']:.6f}")

    # Step 4: apply modification
    print(f"\nApplying modification: w *= {MODIFICATION_FACTOR}")
    w_modified = w_real * MODIFICATION_FACTOR

    modified_stats = {
        "mean": w_modified.mean().item(),
        "std": w_modified.std().item(),
        "abs_max": w_modified.abs().max().item(),
        "l2_norm": w_modified.pow(2).sum().sqrt().item(),
    }
    print(f"  mean={modified_stats['mean']:.6f}  std={modified_stats['std']:.6f}  "
          f"abs_max={modified_stats['abs_max']:.6f}")

    # Step 5: requantize
    print("\nRequantizing to FP8...")
    w_fp8_new, new_scale = requantize_fp8(w_modified)

    print(f"  Old scale: {scale.item():.8f}")
    print(f"  New scale: {new_scale.item():.8f}")
    print(f"  Scale ratio (new/old): {new_scale.item() / scale.item():.8f}")

    # Step 6: compute roundtrip error (dequantize the new fp8 and compare to w_modified)
    print("\nComputing roundtrip error...")
    w_roundtrip = dequantize_fp8(w_fp8_new, new_scale)
    error_stats = compute_roundtrip_error(w_modified, w_roundtrip)

    print(f"  max_abs_error:      {error_stats['max_abs_error']:.2e}")
    print(f"  mean_abs_error:     {error_stats['mean_abs_error']:.2e}")
    print(f"  relative_error_pct: {error_stats['relative_error_pct']:.4f}%")

    # Also compute error vs original (unmodified) for reference
    error_vs_original = compute_roundtrip_error(w_real, w_roundtrip)

    result = {
        "target_tensor": TARGET_TENSOR,
        "target_scale": TARGET_SCALE,
        "shard_name": shard_name,
        "scale_shard_name": scale_shard_name,
        "modification_factor": MODIFICATION_FACTOR,
        "dry_run": dry_run,
        "original_stats": original_stats,
        "modified_stats": modified_stats,
        "scale_change": {
            "old_scale": scale.item(),
            "new_scale": new_scale.item(),
            "ratio": new_scale.item() / scale.item(),
        },
        "roundtrip_error_vs_modified": error_stats,
        "roundtrip_error_vs_original": error_vs_original,
        "shape": list(w_fp8.shape),
        "dtype_original": str(w_fp8.dtype),
        "elapsed_seconds": round(time.time() - t_start, 2),
    }

    # Step 7: write back (unless dry run)
    if dry_run:
        print("\nDRY RUN — skipping write.")
        result["written"] = False
    else:
        print(f"\nBacking up shard: {shard_name}")
        make_backup(shard_path)

        print(f"Writing modified tensors to shard...")
        modifications = {
            TARGET_TENSOR: w_fp8_new,
        }
        # Preserve original scale tensor shape (scalar [] or 1-element [1])
        new_scale_out = new_scale.reshape(scale.shape)
        if shard_name == scale_shard_name:
            # Both weight and scale are in the same shard — one write
            modifications[TARGET_SCALE] = new_scale_out
            rewrite_shard(shard_path, modifications)
        else:
            # Weight and scale are in different shards — write each separately
            rewrite_shard(shard_path, {TARGET_TENSOR: w_fp8_new})
            rewrite_shard(scale_shard_path, {TARGET_SCALE: new_scale_out})

        result["written"] = True
        result["backup_path"] = str(shard_path.with_suffix(shard_path.suffix + ".bak"))
        print(f"  Written successfully.")

    result["elapsed_seconds"] = round(time.time() - t_start, 2)
    return result


# ---------------------------------------------------------------------------
# Restore mode
# ---------------------------------------------------------------------------

def run_restore(model_path: Path) -> None:
    """
    Restore the modified shard from its backup file.

    Reads the weight map to find the shard, then calls restore_from_backup.
    """
    weight_map = load_weight_map(model_path)
    shard_name = find_shard(weight_map, TARGET_TENSOR)
    shard_path = model_path / shard_name

    print(f"Restoring shard: {shard_name}")
    restore_from_backup(shard_path)

    # If scale was in a different shard, restore that too
    scale_shard_name = find_shard(weight_map, TARGET_SCALE)
    if scale_shard_name != shard_name:
        scale_shard_path = model_path / scale_shard_name
        bak = scale_shard_path.with_suffix(scale_shard_path.suffix + ".bak")
        if bak.exists():
            print(f"Restoring scale shard: {scale_shard_name}")
            restore_from_backup(scale_shard_path)

    print("\nRestore complete. Restart vLLM to load the original weights.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the FP8 modification pipeline on a single non-critical tensor. "
            "Tests dequantize → modify → requantize → write-back safety."
        )
    )
    parser.add_argument(
        "--model-path",
        default="/workspace/model",
        help="Path to model directory with safetensors shards "
             "(default: /workspace/model)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute roundtrip error statistics but do not write anything to disk.",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore the shard from backup (.bak) created by a previous run.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path.is_dir():
        print(f"ERROR: Model path is not a directory: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Import check
    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: torch not installed.", file=sys.stderr)
        sys.exit(1)
    try:
        from safetensors import safe_open  # noqa: F401
        from safetensors.torch import save_file  # noqa: F401
    except ImportError:
        print("ERROR: safetensors not installed.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("FP8 Roundtrip Validation — Neuroplastic Phase 2")
    print("=" * 60)
    print()

    if args.restore:
        run_restore(model_path)
        return

    result = run_roundtrip_test(model_path, dry_run=args.dry_run)

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print()

    if result.get("written"):
        print("MODIFIED — restart vLLM and verify model serves correctly.")
        print(f"  Backup at: {result['backup_path']}")
        print(f"  To restore: python {Path(__file__).name} "
              f"--model-path {args.model_path} --restore")
    elif args.dry_run:
        print("DRY RUN complete — no changes written.")
    print()


if __name__ == "__main__":
    main()
