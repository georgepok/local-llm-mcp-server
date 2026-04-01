"""
modify_tensor.py — General-purpose FP8-aware tensor modification for Nemotron safetensors.

Supports both FP8-quantized tensors (float8_e4m3fn + weight_scale) and native-dtype
tensors (BF16, float32). Auto-detects quantization by checking for a companion
weight_scale entry in the weight map.

FP8 Format:
  - {name}.weight:        torch.float8_e4m3fn, values in [-448, 448]
  - {name}.weight_scale:  torch.float32 scalar — per-tensor scale
  - {name}.input_scale:   torch.float32 scalar — activation scale, NEVER modified
  - Dequant: w_real = w_fp8.float() * weight_scale
  - Requant: scale = abs_max / 448.0; w_fp8 = (w_real / scale).clamp(-448, 448).to(fp8)

Protected tensors (require --force to modify):
  - backbone.embeddings.*
  - lm_head.*
  - backbone.norm_f.*

Usage:
    python modify_tensor.py \\
        --model-path /workspace/model \\
        --tensor "backbone.layers.45.mixer.gate.weight" \\
        --modification "t * 1.01" \\
        --backup /workspace/checkpoints/exp001_backup \\
        --output modification_log.json

    # Dry run (compute stats, no write):
    python modify_tensor.py \\
        --model-path /workspace/model \\
        --tensor "backbone.layers.45.mixer.experts.0.up_proj.weight" \\
        --modification "t * 0.99" \\
        --dry-run

    # Force modify a protected tensor:
    python modify_tensor.py \\
        --model-path /workspace/model \\
        --tensor "lm_head.weight" \\
        --modification "t * 1.001" \\
        --force
"""

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FP8_MAX = 448.0

# Tensors that require --force to modify (high-risk: shape the entire output distribution)
PROTECTED_PREFIXES = (
    "backbone.embeddings",
    "lm_head",
    "backbone.norm_f",
)


# ---------------------------------------------------------------------------
# FP8 utilities
# ---------------------------------------------------------------------------

def dequantize_fp8(w_fp8, scale):
    """
    Dequantize FP8 tensor to float32.

    Args:
        w_fp8: torch.Tensor of dtype torch.float8_e4m3fn
        scale: torch.Tensor scalar float32 (per-tensor scale factor)

    Returns:
        torch.Tensor float32
    """
    import torch
    return w_fp8.to(torch.float32) * scale.to(torch.float32)


def requantize_fp8(w_real):
    """
    Requantize a float32 tensor to FP8 with a fresh per-tensor scale.

    Args:
        w_real: torch.Tensor float32

    Returns:
        (w_fp8, new_scale): torch.float8_e4m3fn tensor and float32 scalar scale
    """
    import torch
    abs_max = w_real.abs().max()
    if abs_max < 1e-12:
        new_scale = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
    else:
        new_scale = (abs_max / FP8_MAX).to(torch.float32)
    w_scaled = (w_real / new_scale).clamp(-FP8_MAX, FP8_MAX)
    w_fp8 = w_scaled.to(torch.float8_e4m3fn)
    return w_fp8, new_scale


# ---------------------------------------------------------------------------
# Tensor statistics
# ---------------------------------------------------------------------------

def tensor_stats(t) -> dict:
    """
    Compute basic statistics for a float32 tensor.

    Args:
        t: torch.Tensor (any dtype, will be cast to float32)

    Returns:
        dict with mean, std, norm, abs_max, min, max, numel, shape
    """
    import torch
    with torch.no_grad():
        f = t.to(torch.float32).reshape(-1)
        return {
            "mean": f.mean().item(),
            "std": f.std().item(),
            "norm": f.pow(2).sum().sqrt().item(),
            "abs_max": f.abs().max().item(),
            "min": f.min().item(),
            "max": f.max().item(),
            "numel": f.numel(),
            "shape": list(t.shape),
        }


# ---------------------------------------------------------------------------
# Expression evaluation (restricted namespace)
# ---------------------------------------------------------------------------

def evaluate_modification(expression: str, tensor):
    """
    Evaluate a modification expression with `t` bound to the tensor.

    Only `torch` and `math` are available in the expression namespace.
    The expression must return a tensor of the same shape.

    Args:
        expression: Python expression string, e.g. "t * 1.01" or "t + 0.001"
        tensor: torch.Tensor (float32) bound as `t`

    Returns:
        Modified torch.Tensor

    Raises:
        ValueError: if the expression is invalid or returns wrong type/shape
    """
    import torch

    namespace = {
        "__builtins__": {},  # no builtins — block file I/O, exec, etc.
        "torch": torch,
        "math": math,
        "t": tensor,
    }

    try:
        result = eval(expression, namespace)  # noqa: S307
    except Exception as e:
        raise ValueError(f"Expression evaluation failed: {e}") from e

    if not hasattr(result, "shape"):
        raise ValueError(
            f"Expression must return a tensor, got {type(result).__name__}"
        )
    if result.shape != tensor.shape:
        raise ValueError(
            f"Expression changed tensor shape: {tensor.shape} → {result.shape}"
        )
    return result


# ---------------------------------------------------------------------------
# Weight map helpers
# ---------------------------------------------------------------------------

def load_weight_map(model_path: Path) -> dict:
    """
    Load weight map from model.safetensors.index.json.

    Returns:
        dict mapping tensor_key -> shard_filename
    """
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"ERROR: Index file not found: {index_path}", file=sys.stderr)
        sys.exit(1)
    with open(index_path) as f:
        index = json.load(f)
    return index.get("weight_map", {})


def find_scale_key(weight_map: dict, tensor_key: str) -> str | None:
    """
    Find the weight_scale companion key for a tensor, if it exists.

    Convention: scale key is exactly {tensor_key_without_weight} + .weight_scale
    e.g. "backbone.layers.45.mixer.experts.0.up_proj.weight"
         -> "backbone.layers.45.mixer.experts.0.up_proj.weight_scale"

    Returns None if no scale exists (native dtype tensor).
    """
    # Replace trailing .weight with .weight_scale
    if tensor_key.endswith(".weight"):
        candidate = tensor_key[:-len(".weight")] + ".weight_scale"
        if candidate in weight_map:
            return candidate

    # Also check direct .weight_scale suffix for keys that already have it
    candidate2 = tensor_key + "_scale"
    if candidate2 in weight_map:
        return candidate2

    return None


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------

def load_tensors_from_shard(shard_path: Path, keys: list) -> dict:
    """
    Load specific tensor keys from a safetensors shard.

    Returns dict of key -> torch.Tensor (CPU, original dtype).
    """
    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: safetensors not installed.", file=sys.stderr)
        sys.exit(1)

    result = {}
    with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
        available = set(sf.keys())
        for key in keys:
            if key not in available:
                print(f"ERROR: Key {key!r} not in shard {shard_path.name}", file=sys.stderr)
                sys.exit(1)
            result[key] = sf.get_tensor(key)
    return result


def rewrite_shard(shard_path: Path, modifications: dict) -> None:
    """
    Rewrite a safetensors shard with specific tensors replaced.

    Loads all tensors, applies modifications, saves back to same path.

    Args:
        shard_path: path to the .safetensors file (will be overwritten in-place)
        modifications: dict of key -> new torch.Tensor
    """
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        print("ERROR: safetensors not installed.", file=sys.stderr)
        sys.exit(1)

    # Load everything
    all_tensors = {}
    with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
        for key in sf.keys():
            all_tensors[key] = sf.get_tensor(key)

    # Apply modifications with shape safety check
    for key, new_tensor in modifications.items():
        if key not in all_tensors:
            print(f"WARNING: {key!r} not found in shard — skipping modification",
                  file=sys.stderr)
            continue
        if new_tensor.shape != all_tensors[key].shape:
            print(
                f"ERROR: Shape mismatch for {key!r}: "
                f"original {all_tensors[key].shape} vs new {new_tensor.shape}",
                file=sys.stderr,
            )
            sys.exit(1)
        all_tensors[key] = new_tensor

    save_file(all_tensors, str(shard_path))


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

def backup_shard(shard_path: Path, backup_dir: Path | None) -> Path:
    """
    Save a backup of a shard file.

    If backup_dir is provided, copies to backup_dir/{shard_name}.
    Otherwise copies to {shard_path}.bak alongside the original.
    Existing backups are NOT overwritten (preserves first-ever state).

    Returns:
        Path to the backup file.
    """
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        dest = backup_dir / shard_path.name
    else:
        dest = shard_path.with_suffix(shard_path.suffix + ".bak")

    if dest.exists():
        print(f"  Backup already exists (not overwriting): {dest}")
    else:
        shutil.copy2(str(shard_path), str(dest))
        print(f"  Backup written: {dest}")

    return dest


# ---------------------------------------------------------------------------
# Core modification function
# ---------------------------------------------------------------------------

def modify_tensor(
    model_path: str,
    tensor_key: str,
    modification: str,
    backup_dir: str = None,
    dry_run: bool = False,
) -> dict:
    """
    Modify a single tensor in the Nemotron safetensors model.

    Auto-detects whether the tensor is FP8 (has a companion weight_scale) or
    native dtype (BF16/FP32). Applies the modification expression, handles
    requantization for FP8 tensors, backs up the affected shard(s), and writes
    back to disk.

    The modification expression is evaluated with `t` as the float32 tensor
    variable. Only `torch` and `math` are in scope — no builtins.

    Args:
        model_path: path to model directory with safetensors files
        tensor_key: dot-separated weight key, e.g.
                    "backbone.layers.45.mixer.gate.weight"
        modification: Python expression applied to tensor t, e.g.
                      "t * 1.01" or "t + 0.001" or "torch.clamp(t, -0.5, 0.5)"
        backup_dir: directory to save shard backup (default: {shard}.bak)
        dry_run: if True, compute stats but do not write to disk

    Returns:
        JSON-serializable dict with:
          - tensor_key, shard_name, is_fp8, dry_run
          - original_stats: {mean, std, norm, abs_max, min, max, numel, shape}
          - modified_stats: same structure
          - scale_change: {old_scale, new_scale, ratio} (FP8 only)
          - roundtrip_error: {max_abs_error, mean_abs_error, relative_error_pct}
          - modification_expression: the expression string
          - written: bool
          - elapsed_seconds: float

    Raises:
        SystemExit on configuration or I/O errors.
        ValueError on invalid modification expression or shape change.
    """
    import torch

    t_start = time.time()
    model_path_ = Path(model_path)
    backup_dir_ = Path(backup_dir) if backup_dir else None

    # --- Safety checks ---

    # Check key exists
    weight_map = load_weight_map(model_path_)
    if tensor_key not in weight_map:
        print(f"ERROR: Tensor key not found in weight map: {tensor_key}", file=sys.stderr)
        similar = [k for k in weight_map if tensor_key.split(".")[-2] in k][:5]
        if similar:
            print(f"  Similar keys: {similar}", file=sys.stderr)
        sys.exit(1)

    # Find shard and verify it exists + is writable
    shard_name = weight_map[tensor_key]
    shard_path = model_path_ / shard_name

    if not shard_path.exists():
        print(f"ERROR: Shard file not found: {shard_path}", file=sys.stderr)
        sys.exit(1)
    if not dry_run and not shard_path.stat().st_mode & 0o200:
        print(f"ERROR: Shard file is not writable: {shard_path}", file=sys.stderr)
        sys.exit(1)

    # Detect FP8: does a weight_scale companion exist?
    scale_key = find_scale_key(weight_map, tensor_key)
    is_fp8 = scale_key is not None

    log: dict[str, Any] = {
        "tensor_key": tensor_key,
        "shard_name": shard_name,
        "is_fp8": is_fp8,
        "scale_key": scale_key,
        "modification_expression": modification,
        "dry_run": dry_run,
        "model_path": str(model_path_),
    }

    print(f"Tensor:      {tensor_key}")
    print(f"Shard:       {shard_name}")
    print(f"Is FP8:      {is_fp8}")
    if is_fp8:
        print(f"Scale key:   {scale_key}")
    print(f"Expression:  {modification}")
    print(f"Dry run:     {dry_run}")
    print()

    # --- Load ---
    keys_to_load = [tensor_key]
    if is_fp8 and scale_key:
        scale_shard_name = weight_map[scale_key]
        # Load scale from its own shard if needed
        if scale_shard_name != shard_name:
            scale_tensors = load_tensors_from_shard(model_path_ / scale_shard_name, [scale_key])
        else:
            keys_to_load.append(scale_key)
            scale_tensors = None  # loaded together below

    raw_tensors = load_tensors_from_shard(shard_path, keys_to_load)
    w_raw = raw_tensors[tensor_key]

    if is_fp8 and scale_key:
        if scale_tensors is not None:
            scale = scale_tensors[scale_key]
        else:
            scale = raw_tensors[scale_key]

    print(f"Loaded: shape={list(w_raw.shape)}, dtype={w_raw.dtype}")

    # --- Convert to float32 for modification ---
    if is_fp8:
        w_f32 = dequantize_fp8(w_raw, scale)
        print(f"Dequantized: scale={scale.item():.8f}")
    else:
        w_f32 = w_raw.to(torch.float32)
        print(f"Loaded as float32 (original dtype: {w_raw.dtype})")

    # --- Compute original stats ---
    orig_stats = tensor_stats(w_f32)
    log["original_stats"] = orig_stats
    print(f"\nOriginal: mean={orig_stats['mean']:.6f}  std={orig_stats['std']:.6f}  "
          f"norm={orig_stats['norm']:.4f}  abs_max={orig_stats['abs_max']:.6f}")

    # --- Apply modification ---
    print(f"\nApplying: {modification}")
    w_modified = evaluate_modification(modification, w_f32)

    mod_stats = tensor_stats(w_modified)
    log["modified_stats"] = mod_stats
    print(f"Modified: mean={mod_stats['mean']:.6f}  std={mod_stats['std']:.6f}  "
          f"norm={mod_stats['norm']:.4f}  abs_max={mod_stats['abs_max']:.6f}")

    # --- Requantize (FP8 only) ---
    if is_fp8:
        w_out, new_scale = requantize_fp8(w_modified)
        w_out_dtype = w_out.dtype

        # Roundtrip error
        w_roundtrip = dequantize_fp8(w_out, new_scale)
        diff = (w_roundtrip - w_modified).abs()
        orig_abs_mean = w_modified.abs().mean().item()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        rel_err = (mean_err / orig_abs_mean * 100.0) if orig_abs_mean > 1e-12 else float("inf")

        scale_change = {
            "old_scale": scale.item(),
            "new_scale": new_scale.item(),
            "ratio": new_scale.item() / scale.item() if scale.item() != 0 else float("inf"),
        }
        roundtrip_error = {
            "max_abs_error": max_err,
            "mean_abs_error": mean_err,
            "relative_error_pct": round(rel_err, 6),
        }
        log["scale_change"] = scale_change
        log["roundtrip_error"] = roundtrip_error

        print(f"\nRequantized: old_scale={scale_change['old_scale']:.8f}  "
              f"new_scale={scale_change['new_scale']:.8f}  "
              f"ratio={scale_change['ratio']:.8f}")
        print(f"Roundtrip:   max_err={max_err:.2e}  mean_err={mean_err:.2e}  "
              f"rel_err={rel_err:.4f}%")
    else:
        # Native dtype: convert back to original dtype
        w_out = w_modified.to(w_raw.dtype)
        w_out_dtype = w_out.dtype
        log["scale_change"] = None
        log["roundtrip_error"] = None
        print(f"\nNative dtype modification: converted back to {w_out_dtype}")

    # --- Shape verification ---
    if w_out.shape != w_raw.shape:
        print(
            f"ERROR: Output shape {w_out.shape} != input shape {w_raw.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Write ---
    if dry_run:
        print("\nDRY RUN — no changes written.")
        log["written"] = False
        log["backup_path"] = None
    else:
        print(f"\nBacking up shard: {shard_name}")
        bak = backup_shard(shard_path, backup_dir_)
        log["backup_path"] = str(bak)

        # Determine what to write to which shard(s)
        if is_fp8 and scale_key:
            scale_shard_name = weight_map[scale_key]
            # Preserve original scale tensor shape (scalar [] or 1-element [1])
            new_scale_out = new_scale.reshape(scale.shape)

            if scale_shard_name == shard_name:
                print(f"Writing weight + scale to: {shard_name}")
                rewrite_shard(shard_path, {
                    tensor_key: w_out,
                    scale_key: new_scale_out,
                })
            else:
                print(f"Writing weight to: {shard_name}")
                rewrite_shard(shard_path, {tensor_key: w_out})
                scale_shard_path = model_path_ / scale_shard_name
                if not scale_shard_path.exists():
                    print(f"ERROR: Scale shard not found: {scale_shard_path}", file=sys.stderr)
                    sys.exit(1)
                print(f"Writing scale to: {scale_shard_name}")
                bak_scale = backup_shard(scale_shard_path, backup_dir_)
                log["backup_scale_path"] = str(bak_scale)
                rewrite_shard(scale_shard_path, {scale_key: new_scale_out})
        else:
            print(f"Writing modified tensor to: {shard_name}")
            rewrite_shard(shard_path, {tensor_key: w_out})

        log["written"] = True
        print("Written successfully.")

    log["elapsed_seconds"] = round(time.time() - t_start, 2)
    return log


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "General-purpose FP8-aware tensor modification for Nemotron safetensors. "
            "Auto-detects FP8 vs native dtype. Backs up affected shards before writing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scale a gate weight (native BF16) up by 1%:
  python modify_tensor.py \\
      --model-path /workspace/model \\
      --tensor "backbone.layers.45.mixer.gate.weight" \\
      --modification "t * 1.01"

  # Dampen expert weights (FP8) by 0.1%:
  python modify_tensor.py \\
      --model-path /workspace/model \\
      --tensor "backbone.layers.45.mixer.experts.0.up_proj.weight" \\
      --modification "t * 0.999" \\
      --backup /workspace/checkpoints/exp001_backup \\
      --output modification_log.json

  # Dry run — compute stats without writing:
  python modify_tensor.py \\
      --model-path /workspace/model \\
      --tensor "backbone.layers.5.mixer.q_proj.weight" \\
      --modification "t * 1.001" \\
      --dry-run

  # Force modify a protected tensor (use with caution):
  python modify_tensor.py \\
      --model-path /workspace/model \\
      --tensor "lm_head.weight" \\
      --modification "t * 1.001" \\
      --force
""",
    )
    parser.add_argument(
        "--model-path",
        default="/workspace/model",
        help="Path to model directory with safetensors files "
             "(default: /workspace/model)",
    )
    parser.add_argument(
        "--tensor",
        required=True,
        help="Dot-separated tensor key, e.g. "
             "'backbone.layers.45.mixer.gate.weight'",
    )
    parser.add_argument(
        "--modification",
        required=True,
        help="Python expression applied to tensor t (float32), e.g. "
             "'t * 1.01' or 't + 0.001'. "
             "Only torch and math are in scope.",
    )
    parser.add_argument(
        "--backup",
        default=None,
        metavar="DIR",
        help="Directory to save shard backup before modification "
             "(default: writes {shard}.bak alongside the shard)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write modification log JSON to this file "
             "(default: print to stdout only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats and verify expression but do not write to disk.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow modification of protected tensors (embeddings, lm_head, norm_f).",
    )
    args = parser.parse_args()

    # Protected tensor check
    tensor_key = args.tensor
    if not args.force:
        for prefix in PROTECTED_PREFIXES:
            if tensor_key.startswith(prefix):
                print(
                    f"ERROR: {tensor_key!r} is a protected tensor "
                    f"(matches prefix {prefix!r}).\n"
                    f"  Use --force to override this safety check.",
                    file=sys.stderr,
                )
                sys.exit(1)

    # Import checks
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
    print("Tensor Modification — Neuroplastic Phase 2")
    print("=" * 60)
    print()

    log = modify_tensor(
        model_path=args.model_path,
        tensor_key=tensor_key,
        modification=args.modification,
        backup_dir=args.backup,
        dry_run=args.dry_run,
    )

    # Output log
    print()
    print("=" * 60)
    print("MODIFICATION LOG")
    print("=" * 60)

    log_json = json.dumps(log, indent=2, default=_json_default)
    print(log_json)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(log_json)
            f.write("\n")
        print(f"\nLog written to: {output_path}")

    if log.get("written"):
        print(
            f"\nMODIFIED — restart vLLM and verify model serves correctly.\n"
            f"  Tensor:  {tensor_key}\n"
            f"  Expr:    {args.modification}\n"
            f"  Backup:  {log.get('backup_path')}"
        )
    elif args.dry_run:
        print("\nDRY RUN complete — no changes written.")

    print()


def _json_default(obj):
    """JSON fallback for non-standard types."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


if __name__ == "__main__":
    main()
