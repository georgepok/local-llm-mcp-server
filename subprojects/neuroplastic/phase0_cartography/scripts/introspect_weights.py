"""
introspect_weights.py — Compute per-layer weight statistics for Nemotron-3-Nano-30B-A3B-FP8.

Loads safetensors files lazily using safe_open (one tensor at a time) and computes
statistical baselines for each weight tensor. Output is weight_baseline.json.

Memory model:
  - Never hold more than one large tensor in memory at a time
  - FP8 tensors are converted to float32 on load for statistics
  - Scale tensors (weight_scale, input_scale) are loaded alongside the weight for dequant
  - Singular value decomposition only on small projections (Q/K/V/gate), skipped for experts

Usage:
    python introspect_weights.py
    python introspect_weights.py --model-path /workspace/model
    python introspect_weights.py --model-path /workspace/model --output weight_baseline.json
    python introspect_weights.py --model-path /workspace/model --layers-only 5,12,19
    python introspect_weights.py --model-path /workspace/model --dry-run
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Layer classification
# ---------------------------------------------------------------------------

# Verified indices for Nemotron-3-Nano-30B hybrid pattern:
# MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
MAMBA_INDICES = frozenset([0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25,
                            28, 30, 32, 35, 37, 39, 41, 44, 46, 48, 50])
ATTENTION_INDICES = frozenset([5, 12, 19, 26, 33, 42])
MOE_INDICES = frozenset([1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27,
                          29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51])


def classify_key(key: str) -> dict:
    """
    Classify a weight key into layer type, layer index, component, and role.

    Returns a dict with: layer_idx, layer_type, component, role, is_scale
    """
    parts = key.split(".")
    info: dict[str, Any] = {
        "layer_idx": None,
        "layer_type": "global",
        "component": None,
        "role": None,
        "is_scale": False,
        "is_fp8": False,
    }

    # Mark scale tensors
    if key.endswith(".weight_scale") or key.endswith(".input_scale"):
        info["is_scale"] = True

    # Global tensors
    if key.startswith("backbone.embeddings"):
        info["layer_type"] = "embedding"
        info["component"] = "embeddings"
        return info
    if key.startswith("backbone.norm_f"):
        info["layer_type"] = "final_norm"
        info["component"] = "norm_f"
        return info
    if key.startswith("lm_head"):
        info["layer_type"] = "lm_head"
        info["component"] = "lm_head"
        return info

    # Per-layer tensors: backbone.layers.{N}.*
    if not key.startswith("backbone.layers."):
        info["layer_type"] = "unknown"
        return info

    try:
        layer_idx = int(parts[2])
    except (IndexError, ValueError):
        info["layer_type"] = "unknown"
        return info

    info["layer_idx"] = layer_idx

    if layer_idx in ATTENTION_INDICES:
        info["layer_type"] = "attention"
    elif layer_idx in MAMBA_INDICES:
        info["layer_type"] = "mamba"
    elif layer_idx in MOE_INDICES:
        info["layer_type"] = "moe"
    else:
        info["layer_type"] = "unknown"

    # Component parsing
    rest = ".".join(parts[3:])

    if rest == "norm.weight":
        info["component"] = "norm"
        info["role"] = "layer_norm"
    elif rest in ("A_log",):
        info["component"] = "A_log"
        info["role"] = "ssm_decay"
    elif rest in ("D",):
        info["component"] = "D"
        info["role"] = "ssm_skip"
    elif rest in ("dt_bias",):
        info["component"] = "dt_bias"
        info["role"] = "ssm_dt_offset"
    elif rest.startswith("conv1d"):
        info["component"] = "conv1d"
        info["role"] = rest.split(".")[-1]  # weight or bias
    elif rest.startswith("in_proj"):
        info["component"] = "in_proj"
        info["role"] = rest.split(".")[-1]
        info["is_fp8"] = "scale" in rest
    elif rest.startswith("out_proj"):
        info["component"] = "out_proj"
        info["role"] = rest.split(".")[-1]
        info["is_fp8"] = "scale" in rest
    elif rest.startswith("q_proj"):
        info["component"] = "q_proj"
        info["role"] = rest.split(".")[-1]
        info["is_fp8"] = "scale" in rest
    elif rest.startswith("k_proj"):
        info["component"] = "k_proj"
        info["role"] = rest.split(".")[-1]
        info["is_fp8"] = "scale" in rest
    elif rest.startswith("v_proj"):
        info["component"] = "v_proj"
        info["role"] = rest.split(".")[-1]
        info["is_fp8"] = "scale" in rest
    elif rest.startswith("o_proj"):
        info["component"] = "o_proj"
        info["role"] = rest.split(".")[-1]
        info["is_fp8"] = "scale" in rest
    elif rest.startswith("gate.weight"):
        info["component"] = "gate"
        info["role"] = "router"
    elif rest.startswith("experts."):
        expert_parts = rest.split(".")
        try:
            info["expert_idx"] = int(expert_parts[1])
        except (IndexError, ValueError):
            info["expert_idx"] = None
        info["component"] = "routed_expert"
        info["role"] = ".".join(expert_parts[2:]) if len(expert_parts) > 2 else "unknown"
        info["is_fp8"] = "scale" in rest
    elif rest.startswith("shared_experts."):
        info["component"] = "shared_expert"
        info["role"] = rest[len("shared_experts."):]
        info["is_fp8"] = "scale" in rest
    else:
        info["component"] = rest
        info["role"] = "unknown"

    return info


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_tensor_stats(t, compute_svd: bool = False, svd_top_k: int = 3) -> dict:
    """
    Compute statistics for a single tensor.

    Args:
        t: torch.Tensor (float32)
        compute_svd: whether to compute top-k singular values
        svd_top_k: number of singular values to compute

    Returns:
        dict of statistics
    """
    import torch  # type: ignore[import-not-found]

    numel = t.numel()
    if numel == 0:
        return {"error": "empty tensor", "shape": list(t.shape), "dtype": str(t.dtype)}

    # Flatten to 1D for scalar stats
    flat = t.reshape(-1).float()

    stats = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": numel,
    }

    # Basic statistics
    with torch.no_grad():
        mean_val = flat.mean().item()
        std_val = flat.std().item()
        min_val = flat.min().item()
        max_val = flat.max().item()

        # Norms
        l1 = flat.abs().sum().item()
        l2 = flat.pow(2).sum().sqrt().item()
        frob = l2  # same as L2 for flattened vector = Frobenius norm of matrix

        # Sparsity: fraction of values with |x| < 1e-6
        sparse_count = (flat.abs() < 1e-6).sum().item()
        sparsity = sparse_count / numel

        # Kurtosis: E[(x-mu)^4] / std^4 - 3 (excess kurtosis)
        if std_val > 1e-12:
            centered = flat - mean_val
            kurt = (centered.pow(4).mean().item() / (std_val ** 4)) - 3.0
        else:
            kurt = float("nan")

        stats.update({
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "l1_norm": l1,
            "l2_norm": l2,
            "frobenius_norm": frob,
            "kurtosis_excess": kurt,
            "sparsity_frac": sparsity,
            "sparsity_pct": round(sparsity * 100, 4),
        })

    # SVD on 2D projections
    if compute_svd and t.ndim >= 2:
        # Reshape to 2D: (rows, cols)
        rows = t.shape[0]
        cols = t.numel() // rows
        mat = t.reshape(rows, cols).float()

        # Only compute SVD if matrix is reasonably sized
        max_dim = max(rows, cols)
        if max_dim <= 16384:
            try:
                with torch.no_grad():
                    # Use torch.linalg.svdvals for efficiency (no U/V)
                    sv = torch.linalg.svdvals(mat)
                    top_k = min(svd_top_k, sv.numel())
                    stats["svd_top_singular_values"] = sv[:top_k].tolist()
                    stats["svd_condition_number"] = (sv[0] / sv[-1]).item() if sv[-1] > 1e-12 else float("inf")
                    stats["svd_rank_approx"] = int((sv > sv[0] * 1e-5).sum().item())
            except Exception as e:
                stats["svd_error"] = str(e)
        else:
            stats["svd_skipped"] = f"matrix too large ({rows}x{cols})"

    return stats


def load_tensor_as_float32(sf, key: str):
    """
    Load a tensor from a safe_open file handle, converting FP8 to float32.

    Returns (tensor_float32, original_dtype_str)
    """
    import torch  # type: ignore[import-not-found]

    t = sf.get_tensor(key)
    orig_dtype = str(t.dtype)

    # FP8 types: torch.float8_e4m3fn, torch.float8_e5m2, etc.
    dtype_str = str(t.dtype).lower()
    if "float8" in dtype_str or "f8" in dtype_str:
        t = t.to(torch.float32)

    return t, orig_dtype


# ---------------------------------------------------------------------------
# Expert aggregation
# ---------------------------------------------------------------------------

def aggregate_expert_stats(expert_stats: list[dict]) -> dict:
    """
    Aggregate per-expert statistics to find expert diversity.

    Computes: mean/std of norms across experts, min/max expert by norm.
    """
    if not expert_stats:
        return {}

    norms = [e.get("frobenius_norm", 0.0) for e in expert_stats]
    means = [e.get("mean", 0.0) for e in expert_stats]
    stds = [e.get("std", 1.0) for e in expert_stats]

    n = len(norms)
    norm_mean = sum(norms) / n
    norm_std = math.sqrt(sum((x - norm_mean) ** 2 for x in norms) / n) if n > 1 else 0.0
    norm_cv = norm_std / norm_mean if norm_mean > 1e-12 else 0.0

    min_idx = norms.index(min(norms))
    max_idx = norms.index(max(norms))

    return {
        "num_experts": n,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "norm_cv": norm_cv,
        "norm_min": min(norms),
        "norm_max": max(norms),
        "most_dormant_expert": min_idx,
        "most_active_expert": max_idx,
        "mean_of_means": sum(means) / n,
        "mean_of_stds": sum(stds) / n,
    }


# ---------------------------------------------------------------------------
# Main introspection loop
# ---------------------------------------------------------------------------

def get_shard_files(model_path: Path) -> list[Path]:
    """Get ordered list of safetensors shard files."""
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        return [model_path / name for name in shard_names]

    # Fall back to single file or glob
    single = model_path / "model.safetensors"
    if single.exists():
        return [single]

    shards = sorted(model_path.glob("*.safetensors"))
    return shards


def get_weight_map(model_path: Path) -> dict[str, str]:
    """Return dict mapping weight_key -> shard_filename."""
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with open(index_path) as f:
        index = json.load(f)
    return index.get("weight_map", {})


def introspect_model(
    model_path: Path,
    output_path: Path,
    layers_filter: set[int] | None = None,
    dry_run: bool = False,
    svd_top_k: int = 3,
) -> None:
    """
    Main introspection loop. Processes all tensors in the model.

    Args:
        model_path: path to model directory with safetensors files
        output_path: path to write weight_baseline.json
        layers_filter: if set, only process these layer indices
        dry_run: if True, classify keys but don't load tensors
        svd_top_k: number of singular values for SVD tensors
    """
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ImportError:
        print("ERROR: safetensors not installed. Run: pip install safetensors", file=sys.stderr)
        sys.exit(1)

    try:
        import torch  # type: ignore[import-not-found]  # noqa: F811
        _ = torch  # needed for safetensors framework="pt"
    except ImportError:
        print("ERROR: torch not installed.", file=sys.stderr)
        sys.exit(1)

    # Load weight map to know which shard contains each key
    weight_map = get_weight_map(model_path)
    shard_files = get_shard_files(model_path)

    if not shard_files:
        print(f"ERROR: No safetensors files found in {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(shard_files)} shard files")
    if weight_map:
        print(f"Total weight keys in index: {len(weight_map)}")

    # Group keys by shard for efficient sequential access
    shard_to_keys: dict[str, list[str]] = {}
    if weight_map:
        for key, shard_name in weight_map.items():
            if shard_name not in shard_to_keys:
                shard_to_keys[shard_name] = []
            shard_to_keys[shard_name].append(key)
    else:
        # Single file: all keys discovered on open
        shard_to_keys = {str(shard_files[0]): []}  # will be populated below

    # Result accumulator
    results: dict[str, Any] = {
        "model_path": str(model_path),
        "dry_run": dry_run,
        "layers_filter": sorted(layers_filter) if layers_filter else None,
        "layers": {},       # layer_idx -> component -> role -> stats
        "global": {},       # embedding, lm_head, final_norm
        "summary": {},      # aggregated per-layer stats
    }

    # Per-layer accumulators for expert aggregation
    # layer_idx -> component -> list of per-expert stats (one per expert)
    expert_accumulator: dict[int, dict[str, list]] = {}

    total_tensors = len(weight_map) if weight_map else 0
    processed = 0
    skipped_scale = 0
    skipped_filter = 0
    t_start = time.time()
    last_print = t_start

    print(f"\nStarting introspection of {total_tensors} tensors...")
    if dry_run:
        print("  DRY RUN: classifying keys only, not loading tensors")

    # Process shard by shard
    for shard_file in shard_files:
        shard_name = shard_file.name
        keys_in_shard = shard_to_keys.get(shard_name, [])

        if not keys_in_shard and not weight_map:
            # Single file: discover keys
            with safe_open(str(shard_file), framework="pt", device="cpu") as sf:
                keys_in_shard = list(sf.keys())

        if not keys_in_shard:
            continue

        print(f"\n  Shard: {shard_name} ({len(keys_in_shard)} tensors)")

        with safe_open(str(shard_file), framework="pt", device="cpu") as sf:
            for key in keys_in_shard:
                processed += 1

                # Progress reporting every 5 seconds
                now = time.time()
                if now - last_print >= 5.0:
                    elapsed = now - t_start
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total_tensors - processed) / rate if rate > 0 else 0
                    print(f"  Progress: {processed}/{total_tensors} tensors "
                          f"({100*processed/total_tensors:.1f}%) "
                          f"— {rate:.0f}/s — ETA {eta:.0f}s")
                    last_print = now

                # Classify the key
                info = classify_key(key)

                # Skip scale tensors (processed alongside their weight)
                if info["is_scale"]:
                    skipped_scale += 1
                    continue

                layer_idx = info["layer_idx"]
                layer_type = info["layer_type"]
                component = info["component"] or "unknown"
                role = info["role"] or "weight"

                # Apply layer filter
                if layers_filter is not None and layer_idx is not None:
                    if layer_idx not in layers_filter:
                        skipped_filter += 1
                        continue

                # Determine if SVD should be computed for this tensor
                compute_svd = component in ("q_proj", "k_proj", "v_proj", "o_proj", "gate")

                if dry_run:
                    tensor_stats = {
                        "key": key,
                        "layer_type": layer_type,
                        "component": component,
                        "role": role,
                        "dry_run": True,
                    }
                else:
                    # Load tensor
                    try:
                        t, orig_dtype = load_tensor_as_float32(sf, key)
                    except Exception as e:
                        print(f"  WARNING: Failed to load {key}: {e}", file=sys.stderr)
                        tensor_stats = {"key": key, "error": str(e)}
                        _store_result(results, expert_accumulator, info, key, tensor_stats)
                        continue

                    # Compute statistics
                    try:
                        tensor_stats = compute_tensor_stats(
                            t,
                            compute_svd=compute_svd,
                            svd_top_k=svd_top_k,
                        )
                        tensor_stats["original_dtype"] = orig_dtype
                        tensor_stats["key"] = key
                        tensor_stats["layer_type"] = layer_type
                        tensor_stats["component"] = component
                        tensor_stats["role"] = role
                    except Exception as e:
                        print(f"  WARNING: Stats failed for {key}: {e}", file=sys.stderr)
                        tensor_stats = {"key": key, "error": str(e)}

                    # Free tensor immediately
                    del t

                _store_result(results, expert_accumulator, info, key, tensor_stats)

    # Post-process: aggregate expert stats per MoE layer
    print(f"\n  Aggregating expert statistics...")
    for layer_idx, comp_dict in expert_accumulator.items():
        if str(layer_idx) not in results["layers"]:
            continue
        for comp_name, expert_list in comp_dict.items():
            agg = aggregate_expert_stats(expert_list)
            layer_str = str(layer_idx)
            if "expert_aggregates" not in results["layers"][layer_str]:
                results["layers"][layer_str]["expert_aggregates"] = {}
            results["layers"][layer_str]["expert_aggregates"][comp_name] = agg

    # Build per-layer summary
    print("  Building summary...")
    results["summary"] = build_summary(results)

    # Metadata
    elapsed = time.time() - t_start
    results["metadata"] = {
        "total_tensors_in_index": total_tensors,
        "processed": processed,
        "skipped_scale_tensors": skipped_scale,
        "skipped_by_layer_filter": skipped_filter,
        "elapsed_seconds": round(elapsed, 2),
        "tensors_per_second": round(processed / elapsed, 1) if elapsed > 0 else 0,
    }

    print(f"\n  Processed {processed} tensors in {elapsed:.1f}s "
          f"({processed/elapsed:.0f}/s)")
    print(f"  Skipped {skipped_scale} scale tensors, {skipped_filter} filtered layers")

    # Write output
    print(f"\nWriting output to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Written: {size_mb:.1f} MB")


def _json_default(obj):
    """JSON serializer for non-standard types."""
    if hasattr(obj, "item"):  # numpy/torch scalars
        return obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _store_result(
    results: dict,
    expert_accumulator: dict,
    info: dict,
    key: str,
    tensor_stats: dict,
) -> None:
    """Store tensor statistics in the appropriate place in results."""
    layer_idx = info["layer_idx"]
    layer_type = info["layer_type"]
    component = info["component"] or "unknown"

    if layer_type in ("embedding", "lm_head", "final_norm", "global"):
        results["global"][component] = tensor_stats
        return

    if layer_idx is None:
        results["global"][key] = tensor_stats
        return

    layer_str = str(layer_idx)
    if layer_str not in results["layers"]:
        results["layers"][layer_str] = {
            "_layer_type": layer_type,
            "_layer_idx": layer_idx,
        }

    # For routed experts, accumulate into expert list for aggregation
    expert_idx = info.get("expert_idx")
    if expert_idx is not None and component == "routed_expert":
        role = info.get("role", "unknown")
        # Group by proj type: up_proj.weight, down_proj.weight, etc.
        proj_key = role.replace(".weight_scale", "").replace(".input_scale", "")
        if layer_idx not in expert_accumulator:
            expert_accumulator[layer_idx] = {}
        if proj_key not in expert_accumulator[layer_idx]:
            expert_accumulator[layer_idx][proj_key] = []

        # Store with expert index for ordering
        expert_entry = {"expert_idx": expert_idx, **tensor_stats}
        acc = expert_accumulator[layer_idx][proj_key]
        # Maintain sorted order by expert_idx
        acc.append(expert_entry)

        # Store in results under experts sub-dict for direct access
        if "routed_experts" not in results["layers"][layer_str]:
            results["layers"][layer_str]["routed_experts"] = {}
        expert_key = f"expert_{expert_idx:03d}"
        if expert_key not in results["layers"][layer_str]["routed_experts"]:
            results["layers"][layer_str]["routed_experts"][expert_key] = {}
        results["layers"][layer_str]["routed_experts"][expert_key][proj_key] = tensor_stats
        return

    # Store by component name
    if component not in results["layers"][layer_str]:
        results["layers"][layer_str][component] = {}

    role = info.get("role", "stats")
    if role in ("weight", "stats", None):
        # Main weight: store stats directly under component
        results["layers"][layer_str][component] = tensor_stats
    else:
        # Sub-key (e.g. conv1d.weight, conv1d.bias)
        if not isinstance(results["layers"][layer_str][component], dict):
            results["layers"][layer_str][component] = {}
        results["layers"][layer_str][component][role] = tensor_stats


def build_summary(results: dict) -> dict:
    """Build a concise summary of per-layer statistics."""
    summary = {
        "layer_types": {},
        "mamba_layers": [],
        "attention_layers": [],
        "moe_layers": [],
    }

    for _, layer_data in results["layers"].items():
        layer_type = layer_data.get("_layer_type", "unknown")
        layer_idx = layer_data.get("_layer_idx")
        if layer_idx is None:
            continue

        entry = {"layer_idx": layer_idx, "layer_type": layer_type}

        if layer_type == "mamba":
            # Summarize A_log, D, dt_bias
            for comp in ("A_log", "D", "dt_bias", "in_proj", "out_proj"):
                if comp in layer_data and isinstance(layer_data[comp], dict):
                    s = layer_data[comp]
                    entry[comp] = {
                        "mean": s.get("mean"),
                        "std": s.get("std"),
                        "frob": s.get("frobenius_norm"),
                        "sparsity_pct": s.get("sparsity_pct"),
                    }
            summary["mamba_layers"].append(entry)

        elif layer_type == "attention":
            for comp in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if comp in layer_data and isinstance(layer_data[comp], dict):
                    s = layer_data[comp]
                    entry[comp] = {
                        "mean": s.get("mean"),
                        "std": s.get("std"),
                        "frob": s.get("frobenius_norm"),
                        "svd_top3": s.get("svd_top_singular_values"),
                        "condition_number": s.get("svd_condition_number"),
                    }
            summary["attention_layers"].append(entry)

        elif layer_type == "moe":
            # Gate stats + expert aggregate
            if "gate" in layer_data and isinstance(layer_data["gate"], dict):
                s = layer_data["gate"]
                entry["gate"] = {
                    "mean": s.get("mean"),
                    "std": s.get("std"),
                    "frob": s.get("frobenius_norm"),
                    "svd_top3": s.get("svd_top_singular_values"),
                }
            agg = layer_data.get("expert_aggregates", {})
            entry["expert_norm_cv"] = {
                k: v.get("norm_cv") for k, v in agg.items()
            }
            entry["expert_aggregates"] = agg
            summary["moe_layers"].append(entry)

        # Count by type
        summary["layer_types"][layer_type] = summary["layer_types"].get(layer_type, 0) + 1

    # Sort by layer index
    summary["mamba_layers"].sort(key=lambda x: x["layer_idx"])
    summary["attention_layers"].sort(key=lambda x: x["layer_idx"])
    summary["moe_layers"].sort(key=lambda x: x["layer_idx"])

    # Global stats
    global_comps = list(results.get("global", {}).keys())
    summary["global_components"] = global_comps

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-layer weight statistics for Nemotron-3-Nano-30B safetensors model"
    )
    parser.add_argument(
        "--model-path",
        default="/workspace/model",
        help="Path to the model directory containing safetensors files "
             "(default: /workspace/model)",
    )
    parser.add_argument(
        "--output",
        default="weight_baseline.json",
        help="Output JSON file path (default: weight_baseline.json)",
    )
    parser.add_argument(
        "--layers-only",
        type=str,
        default=None,
        help="Comma-separated layer indices to process, e.g. '5,12,19' "
             "(default: all layers)",
    )
    parser.add_argument(
        "--svd-top-k",
        type=int,
        default=3,
        help="Number of top singular values to compute for attention/gate projections "
             "(default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify keys without loading tensors (for testing key parsing)",
    )
    parser.add_argument(
        "--no-experts",
        action="store_true",
        help="Skip per-expert statistics (much faster, only aggregates)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path.is_dir():
        print(f"ERROR: Model path is not a directory: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Parse layer filter
    layers_filter = None
    if args.layers_only:
        try:
            layers_filter = set(int(x.strip()) for x in args.layers_only.split(","))
            print(f"Layer filter: {sorted(layers_filter)}")
        except ValueError as e:
            print(f"ERROR: Invalid --layers-only argument: {e}", file=sys.stderr)
            sys.exit(1)

    output_path = Path(args.output)

    print(f"Nemotron-3-Nano-30B Weight Introspection")
    print(f"  Model path: {model_path}")
    print(f"  Output:     {output_path}")
    print(f"  SVD top-k:  {args.svd_top_k}")
    print(f"  Dry run:    {args.dry_run}")

    introspect_model(
        model_path=model_path,
        output_path=output_path,
        layers_filter=layers_filter,
        dry_run=args.dry_run,
        svd_top_k=args.svd_top_k,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
