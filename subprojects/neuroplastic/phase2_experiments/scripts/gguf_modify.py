#!/usr/bin/env python3
"""GGUF In-Place Tensor Modification for Neuroplastic Project

Modifies tensor data directly in a GGUF file using memory-mapped I/O.
For F32/F16 tensors (gate weights, A_log, D, dt_bias), this is lossless.
For Q8_0 tensors, modification requires dequantize → modify → requantize.

Usage:
    # Inspect a tensor
    python3 gguf_modify.py --model path/to/model.gguf --tensor blk.50.ssm_a --inspect

    # Scale a tensor
    python3 gguf_modify.py --model path/to/model.gguf --tensor blk.50.ssm_a --op scale --value 1.1

    # Add to a tensor
    python3 gguf_modify.py --model path/to/model.gguf --tensor blk.50.ssm_a --op add --value 0.5

    # Apply arbitrary numpy expression
    python3 gguf_modify.py --model path/to/model.gguf --tensor blk.50.ssm_a --op expr --value "t * 1.1 + 0.5"

    # Save/restore checkpoints
    python3 gguf_modify.py --model path/to/model.gguf --tensor blk.50.ssm_a --save-checkpoint pre_exp
    python3 gguf_modify.py --model path/to/model.gguf --tensor blk.50.ssm_a --restore-checkpoint pre_exp
"""

import argparse
import json
import os
import sys
import time
import struct
import mmap
import numpy as np

# GGUF format constants
GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8

GGML_TYPE_SIZES = {
    GGML_TYPE_F32: 4,
    GGML_TYPE_F16: 2,
    GGML_TYPE_Q8_0: 34,  # 32 int8 values + 1 f16 scale per block
}

GGML_TYPE_NAMES = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_Q8_0: "Q8_0",
}


def parse_gguf_header(f):
    """Parse GGUF file header to find tensor metadata."""
    f.seek(0)
    magic = struct.unpack('<I', f.read(4))[0]
    if magic != GGUF_MAGIC:
        raise ValueError(f"Not a GGUF file (magic: {magic:#x})")

    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]

    return version, n_tensors, n_kv


def find_tensor_info(gguf_path: str, target_name: str) -> dict | None:
    """Find tensor info using the gguf Python package."""
    try:
        from gguf import GGUFReader
    except ImportError:
        print("ERROR: gguf package required. Install with: pip install gguf")
        sys.exit(1)

    reader = GGUFReader(gguf_path)
    for t in reader.tensors:
        if str(t.name) == target_name:
            return {
                "name": str(t.name),
                "shape": [int(s) for s in t.shape],
                "type": int(t.tensor_type),
                "type_name": t.tensor_type.name,
                "offset": int(t.data_offset),
                "n_bytes": int(t.n_bytes),
                "n_elements": int(np.prod(t.shape)),
                "data": t.data,  # numpy memmap
            }
    return None


def list_tensors(gguf_path: str, filter_str: str = "") -> list[dict]:
    """List all tensors, optionally filtered."""
    from gguf import GGUFReader
    reader = GGUFReader(gguf_path)
    results = []
    for t in reader.tensors:
        name = str(t.name)
        if filter_str and filter_str not in name:
            continue
        results.append({
            "name": name,
            "shape": [int(s) for s in t.shape],
            "type": t.tensor_type.name,
        })
    return results


def inspect_tensor(gguf_path: str, tensor_name: str) -> dict:
    """Read and report tensor statistics."""
    info = find_tensor_info(gguf_path, tensor_name)
    if info is None:
        return {"error": f"Tensor '{tensor_name}' not found"}

    data = info["data"]
    flat = data.flatten().astype(np.float32)

    stats = {
        "name": info["name"],
        "shape": info["shape"],
        "type": info["type_name"],
        "n_elements": info["n_elements"],
        "n_bytes": info["n_bytes"],
        "offset": info["offset"],
        "stats": {
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "abs_max": float(np.abs(flat).max()),
            "l2_norm": float(np.linalg.norm(flat)),
        },
    }
    return stats


def modify_tensor(
    gguf_path: str,
    tensor_name: str,
    operation: str,
    value: float | str,
    dry_run: bool = False,
) -> dict:
    """Modify a tensor in-place in the GGUF file."""
    info = find_tensor_info(gguf_path, tensor_name)
    if info is None:
        return {"error": f"Tensor '{tensor_name}' not found"}

    if info["type_name"] not in ("F32", "F16"):
        return {"error": f"Only F32/F16 tensors supported for modification. Got: {info['type_name']}"}

    data = info["data"]
    flat = data.flatten()

    # Before stats
    before = {
        "mean": float(flat.astype(np.float32).mean()),
        "std": float(flat.astype(np.float32).std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }

    # Compute modification
    if operation == "scale":
        new_data = flat * float(value)
    elif operation == "add":
        new_data = flat + float(value)
    elif operation == "expr":
        t = flat.astype(np.float32)
        new_data = eval(str(value), {"__builtins__": {}, "np": np, "t": t, "math": __import__("math")})
        new_data = np.asarray(new_data, dtype=np.float32)
    else:
        return {"error": f"Unknown operation: {operation}"}

    # After stats
    new_flat = new_data.flatten().astype(np.float32)
    after = {
        "mean": float(new_flat.mean()),
        "std": float(new_flat.std()),
        "min": float(new_flat.min()),
        "max": float(new_flat.max()),
    }

    result = {
        "tensor": tensor_name,
        "operation": operation,
        "value": value,
        "shape": info["shape"],
        "type": info["type_name"],
        "before": before,
        "after": after,
        "dry_run": dry_run,
    }

    if not dry_run:
        # Write back in the original dtype
        if info["type_name"] == "F32":
            data[:] = new_data.reshape(data.shape).astype(np.float32)
        elif info["type_name"] == "F16":
            data[:] = new_data.reshape(data.shape).astype(np.float16)
        # Force flush to disk
        if hasattr(data, 'flush'):
            data.flush()
        result["written"] = True
    else:
        result["written"] = False

    return result


def save_checkpoint(gguf_path: str, tensor_name: str, checkpoint_name: str) -> dict:
    """Save current tensor values to a checkpoint file."""
    info = find_tensor_info(gguf_path, tensor_name)
    if info is None:
        return {"error": f"Tensor '{tensor_name}' not found"}

    checkpoint_dir = os.path.join(os.path.dirname(gguf_path), "neuroplastic_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    safe_name = tensor_name.replace(".", "_").replace("/", "_")
    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_{safe_name}.npy")

    data = info["data"].flatten().copy()
    np.save(checkpoint_path, data)

    return {
        "action": "save_checkpoint",
        "tensor": tensor_name,
        "checkpoint": checkpoint_name,
        "path": checkpoint_path,
        "shape": info["shape"],
        "type": info["type_name"],
        "n_elements": len(data),
    }


def restore_checkpoint(gguf_path: str, tensor_name: str, checkpoint_name: str) -> dict:
    """Restore tensor values from a checkpoint file."""
    info = find_tensor_info(gguf_path, tensor_name)
    if info is None:
        return {"error": f"Tensor '{tensor_name}' not found"}

    checkpoint_dir = os.path.join(os.path.dirname(gguf_path), "neuroplastic_checkpoints")
    safe_name = tensor_name.replace(".", "_").replace("/", "_")
    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_{safe_name}.npy")

    if not os.path.exists(checkpoint_path):
        return {"error": f"Checkpoint not found: {checkpoint_path}"}

    saved_data = np.load(checkpoint_path)
    data = info["data"]

    if saved_data.size != data.flatten().size:
        return {"error": f"Size mismatch: checkpoint has {saved_data.size}, tensor has {data.flatten().size}"}

    data[:] = saved_data.reshape(data.shape).astype(data.dtype)
    if hasattr(data, 'flush'):
        data.flush()

    return {
        "action": "restore_checkpoint",
        "tensor": tensor_name,
        "checkpoint": checkpoint_name,
        "shape": info["shape"],
        "restored": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GGUF In-Place Tensor Modification")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--tensor", help="Tensor name (e.g., blk.50.ssm_a)")
    parser.add_argument("--inspect", action="store_true", help="Inspect tensor stats")
    parser.add_argument("--list", action="store_true", help="List tensors")
    parser.add_argument("--filter", default="", help="Filter tensor list")
    parser.add_argument("--op", choices=["scale", "add", "expr"], help="Modification operation")
    parser.add_argument("--value", help="Operation value")
    parser.add_argument("--dry-run", action="store_true", help="Compute without writing")
    parser.add_argument("--save-checkpoint", help="Save tensor to named checkpoint")
    parser.add_argument("--restore-checkpoint", help="Restore tensor from named checkpoint")
    args = parser.parse_args()

    if args.list:
        tensors = list_tensors(args.model, args.filter)
        for t in tensors:
            print(f"  {t['name']}: shape={t['shape']}, type={t['type']}")
        print(f"\nTotal: {len(tensors)} tensors")
        return

    if not args.tensor:
        parser.error("--tensor is required for inspect/modify/checkpoint operations")

    t0 = time.time()

    if args.save_checkpoint:
        result = save_checkpoint(args.model, args.tensor, args.save_checkpoint)
    elif args.restore_checkpoint:
        result = restore_checkpoint(args.model, args.tensor, args.restore_checkpoint)
    elif args.inspect:
        result = inspect_tensor(args.model, args.tensor)
    elif args.op:
        if not args.value:
            parser.error("--value is required for modification operations")
        value: float | str = args.value
        if args.op != "expr":
            value = float(args.value)
        result = modify_tensor(args.model, args.tensor, args.op, value, args.dry_run)
    else:
        parser.error("One of --inspect, --op, --list, --save-checkpoint, --restore-checkpoint is required")
        return

    elapsed = time.time() - t0
    result["elapsed_seconds"] = elapsed
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
