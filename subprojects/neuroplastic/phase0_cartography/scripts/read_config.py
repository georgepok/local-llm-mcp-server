"""
read_config.py — Read model config files and output architecture_ground_truth.json.

Works in two modes:
  1. Inside the vLLM container (model at /workspace/model by default)
  2. With --model-path pointing to the model directory on any filesystem

Usage:
    python read_config.py
    python read_config.py --model-path /workspace/model
    python read_config.py --model-path /home/pokazge/models/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
    python read_config.py --model-path /workspace/model --output /workspace/architecture_ground_truth.json
"""

import argparse
import json
import sys
from pathlib import Path


# Layer type classification
HYBRID_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"

PATTERN_MAP = {
    "M": "mamba",
    "E": "moe",
    "*": "attention",
}


def parse_hybrid_pattern(pattern: str) -> dict:
    """Parse the hybrid_override_pattern string into per-layer type assignments."""
    mamba_indices = []
    moe_indices = []
    attention_indices = []

    for i, char in enumerate(pattern):
        layer_type = PATTERN_MAP.get(char)
        if layer_type == "mamba":
            mamba_indices.append(i)
        elif layer_type == "moe":
            moe_indices.append(i)
        elif layer_type == "attention":
            attention_indices.append(i)
        else:
            print(f"  WARNING: Unknown pattern character '{char}' at index {i}", file=sys.stderr)

    return {
        "mamba_indices": mamba_indices,
        "moe_indices": moe_indices,
        "attention_indices": attention_indices,
        "mamba_count": len(mamba_indices),
        "moe_count": len(moe_indices),
        "attention_count": len(attention_indices),
    }


def load_json_file(path: Path, label: str) -> dict:
    """Load a JSON file with clear error reporting."""
    if not path.exists():
        print(f"  WARNING: {label} not found at {path}", file=sys.stderr)
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        print(f"  Loaded {label}: {path}")
        return data
    except json.JSONDecodeError as e:
        print(f"  ERROR: Failed to parse {label}: {e}", file=sys.stderr)
        return {}


def extract_architecture(config: dict, quant_config: dict, tokenizer_config: dict) -> dict:
    """Extract and structure architecture information from raw config dicts."""
    arch = {
        "model_id": config.get("_name_or_path", "unknown"),
        "model_type": config.get("model_type", "unknown"),
        "architecture_class": (config.get("architectures") or ["unknown"])[0],
    }

    # Core dimensions
    arch["core_dimensions"] = {
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "vocab_size": config.get("vocab_size"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "torch_dtype": config.get("torch_dtype"),
        "tie_word_embeddings": config.get("tie_word_embeddings", False),
    }

    # Hybrid layer pattern
    pattern = config.get("hybrid_override_pattern", "")
    pattern_info = parse_hybrid_pattern(pattern)
    arch["hybrid_layer_pattern"] = {
        "pattern_string": pattern,
        "legend": {"M": "Mamba SSM layer", "E": "MoE-FFN layer", "*": "Full Attention layer"},
        "total_layers": config.get("num_hidden_layers"),
        **pattern_info,
    }

    # Attention config
    arch["attention_config"] = {
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
        "attention_bias": config.get("attention_bias", False),
        "rope_theta": config.get("rope_theta"),
        "partial_rotary_factor": config.get("partial_rotary_factor"),
    }

    # Mamba config — may be nested under 'ssm_cfg' or flat
    ssm_cfg = config.get("ssm_cfg", config)
    arch["mamba_config"] = {
        "mamba_num_heads": ssm_cfg.get("mamba_num_heads") or config.get("mamba_num_heads"),
        "mamba_head_dim": ssm_cfg.get("mamba_head_dim") or config.get("mamba_head_dim"),
        "ssm_state_size": ssm_cfg.get("ssm_state_size") or config.get("ssm_state_size"),
        "conv_kernel": ssm_cfg.get("conv_kernel") or config.get("conv_kernel"),
        "expand": ssm_cfg.get("expand") or config.get("expand"),
        "chunk_size": ssm_cfg.get("chunk_size") or config.get("chunk_size"),
        "n_groups": ssm_cfg.get("n_groups") or config.get("n_groups"),
        "mamba_hidden_act": ssm_cfg.get("mamba_hidden_act") or config.get("mamba_hidden_act"),
    }

    # MoE config
    arch["moe_config"] = {
        "n_routed_experts": config.get("n_routed_experts"),
        "num_experts_per_tok": config.get("num_experts_per_tok"),
        "moe_intermediate_size": config.get("moe_intermediate_size"),
        "n_shared_experts": config.get("n_shared_experts"),
        "moe_shared_expert_intermediate_size": config.get("moe_shared_expert_intermediate_size"),
        "routed_scaling_factor": config.get("routed_scaling_factor"),
        "norm_topk_prob": config.get("norm_topk_prob"),
        "topk_group": config.get("topk_group"),
    }

    # FFN config
    arch["ffn_config"] = {
        "intermediate_size": config.get("intermediate_size"),
        "mlp_hidden_act": config.get("mlp_hidden_act") or config.get("hidden_act"),
        "mlp_bias": config.get("mlp_bias", False),
    }

    # Quantization config
    if quant_config:
        arch["quantization_config"] = {
            "quant_algo": quant_config.get("quant_cfg", {}).get("quant_algo")
                         or quant_config.get("quant_algo"),
            "kv_cache_quant_algo": quant_config.get("kv_cache_quant_cfg", {}).get("quant_algo")
                                   or quant_config.get("kv_cache_quant_algo"),
            "group_size": quant_config.get("quant_cfg", {}).get("group_size")
                          or quant_config.get("group_size"),
            "producer": quant_config.get("producer"),
            "raw": quant_config,
        }

    # Tokenizer info
    if tokenizer_config:
        arch["tokenizer_config"] = {
            "model_max_length": tokenizer_config.get("model_max_length"),
            "tokenizer_class": tokenizer_config.get("tokenizer_class"),
            "bos_token": tokenizer_config.get("bos_token"),
            "eos_token": tokenizer_config.get("eos_token"),
            "pad_token": tokenizer_config.get("pad_token"),
            "chat_template": "<see tokenizer_config.json>",
        }

    # Safetensors index info
    arch["raw_config"] = config

    return arch


def check_safetensors_index(model_path: Path) -> dict:
    """Read safetensors index to get weight file inventory."""
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        # Try single-file model
        single = model_path / "model.safetensors"
        if single.exists():
            size = single.stat().st_size
            return {"num_shards": 1, "total_size_bytes": size, "weight_map": None}
        return {}

    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    total_size = index.get("metadata", {}).get("total_size", 0)

    # Count unique shard files
    shards = set(weight_map.values())

    return {
        "num_shards": len(shards),
        "num_weight_keys": len(weight_map),
        "total_size_bytes": total_size,
        "total_size_gib": round(total_size / (1024**3), 3),
        "shard_files": sorted(shards),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Read model config and output architecture_ground_truth.json"
    )
    parser.add_argument(
        "--model-path",
        default="/workspace/model",
        help="Path to the model directory (default: /workspace/model for inside-container use)",
    )
    parser.add_argument(
        "--output",
        default="architecture_ground_truth.json",
        help="Output JSON file path (default: architecture_ground_truth.json)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: True)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path.is_dir():
        print(f"ERROR: Model path is not a directory: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading model config from: {model_path}")

    # Load all config files
    config = load_json_file(model_path / "config.json", "config.json")
    if not config:
        print("ERROR: Could not read config.json — is this a valid model directory?", file=sys.stderr)
        sys.exit(1)

    quant_config = load_json_file(model_path / "hf_quant_config.json", "hf_quant_config.json")
    tokenizer_config = load_json_file(model_path / "tokenizer_config.json", "tokenizer_config.json")

    # Get safetensors inventory
    print("  Reading safetensors index...")
    weight_inventory = check_safetensors_index(model_path)
    if weight_inventory:
        print(f"  Found {weight_inventory.get('num_shards', '?')} shards, "
              f"{weight_inventory.get('num_weight_keys', '?')} weight keys, "
              f"{weight_inventory.get('total_size_gib', '?')} GiB")

    # Build architecture document
    arch = extract_architecture(config, quant_config, tokenizer_config)
    if weight_inventory:
        arch["weight_inventory"] = weight_inventory

    # Write output
    output_path = Path(args.output)
    indent = 2 if args.pretty else None
    with open(output_path, "w") as f:
        json.dump(arch, f, indent=indent, default=str)

    print(f"\nWrote architecture_ground_truth.json to: {output_path}")
    print(f"  Model type: {arch.get('model_type')}")
    print(f"  Architecture: {arch.get('architecture_class')}")
    dim = arch.get("core_dimensions", {})
    print(f"  hidden_size={dim.get('hidden_size')}, "
          f"num_layers={dim.get('num_hidden_layers')}, "
          f"vocab={dim.get('vocab_size')}")
    pattern_info = arch.get("hybrid_layer_pattern", {})
    print(f"  Layer mix: {pattern_info.get('mamba_count')} Mamba + "
          f"{pattern_info.get('attention_count')} Attention + "
          f"{pattern_info.get('moe_count')} MoE")


if __name__ == "__main__":
    main()
