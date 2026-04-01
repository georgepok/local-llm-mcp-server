#!/usr/bin/env python3
"""Merge LoRA adapter into base model and save as bf16.

Usage:
  python3 merge_lora.py --output-dir /workspace/models/neuroplastic-merged
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA into base model")
    parser.add_argument("--model-name",
                        default="/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--lora-path", default="/workspace/finetune/lora_output")
    parser.add_argument("--output-dir",
                        default="/workspace/models/neuroplastic-merged")
    args = parser.parse_args()

    if "TRITON_PTXAS_PATH" not in os.environ:
        ptxas = "/usr/local/cuda/bin/ptxas"
        if os.path.exists(ptxas):
            os.environ["TRITON_PTXAS_PATH"] = ptxas

    # Load base model in bf16 (no quantization — we need full precision for merge)
    print(f"Loading base model (bf16): {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
    )

    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"Base model loaded: {mem_gb:.1f} GB")

    # Load LoRA adapter
    print(f"Loading LoRA adapter: {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path)
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"Model + LoRA loaded: {mem_gb:.1f} GB")

    # Merge LoRA into base weights
    print("Merging LoRA into base model...")
    model = model.merge_and_unload()
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"After merge: {mem_gb:.1f} GB")

    # Save merged model
    print(f"Saving merged model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    # Copy custom modeling code needed for trust_remote_code
    import shutil
    for f in os.listdir(args.model_name):
        if f.endswith(".py") or f == "config.json":
            src = os.path.join(args.model_name, f)
            dst = os.path.join(args.output_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  Copied {f}")

    print("\nMerge complete!")
    print(f"Merged model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
