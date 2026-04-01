"""Oracle embedding extraction — runs in separate container, then stops.

Loads Qwen3.5-9B base model, serializes all ARC tasks × 8 D4 variants,
extracts mean-pooled last hidden states, saves to embeddings.pt.

Container lifecycle: start → download model → precompute → save → stop (frees ~18GB VRAM).

Usage:
    python scripts/precompute.py \
        --model_id Qwen/Qwen3.5-9B \
        --data_dir /workspace/fgn-v3/data/arc \
        --output /workspace/latent-oracle/embeddings.pt \
        --d4_augment --batch_size 4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Add parent for latent_oracle imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latent_oracle.serialize import serialize_task


def load_arc_tasks_raw(data_dir: str):
    """Load raw ARC task dicts from JSON files."""
    result = {}
    for split_name, dir_name in [("train", "training"), ("eval", "evaluation")]:
        split_dir = os.path.join(data_dir, dir_name)
        if not os.path.isdir(split_dir):
            continue
        tasks = []
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(split_dir, fname)) as f:
                task = json.load(f)
            task["task_id"] = fname.replace(".json", "")
            tasks.append(task)
        result[split_name] = tasks
    return result


def precompute(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and tokenizer
    print(f"Loading model: {args.model_id}")
    from transformers import AutoModel, AutoTokenizer, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True
    )

    # Qwen3.5 is a VLM — AutoModel gives the full multimodal model with
    # language_model sub-module. We extract the text backbone for embeddings.
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    # Detect VLM vs pure LLM architecture
    if hasattr(config, "text_config"):
        oracle_dim = config.text_config.hidden_size
        print(f"VLM detected — using text backbone (hidden_size={oracle_dim})")

        # Load full model then extract language_model
        full_model = AutoModel.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval().to(device)
        # The text backbone is at .language_model (Qwen3.5 VLM architecture)
        if hasattr(full_model, "language_model"):
            model = full_model.language_model
            print(f"Extracted language_model: {type(model).__name__}")
        else:
            model = full_model
            print(f"No language_model sub-module, using full model")
        del full_model  # free VLM wrapper memory
    else:
        oracle_dim = config.hidden_size
        model = AutoModel.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval().to(device)

    for p in model.parameters():
        p.requires_grad = False

    print(f"Oracle dim: {oracle_dim}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    # Load ARC tasks
    print(f"Loading ARC tasks from {args.data_dir}")
    all_tasks = load_arc_tasks_raw(args.data_dir)
    n_train = len(all_tasks.get("train", []))
    n_eval = len(all_tasks.get("eval", []))
    print(f"  Train: {n_train}, Eval: {n_eval}")

    # D4 variants
    d4_range = range(8) if args.d4_augment else range(1)

    # Collect all (task, d4_idx, test_idx, split) to process
    work_items = []
    for split in ("train", "eval"):
        for task in all_tasks.get(split, []):
            for test_idx in range(len(task["test"])):
                for d4_idx in d4_range:
                    work_items.append((task, d4_idx, test_idx, split))

    print(f"Total items to embed: {len(work_items)}")

    # Process in batches
    all_embeddings = []
    all_task_ids = []
    all_d4_indices = []
    all_test_indices = []
    all_splits = []

    t0 = time.time()
    batch_texts = []
    batch_meta = []

    for idx, (task, d4_idx, test_idx, split) in enumerate(work_items):
        text = serialize_task(task, d4_idx=d4_idx, test_idx=test_idx)
        batch_texts.append(text)
        batch_meta.append((task["task_id"], d4_idx, test_idx, split))

        if len(batch_texts) == args.batch_size or idx == len(work_items) - 1:
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Handle different model output formats:
                # - Standard LLM: outputs.last_hidden_state
                # - Some models: outputs.hidden_states[-1]
                if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                    hidden = outputs.last_hidden_state
                elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    hidden = outputs.hidden_states[-1]
                else:
                    # Fallback: first tensor output (logits-like)
                    hidden = outputs[0]
                # hidden: [B, seq, oracle_dim]

            # Mean-pool over non-pad positions
            mask = inputs["attention_mask"].unsqueeze(-1).float()  # [B, seq, 1]
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, oracle_dim]

            all_embeddings.append(pooled.cpu())
            for tid, d4, ti, sp in batch_meta:
                all_task_ids.append(tid)
                all_d4_indices.append(d4)
                all_test_indices.append(ti)
                all_splits.append(sp)

            batch_texts = []
            batch_meta = []

            if (idx + 1) % (args.batch_size * 10) == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (len(work_items) - idx - 1) / rate
                print(f"  [{idx + 1}/{len(work_items)}] "
                      f"{rate:.1f} items/s, ETA {eta:.0f}s")

    # Concatenate
    embeddings = torch.cat(all_embeddings, dim=0)  # [N, oracle_dim]
    d4_tensor = torch.tensor(all_d4_indices, dtype=torch.long)
    test_tensor = torch.tensor(all_test_indices, dtype=torch.long)

    elapsed = time.time() - t0
    print(f"\nDone: {embeddings.shape[0]} embeddings in {elapsed:.1f}s")
    print(f"  Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    print(f"  Size: {embeddings.numel() * 2 / 1e6:.1f} MB (bf16)")

    # Save
    save_data = {
        "embeddings": embeddings,       # [N, oracle_dim] bf16
        "task_ids": all_task_ids,        # list[str]
        "d4_indices": d4_tensor,         # [N] long
        "test_indices": test_tensor,     # [N] long
        "splits": all_splits,            # list[str]
        "oracle_dim": oracle_dim,        # int
        "model_id": args.model_id,       # str
    }
    torch.save(save_data, args.output)
    file_size = os.path.getsize(args.output) / 1e6
    print(f"  Saved to {args.output} ({file_size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Precompute oracle embeddings")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-9B-Base",
                        help="HuggingFace model ID for oracle")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ARC-AGI data directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for embeddings.pt")
    parser.add_argument("--d4_augment", action="store_true",
                        help="Generate all 8 D4 variants per task")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for embedding extraction")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max token length for oracle tokenizer")
    args = parser.parse_args()

    precompute(args)


if __name__ == "__main__":
    main()
