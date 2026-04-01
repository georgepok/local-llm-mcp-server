"""Generate a fixed dataset of CW episodes for grokking experiments.

Generates episodes one at a time, stores all tensors needed for training.
Episodes are pre-tokenized and padded to seq_len.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.tasks import get_task


def main():
    parser = argparse.ArgumentParser(description="Generate fixed CW dataset")
    parser.add_argument("--n_episodes", type=int, default=2000)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--task_kwargs", type=str, default="{}",
                        help="JSON dict of CW task kwargs")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pt file path")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    task_kwargs = json.loads(args.task_kwargs)
    task = get_task("CW", tokenizer, seq_len=args.seq_len, **task_kwargs)

    print(f"Generating {args.n_episodes} episodes...")
    print(f"  seq_len={args.seq_len}")
    print(f"  task_kwargs={task_kwargs}")

    all_input_ids = []
    all_labels = []
    all_context_masks = []
    all_room_distances = []
    all_room_positions = []
    all_n_rooms = []

    for i in range(args.n_episodes):
        # Generate single episode
        input_ids, labels, meta = task.generate_batch(1, device=None)

        all_input_ids.append(input_ids[0])       # [seq_len]
        all_labels.append(labels[0])              # [seq_len]
        all_context_masks.append(meta["context_mask"][0])  # [seq_len]
        all_room_distances.append(meta["room_distances"][0])  # [R, R]
        all_room_positions.append(meta["room_token_positions"][0])  # [R]
        all_n_rooms.append(meta["n_rooms"][0])    # scalar

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n_episodes}")

    # Pad room tensors to global R_max
    R_max = max(rd.shape[0] for rd in all_room_distances)
    print(f"  R_max={R_max}")

    padded_distances = torch.ones(args.n_episodes, R_max, R_max)
    padded_positions = torch.full((args.n_episodes, R_max), -1, dtype=torch.long)

    for i in range(args.n_episodes):
        R = all_room_distances[i].shape[0]
        padded_distances[i, :R, :R] = all_room_distances[i]
        padded_positions[i, :R] = all_room_positions[i]

    dataset = {
        "input_ids": torch.stack(all_input_ids),         # [N, seq_len]
        "labels": torch.stack(all_labels),                # [N, seq_len]
        "context_mask": torch.stack(all_context_masks),   # [N, seq_len]
        "room_distances": padded_distances,               # [N, R_max, R_max]
        "room_token_positions": padded_positions,          # [N, R_max]
        "n_rooms": torch.stack(all_n_rooms),              # [N]
        "n_episodes": args.n_episodes,
        "seq_len": args.seq_len,
        "task_kwargs": task_kwargs,
        "R_max": R_max,
    }

    torch.save(dataset, args.output)
    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    print(f"\nSaved {args.n_episodes} episodes to {args.output} ({size_mb:.1f} MB)")
    print(f"  input_ids: {dataset['input_ids'].shape}")
    print(f"  room_distances: {dataset['room_distances'].shape}")
    print(f"  n_rooms range: {dataset['n_rooms'].min().item()}-{dataset['n_rooms'].max().item()}")


if __name__ == "__main__":
    main()
