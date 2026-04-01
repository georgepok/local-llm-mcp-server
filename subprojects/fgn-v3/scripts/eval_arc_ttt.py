"""ARC-AGI evaluation with test-time training (TTT).

For each eval task:
  1. Save model state
  2. Fine-tune on leave-one-out demo pairs with D4/color augmentation
  3. Predict test output with test-time augmentation (TTA) + majority vote
  4. Score grid exact match and cell accuracy
  5. Restore model state

Inspired by TRM (ARC Prize 2025) and Akyürek et al. (TTT for ARC).
"""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model_arc_sandwich import SandwichARC, create_arc_model
from fgn.tasks.arc import (
    N_COLORS, PAD_COLOR,
    load_arc_tasks, build_sequence, random_color_perm, apply_color_perm,
    pad_single_to_batch, invert_d4, invert_color_perm, reconstruct_grid,
)


def ttt_finetune(model, task, device, config, n_steps, lr, grad_clip):
    """Fine-tune model on leave-one-out demo pairs.

    Each step: randomly hold out one demo, treat it as "test",
    use remaining demos as context. Apply D4 + color augmentation.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    demos = task["train"]
    K = len(demos)

    if K < 2:
        # Can't leave-one-out with <2 demos — skip TTT
        return []

    losses = []
    for step in range(n_steps):
        # Random leave-one-out
        held_out = random.randint(0, K - 1)
        remaining = [d for i, d in enumerate(demos) if i != held_out]
        virtual_task = {
            "train": remaining,
            "test": [demos[held_out]],
        }

        # Random augmentation
        d4_idx = random.randint(0, 7)
        color_perm = random_color_perm()
        demo_order = list(range(len(remaining)))
        random.shuffle(demo_order)

        seq = build_sequence(
            virtual_task,
            d4_idx=d4_idx,
            color_perm=color_perm,
            demo_order=demo_order,
            max_seq_len=config.max_seq_len,
        )
        if seq is None:
            continue

        meta = pad_single_to_batch(seq, config.max_seq_len, device)

        optimizer.zero_grad()
        result = model(
            colors=meta["colors"],
            xs=meta["xs"],
            ys=meta["ys"],
            roles=meta["roles"],
            sep_mask=meta["sep_mask"],
            sep_types=meta["sep_types"],
            target_mask=meta["target_mask"],
            target_labels=meta["target_labels"],
            context_mask=meta["context_mask"],
            grid_ids=meta["grid_ids"],
            lengths=meta["lengths"],
            target_input_colors=meta["target_input_colors"],
            transform_only_loss=False,  # All-cell CE during TTT
        )
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        losses.append(result["loss"].item())

    return losses


def predict_with_tta(model, task, device, config, n_augments=32):
    """Predict test output with test-time augmentation + majority vote.

    Runs model under multiple D4/color augmentations, inverts the transform,
    and majority-votes each cell.

    Returns:
        (final_grid, test_output_shape) where final_grid is List[List[int]]
    """
    model.eval()

    # Get true output shape (un-augmented)
    test_pair = task["test"][0]
    true_H = len(test_pair["output"])
    true_W = len(test_pair["output"][0]) if true_H > 0 else 0
    vote_grid = torch.zeros(true_H, true_W, N_COLORS, dtype=torch.long)

    n_valid = 0
    with torch.no_grad():
        for aug_i in range(n_augments):
            d4_idx = aug_i % 8
            color_perm = random_color_perm() if aug_i >= 8 else None

            seq = build_sequence(
                task,
                d4_idx=d4_idx,
                color_perm=color_perm,
                max_seq_len=config.max_seq_len,
            )
            if seq is None:
                continue

            meta = pad_single_to_batch(seq, config.max_seq_len, device)

            result = model(
                colors=meta["colors"],
                xs=meta["xs"],
                ys=meta["ys"],
                roles=meta["roles"],
                sep_mask=meta["sep_mask"],
                sep_types=meta["sep_types"],
                target_mask=meta["target_mask"],
                target_labels=meta["target_labels"],
                context_mask=meta["context_mask"],
                grid_ids=meta["grid_ids"],
                lengths=meta["lengths"],
                target_input_colors=meta["target_input_colors"],
            )

            # Extract predictions at target positions
            target_mask_1d = meta["target_mask"][0]
            preds = result["logits"][0][target_mask_1d].argmax(dim=-1)

            # Reconstruct augmented grid
            aug_shape = seq["test_output_shape"]
            grid_preds = reconstruct_grid(preds, aug_shape)

            # Invert augmentation
            grid_preds = invert_d4(grid_preds, d4_idx)
            if color_perm is not None:
                inv_perm = invert_color_perm(color_perm)
                grid_preds = apply_color_perm(grid_preds, inv_perm)

            # Accumulate votes
            pred_H = len(grid_preds)
            pred_W = len(grid_preds[0]) if pred_H > 0 else 0
            if pred_H == true_H and pred_W == true_W:
                for y in range(true_H):
                    for x in range(true_W):
                        c = grid_preds[y][x]
                        if 0 <= c < N_COLORS:
                            vote_grid[y, x, c] += 1
                n_valid += 1

    # Majority vote
    final_grid = []
    for y in range(true_H):
        row = []
        for x in range(true_W):
            row.append(vote_grid[y, x].argmax().item())
        final_grid.append(row)

    return final_grid, (true_H, true_W), n_valid


def evaluate_with_ttt(model, tasks, device, config, args):
    """Run TTT evaluation on all tasks."""
    stats = {
        "total_grids": 0,
        "exact_grids": 0,
        "total_cells": 0,
        "correct_cells": 0,
        "ttt_loss_curves": [],
        "task_results": [],
    }

    t0 = time.time()
    for task_i, task in enumerate(tasks):
        task_t0 = time.time()

        # Save model state
        saved_state = copy.deepcopy(model.state_dict())

        # TTT fine-tuning
        losses = ttt_finetune(
            model, task, device, config,
            n_steps=args.ttt_steps,
            lr=args.ttt_lr,
            grad_clip=args.ttt_grad_clip,
        )

        # Predict with TTA
        pred_grid, shape, n_augments_used = predict_with_tta(
            model, task, device, config,
            n_augments=args.n_tta_augments,
        )

        # Score
        true_output = task["test"][0]["output"]
        true_H = len(true_output)
        true_W = len(true_output[0]) if true_H > 0 else 0

        n_cells = true_H * true_W
        n_correct = 0
        for y in range(true_H):
            for x in range(true_W):
                if y < len(pred_grid) and x < len(pred_grid[0]):
                    if pred_grid[y][x] == true_output[y][x]:
                        n_correct += 1

        exact_match = (n_correct == n_cells) if n_cells > 0 else False

        stats["total_grids"] += 1
        stats["exact_grids"] += int(exact_match)
        stats["total_cells"] += n_cells
        stats["correct_cells"] += n_correct
        stats["ttt_loss_curves"].append(losses)

        task_result = {
            "task_id": task.get("task_id", f"task_{task_i}"),
            "exact_match": exact_match,
            "cell_accuracy": n_correct / max(n_cells, 1),
            "n_cells": n_cells,
            "ttt_loss_start": losses[0] if losses else 0,
            "ttt_loss_end": losses[-1] if losses else 0,
            "n_augments_used": n_augments_used,
        }
        stats["task_results"].append(task_result)

        # Restore model state
        model.load_state_dict(saved_state)

        # Progress
        elapsed = time.time() - task_t0
        total_elapsed = time.time() - t0
        cell_acc = n_correct / max(n_cells, 1)
        exact_so_far = stats["exact_grids"]
        total_so_far = stats["total_grids"]
        loss_str = f"ttt_loss: {losses[0]:.3f}→{losses[-1]:.3f}" if losses else "no_loss"
        mark = "EXACT" if exact_match else f"cell={cell_acc:.3f}"

        print(f"  [{task_i+1}/{len(tasks)}] {task_result['task_id']}: "
              f"{mark}, {loss_str}, "
              f"exact={exact_so_far}/{total_so_far}, "
              f"{elapsed:.1f}s")

    # Summary
    cell_accuracy = stats["correct_cells"] / max(stats["total_cells"], 1)
    grid_exact_match = stats["exact_grids"] / max(stats["total_grids"], 1)

    print(f"\n{'='*60}")
    print(f"TTT Evaluation Results")
    print(f"{'='*60}")
    print(f"  Grid exact match: {grid_exact_match:.4f} "
          f"({stats['exact_grids']}/{stats['total_grids']})")
    print(f"  Cell accuracy:    {cell_accuracy:.4f}")
    print(f"  Total cells:      {stats['total_cells']}")
    print(f"  TTT steps/task:   {args.ttt_steps}")
    print(f"  TTT LR:           {args.ttt_lr}")
    print(f"  TTA augments:     {args.n_tta_augments}")
    print(f"  Total time:       {time.time() - t0:.0f}s")
    print(f"{'='*60}\n")

    return {
        "grid_exact_match": grid_exact_match,
        "cell_accuracy": cell_accuracy,
        "exact_grids": stats["exact_grids"],
        "total_grids": stats["total_grids"],
        "total_cells": stats["total_cells"],
        "correct_cells": stats["correct_cells"],
        "task_results": stats["task_results"],
    }


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI TTT Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/arc")
    parser.add_argument("--split", type=str, default="eval",
                        choices=["train", "eval"])
    parser.add_argument("--max_tasks", type=int, default=0,
                        help="Max tasks to evaluate (0=all)")

    # TTT hyperparameters
    parser.add_argument("--ttt_steps", type=int, default=100,
                        help="Fine-tuning steps per task")
    parser.add_argument("--ttt_lr", type=float, default=1e-3,
                        help="TTT learning rate")
    parser.add_argument("--ttt_grad_clip", type=float, default=1.0)

    # TTA
    parser.add_argument("--n_tta_augments", type=int, default=32,
                        help="Number of augmented predictions for majority vote")

    # Sequence length override
    parser.add_argument("--max_seq_len", type=int, default=0,
                        help="Override max_seq_len (0=use config value). "
                             "Position embeddings extended if larger than checkpoint.")

    args = parser.parse_args()

    config = FGNConfig.from_yaml(args.config)
    original_seq_len = config.max_seq_len
    if args.max_seq_len > 0:
        config.max_seq_len = args.max_seq_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model (no torch.compile for TTT)
    model = create_arc_model(config, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Handle position embedding resize if max_seq_len increased
    if config.max_seq_len > original_seq_len:
        old_pos_weight = ckpt["model"]["embedding.seq_pos_embed.weight"]
        old_len = old_pos_weight.shape[0]
        new_len = config.max_seq_len
        if new_len > old_len:
            # Extend: keep trained positions, init new ones from normal distribution
            new_weight = torch.randn(new_len, old_pos_weight.shape[1]) * 0.02
            new_weight[:old_len] = old_pos_weight
            ckpt["model"]["embedding.seq_pos_embed.weight"] = new_weight
            print(f"Extended position embeddings: {old_len} → {new_len}")

    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Load tasks
    all_tasks = load_arc_tasks(args.data_dir)
    if args.split == "train":
        tasks = all_tasks.get("train", [])
    else:
        tasks = all_tasks.get("eval", [])

    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]

    print(f"Evaluating {len(tasks)} {args.split} tasks")
    print(f"TTT: {args.ttt_steps} steps, lr={args.ttt_lr}, "
          f"TTA: {args.n_tta_augments} augments")

    # Run TTT evaluation
    results = evaluate_with_ttt(model, tasks, device, config, args)

    # Save results
    out_path = args.checkpoint.replace(".pt", "_ttt_eval.json")
    # Remove loss curves from JSON (too large)
    save_results = {k: v for k, v in results.items()}
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
