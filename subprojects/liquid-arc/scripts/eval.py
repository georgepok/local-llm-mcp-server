"""LiquidARC evaluation script.

Load a checkpoint and evaluate on ARC-AGI eval split.
Reports cell accuracy and transform accuracy.

Usage:
    python scripts/eval.py --checkpoint output/checkpoints/best.pt --data_dir data/arc
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import create_model

FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
sys.path.insert(0, FGN_ROOT)
from fgn.tasks.arc import ARCTask


def evaluate(model, eval_task, device, n_batches, batch_size):
    """Full evaluation: cell accuracy, transform accuracy, CE loss."""
    model.eval()
    total_correct = 0
    total_cells = 0
    total_transform_correct = 0
    total_transform = 0
    total_ce = 0.0
    n_valid = 0

    with torch.no_grad():
        for i in range(n_batches):
            _, _, meta = eval_task.generate_batch(batch_size, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
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
                    grid_ids=meta.get("grid_ids"),
                    lengths=meta.get("lengths"),
                    target_input_colors=meta.get("target_input_colors"),
                )

            # Cell accuracy
            cell_acc = result.get("cell_accuracy", torch.tensor(0.0))
            if isinstance(cell_acc, torch.Tensor):
                cell_acc = cell_acc.item()
            n_tgt = (meta["target_labels"] != -100).sum().item()
            total_correct += int(cell_acc * n_tgt)
            total_cells += n_tgt

            # Transform accuracy
            xform_acc = result.get("transform_accuracy", torch.tensor(0.0))
            if isinstance(xform_acc, torch.Tensor):
                xform_acc = xform_acc.item()
            n_xform = result.get("n_transform", torch.tensor(0))
            if isinstance(n_xform, torch.Tensor):
                n_xform = n_xform.item()
            total_transform_correct += int(xform_acc * n_xform)
            total_transform += n_xform

            total_ce += result["ce_loss"].item()
            n_valid += 1

            if (i + 1) % 10 == 0:
                running_cell = total_correct / max(total_cells, 1)
                running_xform = total_transform_correct / max(total_transform, 1)
                print(f"  [{i+1}/{n_batches}] cell={running_cell:.4f}, "
                      f"xform={running_xform:.4f}")

    cell_acc = total_correct / max(total_cells, 1)
    xform_acc = total_transform_correct / max(total_transform, 1)
    avg_ce = total_ce / max(n_valid, 1)
    return cell_acc, xform_acc, avg_ce


def main():
    parser = argparse.ArgumentParser(description="LiquidARC Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/arc")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_batches", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    if isinstance(config, dict):
        config = LiquidARCConfig(**config)

    model = create_model(config, device)
    model.load_state_dict(ckpt["model"])
    step = ckpt.get("step", "?")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint from step {step}")
    print(f"Model: {config.model_type}, params: {n_params:,}")

    eval_task = ARCTask(
        seq_len=config.max_seq_len,
        data_dir=args.data_dir,
        split="eval",
        augment=False,
    )

    cell_acc, xform_acc, avg_ce = evaluate(
        model, eval_task, device, args.n_batches, args.batch_size)

    print(f"\n{'='*50}")
    print(f"RESULTS (step {step})")
    print(f"  Cell accuracy:      {cell_acc:.4f}")
    print(f"  Transform accuracy: {xform_acc:.4f}")
    print(f"  CE loss:            {avg_ce:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
