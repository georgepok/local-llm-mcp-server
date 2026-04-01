"""ARC-AGI evaluation script.

Computes:
  - Cell accuracy (per-cell color prediction accuracy)
  - Grid exact match (entire output grid correct)
  - Geometric diagnostics (metric CV, curvature, timescales, grid fidelity)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model_arc import FluidNetARC, FlatTransformerARC
from fgn.model_arc_sandwich import SandwichARC, create_arc_model
from fgn.tasks.arc import ARCTask


def evaluate(model, eval_task, device, n_batches=50, batch_size=8, verbose=True):
    """Full evaluation on ARC eval tasks.

    Refinement happens inside the model in latent space.

    Returns:
        Dict with cell_accuracy, grid_exact_match, ce_loss, etc.
    """
    model.eval()
    is_fluid = isinstance(model, FluidNetARC)
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    is_sandwich = isinstance(m, SandwichARC)

    stats = {
        "total_cells": 0,
        "correct_cells": 0,
        "total_grids": 0,
        "exact_grids": 0,
        "ce_sum": 0.0,
        "n_batches": 0,
        "cv_sum": 0.0,
        "kappa_sum": 0.0,
        "t_local_sum": 0.0,
        "t_medium_sum": 0.0,
        "t_global_sum": 0.0,
        "fidelity_corrs": [],
    }

    with torch.no_grad():
        for batch_i in range(n_batches):
            try:
                _, _, meta = eval_task.generate_batch(batch_size, device=device)
            except RuntimeError:
                continue

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
                    grid_ids=meta["grid_ids"],
                    lengths=meta["lengths"],
                    target_input_colors=meta["target_input_colors"],
                )

            logits = result["logits"]
            preds = logits.argmax(dim=-1)
            target_labels = meta["target_labels"]

            for b in range(batch_size):
                tgt_b = target_labels[b]
                valid = tgt_b != -100
                if valid.sum() == 0:
                    continue
                matches = (preds[b][valid] == tgt_b[valid])
                stats["total_cells"] += valid.sum().item()
                stats["correct_cells"] += matches.sum().item()
                stats["total_grids"] += 1
                if matches.all():
                    stats["exact_grids"] += 1

            stats["ce_sum"] += result["ce_loss"].item()
            stats["n_batches"] += 1

            if is_fluid or is_sandwich:
                cv_val = result["metric_cv"]
                if isinstance(cv_val, torch.Tensor):
                    cv_val = cv_val.item()
                stats["cv_sum"] += cv_val
                stats["kappa_sum"] += result["avg_kappa"].item()
                for key, stat_key in [("avg_t_local", "t_local_sum"),
                                       ("avg_t_medium", "t_medium_sum"),
                                       ("avg_t_global", "t_global_sum")]:
                    if key in result:
                        stats[stat_key] += result[key].item()

    n = max(stats["n_batches"], 1)
    results = {
        "cell_accuracy": stats["correct_cells"] / max(stats["total_cells"], 1),
        "grid_exact_match": stats["exact_grids"] / max(stats["total_grids"], 1),
        "ce_loss": stats["ce_sum"] / n,
        "total_cells": stats["total_cells"],
        "total_grids": stats["total_grids"],
        "exact_grids": stats["exact_grids"],
    }

    if is_fluid or is_sandwich:
        results["metric_cv"] = stats["cv_sum"] / n
        results["avg_kappa"] = stats["kappa_sum"] / n
        results["avg_t_local"] = stats["t_local_sum"] / n
        results["avg_t_medium"] = stats["t_medium_sum"] / n
        results["avg_t_global"] = stats["t_global_sum"] / n
        if stats["fidelity_corrs"]:
            results["grid_fidelity_rho"] = float(np.mean(stats["fidelity_corrs"]))
        else:
            results["grid_fidelity_rho"] = 0.0

    if verbose:
        print(f"\n{'='*50}")
        print(f"ARC Evaluation Results")
        print(f"{'='*50}")
        print(f"  Cell accuracy:    {results['cell_accuracy']:.4f}")
        print(f"  Grid exact match: {results['grid_exact_match']:.4f} "
              f"({results['exact_grids']}/{results['total_grids']})")
        print(f"  CE loss:          {results['ce_loss']:.4f}")
        print(f"  Total cells:      {results['total_cells']}")
        if is_fluid or is_sandwich:
            print(f"  Metric CV:        {results['metric_cv']:.4f}")
            print(f"  Avg |kappa|:      {results['avg_kappa']:.4f}")
            print(f"  Timescales:       [{results['avg_t_local']:.3f}, "
                  f"{results['avg_t_medium']:.3f}, {results['avg_t_global']:.3f}]")
            print(f"  Grid fidelity rho: {results['grid_fidelity_rho']:.4f}")
        print(f"{'='*50}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI Evaluation")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--data_dir", type=str, default="data/arc",
                        help="Path to ARC-AGI data directory")
    parser.add_argument("--split", type=str, default="eval",
                        choices=["train", "eval"],
                        help="Which split to evaluate on")
    parser.add_argument("--n_batches", type=int, default=50,
                        help="Number of evaluation batches")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    config = FGNConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = create_arc_model(config, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Load eval task
    eval_task = ARCTask(
        seq_len=config.max_seq_len,
        data_dir=args.data_dir,
        split=args.split,
        augment=False,
    )

    # Run evaluation
    results = evaluate(model, eval_task, device,
                      n_batches=args.n_batches,
                      batch_size=args.batch_size)

    # Save results
    import json
    out_path = args.checkpoint.replace(".pt", "_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
