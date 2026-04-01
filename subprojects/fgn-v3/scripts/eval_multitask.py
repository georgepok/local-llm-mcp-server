"""FGN v3 Phase 1b — Multi-task evaluation script.

Evaluates a trained model (FGN or flat) on all 4 tasks:
  - Per-task CE loss
  - Per-task accuracy (exact match on answer tokens)
  - Per-task curvature patterns (FGN only)
  - Metric CV per task (FGN only)
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model import FGNModel
from fgn.flat_model import FlatTransformerModel
from fgn.tasks import get_task


def load_model(config, checkpoint_path, device):
    """Load model from checkpoint."""
    if config.model_type == "flat":
        model = FlatTransformerModel(config).to(device)
    else:
        model = FGNModel(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    return model


def evaluate_task(model, task, config, device, n_batches=50, batch_size=8):
    """Evaluate model on a single task."""
    model.eval()
    total_ce = 0.0
    total_correct = 0
    total_answer_tokens = 0
    total_metric_cv = 0.0
    total_avg_kappa = 0.0

    with torch.no_grad():
        for _ in range(n_batches):
            input_ids, labels, _ = task.generate_batch(batch_size, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = model(input_ids, labels=labels)

            total_ce += result["ce_loss"].item()
            total_metric_cv += result["metric_cv"].item()
            total_avg_kappa += result["avg_kappa"].item()

            # Accuracy: exact match on supervised positions
            logits = result["logits"]
            preds = logits.argmax(dim=-1)  # [B, N]
            mask = labels != -100
            if mask.any():
                correct = (preds[mask] == labels[mask]).sum().item()
                total_correct += correct
                total_answer_tokens += mask.sum().item()

    n = n_batches
    results = {
        "ce_loss": total_ce / n,
        "accuracy": total_correct / max(total_answer_tokens, 1),
        "metric_cv": total_metric_cv / n,
        "avg_kappa": total_avg_kappa / n,
        "n_answer_tokens": total_answer_tokens,
    }

    # Curvature analysis for FGN models
    if config.model_type == "fgn" and hasattr(model, "get_curvatures"):
        # Run one batch to get curvature tensors
        input_ids, labels, _ = task.generate_batch(batch_size, device=device)
        with torch.no_grad():
            _ = model(input_ids, labels=labels)
        curvatures = model.get_curvatures()
        if curvatures:
            # Per-layer curvature stats
            layer_stats = []
            for i, kappa in enumerate(curvatures):
                layer_stats.append({
                    "layer": i,
                    "mean_abs": kappa.abs().mean().item(),
                    "std": kappa.std().item(),
                    "max_abs": kappa.abs().max().item(),
                })
            results["curvature_layers"] = layer_stats

    return results


def main():
    parser = argparse.ArgumentParser(description="FGN v3 Multi-task Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default=None,
                        help="Override config.model_type")
    parser.add_argument("--tasks", type=str, default="A,B,C,D")
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    config = FGNConfig.from_yaml(args.config)
    if args.model_type:
        config.model_type = args.model_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {config.model_type}")
    print(f"Checkpoint: {args.checkpoint}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = load_model(config, args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    task_names = args.tasks.split(",")

    print(f"\n{'='*70}")
    print(f"{'Task':<6} {'CE Loss':>10} {'Accuracy':>10} {'Metric CV':>10} {'|kappa|':>10}")
    print(f"{'='*70}")

    all_results = {}
    for task_name in task_names:
        task = get_task(task_name, tokenizer, seq_len=config.max_seq_len)
        results = evaluate_task(model, task, config, device,
                              n_batches=args.n_batches, batch_size=args.batch_size)
        all_results[task_name] = results

        print(f"{task_name:<6} {results['ce_loss']:>10.4f} {results['accuracy']:>10.4f} "
              f"{results['metric_cv']:>10.4f} {results['avg_kappa']:>10.4f}")

        if "curvature_layers" in results:
            for ls in results["curvature_layers"]:
                print(f"  L{ls['layer']}: |kappa|={ls['mean_abs']:.4f}, "
                      f"std={ls['std']:.4f}, max={ls['max_abs']:.4f}")

    # Summary
    avg_ce = sum(r["ce_loss"] for r in all_results.values()) / len(all_results)
    avg_acc = sum(r["accuracy"] for r in all_results.values()) / len(all_results)
    print(f"{'='*70}")
    print(f"{'AVG':<6} {avg_ce:>10.4f} {avg_acc:>10.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
