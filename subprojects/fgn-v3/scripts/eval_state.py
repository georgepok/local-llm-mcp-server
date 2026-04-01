"""FGN v3 Phase 2 — State Tracking Evaluation with OOD Conditions.

Evaluates cumulative state tracking accuracy under:
1. In-distribution: 60 events, 50% distractors
2. Length scaling: 120, 180, 240 events (same distractor ratio)
3. Distractor scaling: 70%, 80%, 90% distractors (same event count)
4. Combined: 180 events, 80% distractors

Accuracy = exact match on the number token after each "?" marker.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model import FGNModel
from fgn.flat_model import FlatTransformerModel
from fgn.tasks.state_tracking import StateTrackingTask


def load_model(config, checkpoint_path, device):
    """Load model from checkpoint."""
    if config.model_type == "flat":
        model = FlatTransformerModel(config).to(device)
    else:
        model = FGNModel(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

    pe_key = "pos_embed.weight"
    if pe_key in state and state[pe_key].shape[0] > config.max_seq_len:
        state[pe_key] = state[pe_key][:config.max_seq_len]

    model.load_state_dict(state)
    return model


def evaluate_state_tracking(model, tokenizer, config, device, n_events,
                            distractor_ratio, n_queries=4,
                            n_batches=100, batch_size=16):
    """Evaluate state tracking accuracy at specific conditions.

    Returns dict with per-query accuracy and geometric metrics.
    """
    model.eval()
    task = StateTrackingTask(
        tokenizer, seq_len=config.max_seq_len,
        n_events=n_events, distractor_ratio=distractor_ratio,
        n_queries=n_queries,
    )

    total_queries = 0
    correct_queries = 0
    total_ce = 0.0
    cv_sum = 0.0
    kappa_sum = 0.0
    overflow_count = 0

    with torch.no_grad():
        for _ in range(n_batches):
            input_ids, labels, meta = task.generate_batch(batch_size, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = model(input_ids, labels=labels)

            total_ce += result["ce_loss"].item()
            cv_sum += result["metric_cv"].item()
            kappa_sum += result["avg_kappa"].item()

            # Per-token accuracy at supervised positions
            preds = result["logits"].argmax(dim=-1)

            for b in range(batch_size):
                mask = labels[b] != -100
                if not mask.any():
                    overflow_count += 1
                    continue

                pred_tokens = preds[b][mask]
                true_tokens = labels[b][mask]

                for p, t in zip(pred_tokens, true_tokens):
                    total_queries += 1
                    if p == t:
                        correct_queries += 1

    n = max(total_queries, 1)
    return {
        "accuracy": correct_queries / n,
        "ce_loss": total_ce / max(n_batches, 1),
        "metric_cv": cv_sum / max(n_batches, 1),
        "avg_kappa": kappa_sum / max(n_batches, 1),
        "n_queries": total_queries,
        "overflow": overflow_count,
        "n_events": n_events,
        "distractor_ratio": distractor_ratio,
    }


# Short-seq eval conditions (seq_len <= 512, training events ~60)
EVAL_CONDITIONS_SHORT = [
    ("In-dist (60 events, 50%)", 60, 0.5),
    ("120 events, 50%", 120, 0.5),
    ("180 events, 50%", 180, 0.5),
    ("240 events, 50%", 240, 0.5),
    ("60 events, 70%", 60, 0.7),
    ("60 events, 80%", 60, 0.8),
    ("60 events, 90%", 60, 0.9),
    ("180 events, 80%", 180, 0.8),
]

# Long-seq eval conditions (seq_len=5120, training events ~1433)
EVAL_CONDITIONS_LONG = [
    # In-distribution
    ("In-dist (1433 ev, 50%)", 1433, 0.5),
    # Length scaling (OOD: more events → more chained ops, some truncation)
    ("1800 events, 50%", 1800, 0.5),
    ("2000 events, 50%", 2000, 0.5),
    ("2500 events, 50%", 2500, 0.5),
    # Distractor scaling (OOD: same events, fewer ops visible)
    ("1433 events, 70%", 1433, 0.7),
    ("1433 events, 80%", 1433, 0.8),
    ("1433 events, 90%", 1433, 0.9),
    # Combined stress
    ("2000 events, 80%", 2000, 0.8),
]


def main():
    parser = argparse.ArgumentParser(description="State Tracking OOD Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    config = FGNConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Model: {config.model_type}")
    print(f"Checkpoint: {args.checkpoint}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = load_model(config, args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"max_seq_len: {config.max_seq_len}")

    print(f"\n{'='*75}")
    print(f"State Tracking OOD Evaluation — {config.model_type} ({n_params:,} params)")
    print(f"{'='*75}")
    print(f"{'Condition':<30} {'Acc':>8} {'CE Loss':>10} {'CV':>8} "
          f"{'|κ|':>8} {'Queries':>8} {'OFlow':>6}")
    print(f"{'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    conditions = (EVAL_CONDITIONS_LONG if config.max_seq_len >= 2048
                  else EVAL_CONDITIONS_SHORT)
    print(f"Using {'long' if config.max_seq_len >= 2048 else 'short'}-seq conditions")

    for name, ne, dr in conditions:
        results = evaluate_state_tracking(
            model, tokenizer, config, device,
            n_events=ne, distractor_ratio=dr,
            n_batches=args.n_batches, batch_size=args.batch_size,
        )
        print(f"{name:<30} {results['accuracy']:>8.4f} "
              f"{results['ce_loss']:>10.4f} "
              f"{results['metric_cv']:>8.4f} "
              f"{results['avg_kappa']:>8.4f} "
              f"{results['n_queries']:>8} "
              f"{results['overflow']:>6}")

    print(f"{'='*75}")


if __name__ == "__main__":
    main()
