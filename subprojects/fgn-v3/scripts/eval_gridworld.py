"""FGN v4 — Grid World Evaluation.

Evaluates grid world action prediction accuracy under varying
plan lengths and world complexity (in-distribution and OOD).

Metrics:
- Action accuracy: per-action token prediction accuracy
- Sequence accuracy: fraction of episodes with all actions correct
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model import FGNModel
from fgn.flat_model import FlatTransformerModel
from fgn.model_v4 import FGNv4Model
from fgn.tasks.gridworld import GridWorldTask


def load_model(config, checkpoint_path, device):
    """Load model from checkpoint."""
    if config.model_type == "flat":
        model = FlatTransformerModel(config).to(device)
    elif config.architecture_version == "v4":
        model = FGNv4Model(config).to(device)
    else:
        model = FGNModel(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

    pe_key = "pos_embed.weight"
    if pe_key in state and state[pe_key].shape[0] > config.max_seq_len:
        state[pe_key] = state[pe_key][:config.max_seq_len]

    model.load_state_dict(state)
    return model


def evaluate_gridworld(model, tokenizer, config, device,
                       n_rooms, min_steps, max_steps,
                       min_state_changes=1, n_objects=4,
                       n_batches=50, batch_size=8):
    """Evaluate grid world action prediction accuracy."""
    model.eval()
    task = GridWorldTask(
        tokenizer, seq_len=config.max_seq_len,
        n_rooms=n_rooms, n_objects=n_objects,
        min_steps=min_steps, max_steps=max_steps,
        min_state_changes=min_state_changes,
    )

    total_sequences = 0
    correct_sequences = 0
    total_tokens = 0
    correct_tokens = 0
    total_ce = 0.0
    cv_sum = 0.0
    kappa_sum = 0.0

    with torch.no_grad():
        for _ in range(n_batches):
            input_ids, labels, meta = task.generate_batch(batch_size, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = model(input_ids, labels=labels)

            total_ce += result["ce_loss"].item()
            cv_sum += result["metric_cv"].item()
            kappa_sum += result["avg_kappa"].item()

            preds = result["logits"].argmax(dim=-1)

            for b in range(batch_size):
                mask = labels[b] != -100
                if not mask.any():
                    continue

                pred_tokens = preds[b][mask]
                true_tokens = labels[b][mask]

                total_sequences += 1

                # Per-token accuracy
                matches = (pred_tokens == true_tokens)
                total_tokens += len(true_tokens)
                correct_tokens += matches.sum().item()

                # Full-sequence accuracy
                if matches.all():
                    correct_sequences += 1

    n_seq = max(total_sequences, 1)
    n_tok = max(total_tokens, 1)
    return {
        "seq_acc": correct_sequences / n_seq,
        "token_acc": correct_tokens / n_tok,
        "ce_loss": total_ce / max(n_batches, 1),
        "metric_cv": cv_sum / max(n_batches, 1),
        "avg_kappa": kappa_sum / max(n_batches, 1),
        "n_sequences": total_sequences,
        "n_tokens": total_tokens,
    }


# Evaluation conditions: (name, kwargs)
EVAL_CONDITIONS = [
    # In-distribution (training range)
    ("ID: 5rm 4-7 steps",
     dict(n_rooms=5, min_steps=4, max_steps=7, min_state_changes=1)),
    ("ID: 5rm 5-7 steps",
     dict(n_rooms=5, min_steps=5, max_steps=7, min_state_changes=1)),
    # Near-OOD
    ("Near: 6rm 8-10 steps",
     dict(n_rooms=6, min_steps=8, max_steps=10, min_state_changes=2)),
    ("Near: 6rm 10-12 steps",
     dict(n_rooms=6, min_steps=10, max_steps=12, min_state_changes=2)),
    # Far-OOD
    ("Far: 8rm 13-15 steps",
     dict(n_rooms=8, min_steps=13, max_steps=15, min_state_changes=2)),
    ("Far: 8rm 16-20 steps",
     dict(n_rooms=8, min_steps=16, max_steps=20, min_state_changes=2)),
]


def main():
    parser = argparse.ArgumentParser(description="Grid World Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    config = FGNConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Model: {config.model_type}, arch: {config.architecture_version}")
    if hasattr(config, 'geo_metric_type'):
        print(f"Geo metric: {config.geo_metric_type}")
    print(f"Checkpoint: {args.checkpoint}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = load_model(config, args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    variant = config.model_type
    if config.model_type == "fgn" and config.architecture_version == "v4":
        variant = f"v4-{config.geo_metric_type}"

    print(f"\n{'='*90}")
    print(f"Grid World Navigation — {variant} ({n_params:,} params)")
    print(f"{'='*90}")
    print(f"{'Condition':<24} {'SeqAcc':>8} {'TokAcc':>8} {'CE Loss':>10} "
          f"{'CV':>8} {'|κ|':>8} {'Seqs':>8} {'Tokens':>8}")
    print(f"{'-'*24} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, kwargs in EVAL_CONDITIONS:
        results = evaluate_gridworld(
            model, tokenizer, config, device,
            n_batches=args.n_batches, batch_size=args.batch_size,
            **kwargs,
        )
        print(f"{name:<24} {results['seq_acc']:>8.4f} "
              f"{results['token_acc']:>8.4f} "
              f"{results['ce_loss']:>10.4f} "
              f"{results['metric_cv']:>8.4f} "
              f"{results['avg_kappa']:>8.4f} "
              f"{results['n_sequences']:>8} "
              f"{results['n_tokens']:>8}")

    print(f"{'='*90}")


if __name__ == "__main__":
    main()
