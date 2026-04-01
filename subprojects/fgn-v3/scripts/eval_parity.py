"""FGN v3 Phase 2 — Parity Evaluation with OOD Conditions.

Evaluates parity accuracy under:
1. In-distribution: length 40, p(1) = 0.5
2. Length generalization: 60, 80, 100, 150, 200
3. Distribution shift: p(1) = 0.1, 0.3, 0.7, 0.9 at length 40
4. Combined: length 100, p(1) = 0.3

For FGN models, also reports geometric metrics per condition.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model import FGNModel
from fgn.flat_model import FlatTransformerModel
from fgn.tasks.parity import ParityTask


def load_model(config, checkpoint_path, device):
    """Load model from checkpoint."""
    if config.model_type == "flat":
        model = FlatTransformerModel(config).to(device)
    else:
        model = FGNModel(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

    # Truncate pos_embed if needed
    pe_key = "pos_embed.weight"
    if pe_key in state and state[pe_key].shape[0] > config.max_seq_len:
        state[pe_key] = state[pe_key][:config.max_seq_len]

    model.load_state_dict(state)
    return model


def evaluate_parity(model, tokenizer, config, device, bit_length, p_one,
                    n_batches=100, batch_size=16):
    """Evaluate parity accuracy at specific conditions.

    Returns dict with accuracy and geometric metrics.
    """
    model.eval()
    task = ParityTask(
        tokenizer, seq_len=config.max_seq_len,
        bit_length=bit_length, p_one=p_one
    )

    correct = 0
    total = 0
    total_ce = 0.0
    cv_sum = 0.0
    kappa_sum = 0.0

    with torch.no_grad():
        for _ in range(n_batches):
            input_ids, labels, _ = task.generate_batch(batch_size, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = model(input_ids, labels=labels)

            total_ce += result["ce_loss"].item()
            cv_sum += result["metric_cv"].item()
            kappa_sum += result["avg_kappa"].item()

            # Get predictions at supervised positions
            preds = result["logits"].argmax(dim=-1)  # [B, N]

            for b in range(batch_size):
                mask = labels[b] != -100
                if not mask.any():
                    continue

                pred_tokens = preds[b][mask]
                true_tokens = labels[b][mask]

                # For parity, there's exactly 1 supervised token
                total += 1
                if pred_tokens[0] == true_tokens[0]:
                    correct += 1

    n = max(total, 1)
    return {
        "accuracy": correct / n,
        "ce_loss": total_ce / max(n_batches, 1),
        "metric_cv": cv_sum / max(n_batches, 1),
        "avg_kappa": kappa_sum / max(n_batches, 1),
        "n_samples": total,
        "bit_length": bit_length,
        "p_one": p_one,
    }


# Evaluation conditions
EVAL_CONDITIONS = [
    # (name, bit_length, p_one)
    ("In-dist (L=40, p=0.5)", 40, 0.5),
    # Length generalization
    ("Length 60", 60, 0.5),
    ("Length 80", 80, 0.5),
    ("Length 100", 100, 0.5),
    ("Length 150", 150, 0.5),
    ("Length 200", 200, 0.5),
    # Distribution shift
    ("p=0.1 (L=40)", 40, 0.1),
    ("p=0.3 (L=40)", 40, 0.3),
    ("p=0.7 (L=40)", 40, 0.7),
    ("p=0.9 (L=40)", 40, 0.9),
    # Combined
    ("Combined (L=100, p=0.3)", 100, 0.3),
]


def main():
    parser = argparse.ArgumentParser(description="Parity OOD Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_batches", type=int, default=100,
                        help="Batches per condition")
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

    print(f"\n{'='*70}")
    print(f"Parity OOD Evaluation — {config.model_type} ({n_params:,} params)")
    print(f"{'='*70}")
    print(f"{'Condition':<30} {'Acc':>8} {'CE Loss':>10} {'CV':>8} {'|κ|':>8}")
    print(f"{'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

    for name, bl, p in EVAL_CONDITIONS:
        # Skip conditions that exceed seq_len
        max_tokens = bl + 10  # bits + overhead
        if max_tokens > config.max_seq_len:
            print(f"{name:<30} {'SKIP':>8} (exceeds seq_len={config.max_seq_len})")
            continue

        results = evaluate_parity(
            model, tokenizer, config, device,
            bit_length=bl, p_one=p,
            n_batches=args.n_batches, batch_size=args.batch_size,
        )
        print(f"{name:<30} {results['accuracy']:>8.4f} "
              f"{results['ce_loss']:>10.4f} "
              f"{results['metric_cv']:>8.4f} "
              f"{results['avg_kappa']:>8.4f}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
