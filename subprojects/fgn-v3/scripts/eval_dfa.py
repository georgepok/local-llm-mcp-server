"""FGN v4 — Random DFA Evaluation.

Evaluates Random DFA sequential state propagation accuracy under
varying chain lengths (in-distribution and OOD).

Accuracy = exact match on final state token(s).
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
from fgn.tasks.random_dfa import RandomDFATask


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


def evaluate_dfa(model, tokenizer, config, device,
                 n_steps, n_states=512, n_symbols=16,
                 n_batches=50, batch_size=8):
    """Evaluate DFA state propagation accuracy.

    Returns dict with accuracy and geometric metrics.
    """
    model.eval()
    task = RandomDFATask(
        tokenizer, seq_len=config.max_seq_len,
        n_states=n_states, n_symbols=n_symbols,
        min_steps=n_steps, max_steps=n_steps,
        seed=42,
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
                for p, t in zip(pred_tokens, true_tokens):
                    total_tokens += 1
                    if p == t:
                        correct_tokens += 1

                # Exact sequence match (all answer tokens correct)
                if torch.equal(pred_tokens, true_tokens):
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
        "n_steps": n_steps,
    }


# Evaluation conditions: (name, n_steps)
# Train on 20-50 steps, evaluate across range
EVAL_CONDITIONS = [
    # In-distribution
    ("20 steps", 20),
    ("30 steps", 30),
    ("40 steps", 40),
    ("50 steps", 50),
    # Near-OOD
    ("60 steps", 60),
    ("75 steps", 75),
    # Far-OOD
    ("100 steps", 100),
    ("125 steps", 125),
    ("150 steps", 150),
]


def main():
    parser = argparse.ArgumentParser(description="Random DFA Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_states", type=int, default=512)
    parser.add_argument("--n_symbols", type=int, default=16)
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

    print(f"\n{'='*85}")
    print(f"Random DFA (K={args.n_states}, A={args.n_symbols}) — {variant} ({n_params:,} params)")
    print(f"{'='*85}")
    print(f"{'Condition':<16} {'SeqAcc':>8} {'TokAcc':>8} {'CE Loss':>10} "
          f"{'CV':>8} {'|κ|':>8} {'Seqs':>8} {'Tokens':>8}")
    print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, n_steps in EVAL_CONDITIONS:
        results = evaluate_dfa(
            model, tokenizer, config, device,
            n_steps=n_steps,
            n_states=args.n_states, n_symbols=args.n_symbols,
            n_batches=args.n_batches, batch_size=args.batch_size,
        )
        print(f"{name:<16} {results['seq_acc']:>8.4f} "
              f"{results['token_acc']:>8.4f} "
              f"{results['ce_loss']:>10.4f} "
              f"{results['metric_cv']:>8.4f} "
              f"{results['avg_kappa']:>8.4f} "
              f"{results['n_sequences']:>8} "
              f"{results['n_tokens']:>8}")

    print(f"{'='*85}")


if __name__ == "__main__":
    main()
