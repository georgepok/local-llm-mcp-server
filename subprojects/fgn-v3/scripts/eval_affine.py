"""FGN v3 — Affine Group Composition Evaluation.

Evaluates affine group (Z₉₇) composition accuracy under varying conditions:
1. Supervision sparsity: how many ops between supervised checkpoints
2. Chain length: number of operations per sequence
3. OOD: longer chains or sparser supervision than training

Accuracy = exact match on both components (a, b) of the affine state at
each supervised position.
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
from fgn.tasks.affine import AffineGroupTask


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


def evaluate_affine(model, tokenizer, config, device,
                    n_ops, sup_every, n_batches=50, batch_size=8):
    """Evaluate affine group composition accuracy.

    Returns dict with per-checkpoint accuracy and geometric metrics.
    """
    model.eval()
    task = AffineGroupTask(
        tokenizer, seq_len=config.max_seq_len,
        min_ops=n_ops, max_ops=n_ops,
        sup_every=sup_every,
    )

    total_tokens = 0
    correct_tokens = 0
    total_checkpoints = 0
    correct_checkpoints = 0
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

                # Per-token accuracy
                for p, t in zip(pred_tokens, true_tokens):
                    total_tokens += 1
                    if p == t:
                        correct_tokens += 1

                # Per-checkpoint accuracy
                # Each checkpoint is the state (a, b) — variable number of tokens
                # We check consecutive supervised chunks
                # The state "a b" tokenizes to ~2-4 tokens depending on values
                # Group by checkpoint: supervised tokens between gaps
                sup_positions = mask.nonzero(as_tuple=True)[0]
                if len(sup_positions) == 0:
                    continue

                # Find checkpoint boundaries (gaps > 1 between supervised positions)
                chunks = []
                chunk_start = 0
                for i in range(1, len(sup_positions)):
                    if sup_positions[i] - sup_positions[i - 1] > 1:
                        chunks.append((chunk_start, i))
                        chunk_start = i
                chunks.append((chunk_start, len(sup_positions)))

                for start, end in chunks:
                    total_checkpoints += 1
                    chunk_pred = pred_tokens[start:end]
                    chunk_true = true_tokens[start:end]
                    if torch.equal(chunk_pred, chunk_true):
                        correct_checkpoints += 1

    n_tok = max(total_tokens, 1)
    n_ckpt = max(total_checkpoints, 1)
    return {
        "token_acc": correct_tokens / n_tok,
        "checkpoint_acc": correct_checkpoints / n_ckpt,
        "ce_loss": total_ce / max(n_batches, 1),
        "metric_cv": cv_sum / max(n_batches, 1),
        "avg_kappa": kappa_sum / max(n_batches, 1),
        "n_tokens": total_tokens,
        "n_checkpoints": total_checkpoints,
        "n_ops": n_ops,
        "sup_every": sup_every,
    }


# Evaluation conditions: (name, n_ops, sup_every)
# Designed for length generalization: model trained on 10-20 ops,
# evaluated on 10-120 ops (all final-answer only).
EVAL_CONDITIONS = [
    # In-distribution (training range)
    ("10 ops (final)", 10, 10),
    ("15 ops (final)", 15, 15),
    ("20 ops (final)", 20, 20),
    # Near-OOD
    ("25 ops (final)", 25, 25),
    ("30 ops (final)", 30, 30),
    # Far-OOD
    ("50 ops (final)", 50, 50),
    ("75 ops (final)", 75, 75),
    ("100 ops (final)", 100, 100),
    ("120 ops (final)", 120, 120),
    # With intermediate supervision (tests per-step accuracy)
    ("50 ops sup/10", 50, 10),
    ("100 ops sup/10", 100, 10),
]


def main():
    parser = argparse.ArgumentParser(description="Affine Group Composition Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
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

    print(f"\n{'='*85}")
    print(f"Affine Group Aff(Z₉₇) Composition — {config.model_type} ({n_params:,} params)")
    print(f"{'='*85}")
    print(f"{'Condition':<28} {'TokAcc':>8} {'CkptAcc':>8} {'CE Loss':>10} "
          f"{'CV':>8} {'|κ|':>8} {'Tokens':>8} {'Ckpts':>6}")
    print(f"{'-'*28} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for name, n_ops, se in EVAL_CONDITIONS:
        results = evaluate_affine(
            model, tokenizer, config, device,
            n_ops=n_ops, sup_every=se,
            n_batches=args.n_batches, batch_size=args.batch_size,
        )
        print(f"{name:<28} {results['token_acc']:>8.4f} "
              f"{results['checkpoint_acc']:>8.4f} "
              f"{results['ce_loss']:>10.4f} "
              f"{results['metric_cv']:>8.4f} "
              f"{results['avg_kappa']:>8.4f} "
              f"{results['n_tokens']:>8} "
              f"{results['n_checkpoints']:>6}")

    print(f"{'='*85}")


if __name__ == "__main__":
    main()
