"""FGN v3 — S₅ Permutation Composition Evaluation.

Evaluates permutation composition accuracy under varying supervision sparsity:
1. In-distribution: same sup_every as training
2. Sparser supervision: larger gaps between checkpoints
3. Longer chains: more permutations per sequence (OOD length)

Accuracy = exact match on the 5-element permutation state at each supervised position.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model import FGNModel
from fgn.flat_model import FlatTransformerModel
from fgn.tasks.permutation import PermutationTask


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


def evaluate_permutation(model, tokenizer, config, device,
                         n_perms, sup_every, n_batches=50, batch_size=8):
    """Evaluate permutation composition accuracy at specific conditions.

    Returns dict with accuracy and geometric metrics.
    """
    model.eval()
    task = PermutationTask(
        tokenizer, seq_len=config.max_seq_len,
        min_perms=n_perms, max_perms=n_perms,
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

                # Per-checkpoint accuracy (every 5 consecutive tokens = one permutation state)
                n_sup = pred_tokens.shape[0]
                # Each supervised checkpoint is 5 tokens (the permutation state)
                for start in range(0, n_sup - 4, 5):
                    chunk_pred = pred_tokens[start:start + 5]
                    chunk_true = true_tokens[start:start + 5]
                    total_checkpoints += 1
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
        "n_perms": n_perms,
        "sup_every": sup_every,
    }


# Evaluation conditions: (name, n_perms, sup_every)
EVAL_CONDITIONS = [
    # Supervision sparsity sweep (fixed 50 perms)
    ("50p sup/5", 50, 5),
    ("50p sup/10", 50, 10),
    ("50p sup/25", 50, 25),
    ("50p sup/50 (final)", 50, 50),
    # Longer chains with sparse supervision
    ("30p sup/30 (final)", 30, 30),
    ("40p sup/40 (final)", 40, 40),
    # OOD: more perms than training
    ("60p sup/10", 60, 10),
    ("60p sup/60 (final)", 60, 60),
]


def main():
    parser = argparse.ArgumentParser(description="Permutation Composition OOD Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_sup_every", type=int, default=None,
                        help="If set, only eval conditions matching this training sup_every")
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

    print(f"\n{'='*80}")
    print(f"S₅ Permutation Composition — {config.model_type} ({n_params:,} params)")
    print(f"{'='*80}")
    print(f"{'Condition':<25} {'TokAcc':>8} {'CkptAcc':>8} {'CE Loss':>10} "
          f"{'CV':>8} {'|κ|':>8} {'Tokens':>8} {'Ckpts':>6}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for name, np_, se in EVAL_CONDITIONS:
        results = evaluate_permutation(
            model, tokenizer, config, device,
            n_perms=np_, sup_every=se,
            n_batches=args.n_batches, batch_size=args.batch_size,
        )
        print(f"{name:<25} {results['token_acc']:>8.4f} "
              f"{results['checkpoint_acc']:>8.4f} "
              f"{results['ce_loss']:>10.4f} "
              f"{results['metric_cv']:>8.4f} "
              f"{results['avg_kappa']:>8.4f} "
              f"{results['n_tokens']:>8} "
              f"{results['n_checkpoints']:>6}")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
