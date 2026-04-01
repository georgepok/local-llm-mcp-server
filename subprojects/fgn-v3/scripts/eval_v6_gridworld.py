"""FGN v6 — Grid World Evaluation with correct nav/manip from action_spans.

Uses action_spans metadata from GridWorldTask to correctly classify
navigation vs manipulation accuracy (fixing v5's broken token heuristic).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model_v6 import FGNv6Model
from fgn.flat_model import FlatTransformerModel
from fgn.tasks.gridworld import GridWorldTask


def load_model(config, checkpoint_path, device):
    """Load model from checkpoint."""
    if config.model_type == "flat":
        model = FlatTransformerModel(config).to(device)
    elif config.architecture_version == "v6":
        model = FGNv6Model(config).to(device)
    elif config.architecture_version == "v5":
        from fgn.model_v5 import FGNv5Model
        model = FGNv5Model(config).to(device)
    else:
        from fgn.model_v4 import FGNv4Model
        model = FGNv4Model(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

    pe_key = "pos_embed.weight"
    if pe_key in state and state[pe_key].shape[0] > config.max_seq_len:
        state[pe_key] = state[pe_key][:config.max_seq_len]

    model.load_state_dict(state)
    return model


def evaluate_gridworld(model, tokenizer, config, device,
                       n_rooms_min, n_rooms_max,
                       min_steps, max_steps,
                       min_state_changes=2, n_objects=6,
                       n_batches=50, batch_size=8):
    """Evaluate grid world with nav/manip accuracy from action_spans."""
    model.eval()
    is_v6 = isinstance(model, FGNv6Model)

    task = GridWorldTask(
        tokenizer, seq_len=config.max_seq_len,
        n_rooms_min=n_rooms_min, n_rooms_max=n_rooms_max,
        n_objects=n_objects,
        min_steps=min_steps, max_steps=max_steps,
        min_state_changes=min_state_changes,
        randomize_topology=True,
    )

    total_sequences = 0
    correct_sequences = 0
    total_tokens = 0
    correct_tokens = 0
    total_nav_tokens = 0
    correct_nav_tokens = 0
    total_manip_tokens = 0
    correct_manip_tokens = 0
    total_ce = 0.0
    cv_sum = 0.0
    kappa_sum = 0.0
    esc_rate_sum = 0.0
    entropy_sum = 0.0

    with torch.no_grad():
        for _ in range(n_batches):
            input_ids, labels, meta = task.generate_batch(batch_size, device=device)

            context_mask = meta.get("context_mask")
            action_spans = meta.get("action_spans", [[] for _ in range(batch_size)])

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                if is_v6:
                    result = model(input_ids, labels=labels,
                                   context_mask=context_mask)
                else:
                    result = model(input_ids, labels=labels)

            total_ce += result["ce_loss"].item()
            cv_sum += result["metric_cv"].item()
            kappa_sum += result["avg_kappa"].item()
            if "escalation_rate" in result:
                esc_rate_sum += result["escalation_rate"].item()
                entropy_sum += result["avg_entropy"].item()

            preds = result["logits"].argmax(dim=-1)

            for b in range(batch_size):
                mask = labels[b] != -100
                if not mask.any():
                    continue

                total_sequences += 1

                pred_tokens = preds[b][mask]
                true_tokens = labels[b][mask]

                matches = (pred_tokens == true_tokens)
                total_tokens += len(true_tokens)
                correct_tokens += matches.sum().item()

                if matches.all():
                    correct_sequences += 1

                # Use action_spans for correct nav/manip classification
                spans = action_spans[b] if b < len(action_spans) else []
                supervised_positions = mask.nonzero(as_tuple=True)[0]

                for start, end, action_type in spans:
                    # Find supervised positions within this span
                    span_mask = (supervised_positions >= start) & (supervised_positions < end)
                    span_indices = span_mask.nonzero(as_tuple=True)[0]

                    for idx in span_indices:
                        if action_type == "nav":
                            total_nav_tokens += 1
                            if matches[idx]:
                                correct_nav_tokens += 1
                        else:
                            total_manip_tokens += 1
                            if matches[idx]:
                                correct_manip_tokens += 1

    n_seq = max(total_sequences, 1)
    return {
        "seq_acc": correct_sequences / n_seq,
        "token_acc": correct_tokens / max(total_tokens, 1),
        "nav_acc": correct_nav_tokens / max(total_nav_tokens, 1),
        "manip_acc": correct_manip_tokens / max(total_manip_tokens, 1),
        "ce_loss": total_ce / max(n_batches, 1),
        "metric_cv": cv_sum / max(n_batches, 1),
        "avg_kappa": kappa_sum / max(n_batches, 1),
        "esc_rate": esc_rate_sum / max(n_batches, 1),
        "avg_entropy": entropy_sum / max(n_batches, 1),
        "n_sequences": total_sequences,
        "n_tokens": total_tokens,
        "n_nav_tokens": total_nav_tokens,
        "n_manip_tokens": total_manip_tokens,
    }


EVAL_CONDITIONS = [
    # In-distribution
    ("ID: 8-12rm 8-15st",
     dict(n_rooms_min=8, n_rooms_max=12, min_steps=8, max_steps=15,
          min_state_changes=2)),

    # Near-OOD: same world, fewer state changes (easier)
    ("Near: 8-12rm 8-15st 4sc",
     dict(n_rooms_min=8, n_rooms_max=12, min_steps=8, max_steps=15,
          min_state_changes=4)),

    # Near-OOD: longer plans (same world size)
    ("Near: 8-12rm 12-17st",
     dict(n_rooms_min=8, n_rooms_max=12, min_steps=12, max_steps=17,
          min_state_changes=2)),

    # Near-OOD: bigger world (same plan length)
    ("Near: 13-15rm 8-15st",
     dict(n_rooms_min=13, n_rooms_max=15, min_steps=8, max_steps=15,
          min_state_changes=2)),

    # Far-OOD: bigger + longer
    ("Far: 15-18rm 12-17st",
     dict(n_rooms_min=15, n_rooms_max=18, min_steps=12, max_steps=17,
          min_state_changes=2)),

    # Far-OOD: biggest + longest achievable
    ("Far: 15-18rm 15-20st",
     dict(n_rooms_min=15, n_rooms_max=18, min_steps=15, max_steps=20,
          min_state_changes=2)),
]


def main():
    parser = argparse.ArgumentParser(description="Grid World v6 Evaluation")
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
    if config.model_type == "fgn":
        variant = f"v6-{config.geo_metric_type}"
        if hasattr(config, 'attention_budgets'):
            budgets = config.attention_budgets
            print(f"Budgets: [{', '.join(f'{b:.2f}' for b in budgets)}]")

    print(f"\n{'='*110}")
    print(f"Grid World Navigation — {variant} ({n_params:,} params)")
    print(f"{'='*110}")
    print(f"{'Condition':<26} {'SeqAcc':>8} {'TokAcc':>8} {'NavAcc':>8} "
          f"{'ManipAcc':>8} {'CE Loss':>10} {'EscRate':>8} {'CV':>8} "
          f"{'|κ|':>8} {'Seqs':>6} {'Tok':>8}")
    print(f"{'-'*26} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} "
          f"{'-'*8} {'-'*8} {'-'*6} {'-'*8}")

    for name, kwargs in EVAL_CONDITIONS:
        results = evaluate_gridworld(
            model, tokenizer, config, device,
            n_batches=args.n_batches, batch_size=args.batch_size,
            **kwargs,
        )
        print(f"{name:<26} "
              f"{results['seq_acc']:>8.4f} "
              f"{results['token_acc']:>8.4f} "
              f"{results['nav_acc']:>8.4f} "
              f"{results['manip_acc']:>8.4f} "
              f"{results['ce_loss']:>10.4f} "
              f"{results['esc_rate']:>8.4f} "
              f"{results['metric_cv']:>8.4f} "
              f"{results['avg_kappa']:>8.4f} "
              f"{results['n_sequences']:>6} "
              f"{results['n_tokens']:>8}")

    print(f"{'='*110}")


if __name__ == "__main__":
    main()
