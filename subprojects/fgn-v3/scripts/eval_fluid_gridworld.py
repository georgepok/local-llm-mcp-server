"""FluidNet v1 — Continuous Grid World Evaluation.

Evaluates FluidNet, v6-metric, and flat models on continuous gridworld
with PathOptimality and DistanceError metrics in addition to standard
SeqAcc, TokAcc, NavAcc, ManipAcc.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model_fluid import FluidNetModel
from fgn.model_v6 import FGNv6Model
from fgn.flat_model import FlatTransformerModel
from fgn.tasks.continuous_gridworld import ContinuousGridWorldTask


def load_model(config, checkpoint_path, device):
    """Load model from checkpoint."""
    if config.model_type == "flat":
        model = FlatTransformerModel(config).to(device)
    elif config.architecture_version == "fluid":
        model = FluidNetModel(config).to(device)
    elif config.architecture_version == "v6":
        model = FGNv6Model(config).to(device)
    else:
        model = FlatTransformerModel(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

    pe_key = "pos_embed.weight"
    if pe_key in state and state[pe_key].shape[0] > config.max_seq_len:
        state[pe_key] = state[pe_key][:config.max_seq_len]

    model.load_state_dict(state)
    return model


def evaluate_cw(model, tokenizer, config, device,
                n_rooms_min, n_rooms_max,
                space_size=100.0, connect_radius=30.0,
                min_steps=4, max_steps=10,
                min_state_changes=1, n_objects=4,
                n_batches=50, batch_size=8):
    """Evaluate on continuous gridworld."""
    model.eval()
    is_fluid = isinstance(model, FluidNetModel)
    is_v6 = isinstance(model, FGNv6Model)

    task = ContinuousGridWorldTask(
        tokenizer, seq_len=config.max_seq_len,
        n_rooms_min=n_rooms_min, n_rooms_max=n_rooms_max,
        space_size=space_size, connect_radius=connect_radius,
        n_objects=n_objects,
        min_steps=min_steps, max_steps=max_steps,
        min_state_changes=min_state_changes,
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
    t_local_sum = 0.0
    t_medium_sum = 0.0
    t_global_sum = 0.0

    with torch.no_grad():
        for _ in range(n_batches):
            input_ids, labels, meta = task.generate_batch(batch_size, device=device)
            context_mask = meta.get("context_mask")
            action_spans = meta.get("action_spans", [[] for _ in range(batch_size)])

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                if is_fluid or is_v6:
                    result = model(input_ids, labels=labels,
                                   context_mask=context_mask)
                else:
                    result = model(input_ids, labels=labels)

            total_ce += result["ce_loss"].item()

            cv_val = result["metric_cv"]
            if isinstance(cv_val, torch.Tensor):
                cv_val = cv_val.item()
            cv_sum += cv_val

            kappa_sum += result["avg_kappa"].item()

            if is_fluid:
                t_local_sum += result.get("avg_t_local", torch.tensor(0.0)).item()
                t_medium_sum += result.get("avg_t_medium", torch.tensor(0.0)).item()
                t_global_sum += result.get("avg_t_global", torch.tensor(0.0)).item()

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

                spans = action_spans[b] if b < len(action_spans) else []
                supervised_positions = mask.nonzero(as_tuple=True)[0]

                for start, end, action_type in spans:
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
        "avg_t_local": t_local_sum / max(n_batches, 1),
        "avg_t_medium": t_medium_sum / max(n_batches, 1),
        "avg_t_global": t_global_sum / max(n_batches, 1),
        "n_sequences": total_sequences,
        "n_tokens": total_tokens,
        "n_nav_tokens": total_nav_tokens,
        "n_manip_tokens": total_manip_tokens,
    }


EVAL_CONDITIONS = [
    # In-distribution
    ("ID: 10-15rm [0,100] R=30",
     dict(n_rooms_min=10, n_rooms_max=15, space_size=100.0, connect_radius=30.0,
          min_steps=4, max_steps=10, min_state_changes=1)),

    # Near-OOD: more rooms
    ("Near: 15-20rm [0,100] R=30",
     dict(n_rooms_min=15, n_rooms_max=20, space_size=100.0, connect_radius=30.0,
          min_steps=4, max_steps=12, min_state_changes=1)),

    # Near-OOD: bigger space
    ("Near: 10-15rm [0,150] R=40",
     dict(n_rooms_min=10, n_rooms_max=15, space_size=150.0, connect_radius=40.0,
          min_steps=4, max_steps=12, min_state_changes=1)),

    # Far-OOD: both bigger
    ("Far: 20-25rm [0,200] R=50",
     dict(n_rooms_min=20, n_rooms_max=25, space_size=200.0, connect_radius=50.0,
          min_steps=4, max_steps=15, min_state_changes=1)),
]


def main():
    parser = argparse.ArgumentParser(description="FluidNet Continuous GridWorld Eval")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    config = FGNConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Model: {config.model_type}, arch: {config.architecture_version}")
    print(f"Checkpoint: {args.checkpoint}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = load_model(config, args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    variant = config.model_type
    if config.architecture_version == "fluid":
        variant = "FluidNet"
    elif config.model_type == "fgn":
        variant = f"v6-{config.geo_metric_type}"

    print(f"\n{'='*120}")
    print(f"Continuous Grid World — {variant} ({n_params:,} params)")
    print(f"{'='*120}")
    print(f"{'Condition':<28} {'SeqAcc':>8} {'TokAcc':>8} {'NavAcc':>8} "
          f"{'ManipAcc':>8} {'CE Loss':>10} {'CV':>8} "
          f"{'|κ|':>8} {'t_loc':>6} {'t_med':>6} {'t_glo':>6} {'Seqs':>6}")
    print(f"{'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} "
          f"{'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for name, kwargs in EVAL_CONDITIONS:
        results = evaluate_cw(
            model, tokenizer, config, device,
            n_batches=args.n_batches, batch_size=args.batch_size,
            **kwargs,
        )
        print(f"{name:<28} "
              f"{results['seq_acc']:>8.4f} "
              f"{results['token_acc']:>8.4f} "
              f"{results['nav_acc']:>8.4f} "
              f"{results['manip_acc']:>8.4f} "
              f"{results['ce_loss']:>10.4f} "
              f"{results['metric_cv']:>8.4f} "
              f"{results['avg_kappa']:>8.4f} "
              f"{results['avg_t_local']:>6.2f} "
              f"{results['avg_t_medium']:>6.2f} "
              f"{results['avg_t_global']:>6.2f} "
              f"{results['n_sequences']:>6}")

    print(f"{'='*120}")


if __name__ == "__main__":
    main()
