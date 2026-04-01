#!/usr/bin/env python3
"""Test generalization: evaluate checkpoints on FRESH (unseen) episodes.

Compares curved vs flat geometry checkpoints to see if curvature
helps generalization or is purely penalty-driven artifact.
"""

import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fgn.config import FGNConfig
from fgn.model_fluid import FluidNetModel as FluidNet
from fgn.tasks import get_task


def _get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


def generate_fresh_episodes(task, n_episodes, seq_len, seed):
    """Generate fresh episodes — same logic as train_grokking.generate_fixed_dataset."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)

    pad_id = task.tokenizer.eos_token_id or 0
    dataset = []

    for i in range(n_episodes):
        for _retry in range(200):
            ep_result = task._generate_valid_episode()
            if ep_result is None:
                continue
            episode_text, _actions, n_steps, optimal_cost, step_costs, world = ep_result
            input_ids, labels, context_end_pos, action_spans, room_token_pos = \
                task._tokenize_episode(episode_text)

            if len(input_ids) > seq_len:
                input_ids = input_ids[:seq_len]
                labels = labels[:seq_len]
            else:
                pad_len = seq_len - len(input_ids)
                input_ids += [pad_id] * pad_len
                labels += [-100] * pad_len

            n_supervised = sum(1 for l in labels if l != -100)
            if n_supervised >= 5:
                break
        else:
            continue

        context_mask = [False] * seq_len
        for j in range(min(context_end_pos, seq_len)):
            context_mask[j] = True

        dataset.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "context_mask": torch.tensor(context_mask, dtype=torch.bool),
        })

    return dataset


def evaluate_checkpoint(ckpt_path, config, dataset, device):
    """Load checkpoint and evaluate CE on fresh episodes."""
    model = FluidNet(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_key = "model" if "model" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[state_key])
    model.eval()

    total_ce = 0.0
    total_tokens = 0
    per_episode_ce = []
    result = None

    with torch.no_grad():
        for ep in dataset:
            input_ids = ep["input_ids"].unsqueeze(0).to(device)
            labels = ep["labels"].unsqueeze(0).to(device)

            result = model(input_ids)
            logits = result["logits"]

            # CE on supervised tokens only
            logits_flat = logits[:, :-1].contiguous().view(-1, logits.size(-1))
            labels_flat = labels[:, 1:].contiguous().view(-1)

            mask = labels_flat != -100
            if mask.sum() == 0:
                continue

            ce = F.cross_entropy(logits_flat[mask], labels_flat[mask])
            n_tok = mask.sum().item()

            total_ce += ce.item() * n_tok
            total_tokens += n_tok
            per_episode_ce.append(ce.item())

    avg_ce = total_ce / max(total_tokens, 1)

    kappa_mean = 0.0
    cv_mean = 0.0
    if result is not None:
        if "kappa" in result:
            kappa_mean = result["kappa"].abs().mean().item()
        if "metric_cv" in result:
            v = result["metric_cv"]
            cv_mean = v.item() if v.dim() == 0 else v.mean().item()

    return {
        "avg_ce": avg_ce,
        "total_tokens": total_tokens,
        "n_episodes": len(per_episode_ce),
        "per_episode_ce": per_episode_ce,
        "kappa": kappa_mean,
        "cv": cv_mean,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--n_episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument("--task_kwargs", type=str, default="{}")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = FGNConfig.from_yaml(args.config)
    task_kwargs = json.loads(args.task_kwargs)
    tokenizer = _get_tokenizer()
    task = get_task("CW", tokenizer, seq_len=config.max_seq_len, **task_kwargs)

    print(f"Generating {args.n_episodes} FRESH episodes (seed={args.seed})...")
    ds1 = generate_fresh_episodes(task, args.n_episodes, config.max_seq_len, args.seed)
    print(f"  Generated {len(ds1)} episodes")

    ds2 = generate_fresh_episodes(task, args.n_episodes, config.max_seq_len, args.seed + 1)
    print(f"  Generated {len(ds2)} episodes (seed2)")

    print()
    print("=" * 70)
    print(f"  Generalization Test — {len(ds1)} fresh episodes")
    print("=" * 70)

    for ckpt_name in args.checkpoints:
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"\n  {ckpt_name}: NOT FOUND, skipping")
            continue

        r1 = evaluate_checkpoint(ckpt_path, config, ds1, device)
        r2 = evaluate_checkpoint(ckpt_path, config, ds2, device)

        avg = (r1["avg_ce"] + r2["avg_ce"]) / 2
        spread = abs(r1["avg_ce"] - r2["avg_ce"])

        ces = sorted(r1["per_episode_ce"])
        p25 = ces[len(ces) // 4] if ces else 0
        p50 = ces[len(ces) // 2] if ces else 0
        p75 = ces[3 * len(ces) // 4] if ces else 0

        print(f"\n  {ckpt_name}:")
        print(f"    |κ|={r1['kappa']:.4f}  CV={r1['cv']:.4f}")
        print(f"    CE (seed1): {r1['avg_ce']:.4f}  ({r1['total_tokens']} tokens, {r1['n_episodes']} eps)")
        print(f"    CE (seed2): {r2['avg_ce']:.4f}  ({r2['total_tokens']} tokens, {r2['n_episodes']} eps)")
        print(f"    CE (avg):   {avg:.4f}  (±{spread:.4f})")
        print(f"    Percentiles: p25={p25:.4f}  p50={p50:.4f}  p75={p75:.4f}")

    print()
    print("=" * 70)
    print("  If curved checkpoints generalize better (lower CE),")
    print("  geometry is functional. If equal, it's penalty artifact.")
    print("=" * 70)


if __name__ == "__main__":
    main()
