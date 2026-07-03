"""Evaluate a TSP (temporal shortest-path) checkpoint on held-out batches.

Reports:
  - Overall accuracy
  - Non-unreach accuracy (excludes the degenerate "always predict unreach" shortcut)
  - Degeneracy score (max prediction fraction for any single bucket)
  - Prediction entropy (Shannon, base 2, over bucket prediction distribution)
  - Per-bucket accuracy (1, 2, 3, 4, 5, 6+, unreach)
  - CV diagnostics from the Liquid substrate
  - Confusion matrix summary (off-by-one count, large errors)

Supports arbitrary graph file via --graph_file, multiple eval seeds via --seed.
Optional --json_out writes a structured result row for aggregation across runs.

Usage:
    python scripts/eval_tsp.py \\
        --config configs/tr_liquid_n512.yaml \\
        --checkpoint output_liquid_final/stage2_mixed/checkpoints/final.pt \\
        --n_batches 50 --batch_size 14 \\
        --graph_file /workspace/fgn-v3/data/real_graphs/email-Eu-core-temporal.txt \\
        --seed 0 --json_out results.jsonl
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.flat_model import FlatTransformerModel
from fgn.model import FGNModel
try:
    from fgn.liquid_model import LiquidSequenceModel
except Exception:
    LiquidSequenceModel = None
from fgn.tasks import get_task


def _tokenizer():
    from transformers import GPT2Tokenizer
    t = GPT2Tokenizer.from_pretrained("gpt2")
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


def load_model(config, ckpt_path, device):
    if config.model_type == "flat":
        model = FlatTransformerModel(config).to(device)
    elif config.model_type == "liquid":
        assert LiquidSequenceModel is not None
        model = LiquidSequenceModel(config).to(device)
    else:
        model = FGNModel(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    pe_key = "pos_embed.weight"
    if pe_key in state and state[pe_key].shape[0] > config.max_seq_len:
        state[pe_key] = state[pe_key][:config.max_seq_len]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"  unexpected: {len(unexpected)} (first 3: {unexpected[:3]})")
    model.eval()
    return model


@torch.no_grad()
def eval_tsp(model, task, n_batches, batch_size, device, label_tokens: list):
    """Return metrics dict. Includes non-unreach acc + degeneracy diagnostics."""
    n_classes = len(label_tokens)
    unreach_idx = n_classes - 1  # last bucket = "unreach" per task construction
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    total_by_bucket = [0] * n_classes
    correct_by_bucket = [0] * n_classes
    pred_count_by_bucket = [0] * n_classes
    cv_sum = 0.0
    off_by_one = 0
    off_by_many = 0

    for _ in range(n_batches):
        ids, labels, _ = task.generate_batch(batch_size, device)
        out = model(ids)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        pos = (labels != -100).int().argmax(dim=1)
        row = torch.arange(ids.shape[0], device=device)
        truth = labels[row, pos]
        answer_logits = torch.stack(
            [logits[row, pos, t] for t in label_tokens], dim=-1)
        pred_bucket = answer_logits.argmax(dim=-1)
        truth_bucket = torch.full_like(truth, -1)
        for i, tid in enumerate(label_tokens):
            truth_bucket[truth == tid] = i

        for tb, pb in zip(truth_bucket.tolist(), pred_bucket.tolist()):
            if tb < 0:
                continue
            total_by_bucket[tb] += 1
            pred_count_by_bucket[pb] += 1
            confusion[tb, pb] += 1
            if tb == pb:
                correct_by_bucket[tb] += 1
            elif abs(tb - pb) == 1 and tb < 5 and pb < 5:
                off_by_one += 1
            else:
                off_by_many += 1

        if "metric_cv" in out:
            cv_sum += float(out["metric_cv"].item())

    total = sum(total_by_bucket)
    correct = sum(correct_by_bucket)

    # Non-unreach accuracy: exclude unreach from both numerator and denominator.
    # This is the key metric for distinguishing genuine reasoning from the
    # "always predict unreach" degenerate strategy.
    non_unreach_total = total - total_by_bucket[unreach_idx]
    non_unreach_correct = correct - correct_by_bucket[unreach_idx]
    non_unreach_acc = non_unreach_correct / max(1, non_unreach_total)

    # Degeneracy: the max fraction that any single bucket received as prediction.
    # 1.0 = fully degenerate (always one class). 1/n_classes = uniform.
    pred_fractions = [c / max(1, total) for c in pred_count_by_bucket]
    degeneracy = max(pred_fractions)
    degenerate_bucket = pred_fractions.index(degeneracy)

    # Prediction entropy (Shannon, log2): 0 = fully degenerate, log2(7) ≈ 2.81 = uniform.
    pred_entropy = -sum(
        p * math.log2(p) for p in pred_fractions if p > 0)

    return {
        "overall_acc": correct / max(1, total),
        "non_unreach_acc": non_unreach_acc,
        "non_unreach_total": non_unreach_total,
        "non_unreach_correct": non_unreach_correct,
        "degeneracy": degeneracy,
        "degenerate_bucket": degenerate_bucket,
        "pred_entropy": pred_entropy,
        "pred_fractions": pred_fractions,
        "per_bucket": [correct_by_bucket[i] / max(1, total_by_bucket[i])
                        for i in range(n_classes)],
        "per_bucket_n": total_by_bucket,
        "pred_count_by_bucket": pred_count_by_bucket,
        "cv_avg": cv_sum / n_batches,
        "off_by_one": off_by_one / max(1, total),
        "off_by_many": off_by_many / max(1, total),
        "total": total,
        "confusion": confusion.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_batches", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=14)
    p.add_argument("--n_nodes", type=int, default=1024)
    p.add_argument("--min_edges", type=int, default=50)
    p.add_argument("--max_edges", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--graph_file", type=str, default=None,
                   help="Override task data_path. None = TSP default (SNAP email-Eu-core).")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for sample ordering. Does not affect graph contents.")
    p.add_argument("--json_out", type=str, default=None,
                   help="If set, append a JSON record (one line) to this file.")
    p.add_argument("--run_label", type=str, default="",
                   help="Free-form label stored in JSON record (e.g. 'liquid_attn_step5k').")
    args = p.parse_args()

    # Fix seeds for batch-sampling reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = FGNConfig.from_yaml(args.config)
    device = torch.device(args.device)
    tok = _tokenizer()
    model = load_model(config, args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.model_type}  params: {n_params:,}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Graph: {args.graph_file or '(task default)'}")
    print(f"Seed: {args.seed}")

    task_kwargs = dict(
        seq_len=config.max_seq_len,
        n_nodes=args.n_nodes,
        min_edges=args.min_edges, max_edges=args.max_edges,
    )
    if args.graph_file is not None:
        task_kwargs["data_path"] = args.graph_file
    task = get_task("TSP", tok, **task_kwargs)
    label_tokens = task.answer_tokens

    r = eval_tsp(model, task, args.n_batches, args.batch_size, device,
                  label_tokens=label_tokens)
    bucket_names = task.bucket_names
    print(f"\n{'='*60}")
    print(f"TSP evaluation: {r['total']} examples "
          f"({args.n_batches} batches × {args.batch_size})")
    print(f"Edge window: {args.min_edges}-{args.max_edges}")
    print(f"{'='*60}")
    print(f"  Overall accuracy:    {r['overall_acc']:.3f}")
    print(f"  Non-unreach accuracy:{r['non_unreach_acc']:.3f}  "
          f"({r['non_unreach_correct']}/{r['non_unreach_total']})")
    print(f"  Degeneracy score:    {r['degeneracy']:.3f}  "
          f"(bucket: {bucket_names[r['degenerate_bucket']]})")
    print(f"  Prediction entropy:  {r['pred_entropy']:.3f}  "
          f"(uniform={math.log2(len(bucket_names)):.3f})")
    print(f"  Off-by-one rate:     {r['off_by_one']:.3f}")
    print(f"  Large error rate:    {r['off_by_many']:.3f}")
    print(f"  Avg CV (Liquid):     {r['cv_avg']:.3f}")
    print(f"  Random baseline:     {1/len(bucket_names):.3f}")
    print()
    print(f"  Per-bucket accuracy:")
    for name, acc, n in zip(bucket_names, r['per_bucket'], r['per_bucket_n']):
        print(f"    {name:>10} ({n:>4} ex):  {acc:.3f}")
    print()
    print(f"  Prediction distribution (how model distributes guesses):")
    for name, frac, cnt in zip(bucket_names, r['pred_fractions'],
                                r['pred_count_by_bucket']):
        print(f"    {name:>10} ({cnt:>4} pred):  {frac:.3f}")
    print()
    print(f"  Confusion matrix (rows=true, cols=predicted):")
    print(f"  {'':>10}", end="")
    for n in bucket_names:
        print(f"{n:>8}", end="")
    print()
    for name, row in zip(bucket_names, r['confusion']):
        print(f"  {name:>10}", end="")
        for v in row:
            print(f"{v:>8}", end="")
        print()

    if args.json_out:
        record = {
            "run_label": args.run_label,
            "checkpoint": args.checkpoint,
            "config": args.config,
            "model_type": config.model_type,
            "n_params": n_params,
            "graph_file": args.graph_file,
            "seed": args.seed,
            "n_batches": args.n_batches,
            "batch_size": args.batch_size,
            "min_edges": args.min_edges,
            "max_edges": args.max_edges,
            **r,
        }
        with open(args.json_out, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  Appended JSON record to {args.json_out}")


if __name__ == "__main__":
    main()
