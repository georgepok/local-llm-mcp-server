"""Eval SBF model on a chosen curriculum phase (held-out batches).

Loads a checkpoint, runs N batches on Phase-X distribution (graph size +
allowed buckets), reports:
  - Per-bucket accuracy + CE
  - Random baseline log(num_active_buckets)
  - Aggregate accuracy
  - Confusion matrix
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.liquid_model import LiquidSequenceModel
from fgn.tasks import get_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--curriculum_step", type=int, default=800,
                    help="step to set on the task (selects phase)")
    ap.add_argument("--n_batches", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=14)
    args = ap.parse_args()

    cfg = FGNConfig.from_yaml(args.config)
    device = torch.device("cuda")

    print(f"Loading {args.checkpoint}")
    model = LiquidSequenceModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing: print(f"  MISSING: {len(missing)}")
    if unexpected: print(f"  UNEXPECTED: {len(unexpected)}")
    model.eval()

    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    task = get_task("SBF", tok, seq_len=cfg.max_seq_len, curriculum_enabled=True)
    task.set_curriculum_step(args.curriculum_step)
    allowed = list(task._allowed_buckets)
    n_active = len(allowed)
    rand_baseline = float(np.log(n_active))
    print(f"\nPhase: curriculum_step={args.curriculum_step}")
    print(f"  graph nodes={task.min_nodes}-{task.max_nodes}, "
          f"edges={task.min_edges}-{task.max_edges}")
    print(f"  active buckets: {[task.bucket_names[b] for b in allowed]} "
          f"(n={n_active}, log(n)={rand_baseline:.3f})")

    answer_ids = task.answer_tokens
    bucket_names = task.bucket_names

    confusion = np.zeros((7, 7), dtype=np.int64)
    pred_counts = np.zeros(7, dtype=np.int64)
    true_counts = np.zeros(7, dtype=np.int64)
    ce_per_bucket = [[] for _ in range(7)]
    answer_logits_sum = torch.zeros(7, device=device)
    answer_logits_n = 0

    with torch.no_grad():
        for bi in range(args.n_batches):
            ids, labels, _ = task.generate_batch(args.batch_size, device=device)
            out = model(ids, labels=labels)
            logits = out["logits"]
            mask = (labels != -100)
            label_pos = mask.nonzero(as_tuple=False)
            for k in range(label_pos.shape[0]):
                b, s = label_pos[k, 0].item(), label_pos[k, 1].item()
                true_tok = labels[b, s].item()
                step_logits = logits[b, s]
                # Restrict prediction to ACTIVE buckets only
                ans_l_full = step_logits[answer_ids]  # [7]
                # Mask inactive buckets to -inf for argmax
                mask_active = torch.full((7,), float('-inf'), device=device)
                for j in allowed:
                    mask_active[j] = ans_l_full[j]
                pred_idx = mask_active.argmax().item()
                true_idx = answer_ids.index(true_tok)
                confusion[true_idx, pred_idx] += 1
                pred_counts[pred_idx] += 1
                true_counts[true_idx] += 1
                answer_logits_sum += ans_l_full.detach()
                answer_logits_n += 1
                # CE over the full 7-class softmax (matches training)
                bucket_ce = F.cross_entropy(
                    ans_l_full.unsqueeze(0),
                    torch.tensor([true_idx], device=device))
                ce_per_bucket[true_idx].append(bucket_ce.item())

    total = answer_logits_n
    correct = int(np.sum(np.diag(confusion)))
    print(f"\n=== Held-out eval ===")
    print(f"Total: {total} predictions")
    print(f"Aggregate accuracy: {correct}/{total} = {correct/total:.3f}")
    print(f"Random baseline:    {1.0/n_active:.3f}")
    mean_l = (answer_logits_sum / total).cpu().numpy()
    print(f"\nMean logits per bucket:")
    for i, name in enumerate(bucket_names):
        marker = " *" if i in allowed else "  "
        print(f"  {marker}{name:>8s}: {mean_l[i]:+.3f}")

    print(f"\nPer-bucket accuracy + CE (active only):")
    for i in allowed:
        tot = confusion[i].sum()
        corr = confusion[i, i]
        acc = corr/max(tot, 1)
        ce_b = np.mean(ce_per_bucket[i]) if ce_per_bucket[i] else float("nan")
        print(f"  {bucket_names[i]:>8s}: acc={acc:.3f} ({corr}/{tot}) ce={ce_b:.3f}")

    print(f"\nConfusion (rows=true, cols=pred, only active buckets):")
    header = "        " + " ".join(f"{bucket_names[j]:>5s}" for j in allowed)
    print(header)
    for i in allowed:
        row = " ".join(f"{confusion[i, j]:>5d}" for j in allowed)
        print(f"  {bucket_names[i]:>5s} {row}")


if __name__ == "__main__":
    main()
