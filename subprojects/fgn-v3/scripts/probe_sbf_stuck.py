"""Probe what the SBF d=768 model is doing at step 2000.

Training is plateaued at CE≈1.94 (random baseline log(7)=1.945) with CV=0.
We need to know:
  1. Prediction distribution — uniform random or collapsed to one class?
  2. Confusion matrix — does it favor a particular bucket?
  3. Gradient flow — is metric receiving gradient?
  4. h-state stats — is the residual essentially identity? (gate=0.14)
  5. CE per-bucket — are some buckets harder?
  6. Logit dynamics across ODE steps — do logits change at all?
"""

import argparse
import os
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
    ap.add_argument("--task", default="SBF")
    ap.add_argument("--n_batches", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=14)
    args = ap.parse_args()

    cfg = FGNConfig.from_yaml(args.config)
    device = torch.device("cuda")

    print(f"Loading model from {args.checkpoint}")
    model = LiquidSequenceModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing: print(f"  MISSING keys: {missing[:5]} ...")
    if unexpected: print(f"  UNEXPECTED keys: {unexpected[:5]} ...")
    model.eval()

    # Tokenizer + task
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    task = get_task(args.task, tok, seq_len=cfg.max_seq_len)

    # Answer token IDs
    answer_ids = task.answer_tokens
    bucket_names = task.bucket_names

    # Confusion: rows=true, cols=pred
    confusion = np.zeros((7, 7), dtype=np.int64)
    pred_counts = np.zeros(7, dtype=np.int64)
    true_counts = np.zeros(7, dtype=np.int64)

    # Logit stats for answer tokens
    answer_logits_sum = torch.zeros(7, device=device)
    answer_logits_n = 0

    # Per-bucket CE
    ce_per_bucket = [[] for _ in range(7)]

    # Gate stats from logs say gate=0.14 → sigmoid≈0.535 means alpha for SDPA
    # vs identity. Let's verify by reading model state.
    with torch.no_grad():
        # 1. Inspect step-conditional FiLM weights
        dyn = model.layers[0].dynamics if hasattr(model, 'layers') else None
        if dyn is None:
            for n, m in model.named_modules():
                if 'dynamics' in n.lower() and 'film' not in n.lower():
                    print(f"  found dynamics-like: {n}")
                    break
        # Find dynamics module
        from liquid_arc.dynamics import ContinuousDynamics
        dyns = [m for m in model.modules() if isinstance(m, ContinuousDynamics)]
        if dyns:
            dyn = dyns[0]
            print(f"\n=== Dynamics state ===")
            print(f"  alpha_logit={dyn.alpha_logit.item():.3f} → α={torch.sigmoid(dyn.alpha_logit).item():.3f}")
            print(f"  tau_frozen={getattr(dyn, '_tau_frozen', 'n/a')}")
            if hasattr(dyn, 'metric_film_gamma'):
                g = dyn.metric_film_gamma.weight  # [n_max, d_metric_bn]
                b = dyn.metric_film_beta.weight
                print(f"  metric_film γ: mean={g.mean().item():.4f} std={g.std().item():.4f}")
                print(f"  metric_film β: mean={b.mean().item():.4f} std={b.std().item():.4f}")
            if hasattr(dyn, 'tau_film_gamma'):
                g = dyn.tau_film_gamma.weight
                b = dyn.tau_film_beta.weight
                print(f"  tau_film    γ: mean={g.mean().item():.4f} std={g.std().item():.4f}")
                print(f"  tau_film    β: mean={b.mean().item():.4f} std={b.std().item():.4f}")
            if hasattr(dyn, 't_diff_per_step'):
                td = dyn.t_diff_per_step.weight
                print(f"  t_diff_per_step: min={td.min().item():.3f} max={td.max().item():.3f} mean={td.mean().item():.3f}")
            if hasattr(dyn, 'halt_head'):
                hh = dyn.halt_head
                print(f"  halt_head: bias={hh.bias.item():.3f} weight_norm={hh.weight.norm().item():.3f}")

        # 2. Run batches and accumulate stats
        print(f"\n=== Inference ({args.n_batches} batches × {args.batch_size}) ===")
        total_loss = 0.0
        n = 0
        for bi in range(args.n_batches):
            ids, labels, meta = task.generate_batch(args.batch_size, device=device)
            out = model(ids, labels=labels)
            logits = out["logits"]  # [B, S, V]
            B, S, V = logits.shape

            # Find label positions (label != -100)
            mask = (labels != -100)  # [B, S]
            # there should be exactly 1 per row
            label_pos = mask.nonzero(as_tuple=False)  # [B, 2] (b, s)

            for k in range(label_pos.shape[0]):
                b, s = label_pos[k, 0].item(), label_pos[k, 1].item()
                true_tok = labels[b, s].item()
                step_logits = logits[b, s]  # [V]
                # Logits restricted to the 7 answer tokens
                ans_l = step_logits[answer_ids]  # [7]
                pred_idx = ans_l.argmax().item()
                pred_tok = answer_ids[pred_idx]

                true_idx = answer_ids.index(true_tok)
                confusion[true_idx, pred_idx] += 1
                pred_counts[pred_idx] += 1
                true_counts[true_idx] += 1
                answer_logits_sum += ans_l.detach()
                answer_logits_n += 1

                # CE just over the 7-class softmax
                bucket_ce = F.cross_entropy(ans_l.unsqueeze(0), torch.tensor([true_idx], device=device))
                ce_per_bucket[true_idx].append(bucket_ce.item())

            if 'loss' in out and out['loss'] is not None:
                total_loss += out['loss'].item()
                n += 1

        # 3. Print results
        print(f"\nFull-vocab loss: {total_loss/max(n,1):.4f}")
        print(f"Total predictions: {answer_logits_n}")

        print(f"\n=== Mean answer logits (avg over all label positions) ===")
        mean_l = (answer_logits_sum / answer_logits_n).cpu().numpy()
        for i, name in enumerate(bucket_names):
            print(f"  bucket {name:>8s}: {mean_l[i]:+.3f}")

        print(f"\n=== Prediction distribution (over 7 answer tokens) ===")
        for i, name in enumerate(bucket_names):
            print(f"  pred {name:>8s}: {pred_counts[i]:>5d} ({100*pred_counts[i]/answer_logits_n:5.2f}%)")

        print(f"\n=== True distribution (sanity: should be ~uniform 14.3%) ===")
        for i, name in enumerate(bucket_names):
            print(f"  true {name:>8s}: {true_counts[i]:>5d} ({100*true_counts[i]/answer_logits_n:5.2f}%)")

        print(f"\n=== Confusion matrix (rows=true, cols=pred) ===")
        col_widths = [max(8, len(n)+1) for n in bucket_names]
        header = "        " + " ".join(f"{n:>7s}" for n in bucket_names)
        print(header)
        for i, name in enumerate(bucket_names):
            row = " ".join(f"{confusion[i, j]:>7d}" for j in range(7))
            print(f"  {name:>5s} {row}")

        # Per-row accuracy
        print(f"\n=== Per-bucket accuracy + CE ===")
        for i, name in enumerate(bucket_names):
            tot = confusion[i].sum()
            corr = confusion[i, i]
            acc = corr/max(tot,1)
            ce_b = np.mean(ce_per_bucket[i]) if ce_per_bucket[i] else float("nan")
            print(f"  {name:>8s}: acc={acc:.3f} ({corr}/{tot}) ce={ce_b:.3f}")

        # 4. h-state probe: how much does h change over ODE?
        print(f"\n=== h evolution across ODE steps (1 batch) ===")
        ids, labels, _ = task.generate_batch(args.batch_size, device=device)
        # Get hidden state pre-ODE and post-ODE
        # Use forward hooks
        h_pre = None
        h_post = None
        def pre_hook(mod, inp):
            nonlocal h_pre
            h_pre = inp[0].detach().clone()
        def post_hook(mod, inp, out):
            nonlocal h_post
            # output is what after ODE
            if isinstance(out, tuple):
                h_post = out[0].detach().clone() if torch.is_tensor(out[0]) else None
            else:
                h_post = out.detach().clone() if torch.is_tensor(out) else None
        # Hook on the LiquidARC backbone
        for n, m in model.named_modules():
            if hasattr(m, 'la_cfg') or 'arc_backbone' in n.lower() or 'liquid_arc' in n.lower():
                print(f"  hooking module: {n} ({type(m).__name__})")
                h1 = m.register_forward_pre_hook(pre_hook)
                h2 = m.register_forward_hook(post_hook)
                _ = model(ids, labels=labels)
                h1.remove(); h2.remove()
                break

        if h_pre is not None and h_post is not None:
            print(f"  h_pre  shape={tuple(h_pre.shape)} norm={h_pre.norm(dim=-1).mean().item():.3f}")
            print(f"  h_post shape={tuple(h_post.shape)} norm={h_post.norm(dim=-1).mean().item():.3f}")
            delta = (h_post - h_pre)
            print(f"  Δh L2={delta.norm(dim=-1).mean().item():.4f}  (should be >> 0 for learning)")
            cos = F.cosine_similarity(h_pre, h_post, dim=-1)
            print(f"  cos(h_pre, h_post)={cos.mean().item():.4f}  (1.0 = no change)")


if __name__ == "__main__":
    main()
