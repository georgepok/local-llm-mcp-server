"""Real-code repair training using v15 distilled checkpoint as bootstrap.

Mirrors train_python_repair.py but loads geometric/dynamics weights from
v15's step_2500.pt (ARC-distilled, d=640, CV~8) into a LiquidSequenceModel
configured to match v15's architecture (d_liquid_metric=192, d_liquid_ffn=1280).

The ARC-specific embedding + LM head are NOT loaded; fresh GPT-2 token
embedding + LM head are initialized.

Goal: empirically test whether v15's ARC-distilled geometry helps Python
code repair, vs the Phase 2a from-scratch baseline (K=1 d=128: 47.98% EM).

Usage:
    python scripts/train_python_repair_from_v15.py \
        --corpus /workspace/data/py_bugs_5k.jsonl \
        --teacher_ckpt /tmp/distill_n1024_v15/checkpoints/step_2500.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS))

import torch

# TF32 for fp32 matmul speedup on GB10
torch.set_float32_matmul_precision('high')

from fgn.config import FGNConfig
from fgn.liquid_model import LiquidSequenceModel
from fgn.tasks.python_repair import PythonRepairTask


def transfer_dynamics_weights(student: LiquidSequenceModel,
                              teacher_ckpt_path: str,
                              device: torch.device):
    """Load v15 ContinuousDynamics weights into student.dynamics."""
    ckpt = torch.load(teacher_ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    sd = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
    dyn_sd = {}
    for k, v in sd.items():
        if k.startswith("dynamics."):
            dyn_sd[k[len("dynamics."):]] = v
    target = student.dynamics
    # If torch.compile wrapped dynamics, get the inner module
    inner = getattr(target, "_orig_mod", target)
    print(f"  v15 dynamics keys: {len(dyn_sd)}")
    missing, unexpected = inner.load_state_dict(dyn_sd, strict=False)
    print(f"  loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        print(f"    missing[:5]: {missing[:5]}")
    if unexpected:
        print(f"    unexpected[:5]: {unexpected[:5]}")
    return {"dyn_keys": len(dyn_sd), "missing": len(missing),
            "unexpected": len(unexpected)}


def build_v15_compatible_config(seq_len: int, vocab_size: int,
                                d_model: int = 640,
                                n_ode_steps: int = 512) -> FGNConfig:
    """Build FGNConfig matching v15's architecture so dynamics weights load."""
    return FGNConfig(
        d_model=d_model,
        n_heads=4,
        n_layers=1,
        d_ff=d_model * 4,            # FGNConfig field, ignored by LiquidSequenceModel
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        model_type="liquid",
        n_ode_steps=n_ode_steps,
        liquid_routing="metric",
        liquid_tau_min=0.5,
        liquid_tau_max=1.0,
        halting_enabled=False,
        deep_supervision_enabled=False,
        ponder_kl_lambda=0.0,
        k_substrates=1,
        use_torch_compile=True,
        dropout=0.0,
        metric_type="diagonal",
        metric_rank=0,
        metric_lr_mult=1.0,
        rezero_enabled=False,
        # Match v15's narrow bottleneck (NOT the LiquidSequenceModel default)
        d_liquid_metric=192,
        d_liquid_ffn=2 * d_model,
    )


def run(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Teacher: {args.teacher_ckpt}")
    print(f"Corpus: {args.corpus}")

    # Load tasks
    train_task = PythonRepairTask(args.corpus, seq_len=args.seq_len,
                                   seed=args.seed, split="train")
    eval_task = PythonRepairTask(args.corpus, seq_len=args.seq_len,
                                  seed=args.seed, split="eval")

    print(f"\n=== Building model d={args.d_model}, n_ode_steps={args.n_ode_steps} "
          f"(v15-matched architecture) ===")
    cfg = build_v15_compatible_config(
        seq_len=args.seq_len, vocab_size=train_task.vocab_size,
        d_model=args.d_model, n_ode_steps=args.n_ode_steps,
    )
    model = LiquidSequenceModel(cfg).to(device)
    # Norm homeostasis (matching v15 training)
    model.dynamics._norm_ref = 1.0
    model.dynamics._norm_lambda = 0.1
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {n_params:,}")

    print(f"\n=== Transferring v15 dynamics weights ===")
    stats = transfer_dynamics_weights(model, args.teacher_ckpt, device)

    # Param groups: dynamics gets slow LR if user requests
    geo_params = []
    content_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("dynamics."):
            geo_params.append(p)
        else:
            content_params.append(p)
    print(f"  Geo params: {sum(p.numel() for p in geo_params):,} | "
          f"Content params: {sum(p.numel() for p in content_params):,}")

    optim = torch.optim.AdamW([
        {"params": geo_params, "lr": args.lr * args.geo_lr_mult},
        {"params": content_params, "lr": args.lr},
    ], weight_decay=0.0)
    print(f"  LR: geo={args.lr * args.geo_lr_mult:.5f}, content={args.lr:.5f} "
          f"({1.0/args.geo_lr_mult:.0f}x ratio)")

    print(f"\n=== Training {args.n_steps} steps on python_repair ===")
    history: List[Dict] = []
    losses_window: List[float] = []
    t0 = time.time()
    for step in range(1, args.n_steps + 1):
        ids, labels, _ = train_task.generate_batch(args.batch_size,
                                                    device=device)
        out = model(ids, labels=labels)
        loss = out["ce_loss"]
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        losses_window.append(float(loss.item()))
        if len(losses_window) > 200:
            losses_window = losses_window[-200:]
        if step == 1 or step % args.eval_every == 0 or step == args.n_steps:
            metrics = eval_task.evaluate(model, args.eval_examples,
                                          args.batch_size, device)
            avg_loss = sum(losses_window) / len(losses_window)
            elapsed = time.time() - t0
            log = {"step": step, "loss": avg_loss, **metrics,
                   "elapsed_s": elapsed}
            history.append(log)
            print(f"  step {step:5d}  loss {avg_loss:.4f}  "
                  f"EM {metrics['em']*100:6.2f}%  "
                  f"tok_acc {metrics['tok_acc']*100:5.1f}%  "
                  f"({elapsed:.1f}s)")

    print(f"\n  FINAL  EM={history[-1]['em']*100:6.2f}%  "
          f"tok_acc={history[-1]['tok_acc']*100:5.1f}%  "
          f"({time.time()-t0:.1f}s)")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "result.json")
    with open(out_path, "w") as f:
        json.dump({
            "teacher_ckpt": args.teacher_ckpt,
            "corpus": args.corpus,
            "transfer_stats": stats,
            "config": {"d_model": args.d_model, "n_ode_steps": args.n_ode_steps,
                       "n_steps": args.n_steps, "batch_size": args.batch_size,
                       "lr": args.lr, "geo_lr_mult": args.geo_lr_mult},
            "params": n_params,
            "history": history,
        }, f, indent=2)
    ckpt_path = os.path.join(args.out_dir, "final.pt")
    torch.save({"model": model.state_dict(), "config": vars(args)}, ckpt_path)
    print(f"  wrote {out_path}")
    print(f"  saved {ckpt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/workspace/data/py_bugs_5k.jsonl")
    ap.add_argument("--teacher_ckpt", default="/tmp/distill_n1024_v15/checkpoints/step_2500.pt")
    ap.add_argument("--d_model", type=int, default=640)
    ap.add_argument("--n_ode_steps", type=int, default=512)
    ap.add_argument("--seq_len", type=int, default=384)
    ap.add_argument("--n_steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--geo_lr_mult", type=float, default=1.0,
                    help="Geo LR multiplier (1.0 = unfrozen, 0.1 = 10× slower)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_examples", type=int, default=128)
    ap.add_argument("--out_dir", default="/tmp/python_repair_from_v15")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
