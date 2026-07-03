"""ODE-faithfulness validation: train a small LiquidSequenceModel with finer
dt + norm homeostasis, then verify whether the trained dynamics has
discretisation-invariant time-T flow.

Hypothesis (from the SOC-vs-ODE analysis): the project's "explicit Euler at
coarse dt" is mathematically inconsistent with critical SOC dynamics. Fixes:
   1. Smaller dt — train with n_ode_steps=128 (was 16)
   2. Norm homeostasis — keep ||h|| bounded so a time-T endpoint is well-defined

Test:
   A. Train a small model on a non-trivial task (parity) with the above
   B. Run convergence diagnostic on the trained model — measure
      h(T) at varying n_steps. If the trained model is ODE-faithful,
      h(T=2.0) should be approximately invariant to n_steps refinement
      (e.g. n=64, 128, 256, 512 should give similar h_final).
   C. Quality check — task accuracy at the trained n_steps should be
      reasonable.

If A+B+C pass: explicit Euler with finer dt + norm homeostasis is the
fixed-step ODE-faithful solver path the project needs. If they fail,
implicit / Rosenbrock methods become unavoidable.

Run:
    python scripts/train_and_test_ode_faithful.py --steps 2000 --device cuda
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
import torch.nn.functional as F

from fgn.config import FGNConfig
from fgn.liquid_model import LiquidSequenceModel
from fgn.tasks.parity import ParityTask
from fgn.tasks.affine import AffineGroupTask


# Resolve the GPT-2 tokenizer once
def _tokenizer():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained("gpt2")


def build_model(d_model: int, n_ode_steps: int, vocab_size: int, seq_len: int,
                norm_ref: float, norm_lambda: float, device: torch.device):
    cfg = FGNConfig(
        d_model=d_model,
        n_heads=4,
        n_layers=1,
        d_ff=d_model * 4,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        model_type="liquid",
        n_ode_steps=n_ode_steps,
        liquid_routing="metric",
        liquid_tau_min=0.5,
        liquid_tau_max=1.0,
        halting_enabled=False,            # use plain euler_solve (has norm homeostasis)
        deep_supervision_enabled=False,
        ponder_kl_lambda=0.0,
        k_substrates=1,
        use_torch_compile=False,           # explicit no-compile for diagnostic clarity
        dropout=0.0,
        metric_type="diagonal",
        metric_rank=0,
        metric_lr_mult=1.0,
        rezero_enabled=False,
    )
    model = LiquidSequenceModel(cfg).to(device)
    # Set norm homeostasis on the dynamics (read by euler_solve via getattr)
    model.dynamics._norm_ref = norm_ref
    model.dynamics._norm_lambda = norm_lambda
    return model, cfg


@torch.no_grad()
def manual_unroll(model, h0, max_steps: int, integration_time: float,
                  record_every: int = 1):
    """Step-by-step Euler with norm homeostasis active. Records per-step
    relative residual and h_norm, plus snapshots at sampled points."""
    dt = integration_time / max_steps
    t = 0.0
    h = h0
    norm_ref = float(getattr(model.dynamics, "_norm_ref", 0.0))
    norm_lambda = float(getattr(model.dynamics, "_norm_lambda", 0.0))
    inner = model.dynamics._orig_mod if hasattr(model.dynamics, "_orig_mod") else model.dynamics
    if hasattr(inner, "reset_fast_weights"):
        inner.reset_fast_weights(h0.shape[0], h0.device, h0.dtype)
    if hasattr(inner, "reset_id_history"):
        inner.reset_id_history(h0.shape[0], h0.shape[1], h0.device, h0.dtype)
    residuals = []
    snapshots = {}
    for k in range(max_steps):
        if hasattr(model.dynamics, "set_step_index"):
            model.dynamics.set_step_index(k, max_steps)
        if hasattr(model.dynamics, "set_step_embed"):
            model.dynamics.set_step_embed(k, max_steps)
        dh = model.dynamics(t, h)
        if isinstance(dh, tuple):
            dh = dh[0]
        h_new = h + dt * dh
        if norm_ref > 0 and norm_lambda > 0:
            pos_norm = h_new.detach().norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.where(
                pos_norm > norm_ref,
                1.0 - norm_lambda * (1.0 - norm_ref / pos_norm),
                torch.ones_like(pos_norm),
            )
            h_new = h_new * scale
        diff = (h_new - h).norm(dim=-1).mean().item()
        h_norm = h.norm(dim=-1).mean().item()
        rel = diff / max(h_norm, 1e-8)
        if (k + 1) % record_every == 0 or k == 0:
            residuals.append((k + 1, rel, h_norm))
        # Save snapshots at k+1 = 64, 128, 256, max_steps (whichever exist)
        if (k + 1) in (64, 128, 256, 512, max_steps):
            snapshots[k + 1] = h_new.detach().clone()
        h = h_new
        t = t + dt
    return residuals, snapshots, h


def setup_dynamics_context(model, input_ids):
    """Replicate LiquidSequenceModel.forward's pre-ODE setup so we can run
    the dynamics manually and observe each step."""
    B, N = input_ids.shape
    device = input_ids.device
    pos = torch.arange(N, device=device).unsqueeze(0)
    h0 = model.embed(input_ids) + model.pos_embed(pos)
    mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
    context = h0.mean(dim=1)
    model.dynamics.set_context(context, mask=mask)
    model.dynamics.set_n_steps(model.n_ode_steps)
    return h0


def evaluate(model, eval_task, n_batches: int, batch_size: int,
             device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            ids, labels, _ = eval_task.generate_batch(batch_size, device=device)
            out = model(ids, labels=labels)
            preds = out["logits"].argmax(dim=-1)
            mask = labels != -100
            correct += int(((preds == labels) & mask).sum().item())
            total += int(mask.sum().item())
    model.train()
    return {"acc": correct / max(total, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_ode_steps", type=int, default=128)
    ap.add_argument("--norm_ref", type=float, default=1.0)
    ap.add_argument("--norm_lambda", type=float, default=0.1)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--bit_length", type=int, default=40)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default="ode_faithful_result.json")
    ap.add_argument("--task", type=str, default="parity",
                    choices=["parity", "affine"])
    ap.add_argument("--diag_n_steps", type=str, default="64,128,256,512,1024,2048")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Config: d={args.d_model}, n_ode_steps={args.n_ode_steps}, "
          f"norm_ref={args.norm_ref}, norm_lambda={args.norm_lambda}")

    tok = _tokenizer()
    if args.task == "parity":
        train_task = ParityTask(tokenizer=tok, seq_len=args.seq_len,
                                bit_length=args.bit_length, p_one=0.5)
        eval_task = ParityTask(tokenizer=tok, seq_len=args.seq_len,
                               bit_length=args.bit_length, p_one=0.5)
    elif args.task == "affine":
        train_task = AffineGroupTask(tokenizer=tok, seq_len=args.seq_len,
                                     min_ops=20, max_ops=40, sup_every=5)
        eval_task = AffineGroupTask(tokenizer=tok, seq_len=args.seq_len,
                                    min_ops=20, max_ops=40, sup_every=5)
    else:
        raise ValueError(f"unknown task: {args.task}")

    model, cfg = build_model(
        d_model=args.d_model, n_ode_steps=args.n_ode_steps,
        vocab_size=tok.vocab_size, seq_len=args.seq_len,
        norm_ref=args.norm_ref, norm_lambda=args.norm_lambda, device=device,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params={n_params:,}\n")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    history: List[Dict] = []
    losses_window: List[float] = []
    t0 = time.time()

    print(f"=== Training {args.steps} steps on parity (bit_length={args.bit_length}) ===")
    for step in range(1, args.steps + 1):
        ids, labels, _ = train_task.generate_batch(args.batch_size, device=device)
        out = model(ids, labels=labels)
        loss = out["ce_loss"]
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        losses_window.append(float(loss.item()))
        if len(losses_window) > 100:
            losses_window = losses_window[-100:]
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, eval_task, n_batches=8,
                                batch_size=args.batch_size, device=device)
            avg_loss = sum(losses_window) / len(losses_window)
            elapsed = time.time() - t0
            print(f"  step {step:5d}  loss {avg_loss:.4f}  acc {metrics['acc']*100:5.1f}%  "
                  f"({elapsed:.1f}s)")
            history.append({"step": step, "loss": avg_loss, **metrics, "elapsed_s": elapsed})

    train_time = time.time() - t0
    final_acc = history[-1]["acc"]
    print(f"\nTraining done in {train_time:.1f}s. Final acc: {final_acc*100:.2f}%")

    # === Convergence diagnostic ===
    print(f"\n=== Convergence diagnostic ===")
    diag_batch = 4
    ids, labels, _ = eval_task.generate_batch(diag_batch, device=device)
    h0 = setup_dynamics_context(model, ids)
    integration_time = 2.0  # default in la_cfg

    # Run at varying n_steps and check h(T) invariance
    diag_n_steps = [int(x) for x in args.diag_n_steps.split(",")]
    diag_n_steps = sorted(set(diag_n_steps))
    n_ref = diag_n_steps[-1]
    print(f"\nh(T=2.0) invariance to n_steps refinement (reference: n={n_ref}):")
    print(f"  {'n_steps':>10}  {'mean(h_norm)':>14}  {'rel_diff vs n_ref':>20}")
    h_finals = {}
    for n in diag_n_steps:
        residuals, snapshots, h_final = manual_unroll(
            model, h0, max_steps=n, integration_time=integration_time,
            record_every=max(1, n // 16),
        )
        h_finals[n] = h_final
    h_ref = h_finals[n_ref]
    norm_ref_val = h_ref.norm(dim=-1).mean().item()
    invariance_rows = []
    for n in diag_n_steps:
        diff = (h_finals[n] - h_ref).norm(dim=-1).mean().item()
        rel = diff / max(norm_ref_val, 1e-8)
        h_n_norm = h_finals[n].norm(dim=-1).mean().item()
        print(f"  {n:>10d}  {h_n_norm:>14.4f}  {rel:>20.6f}")
        invariance_rows.append({"n_steps": n, "h_norm": h_n_norm,
                                f"rel_diff_vs_{n_ref}": rel})

    # Convergence-rate analysis: ratio of consecutive rel_diffs
    # If Euler converges as O(h), doubling n should halve the error.
    # rel_diff(n)/rel_diff(2n) ≈ 2 for clean O(h) convergence.
    print("\nO(h) convergence check (consecutive doublings, expect ratio ≈ 2):")
    rel_diffs = [r[f"rel_diff_vs_{n_ref}"] for r in invariance_rows]
    for i in range(len(diag_n_steps) - 2):
        n_a = diag_n_steps[i]
        n_b = diag_n_steps[i + 1]
        a = rel_diffs[i]
        b = rel_diffs[i + 1]
        if b > 1e-8:
            ratio = a / b
        else:
            ratio = float("inf")
        print(f"  n={n_a:4d} → n={n_b:4d}: rel_diff {a:.4f} → {b:.4f}, "
              f"ratio = {ratio:.3f}")

    # Run a long unroll to inspect per-step residuals at fine dt
    print(f"\nLong unroll (n_steps=512) — per-step residual at fine dt should "
          f"decay if dynamics is bounded:")
    residuals_long, _, _ = manual_unroll(
        model, h0, max_steps=512, integration_time=integration_time,
        record_every=32,
    )
    print(f"  {'step':>6}  {'residual':>15}  {'h_norm':>10}")
    for k, rel, hn in residuals_long:
        print(f"  {k:>6d}  {rel:>15.6f}  {hn:>10.4f}")

    # Verdict
    print(f"\n{'='*60}")
    print("Verdict")
    print(f"{'='*60}")
    rel_at_trained = next(
        (r[f"rel_diff_vs_{n_ref}"] for r in invariance_rows
         if r["n_steps"] == args.n_ode_steps), None)
    if rel_at_trained is not None:
        print(f"  h(T=2.0) at trained n_steps={args.n_ode_steps} vs reference "
              f"n={n_ref}: rel_diff = {rel_at_trained:.4f}")
        if rel_at_trained < 0.05:
            print(f"  → ODE-faithful at trained n. Explicit Euler is sufficient.")
        elif rel_at_trained < 0.2:
            print(f"  → MARGINAL: trained n is closer to reference but not "
                  f"converged. Production would need finer dt.")
        else:
            print(f"  → NOT YET ODE-faithful at trained n. Need either much "
                  f"finer dt or implicit method.")
    print(f"  Final task acc at trained n_ode_steps: {final_acc*100:.2f}%")

    # Save
    summary = {
        "config": {
            "d_model": args.d_model, "n_ode_steps": args.n_ode_steps,
            "norm_ref": args.norm_ref, "norm_lambda": args.norm_lambda,
            "steps": args.steps, "lr": args.lr,
        },
        "params": n_params,
        "train_time_s": train_time,
        "final_acc": final_acc,
        "history": history,
        "invariance": invariance_rows,
        "long_residuals": [(int(k), float(r), float(n)) for (k, r, n) in residuals_long],
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
