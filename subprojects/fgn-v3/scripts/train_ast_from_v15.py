"""First AST experiment using v15 distilled checkpoint as bootstrap.

Loads geometric/dynamics weights from v15's step_2500.pt (the latest distillation
state at d=640 with CV~8 matching teacher) and uses them as initialization for
an ASTEditorModel that runs the synthetic AST task. ARC-specific embedding +
output head are discarded; AST-specific input embedding + 3 emit heads are
fresh-initialized.

Goal: empirically test whether the geometric structure transferred from the
ARC teacher (via v15 distillation) helps with AST editor learning, vs. v3
(small student, low CV) or training from scratch (already known dismal).

Usage:
    python scripts/train_ast_from_v15.py
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

# Enable TF32 cores on Ampere+ for fp32 matmul speedup
torch.set_float32_matmul_precision('high')

from fgn.config import FGNConfig
from fgn.ast_editor_model import ASTEditorModel
from fgn.tasks.synthetic_ast import (
    SyntheticASTTask, VOCAB_SIZE, SEQ_LEN, SCRIPT_START,
    N_NODES, N_EDIT_OPS, PAY_RANGE, K_FIX,
)


def transfer_dynamics_weights(student: ASTEditorModel,
                              teacher_ckpt_path: str,
                              device: torch.device) -> Dict[str, str]:
    """Load v15 student's ContinuousDynamics weights into our AST model.

    Returns: dict of {key: status} for diagnostic.
    """
    ckpt = torch.load(teacher_ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    sd = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}

    # Filter to dynamics keys only (skip embedding, output_head, persistent, etc.)
    dyn_sd = {}
    for k, v in sd.items():
        if k.startswith("dynamics."):
            new_k = k[len("dynamics."):]   # strip prefix to land on student.dynamics
            dyn_sd[new_k] = v

    print(f"  Found {len(dyn_sd)} dynamics keys in v15 checkpoint")
    target = student.dynamics
    missing, unexpected = target.load_state_dict(dyn_sd, strict=False)
    print(f"  Loaded into student.dynamics — missing: {len(missing)}, "
          f"unexpected: {len(unexpected)}")
    if missing:
        print(f"    missing[:5]: {missing[:5]}")
    if unexpected:
        print(f"    unexpected[:5]: {unexpected[:5]}")
    return {"dyn_keys": len(dyn_sd), "missing": len(missing),
            "unexpected": len(unexpected)}


def build_ast_model_for_v15(device: torch.device,
                            d_model: int = 640) -> ASTEditorModel:
    """Build ASTEditorModel at v15's dimensionality and metric/ffn shape so
    weights load cleanly. v15 used distill_geometry.py defaults:
        d_metric = 192 (kept at teacher's value, NOT scaled with d)
        d_ffn = 2 * d_model = 1280 (NOT 4× like LiquidSequenceModel default)
    """
    cfg = FGNConfig(
        d_model=d_model,
        n_heads=4,
        n_layers=1,
        d_ff=d_model * 4,
        vocab_size=VOCAB_SIZE,        # synthetic_ast vocab (12)
        max_seq_len=SEQ_LEN,
        model_type="liquid",
        n_ode_steps=1024,             # match v15
        liquid_routing="metric",
        liquid_tau_min=0.5,
        liquid_tau_max=1.0,
        halting_enabled=False,
        deep_supervision_enabled=False,
        ponder_kl_lambda=0.0,
        k_substrates=1,
        use_torch_compile=False,
        dropout=0.0,
        metric_type="diagonal",
        metric_rank=0,
        metric_lr_mult=1.0,
        rezero_enabled=False,
        # CRITICAL: must match v15's distill_geometry.py student config
        d_liquid_metric=192,         # teacher's d_metric, kept as-is
        d_liquid_ffn=2 * d_model,    # 1280 for d=640 (not 4x)
    )
    model = ASTEditorModel(
        cfg,
        n_nodes=N_NODES, n_edit_ops=N_EDIT_OPS, payload_range=PAY_RANGE,
        script_start=SCRIPT_START, k_fix=K_FIX,
    ).to(device)
    # Norm homeostasis (matching v15 training-time setup)
    model.dynamics._norm_ref = 1.0
    model.dynamics._norm_lambda = 0.1
    return model


def evaluate(model, task, n_examples, batch_size, device):
    model.eval()
    em_total = 0.0
    ptr_total = 0.0
    op_total = 0.0
    n_batches = 0
    with torch.no_grad():
        seen = 0
        while seen < n_examples:
            B = min(batch_size, n_examples - seen)
            ids, gt_p, gt_o, gt_y, _ = task.generate_batch(B, device=device)
            out = model(ids, gt_ptr=gt_p, gt_op=gt_o, gt_pay=gt_y)
            em, ptr_acc, op_acc = SyntheticASTTask.exact_match(
                ids, out["ptr_logits"], out["op_logits"], out["pay_logits"],
                gt_p, gt_o)
            em_total += em
            ptr_total += ptr_acc
            op_total += op_acc
            n_batches += 1
            seen += B
    model.train()
    return {
        "em": em_total / n_batches,
        "ptr_acc": ptr_total / n_batches,
        "op_acc": op_total / n_batches,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", type=str,
                    default="/tmp/distill_n1024_v15/checkpoints/step_2500.pt")
    ap.add_argument("--n_steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--geo_lr_mult", type=float, default=0.1,
                    help="LR multiplier for geometric params (lower = preserve "
                         "transferred geometry). 0.1 = 10x slower than content.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_examples", type=int, default=512)
    ap.add_argument("--out_dir", type=str, default="/tmp/ast_from_v15")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Teacher checkpoint: {args.teacher_ckpt}")
    print(f"AST task: synthetic, VOCAB_SIZE={VOCAB_SIZE}, SEQ_LEN={SEQ_LEN}, "
          f"K_FIX={K_FIX}")

    print("\n=== Building AST model at d=640 (v15 dimensionality) ===")
    model = build_ast_model_for_v15(device, d_model=640)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {n_params:,}")

    print("\n=== Transferring v15 dynamics weights ===")
    stats = transfer_dynamics_weights(model, args.teacher_ckpt, device)

    # Geometric params (dynamics) get slow LR; content (embeddings, heads) fast.
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

    task = SyntheticASTTask(seed=args.seed)
    print(f"\n=== Training {args.n_steps} steps on synthetic AST ===")
    history: List[Dict] = []
    losses_window: List[float] = []
    t0 = time.time()
    for step in range(1, args.n_steps + 1):
        ids, gt_p, gt_o, gt_y, _ = task.generate_batch(args.batch_size,
                                                       device=device)
        out = model(ids, gt_ptr=gt_p, gt_op=gt_o, gt_pay=gt_y)
        loss = out["loss"]
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        losses_window.append(float(loss.item()))
        if len(losses_window) > 200:
            losses_window = losses_window[-200:]
        if step == 1 or step % args.eval_every == 0 or step == args.n_steps:
            metrics = evaluate(model, task, args.eval_examples,
                                args.batch_size, device)
            avg_loss = sum(losses_window) / len(losses_window)
            elapsed = time.time() - t0
            log = {"step": step, "loss": avg_loss, **metrics,
                   "elapsed_s": elapsed}
            history.append(log)
            print(f"  step {step:5d}  loss {avg_loss:.4f}  "
                  f"EM {metrics['em']*100:6.2f}%  "
                  f"ptr {metrics['ptr_acc']*100:5.1f}%  "
                  f"op {metrics['op_acc']*100:5.1f}%  ({elapsed:.1f}s)")

    final = evaluate(model, task, args.eval_examples * 4, args.batch_size,
                     device)
    print(f"\n  FINAL  EM={final['em']*100:6.2f}%  "
          f"ptr={final['ptr_acc']*100:5.1f}%  op={final['op_acc']*100:5.1f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "result.json")
    with open(out_path, "w") as f:
        json.dump({
            "teacher_ckpt": args.teacher_ckpt,
            "transfer_stats": stats,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "geo_lr_mult": args.geo_lr_mult,
            "params": n_params,
            "history": history,
            "final": final,
        }, f, indent=2)
    # Save final model
    ckpt_path = os.path.join(args.out_dir, "final.pt")
    torch.save({"model": model.state_dict(), "config": vars(args)}, ckpt_path)
    print(f"\n  wrote {out_path}")
    print(f"  saved {ckpt_path}")


if __name__ == "__main__":
    main()
