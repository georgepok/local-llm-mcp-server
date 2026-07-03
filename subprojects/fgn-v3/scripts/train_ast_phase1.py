"""Phase 1 trainer — synthetic AST editor on LiquidARC + 3 emit heads.

Tests whether the K=2 coupled-substrate specialisation observed in Phase 0
(NumPy MLP toy with three independent heads) transfers to LiquidARC's
continuous ODE substrate. Three heads (pointer, op, payload) emit each edit
from a SINGLE substrate hidden vector — no AR teacher-forcing easy path.

Conditions: K=1 baseline, K=2 coupled. Param count near-matched.

Run:
    python scripts/train_ast_phase1.py
    python scripts/train_ast_phase1.py --condition k1 --seed 0
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

from fgn.config import FGNConfig
from fgn.ast_editor_model import ASTEditorModel
from fgn.tasks.synthetic_ast import (
    SyntheticASTTask, VOCAB_SIZE, SEQ_LEN, SCRIPT_START,
    N_NODES, N_EDIT_OPS, PAY_RANGE, K_FIX,
)


CONDITIONS: Dict[str, Dict] = {
    "k1": {"k_substrates": 1, "lateral_weight": 0.0, "d_model": 48},
    "k2": {"k_substrates": 2, "lateral_weight": 0.5, "d_model": 32},
}


def build_config(condition: str) -> FGNConfig:
    cond = CONDITIONS[condition]
    return FGNConfig(
        d_model=cond["d_model"],
        n_heads=4,
        n_layers=1,
        d_ff=cond["d_model"] * 4,
        vocab_size=VOCAB_SIZE,
        max_seq_len=SEQ_LEN,
        model_type="liquid",
        n_ode_steps=8,
        liquid_routing="metric",
        liquid_tau_min=0.5,
        liquid_tau_max=1.0,
        halting_enabled=False,
        deep_supervision_enabled=False,
        ponder_kl_lambda=0.0,
        k_substrates=cond["k_substrates"],
        lateral_weight=cond["lateral_weight"],
        use_torch_compile=False,
        dropout=0.0,
        metric_type="diagonal",
        metric_rank=0,
        metric_lr_mult=1.0,
        rezero_enabled=False,
    )


def build_model(condition: str, device: torch.device) -> ASTEditorModel:
    cfg = build_config(condition)
    model = ASTEditorModel(
        cfg,
        n_nodes=N_NODES, n_edit_ops=N_EDIT_OPS, payload_range=PAY_RANGE,
        script_start=SCRIPT_START, k_fix=K_FIX,
    ).to(device)
    return model


def evaluate(model: ASTEditorModel, task: SyntheticASTTask,
             n_examples: int, batch_size: int, device: torch.device):
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


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_condition(condition: str, seed: int, n_steps: int, batch_size: int,
                  lr: float, eval_every: int, eval_examples: int,
                  device: torch.device, out_dir: str):
    torch.manual_seed(seed)
    model = build_model(condition, device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    task = SyntheticASTTask(seed=seed)
    cfg = model.config
    print(f"\n=== condition={condition} seed={seed} "
          f"params={n_params(model):,} d_model={cfg.d_model} K={cfg.k_substrates} ===")

    history: List[Dict] = []
    losses_window: List[float] = []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        ids, gt_p, gt_o, gt_y, _ = task.generate_batch(batch_size, device=device)
        out = model(ids, gt_ptr=gt_p, gt_op=gt_o, gt_pay=gt_y)
        loss = out["loss"]
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        losses_window.append(float(loss.item()))
        if len(losses_window) > 200:
            losses_window = losses_window[-200:]
        if step == 1 or step % eval_every == 0 or step == n_steps:
            metrics = evaluate(model, task, eval_examples, batch_size, device)
            avg_loss = sum(losses_window) / len(losses_window)
            elapsed = time.time() - t0
            log = {"step": step, "loss": avg_loss, **metrics, "elapsed_s": elapsed}
            history.append(log)
            print(f"  step {step:5d}  loss {avg_loss:.4f}  EM {metrics['em']*100:6.2f}%  "
                  f"ptr {metrics['ptr_acc']*100:5.1f}%  op {metrics['op_acc']*100:5.1f}%  "
                  f"({elapsed:.1f}s)")
    final = evaluate(model, task, eval_examples * 4, batch_size, device)
    print(f"  FINAL  EM={final['em']*100:6.2f}%  ptr={final['ptr_acc']*100:5.1f}%  "
          f"op={final['op_acc']*100:5.1f}%  params={n_params(model):,}")
    result = {
        "condition": condition, "seed": seed,
        "config": {
            "d_model": cfg.d_model, "k_substrates": cfg.k_substrates,
            "lateral_weight": getattr(cfg, "lateral_weight", 0.0),
            "n_ode_steps": cfg.n_ode_steps,
        },
        "params": n_params(model),
        "n_steps": n_steps, "batch_size": batch_size, "lr": lr,
        "history": history, "final": final,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"phase1a_{condition}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(CONDITIONS.keys()), default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_examples", type=int, default=512)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="output_ast_phase1")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"VOCAB_SIZE={VOCAB_SIZE}, SEQ_LEN={SEQ_LEN}, SCRIPT_START={SCRIPT_START}, "
          f"K_FIX={K_FIX}, N_NODES={N_NODES}, N_EDIT_OPS={N_EDIT_OPS}, "
          f"PAY_RANGE={PAY_RANGE}")

    conds = [args.condition] if args.condition else list(CONDITIONS.keys())
    results = {}
    for cond in conds:
        results[cond] = run_condition(
            condition=cond, seed=args.seed, n_steps=args.n_steps,
            batch_size=args.batch_size, lr=args.lr, eval_every=args.eval_every,
            eval_examples=args.eval_examples, device=device, out_dir=args.out_dir,
        )

    if len(conds) == 2:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for cond in conds:
            r = results[cond]
            f = r["final"]
            print(f"  {cond:6s}  EM={f['em']*100:6.2f}%  ptr={f['ptr_acc']*100:5.1f}%  "
                  f"op={f['op_acc']*100:5.1f}%  params={r['params']:>8,}")
        delta = results["k2"]["final"]["em"] - results["k1"]["final"]["em"]
        print(f"\nDelta (K2 - K1): {delta*100:+.2f}pp")
        print(f"Phase 1a transfer (K2 - K1 ≥ 5pp): "
              f"{'PASS' if delta >= 0.05 else 'FAIL'}")


if __name__ == "__main__":
    main()
