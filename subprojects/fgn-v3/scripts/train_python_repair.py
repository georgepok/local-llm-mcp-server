"""Phase 2 trainer — Python code repair on LiquidARC.

Tests whether the K=2 coupled-substrate mechanism (validated on Phase 0
NumPy MLP toy) gives real signal on a real-data task: programmatic
bug-fix on Python stdlib functions, GPT-2 tokenisation, AR teacher-forced
training with held-out exact-match recovery as the eval.

Run a single condition:
    python scripts/train_python_repair.py --condition k1 --corpus /workspace/data/py_bugs.jsonl
Or sweep K=1 vs K=2:
    python scripts/train_python_repair.py --corpus /workspace/data/py_bugs.jsonl
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
from fgn.liquid_model import LiquidSequenceModel
from fgn.tasks.python_repair import PythonRepairTask


CONDITIONS: Dict[str, Dict] = {
    "k1": {"k_substrates": 1, "lateral_weight": 0.0, "d_model": 128},
    "k2": {"k_substrates": 2, "lateral_weight": 0.5, "d_model": 96},
}


def build_config(condition: str, seq_len: int, vocab_size: int,
                 d_model_override: int = 0,
                 n_ode_steps: int = 8,
                 halting_enabled: bool = False,
                 halting_min_steps: int = 4,
                 ponder_kl_lambda: float = 0.0,
                 deep_supervision_enabled: bool = False,
                 ponder_kl_prior_rate: float = 0.0625,
                 halting_ponder_lambda: float = 0.01) -> FGNConfig:
    cond = CONDITIONS[condition]
    d_model = d_model_override if d_model_override > 0 else cond["d_model"]
    return FGNConfig(
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
        halting_enabled=halting_enabled,
        halting_min_steps=halting_min_steps,
        deep_supervision_enabled=deep_supervision_enabled,
        ponder_kl_lambda=ponder_kl_lambda,
        ponder_kl_prior_rate=ponder_kl_prior_rate,
        halting_ponder_lambda=halting_ponder_lambda,
        k_substrates=cond["k_substrates"],
        lateral_weight=cond["lateral_weight"],
        use_torch_compile=False,
        dropout=0.0,
        metric_type="diagonal",
        metric_rank=0,
        metric_lr_mult=1.0,
        rezero_enabled=False,
    )


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_condition(condition: str, seed: int, corpus: str, n_steps: int,
                  batch_size: int, lr: float, eval_every: int,
                  eval_examples: int, seq_len: int,
                  device: torch.device, out_dir: str,
                  d_model_override: int = 0,
                  n_ode_steps: int = 8,
                  log_every: int = 50,
                  halting_enabled: bool = False,
                  halting_min_steps: int = 4,
                  ponder_kl_lambda: float = 0.0,
                  deep_supervision_enabled: bool = False,
                  ponder_kl_prior_rate: float = 0.0625,
                  halting_ponder_lambda: float = 0.01) -> Dict:
    torch.manual_seed(seed)

    train_task = PythonRepairTask(corpus, seq_len=seq_len, seed=seed,
                                  split="train")
    eval_task = PythonRepairTask(corpus, seq_len=seq_len, seed=seed,
                                 split="eval")

    cfg = build_config(condition, seq_len=seq_len,
                       vocab_size=train_task.vocab_size,
                       d_model_override=d_model_override,
                       n_ode_steps=n_ode_steps,
                       halting_enabled=halting_enabled,
                       halting_min_steps=halting_min_steps,
                       ponder_kl_lambda=ponder_kl_lambda,
                       deep_supervision_enabled=deep_supervision_enabled,
                       ponder_kl_prior_rate=ponder_kl_prior_rate,
                       halting_ponder_lambda=halting_ponder_lambda)
    model = LiquidSequenceModel(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    print(f"\n=== condition={condition} seed={seed} "
          f"params={n_params(model):,} d={cfg.d_model} K={cfg.k_substrates} "
          f"train_n={len(train_task)} eval_n={len(eval_task)} ===")

    history: List[Dict] = []
    losses_window: List[float] = []
    t0 = time.time()

    steps_window: List[float] = []
    for step in range(1, n_steps + 1):
        ids, labels, _ = train_task.generate_batch(batch_size, device=device)
        out = model(ids, labels=labels)
        # When halting is enabled, model returns task_loss (deep-sup CE +
        # ponder KL). Use that as the optimizer's loss; otherwise plain ce_loss.
        loss = out.get("loss", out["ce_loss"])
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        loss_val = float(loss.item())
        losses_window.append(loss_val)
        if len(losses_window) > 200:
            losses_window = losses_window[-200:]
        if halting_enabled and "steps_mean" in out:
            steps_window.append(float(out["steps_mean"].item()))
            if len(steps_window) > 200:
                steps_window = steps_window[-200:]
        # Intermediate "heartbeat" log — cheap, every log_every steps. Lets
        # us see divergence / NaN / hangs without waiting for full eval.
        # Uses the same line format as the eval print (EM/tok_acc shown as
        # last-known cached values, marked with [hb] suffix).
        if log_every > 0 and step % log_every == 0 and step % eval_every != 0:
            avg_loss = sum(losses_window) / len(losses_window)
            elapsed = time.time() - t0
            tps = (step * batch_size * seq_len) / max(elapsed, 1)
            nan = " NaN!" if (loss_val != loss_val) else ""
            last_em = history[-1]["em"] if history else 0.0
            last_tok = history[-1]["tok_acc"] if history else 0.0
            steps_str = ""
            if steps_window:
                avg_n = sum(steps_window) / len(steps_window)
                steps_str = f"  n_used {avg_n:5.1f}"
            print(f"  step {step:5d}  loss {avg_loss:.4f}  "
                  f"EM {last_em*100:6.2f}%  "
                  f"tok_acc {last_tok*100:5.1f}%{steps_str}  "
                  f"({elapsed:.0f}s, {tps:.0f} tok/s) [hb]{nan}", flush=True)
        if step == 1 or step % eval_every == 0 or step == n_steps:
            metrics = eval_task.evaluate(model, eval_examples, batch_size,
                                         device)
            avg_loss = sum(losses_window) / len(losses_window)
            elapsed = time.time() - t0
            log = {"step": step, "loss": avg_loss, **metrics,
                   "elapsed_s": elapsed}
            if steps_window:
                log["n_used_train"] = sum(steps_window) / len(steps_window)
            history.append(log)
            steps_str = ""
            if steps_window:
                avg_n = sum(steps_window) / len(steps_window)
                steps_str = f"  n_used {avg_n:5.1f}"
            print(f"  step {step:5d}  loss {avg_loss:.4f}  "
                  f"EM {metrics['em']*100:6.2f}%  "
                  f"tok_acc {metrics['tok_acc']*100:5.1f}%{steps_str}  "
                  f"({elapsed:.1f}s)", flush=True)

    final = eval_task.evaluate(model, len(eval_task), batch_size, device)
    print(f"  FINAL  EM={final['em']*100:6.2f}%  "
          f"tok_acc={final['tok_acc']*100:5.1f}%  "
          f"params={n_params(model):,}  ({time.time() - t0:.1f}s)")

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
        "corpus": corpus,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"phase2a_{condition}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  wrote {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/workspace/data/py_bugs_smoke.jsonl")
    ap.add_argument("--condition", choices=list(CONDITIONS.keys()),
                    default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=400)
    ap.add_argument("--eval_examples", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=384)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="output_python_repair")
    ap.add_argument("--d_model", type=int, default=0,
                    help="Override condition's d_model (0 = use condition default)")
    ap.add_argument("--n_ode_steps", type=int, default=8,
                    help="Override default n_ode_steps")
    ap.add_argument("--log_every", type=int, default=50,
                    help="Print loss/throughput every N steps (0 to disable)")
    ap.add_argument("--halting_enabled", action="store_true",
                    help="Enable adaptive PonderNet halting (per-position halt head)")
    ap.add_argument("--halting_min_steps", type=int, default=4,
                    help="Minimum ODE steps before halt is allowed")
    ap.add_argument("--ponder_kl_lambda", type=float, default=0.0,
                    help="KL(p_halt || Geom(prior_rate)) weight (regularizes against using full budget)")
    ap.add_argument("--deep_supervision_enabled", action="store_true",
                    help="Apply per-step CE weighted by halt distribution")
    ap.add_argument("--ponder_kl_prior_rate", type=float, default=0.0625,
                    help="Geometric prior rate for halting depth (1/16 → mean depth 16)")
    ap.add_argument("--halting_ponder_lambda", type=float, default=0.01,
                    help="Compute-cost penalty weight (λ × E[steps_used/K]) — explicit pressure to halt early")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"corpus = {args.corpus}, seq_len = {args.seq_len}")

    conds = [args.condition] if args.condition else list(CONDITIONS.keys())
    results = {}
    for cond in conds:
        results[cond] = run_condition(
            d_model_override=args.d_model,
            n_ode_steps=args.n_ode_steps,
            log_every=args.log_every,
            halting_enabled=args.halting_enabled,
            halting_min_steps=args.halting_min_steps,
            ponder_kl_lambda=args.ponder_kl_lambda,
            deep_supervision_enabled=args.deep_supervision_enabled,
            ponder_kl_prior_rate=args.ponder_kl_prior_rate,
            halting_ponder_lambda=args.halting_ponder_lambda,
            condition=cond, seed=args.seed, corpus=args.corpus,
            n_steps=args.n_steps, batch_size=args.batch_size, lr=args.lr,
            eval_every=args.eval_every, eval_examples=args.eval_examples,
            seq_len=args.seq_len, device=device, out_dir=args.out_dir,
        )

    if len(conds) == 2:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for cond in conds:
            r = results[cond]
            f = r["final"]
            print(f"  {cond:6s}  EM={f['em']*100:6.2f}%  "
                  f"tok_acc={f['tok_acc']*100:5.1f}%  "
                  f"params={r['params']:>10,}")
        delta = results["k2"]["final"]["em"] - results["k1"]["final"]["em"]
        print(f"\nDelta (K2 - K1): {delta*100:+.2f}pp")
        print(f"Phase 2a transfer (K2 - K1 ≥ 5pp): "
              f"{'PASS' if delta >= 0.05 else 'FAIL'}")


if __name__ == "__main__":
    main()
