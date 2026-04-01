"""Verified TTT — Gated Test-Time Training for LiquidARC.

Runs standard TTT (adapt MetricNet + TauNet + W_o on demo pairs), then gates
the adaptation based on TTT convergence. If TTT converged (low final loss),
the adapted model likely found the right rule. If not, fall back to base model.

Usage:
    python scripts/eval_verified_ttt.py \
        --checkpoint output_30to50/checkpoints/best.pt \
        --config configs/liquid_arc_zero_scaffold.yaml \
        --data_dir /workspace/fgn-v3/data/arc-repo/data \
        --n_tasks 400
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel, create_model

FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if not Path(FGN_ROOT).exists():
    FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, FGN_ROOT)
from fgn.tasks.arc import (
    load_arc_tasks, build_sequence, pad_single_to_batch,
    ROLE_OUTPUT_DEMO, ROLE_TEST_OUTPUT,
)
from liquid_arc.ttt import make_ttt_training_meta, _forward_model


def run_ttt_with_tracking(
    base_model: LiquidARCModel,
    task: dict,
    config: LiquidARCConfig,
    device: torch.device,
    ttt_steps: int = 100,
    ttt_lr: float = 0.001,
    curv_lambda: float = 0.01,
    early_stop: float = 0.01,
):
    """Run TTT and return adapted model + convergence info."""

    seq = build_sequence(task, d4_idx=0, color_perm=None, test_idx=0,
                         max_seq_len=config.max_seq_len)
    if seq is None:
        return None, None, {"skipped": True, "reason": "seq_too_long"}

    meta = pad_single_to_batch(seq, config.max_seq_len, device)
    ttt_meta = make_ttt_training_meta(meta, seq)
    n_ttt_targets = (ttt_meta["target_labels"] != -100).sum().item()
    if n_ttt_targets == 0:
        return None, None, {"skipped": True, "reason": "no_ttt_targets"}

    # Deep copy + selective unfreeze
    adapted = copy.deepcopy(base_model)
    adapted.to(device)
    adapted.train()

    for p in adapted.parameters():
        p.requires_grad = False

    melt_modules = [
        adapted.dynamics.metric_net_linear1,
        adapted.dynamics.metric_net_linear2,
        adapted.dynamics.W_o,
    ]
    if getattr(config, 'channel_gate_enabled', False):
        melt_modules.extend([
            adapted.dynamics.gate_net_linear1,
            adapted.dynamics.gate_net_linear2,
        ])
    else:
        melt_modules.extend([
            adapted.dynamics.tau_net_linear1,
            adapted.dynamics.tau_net_linear2,
        ])

    n_unfrozen = 0
    for mod in melt_modules:
        for p in mod.parameters():
            p.requires_grad = True
            n_unfrozen += p.numel()

    optimizer = torch.optim.AdamW(
        [p for p in adapted.parameters() if p.requires_grad],
        lr=ttt_lr, weight_decay=0.0,
    )

    # TTT inner loop
    ce_history = []
    for step in range(ttt_steps):
        optimizer.zero_grad()
        result = _forward_model(adapted, ttt_meta, device,
                                n_steps=config.n_ode_steps, geo_phase=0)
        xf = result["xform_loss"]
        ttt_ce = xf if xf.item() > 0 else result["ce_loss"]
        curv = curv_lambda * result["avg_kappa"]
        loss = ttt_ce + curv
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in adapted.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        ce_val = ttt_ce.item()
        ce_history.append(ce_val)
        if ce_val < early_stop:
            break

    adapted.eval()
    info = {
        "ttt_steps_used": len(ce_history),
        "ttt_ce_start": ce_history[0] if ce_history else 0.0,
        "ttt_ce_end": ce_history[-1] if ce_history else 0.0,
        "n_unfrozen": n_unfrozen,
    }
    return adapted, base_model, info


def evaluate_single_task(model, task, config, device, test_idx=0):
    """Run inference on a single task, return xform_acc, cell_acc, solved."""
    seq = build_sequence(task, d4_idx=0, color_perm=None,
                         test_idx=test_idx, max_seq_len=config.max_seq_len)
    if seq is None:
        return None

    meta = pad_single_to_batch(seq, config.max_seq_len, device)
    with torch.no_grad():
        result = _forward_model(model, meta, device, n_steps=config.n_ode_steps)

    # Extract predictions
    logits = result["logits"]  # [1, N, 10]
    target_mask = meta["target_mask"][0]  # [N]
    targets = meta["target_labels"][0, target_mask]  # [N_target]
    target_input = meta["target_input_colors"][0, target_mask]
    preds = logits[0, target_mask].argmax(dim=-1)

    cell_correct = (preds == targets)
    cell_acc = cell_correct.float().mean().item()
    solved = cell_correct.all().item()

    xform_mask = targets != target_input
    n_xform = xform_mask.sum().item()
    if n_xform > 0:
        xform_acc = cell_correct[xform_mask].float().mean().item()
    else:
        xform_acc = 1.0

    return {
        "cell_acc": cell_acc,
        "xform_acc": xform_acc,
        "solved": solved,
        "n_xform": n_xform,
        "n_cells": len(targets),
    }


def main():
    parser = argparse.ArgumentParser(description="Verified TTT evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument("--n_tasks", type=int, default=400)
    parser.add_argument("--ttt_steps", type=int, default=100)
    parser.add_argument("--ttt_lr", type=float, default=0.001)
    parser.add_argument("--curv_lambda", type=float, default=0.01)
    parser.add_argument("--early_stop", type=float, default=0.01)
    parser.add_argument("--strict_threshold", type=float, default=0.01,
                        help="TTT loss threshold for 'verified' gate")
    parser.add_argument("--partial_threshold", type=float, default=0.1,
                        help="TTT loss threshold for 'partial' gate")
    parser.add_argument("--output_dir", type=str, default="output_verified_ttt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LiquidARCConfig.from_yaml(args.config)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = create_model(config, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"]
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    model.eval()
    print(f"  Loaded at step {ckpt.get('step', '?')}")

    # Load tasks
    all_tasks = load_arc_tasks(args.data_dir)
    tasks = all_tasks.get(args.split, [])
    if args.n_tasks > 0:
        tasks = tasks[:args.n_tasks]
    print(f"  Tasks: {len(tasks)} ({args.split} split)")
    print(f"  TTT: {args.ttt_steps} steps, lr={args.ttt_lr}")
    print(f"  Gates: strict<{args.strict_threshold}, partial<{args.partial_threshold}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Accumulators
    all_results = []
    n_skipped = 0
    n_evaluated = 0

    # Totals for summary
    totals = {
        "base": {"xform_n": 0, "xform_d": 0, "solved": 0},
        "ttt": {"xform_n": 0, "xform_d": 0, "solved": 0},
        "verified": {"xform_n": 0, "xform_d": 0, "solved": 0},
        "partial": {"xform_n": 0, "xform_d": 0, "solved": 0},
    }
    gate_counts = {"verified_ttt": 0, "partial_ttt": 0, "base_fallback": 0}

    t0 = time.time()
    for task_idx, task in enumerate(tasks):
        task_id = task.get("task_id", f"task_{task_idx}")

        # Base model prediction
        base_result = evaluate_single_task(model, task, config, device, test_idx=0)
        if base_result is None:
            n_skipped += 1
            continue

        # TTT adaptation
        adapted, _, ttt_info = run_ttt_with_tracking(
            model, task, config, device,
            ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr,
            curv_lambda=args.curv_lambda, early_stop=args.early_stop)

        if adapted is None:
            # TTT couldn't run — use base result
            task_result = {
                "task_id": task_id,
                "base_xform_acc": base_result["xform_acc"],
                "base_solved": base_result["solved"],
                "ttt_xform_acc": base_result["xform_acc"],
                "ttt_solved": base_result["solved"],
                "verified_ttt_xform_acc": base_result["xform_acc"],
                "verified_ttt_solved": base_result["solved"],
                "gate_decision": "base_fallback",
                "ttt_skipped": True,
                "skip_reason": ttt_info.get("reason", "unknown"),
            }
            gate_counts["base_fallback"] += 1
            n_xform = base_result["n_xform"]

            for key in ["base", "ttt", "verified", "partial"]:
                totals[key]["xform_n"] += int(base_result["xform_acc"] * n_xform)
                totals[key]["xform_d"] += n_xform
                totals[key]["solved"] += int(base_result["solved"])

            all_results.append(task_result)
            n_evaluated += 1
            continue

        # TTT prediction (always apply)
        ttt_result = evaluate_single_task(adapted, task, config, device, test_idx=0)
        if ttt_result is None:
            ttt_result = base_result  # fallback

        # Gate decision
        final_loss = ttt_info["ttt_ce_end"]
        if final_loss < args.strict_threshold:
            gate = "verified_ttt"
            verified_result = ttt_result
        elif final_loss < args.partial_threshold:
            gate = "partial_ttt"
            verified_result = ttt_result
        else:
            gate = "base_fallback"
            verified_result = base_result

        gate_counts[gate] += 1

        # Partial gate result: use TTT if loss < partial_threshold, else base
        if final_loss < args.partial_threshold:
            partial_result = ttt_result
        else:
            partial_result = base_result

        n_xform = base_result["n_xform"]
        task_result = {
            "task_id": task_id,
            "base_xform_acc": base_result["xform_acc"],
            "base_cell_acc": base_result["cell_acc"],
            "base_solved": base_result["solved"],
            "ttt_xform_acc": ttt_result["xform_acc"],
            "ttt_cell_acc": ttt_result["cell_acc"],
            "ttt_solved": ttt_result["solved"],
            "verified_ttt_xform_acc": verified_result["xform_acc"],
            "verified_ttt_solved": verified_result["solved"],
            "partial_ttt_xform_acc": partial_result["xform_acc"],
            "partial_ttt_solved": partial_result["solved"],
            "ttt_final_loss": final_loss,
            "ttt_steps_used": ttt_info["ttt_steps_used"],
            "ttt_ce_start": ttt_info["ttt_ce_start"],
            "gate_decision": gate,
            "n_xform": n_xform,
        }

        totals["base"]["xform_n"] += int(base_result["xform_acc"] * n_xform)
        totals["base"]["xform_d"] += n_xform
        totals["base"]["solved"] += int(base_result["solved"])

        totals["ttt"]["xform_n"] += int(ttt_result["xform_acc"] * n_xform)
        totals["ttt"]["xform_d"] += n_xform
        totals["ttt"]["solved"] += int(ttt_result["solved"])

        totals["verified"]["xform_n"] += int(verified_result["xform_acc"] * n_xform)
        totals["verified"]["xform_d"] += n_xform
        totals["verified"]["solved"] += int(verified_result["solved"])

        totals["partial"]["xform_n"] += int(partial_result["xform_acc"] * n_xform)
        totals["partial"]["xform_d"] += n_xform
        totals["partial"]["solved"] += int(partial_result["solved"])

        all_results.append(task_result)
        n_evaluated += 1

        # Clean up adapted model
        del adapted
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if n_evaluated % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_evaluated}/{len(tasks)}] {elapsed:.1f}s "
                  f"({elapsed/n_evaluated:.1f}s/task) "
                  f"gates: v={gate_counts['verified_ttt']} "
                  f"p={gate_counts['partial_ttt']} "
                  f"f={gate_counts['base_fallback']}")

    elapsed = time.time() - t0

    # Summary
    def safe_div(n, d):
        return n / d if d > 0 else 0.0

    lines = []
    lines.append("=" * 60)
    lines.append("Verified TTT Results")
    lines.append("=" * 60)
    lines.append(f"Checkpoint: {args.checkpoint}")
    lines.append(f"Tasks evaluated: {n_evaluated}")
    lines.append(f"Tasks skipped: {n_skipped}")
    lines.append(f"Time: {elapsed:.1f}s ({elapsed/max(n_evaluated,1):.1f}s/task)")
    lines.append("")

    lines.append("--- Transform Accuracy ---")
    for label, key in [("Base model", "base"), ("Standard TTT", "ttt"),
                        (f"Verified TTT (loss<{args.strict_threshold})", "verified"),
                        (f"Partial TTT (loss<{args.partial_threshold})", "partial")]:
        d = totals[key]
        acc = safe_div(d["xform_n"], d["xform_d"])
        lines.append(f"{label:40s}: {acc*100:.1f}%")

    lines.append("")
    lines.append("--- Task Solve Rate ---")
    for label, key in [("Base model", "base"), ("Standard TTT", "ttt"),
                        ("Verified TTT", "verified"), ("Partial TTT", "partial")]:
        d = totals[key]
        lines.append(f"{label:40s}: {d['solved']}/{n_evaluated} "
                     f"({safe_div(d['solved'], n_evaluated)*100:.1f}%)")

    lines.append("")
    lines.append("--- Gate Statistics ---")
    lines.append(f"TTT converged (loss < {args.strict_threshold}): "
                 f"{gate_counts['verified_ttt']} ({safe_div(gate_counts['verified_ttt'], n_evaluated)*100:.1f}%)")
    lines.append(f"TTT partially converged (loss < {args.partial_threshold}): "
                 f"{gate_counts['partial_ttt']} ({safe_div(gate_counts['partial_ttt'], n_evaluated)*100:.1f}%)")
    lines.append(f"TTT failed (loss >= {args.partial_threshold}): "
                 f"{gate_counts['base_fallback']} ({safe_div(gate_counts['base_fallback'], n_evaluated)*100:.1f}%)")

    # New solves / regressions
    lines.append("")
    lines.append("--- Verified TTT vs Base ---")
    new_solves_v = sum(1 for r in all_results
                       if r.get("verified_ttt_solved") and not r.get("base_solved"))
    regressions_v = sum(1 for r in all_results
                        if r.get("base_solved") and not r.get("verified_ttt_solved"))
    lines.append(f"New solves (verified TTT solves, base doesn't): {new_solves_v}")
    lines.append(f"Regressions (base solves, verified TTT doesn't): {regressions_v}")
    lines.append(f"Net new solves: {new_solves_v - regressions_v}")

    lines.append("")
    lines.append("--- Standard TTT vs Base ---")
    new_solves_t = sum(1 for r in all_results
                       if r.get("ttt_solved") and not r.get("base_solved"))
    regressions_t = sum(1 for r in all_results
                        if r.get("base_solved") and not r.get("ttt_solved"))
    lines.append(f"New solves (TTT solves, base doesn't): {new_solves_t}")
    lines.append(f"Regressions (base solves, TTT doesn't): {regressions_t}")
    lines.append(f"Net new solves: {new_solves_t - regressions_t}")

    # Per-gate accuracy breakdown
    lines.append("")
    lines.append("--- Per-Gate Accuracy (TTT result only, not gated) ---")
    for gate_type in ["verified_ttt", "partial_ttt", "base_fallback"]:
        gate_tasks = [r for r in all_results if r.get("gate_decision") == gate_type]
        if gate_tasks:
            ttt_accs = [r["ttt_xform_acc"] for r in gate_tasks if "ttt_xform_acc" in r]
            base_accs = [r["base_xform_acc"] for r in gate_tasks if "base_xform_acc" in r]
            avg_ttt = sum(ttt_accs) / len(ttt_accs) if ttt_accs else 0
            avg_base = sum(base_accs) / len(base_accs) if base_accs else 0
            lines.append(f"{gate_type} ({len(gate_tasks)} tasks): "
                         f"base={avg_base*100:.1f}%, ttt={avg_ttt*100:.1f}%, "
                         f"delta={'+' if avg_ttt >= avg_base else ''}{(avg_ttt-avg_base)*100:.1f}%")

    summary = "\n".join(lines)
    print(summary)

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "tasks": all_results}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save report
    report_path = os.path.join(args.output_dir, "VERIFIED_TTT_REPORT.md")
    with open(report_path, "w") as f:
        f.write(f"# Verified TTT Evaluation Report\n\n```\n{summary}\n```\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
