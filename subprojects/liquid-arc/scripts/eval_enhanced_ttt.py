"""Enhanced Verified TTT — Grid search over steps and unfrozen params.

Tests 4 priority configurations:
  A: 100 steps, geo only (baseline)
  C: 500 steps, geo only (more steps)
  D: 100 steps, geo + FFN[-1] (more params)
  F: 500 steps, geo + FFN[-1] (maximum effort)

Each config applies the verification gate (loss < 0.01).

Usage:
    python scripts/eval_enhanced_ttt.py \
        --checkpoint output_30to50/checkpoints/best.pt \
        --config configs/liquid_arc_zero_scaffold.yaml \
        --data_dir /workspace/fgn-v3/data/arc-repo/data
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel, create_model

FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if not Path(FGN_ROOT).exists():
    FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, FGN_ROOT)
from fgn.tasks.arc import load_arc_tasks, build_sequence, pad_single_to_batch
from liquid_arc.ttt import make_ttt_training_meta, _forward_model


def run_ttt(base_model, task, config, device, ttt_steps, ttt_lr, curv_lambda,
            early_stop, unfreeze_ffn):
    """Run TTT with configurable steps and FFN unfreeze. Returns adapted model + info."""
    seq = build_sequence(task, d4_idx=0, color_perm=None, test_idx=0,
                         max_seq_len=config.max_seq_len)
    if seq is None:
        return None, {"skipped": True, "reason": "seq_too_long"}

    meta = pad_single_to_batch(seq, config.max_seq_len, device)
    ttt_meta = make_ttt_training_meta(meta, seq)
    n_targets = (ttt_meta["target_labels"] != -100).sum().item()
    if n_targets == 0:
        return None, {"skipped": True, "reason": "no_ttt_targets"}

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
        melt_modules.extend([adapted.dynamics.gate_net_linear1,
                             adapted.dynamics.gate_net_linear2])
    else:
        melt_modules.extend([adapted.dynamics.tau_net_linear1,
                             adapted.dynamics.tau_net_linear2])

    if unfreeze_ffn:
        melt_modules.append(adapted.dynamics.ffn[-1])

    n_unfrozen = 0
    for mod in melt_modules:
        for p in mod.parameters():
            p.requires_grad = True
            n_unfrozen += p.numel()

    optimizer = torch.optim.AdamW(
        [p for p in adapted.parameters() if p.requires_grad],
        lr=ttt_lr, weight_decay=0.0)

    ce_history = []
    for step in range(ttt_steps):
        optimizer.zero_grad()
        result = _forward_model(adapted, ttt_meta, device,
                                n_steps=config.n_ode_steps, geo_phase=0)
        xf = result["xform_loss"]
        ttt_ce = xf if xf.item() > 0 else result["ce_loss"]
        loss = ttt_ce + curv_lambda * result["avg_kappa"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in adapted.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        ce_val = ttt_ce.item()
        ce_history.append(ce_val)
        if ce_val < early_stop:
            break

    adapted.eval()
    return adapted, {
        "ttt_steps_used": len(ce_history),
        "ttt_ce_start": ce_history[0],
        "ttt_ce_end": ce_history[-1],
        "n_unfrozen": n_unfrozen,
        "ce_history": ce_history,
    }


def eval_task(model, task, config, device, test_idx=0):
    """Evaluate model on single task. Returns metrics dict or None."""
    seq = build_sequence(task, d4_idx=0, color_perm=None, test_idx=test_idx,
                         max_seq_len=config.max_seq_len)
    if seq is None:
        return None

    meta = pad_single_to_batch(seq, config.max_seq_len, device)
    with torch.no_grad():
        result = _forward_model(model, meta, device, n_steps=config.n_ode_steps)

    logits = result["logits"]
    target_mask = meta["target_mask"][0]
    targets = meta["target_labels"][0, target_mask]
    target_input = meta["target_input_colors"][0, target_mask]
    preds = logits[0, target_mask].argmax(dim=-1)

    correct = (preds == targets)
    cell_acc = correct.float().mean().item()
    solved = correct.all().item()

    xform_mask = targets != target_input
    n_xform = xform_mask.sum().item()
    xform_acc = correct[xform_mask].float().mean().item() if n_xform > 0 else 1.0

    return {"cell_acc": cell_acc, "xform_acc": xform_acc, "solved": solved,
            "n_xform": n_xform, "n_cells": len(targets)}


def run_config(name, base_model, tasks, config, device, ttt_steps, unfreeze_ffn,
               ttt_lr, curv_lambda, early_stop, strict_thresh):
    """Run one configuration across all tasks."""
    params_desc = "geo+ffn" if unfreeze_ffn else "geo"
    print(f"\n{'='*60}")
    print(f"Config {name}: {ttt_steps} steps, {params_desc}")
    print(f"{'='*60}")

    results = []
    n_skipped = 0
    gate = {"verified": 0, "partial": 0, "fallback": 0}
    totals = {k: {"xform_n": 0, "xform_d": 0, "solved": 0}
              for k in ["base", "ttt", "verified"]}
    verified_base_accs = []
    verified_ttt_accs = []

    t0 = time.time()
    for i, task in enumerate(tasks):
        task_id = task.get("task_id", f"task_{i}")

        base = eval_task(base_model, task, config, device)
        if base is None:
            n_skipped += 1
            continue

        adapted, info = run_ttt(base_model, task, config, device,
                                ttt_steps, ttt_lr, curv_lambda,
                                early_stop, unfreeze_ffn)
        if adapted is None:
            n_skipped += 1
            continue

        ttt = eval_task(adapted, task, config, device)
        if ttt is None:
            ttt = base

        final_loss = info["ttt_ce_end"]
        if final_loss < strict_thresh:
            g = "verified"
            verified = ttt
            verified_base_accs.append(base["xform_acc"])
            verified_ttt_accs.append(ttt["xform_acc"])
        elif final_loss < 0.1:
            g = "partial"
            verified = base
        else:
            g = "fallback"
            verified = base
        gate[g] += 1

        n_xform = base["n_xform"]
        totals["base"]["xform_n"] += int(base["xform_acc"] * n_xform)
        totals["base"]["xform_d"] += n_xform
        totals["base"]["solved"] += int(base["solved"])
        totals["ttt"]["xform_n"] += int(ttt["xform_acc"] * n_xform)
        totals["ttt"]["xform_d"] += n_xform
        totals["ttt"]["solved"] += int(ttt["solved"])
        totals["verified"]["xform_n"] += int(verified["xform_acc"] * n_xform)
        totals["verified"]["xform_d"] += n_xform
        totals["verified"]["solved"] += int(verified["solved"])

        results.append({
            "task_id": task_id, "gate": g,
            "base_xform": base["xform_acc"], "ttt_xform": ttt["xform_acc"],
            "base_solved": base["solved"], "ttt_solved": ttt["solved"],
            "ttt_loss": final_loss, "ttt_steps": info["ttt_steps_used"],
        })

        del adapted
        if device.type == "cuda":
            torch.cuda.empty_cache()

        n_eval = len(results)
        if n_eval % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_eval}/{len(tasks)}] {elapsed:.0f}s "
                  f"({elapsed/n_eval:.1f}s/task) "
                  f"v={gate['verified']} p={gate['partial']} f={gate['fallback']}")

    elapsed = time.time() - t0
    n_eval = len(results)

    def sdiv(n, d): return n / d if d > 0 else 0.0

    # Verified-only stats
    v_base = sum(verified_base_accs) / len(verified_base_accs) if verified_base_accs else 0
    v_ttt = sum(verified_ttt_accs) / len(verified_ttt_accs) if verified_ttt_accs else 0

    new_solves = sum(1 for r in results if r["ttt_solved"] and not r["base_solved"]
                     and r["gate"] == "verified")
    regressions = sum(1 for r in results if r["base_solved"] and not r["ttt_solved"]
                      and r["gate"] == "verified")

    summary = {
        "name": name,
        "steps": ttt_steps,
        "params": params_desc,
        "n_eval": n_eval,
        "n_skipped": n_skipped,
        "time": elapsed,
        "gate_verified": gate["verified"],
        "gate_partial": gate["partial"],
        "gate_fallback": gate["fallback"],
        "base_xform": sdiv(totals["base"]["xform_n"], totals["base"]["xform_d"]),
        "ttt_xform": sdiv(totals["ttt"]["xform_n"], totals["ttt"]["xform_d"]),
        "verified_xform": sdiv(totals["verified"]["xform_n"], totals["verified"]["xform_d"]),
        "base_solved": totals["base"]["solved"],
        "ttt_solved": totals["ttt"]["solved"],
        "verified_solved": totals["verified"]["solved"],
        "verified_only_base_acc": v_base,
        "verified_only_ttt_acc": v_ttt,
        "new_solves": new_solves,
        "regressions": regressions,
    }

    print(f"\n  Gate: verified={gate['verified']} partial={gate['partial']} fallback={gate['fallback']}")
    print(f"  Verified tasks: base={v_base*100:.1f}% → ttt={v_ttt*100:.1f}% (Δ={'+' if v_ttt>=v_base else ''}{(v_ttt-v_base)*100:.1f}%)")
    print(f"  Overall verified TTT: {summary['verified_xform']*100:.1f}%")
    print(f"  Solved: base={totals['base']['solved']} ttt={totals['ttt']['solved']} verified={totals['verified']['solved']}")
    print(f"  New solves: {new_solves}, regressions: {regressions}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/max(n_eval,1):.1f}s/task)")

    return summary, results


def main():
    parser = argparse.ArgumentParser(description="Enhanced verified TTT")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument("--max_tasks", type=int, default=400)
    parser.add_argument("--ttt_lr", type=float, default=0.001)
    parser.add_argument("--curv_lambda", type=float, default=0.01)
    parser.add_argument("--early_stop", type=float, default=0.01)
    parser.add_argument("--strict_threshold", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="output_enhanced_ttt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LiquidARCConfig.from_yaml(args.config)

    print(f"Loading model from {args.checkpoint}...")
    model = create_model(config, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"]
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    model.eval()
    print(f"  Loaded at step {ckpt.get('step', '?')}")

    all_tasks = load_arc_tasks(args.data_dir)
    tasks = all_tasks.get(args.split, [])
    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]
    print(f"  Tasks: {len(tasks)} ({args.split} split)")

    os.makedirs(args.output_dir, exist_ok=True)

    # Priority configs: A, C, D, F
    configs_to_run = [
        ("A", 100, False),   # baseline
        ("C", 500, False),   # more steps
        ("D", 100, True),    # more params
        ("F", 500, True),    # maximum effort
    ]

    all_summaries = []
    all_details = {}

    for name, steps, ffn in configs_to_run:
        summary, details = run_config(
            name, model, tasks, config, device,
            ttt_steps=steps, unfreeze_ffn=ffn,
            ttt_lr=args.ttt_lr, curv_lambda=args.curv_lambda,
            early_stop=args.early_stop, strict_thresh=args.strict_threshold)
        all_summaries.append(summary)
        all_details[name] = details

    # Comparison table
    print(f"\n{'='*60}")
    print("Comparison Across Configs")
    print(f"{'='*60}")
    print(f"{'Config':<8} {'Steps':<7} {'Params':<10} {'Verified':<10} "
          f"{'V.Base%':<9} {'V.TTT%':<9} {'Δ':<8} {'Solved':<8} {'New':<5}")
    print("-" * 80)
    for s in all_summaries:
        delta = s["verified_only_ttt_acc"] - s["verified_only_base_acc"]
        print(f"{s['name']:<8} {s['steps']:<7} {s['params']:<10} "
              f"{s['gate_verified']:<10} "
              f"{s['verified_only_base_acc']*100:<9.1f} "
              f"{s['verified_only_ttt_acc']*100:<9.1f} "
              f"{'+' if delta>=0 else ''}{delta*100:<7.1f} "
              f"{s['verified_solved']:<8} {s['new_solves']:<5}")

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"summaries": all_summaries, "details": all_details}, f, indent=2)

    # Save report
    report_lines = ["# Enhanced TTT Evaluation Report\n"]
    report_lines.append("## Comparison\n")
    report_lines.append("| Config | Steps | Params | Verified | V.Base | V.TTT | Delta | Solved | New Solves |")
    report_lines.append("|--------|-------|--------|----------|--------|-------|-------|--------|-----------|")
    for s in all_summaries:
        delta = s["verified_only_ttt_acc"] - s["verified_only_base_acc"]
        report_lines.append(
            f"| {s['name']} | {s['steps']} | {s['params']} | "
            f"{s['gate_verified']} | {s['verified_only_base_acc']*100:.1f}% | "
            f"{s['verified_only_ttt_acc']*100:.1f}% | "
            f"{'+' if delta>=0 else ''}{delta*100:.1f}% | "
            f"{s['verified_solved']} | {s['new_solves']} |")

    report_lines.append("\n## Per-Config Details\n")
    for s in all_summaries:
        report_lines.append(f"### Config {s['name']}: {s['steps']} steps, {s['params']}")
        report_lines.append(f"- Evaluated: {s['n_eval']}, Skipped: {s['n_skipped']}")
        report_lines.append(f"- Gate: verified={s['gate_verified']}, partial={s['gate_partial']}, fallback={s['gate_fallback']}")
        report_lines.append(f"- Overall: base={s['base_xform']*100:.1f}%, ttt={s['ttt_xform']*100:.1f}%, verified={s['verified_xform']*100:.1f}%")
        report_lines.append(f"- Solved: base={s['base_solved']}, ttt={s['ttt_solved']}, verified={s['verified_solved']}")
        report_lines.append(f"- Time: {s['time']:.0f}s ({s['time']/max(s['n_eval'],1):.1f}s/task)")
        report_lines.append("")

    report_path = os.path.join(args.output_dir, "ENHANCED_TTT_REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nResults saved to {results_path}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
