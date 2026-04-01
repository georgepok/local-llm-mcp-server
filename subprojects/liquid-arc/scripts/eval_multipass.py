"""Multi-pass inference diagnostic for LiquidARC.

Evaluates whether the model's errors are refinable (fixable by seeing its own output)
or fundamental (wrong rule entirely).

Pass 1: [demos, test_in] → predicted_out_1
Pass 2: [demos, test_in, predicted_out_1 as extra demo, test_in] → predicted_out_2
Pass 3: [demos, test_in, predicted_out_2 as extra demo, test_in] → predicted_out_3

Uses frozen model weights — no training, no gradients, no TTT.

Usage:
    python scripts/eval_multipass.py \
        --checkpoint output_30to50/checkpoints/best.pt \
        --config configs/liquid_arc_zero_scaffold.yaml \
        --data_dir /workspace/fgn-v3/data/arc-repo/data \
        --n_passes 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel, create_model

FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if not Path(FGN_ROOT).exists():
    FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, FGN_ROOT)
from fgn.tasks.arc import (
    load_arc_tasks, build_sequence, pad_single_to_batch,
    grid_to_cells, apply_d4, apply_color_perm,
    ROLE_INPUT_DEMO, ROLE_OUTPUT_DEMO, ROLE_TEST_INPUT, ROLE_TEST_OUTPUT,
    SEP_DEMO_IO, SEP_BETWEEN_DEMOS, SEP_BEFORE_TEST_IN, SEP_BEFORE_TEST_OUT,
    PAD_COLOR, PAD_COORD,
)


def build_sequence_with_extra_demo(
    task: dict,
    extra_input: List[List[int]],
    extra_output: List[List[int]],
    test_idx: int = 0,
    max_seq_len: int = 2048,
) -> Optional[Dict[str, torch.Tensor]]:
    """Build sequence with an extra demo pair (the model's own prediction).

    Adds (extra_input → extra_output) as the LAST demo pair before the test,
    then appends the test input/output as usual.

    If the full sequence exceeds max_seq_len, drops the FIRST original demo
    to make room (keeps the extra demo + as many originals as fit).
    """
    demos = task["train"]
    tests = task["test"]
    if test_idx >= len(tests):
        test_idx = 0

    # Estimate sequence length with extra demo
    test_pair = tests[test_idx]
    extra_cells = sum(len(r) for r in extra_input) + sum(len(r) for r in extra_output)
    test_cells = sum(len(r) for r in test_pair["input"]) + sum(len(r) for r in test_pair["output"])
    demo_cells = sum(
        sum(len(r) for r in d["input"]) + sum(len(r) for r in d["output"])
        for d in demos
    )
    # +separators: 1 per grid boundary + between demos
    total_est = demo_cells + extra_cells + test_cells + (len(demos) + 1) * 3 + 5

    # If too long, build a reduced task dropping earliest demos
    demo_list = list(demos)
    while total_est > max_seq_len and len(demo_list) > 1:
        dropped = demo_list.pop(0)
        dropped_cells = sum(len(r) for r in dropped["input"]) + sum(len(r) for r in dropped["output"])
        total_est -= dropped_cells + 3  # cells + separators

    # Build the augmented task dict
    augmented_task = {
        "train": demo_list + [{"input": extra_input, "output": extra_output}],
        "test": tests,
        "task_id": task.get("task_id", "unknown"),
    }

    return build_sequence(augmented_task, d4_idx=0, color_perm=None,
                          test_idx=test_idx, max_seq_len=max_seq_len)


def run_inference(model, meta, device, n_steps=16):
    """Run a single forward pass, return logits and result dict."""
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            result = model(
                colors=meta["colors"],
                xs=meta["xs"],
                ys=meta["ys"],
                roles=meta["roles"],
                sep_mask=meta["sep_mask"],
                sep_types=meta["sep_types"],
                target_mask=meta["target_mask"],
                target_labels=meta["target_labels"],
                context_mask=meta["context_mask"],
                grid_ids=meta.get("grid_ids"),
                lengths=meta.get("lengths"),
                target_input_colors=meta.get("target_input_colors"),
                n_steps=n_steps,
            )
    return result


def extract_predictions(result, meta):
    """Extract predicted colors from model logits at target positions."""
    logits = result["logits"]  # [1, N, 10]
    target_mask = meta["target_mask"][0]  # [N]
    target_logits = logits[0, target_mask]  # [N_target, 10]
    preds = target_logits.argmax(dim=-1)  # [N_target]
    return preds


def preds_to_grid(preds: torch.Tensor, shape: Tuple[int, int]) -> List[List[int]]:
    """Convert flat predictions to 2D grid (row-major)."""
    H, W = shape
    grid = []
    idx = 0
    for y in range(H):
        row = []
        for x in range(W):
            if idx < len(preds):
                row.append(preds[idx].item())
            else:
                row.append(0)
            idx += 1
        grid.append(row)
    return grid


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor,
                    input_colors: torch.Tensor):
    """Compute cell accuracy, transform accuracy, and solve status."""
    correct = (preds == targets)
    cell_acc = correct.float().mean().item()

    # Transform cells: where target != input
    xform_mask = targets != input_colors
    n_xform = xform_mask.sum().item()
    if n_xform > 0:
        xform_acc = correct[xform_mask].float().mean().item()
    else:
        xform_acc = 1.0  # no transforms needed

    solved = correct.all().item()
    return cell_acc, xform_acc, solved, n_xform


def compare_passes(preds_a: torch.Tensor, preds_b: torch.Tensor,
                   targets: torch.Tensor):
    """Compare two passes: cells changed, improved, worsened."""
    changed = preds_a != preds_b
    n_changed = changed.sum().item()

    # Among changed cells: did they get closer to or further from target?
    a_correct = preds_a == targets
    b_correct = preds_b == targets

    improved = (changed & ~a_correct & b_correct).sum().item()
    worsened = (changed & a_correct & ~b_correct).sum().item()

    return n_changed, improved, worsened


def main():
    parser = argparse.ArgumentParser(description="Multi-pass inference diagnostic")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_passes", type=int, default=3)
    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument("--max_tasks", type=int, default=400)
    parser.add_argument("--output_dir", type=str, default="output_multipass")
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
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load tasks
    all_tasks = load_arc_tasks(args.data_dir)
    tasks = all_tasks.get(args.split, [])
    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]
    print(f"  Tasks: {len(tasks)} ({args.split} split)")
    print(f"  Passes: {args.n_passes}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Run evaluation
    all_results = []
    total_by_pass = {p: {"xform_correct": 0, "xform_total": 0,
                          "cell_correct": 0, "cell_total": 0,
                          "solved": 0} for p in range(1, args.n_passes + 1)}
    total_changes = {f"{a}to{b}": {"changed": 0, "improved": 0, "worsened": 0}
                     for a, b in zip(range(1, args.n_passes), range(2, args.n_passes + 1))}
    n_skipped = 0
    n_evaluated = 0

    t0 = time.time()
    for task_idx, task in enumerate(tasks):
        task_id = task.get("task_id", f"task_{task_idx}")

        for test_idx in range(len(task["test"])):
            # Pass 1: standard inference
            seq = build_sequence(task, d4_idx=0, color_perm=None,
                                 test_idx=test_idx, max_seq_len=config.max_seq_len)
            if seq is None:
                n_skipped += 1
                continue

            meta = pad_single_to_batch(seq, config.max_seq_len, device)
            result = run_inference(model, meta, device, n_steps=config.n_ode_steps)
            preds_1 = extract_predictions(result, meta)

            # Ground truth
            target_mask = meta["target_mask"][0]
            targets = meta["target_labels"][0, target_mask]
            input_colors = meta["target_input_colors"][0, target_mask]
            output_shape = seq["test_output_shape"]

            cell_acc_1, xform_acc_1, solved_1, n_xform = compute_metrics(
                preds_1, targets, input_colors)

            task_result = {
                "task_id": task_id,
                "test_idx": test_idx,
                "output_shape": list(output_shape),
                "n_xform_cells": n_xform,
                "pass1_cell_acc": cell_acc_1,
                "pass1_xform_acc": xform_acc_1,
                "pass1_solved": solved_1,
            }

            total_by_pass[1]["xform_correct"] += int(xform_acc_1 * n_xform)
            total_by_pass[1]["xform_total"] += n_xform
            total_by_pass[1]["cell_correct"] += int(cell_acc_1 * len(targets))
            total_by_pass[1]["cell_total"] += len(targets)
            total_by_pass[1]["solved"] += int(solved_1)

            # Subsequent passes
            prev_preds = preds_1
            test_input = task["test"][test_idx]["input"]

            for pass_num in range(2, args.n_passes + 1):
                pred_grid = preds_to_grid(prev_preds, output_shape)

                # Build sequence with previous prediction as extra demo
                seq_mp = build_sequence_with_extra_demo(
                    task, extra_input=test_input, extra_output=pred_grid,
                    test_idx=test_idx, max_seq_len=config.max_seq_len)

                if seq_mp is None:
                    # Can't fit — use pass 1 result
                    task_result[f"pass{pass_num}_cell_acc"] = cell_acc_1
                    task_result[f"pass{pass_num}_xform_acc"] = xform_acc_1
                    task_result[f"pass{pass_num}_solved"] = solved_1
                    task_result[f"pass{pass_num}_skipped"] = True

                    total_by_pass[pass_num]["xform_correct"] += int(xform_acc_1 * n_xform)
                    total_by_pass[pass_num]["xform_total"] += n_xform
                    total_by_pass[pass_num]["cell_correct"] += int(cell_acc_1 * len(targets))
                    total_by_pass[pass_num]["cell_total"] += len(targets)
                    total_by_pass[pass_num]["solved"] += int(solved_1)
                    continue

                meta_mp = pad_single_to_batch(seq_mp, config.max_seq_len, device)
                result_mp = run_inference(model, meta_mp, device,
                                          n_steps=config.n_ode_steps)
                preds_p = extract_predictions(result_mp, meta_mp)

                # Recompute targets from the new sequence (same ground truth)
                target_mask_mp = meta_mp["target_mask"][0]
                targets_mp = meta_mp["target_labels"][0, target_mask_mp]
                input_colors_mp = meta_mp["target_input_colors"][0, target_mask_mp]

                cell_acc_p, xform_acc_p, solved_p, _ = compute_metrics(
                    preds_p, targets_mp, input_colors_mp)

                # Compare with previous pass
                # Ensure same length (should be — same test output)
                min_len = min(len(prev_preds), len(preds_p))
                n_changed, n_improved, n_worsened = compare_passes(
                    prev_preds[:min_len], preds_p[:min_len], targets_mp[:min_len])

                key = f"{pass_num-1}to{pass_num}"
                total_changes[key]["changed"] += n_changed
                total_changes[key]["improved"] += n_improved
                total_changes[key]["worsened"] += n_worsened

                task_result[f"pass{pass_num}_cell_acc"] = cell_acc_p
                task_result[f"pass{pass_num}_xform_acc"] = xform_acc_p
                task_result[f"pass{pass_num}_solved"] = solved_p
                task_result[f"cells_changed_{pass_num-1}to{pass_num}"] = n_changed
                task_result[f"cells_improved_{pass_num-1}to{pass_num}"] = n_improved
                task_result[f"cells_worsened_{pass_num-1}to{pass_num}"] = n_worsened

                total_by_pass[pass_num]["xform_correct"] += int(xform_acc_p * n_xform)
                total_by_pass[pass_num]["xform_total"] += n_xform
                total_by_pass[pass_num]["cell_correct"] += int(cell_acc_p * len(targets_mp))
                total_by_pass[pass_num]["cell_total"] += len(targets_mp)
                total_by_pass[pass_num]["solved"] += int(solved_p)

                prev_preds = preds_p

            all_results.append(task_result)
            n_evaluated += 1

            if n_evaluated % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{n_evaluated}/{len(tasks)}] {elapsed:.1f}s "
                      f"({elapsed/n_evaluated:.2f}s/task)")

    elapsed = time.time() - t0

    # Summary
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("Multi-Pass Inference Results")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Checkpoint: {args.checkpoint}")
    summary_lines.append(f"Tasks evaluated: {n_evaluated}")
    summary_lines.append(f"Tasks skipped (seq_len): {n_skipped}")
    summary_lines.append(f"Time: {elapsed:.1f}s ({elapsed/max(n_evaluated,1):.2f}s/task)")
    summary_lines.append("")

    for p in range(1, args.n_passes + 1):
        d = total_by_pass[p]
        xform_acc = d["xform_correct"] / max(d["xform_total"], 1)
        cell_acc = d["cell_correct"] / max(d["cell_total"], 1)
        summary_lines.append(
            f"Pass {p}: xform_acc={xform_acc*100:.1f}%, "
            f"cell_acc={cell_acc*100:.1f}%, "
            f"tasks_solved={d['solved']}/{n_evaluated} ({d['solved']/max(n_evaluated,1)*100:.1f}%)")

    summary_lines.append("")
    for key, d in total_changes.items():
        net = d["improved"] - d["worsened"]
        summary_lines.append(
            f"Cell changes {key.replace('to', '→')}: "
            f"{d['changed']} changed, {d['improved']} improved, "
            f"{d['worsened']} worsened (net: {'+' if net >= 0 else ''}{net})")

    # New solves / regressions between passes
    summary_lines.append("")
    for p in range(2, args.n_passes + 1):
        new_solves = sum(
            1 for r in all_results
            if r.get(f"pass{p}_solved") and not r.get(f"pass{p-1}_solved")
        )
        regressions = sum(
            1 for r in all_results
            if r.get(f"pass{p-1}_solved") and not r.get(f"pass{p}_solved")
        )
        summary_lines.append(
            f"Pass {p-1}→{p}: {new_solves} new solves, {regressions} regressions")

    summary = "\n".join(summary_lines)
    print(summary)

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "tasks": all_results}, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")

    # Save summary report
    report_path = os.path.join(args.output_dir, "MULTIPASS_EVAL_REPORT.md")
    with open(report_path, "w") as f:
        f.write(f"# Multi-Pass Inference Diagnostic\n\n```\n{summary}\n```\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
