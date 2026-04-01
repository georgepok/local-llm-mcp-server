"""Test-Time Training (TTT) for LiquidARC.

Clone the base model, run aggressive gradient descent on support examples
to rewire MetricNet/TauNet for that specific task, then predict.
V3 adds: FFN plasticity (A), D4 augmentation (B), amortized TTT via hypernetwork (C).

Usage:
    from liquid_arc.ttt import test_time_adapt, evaluate_ttt
    from liquid_arc.ttt import test_time_adapt_amortized, evaluate_ttt_amortized
"""

import copy
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

# Import ARC data utilities from fgn-v3
FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if FGN_ROOT not in sys.path:
    sys.path.insert(0, FGN_ROOT)

from fgn.tasks.arc import (
    ROLE_INPUT_DEMO, ROLE_OUTPUT_DEMO, ROLE_TEST_OUTPUT,
    PAD_COLOR,
    build_sequence, pad_single_to_batch, load_arc_tasks,
    random_color_perm,
)

from .config import LiquidARCConfig
from .model import LiquidARCModel


def make_ttt_training_meta(
    meta: Dict[str, torch.Tensor],
    seq: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Flip metadata so demo outputs are targets for TTT inner loop.

    In standard eval, only test_output positions are targets.
    In TTT, demo_output positions become targets (with their actual colors as labels),
    and test_output positions are masked (no loss, but colors replaced to prevent cheating).

    Args:
        meta: padded batch metadata from pad_single_to_batch() — [1, max_seq_len] tensors
        seq: raw build_sequence() output — [N] tensors (unpadded)

    Returns:
        New metadata dict with TTT-specific target_mask, target_labels,
        target_input_colors, and context_mask.
    """
    ttt_meta = {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in meta.items()}

    device = meta["target_mask"].device
    N = seq["colors"].shape[0]
    roles = seq["roles"]  # [N]
    colors = seq["colors"]  # [N]
    grid_ids = seq["grid_ids"]  # [N]

    # Build TTT target mask: demo output positions
    demo_out_mask_raw = (roles == ROLE_OUTPUT_DEMO)  # [N]
    test_out_mask_raw = (roles == ROLE_TEST_OUTPUT)  # [N]

    # Pad to batch format
    max_len = ttt_meta["target_mask"].shape[1]
    ttt_target_mask = torch.zeros(1, max_len, dtype=torch.bool, device=device)
    ttt_target_mask[0, :N] = demo_out_mask_raw | test_out_mask_raw

    # Target labels: demo output colors for demo positions, -100 for test output (no loss)
    ttt_labels = torch.full((1, max_len), -100, dtype=torch.long, device=device)
    demo_out_indices = demo_out_mask_raw.nonzero(as_tuple=True)[0]
    ttt_labels[0, demo_out_indices] = colors[demo_out_indices].to(device)
    # test_output positions stay -100 (no loss contribution during TTT)

    # Target input colors: for each demo output at (x,y,grid_id=g),
    # look up the demo input color at (x,y,grid_id=g-1)
    ttt_input_colors = torch.full((1, max_len), PAD_COLOR, dtype=torch.long,
                                  device=device)

    xs = seq["xs"]  # [N]
    ys = seq["ys"]  # [N]

    # Build lookup: (grid_id, x, y) -> color for input grids
    input_lookup = {}
    for i in range(N):
        if roles[i] == ROLE_INPUT_DEMO:
            gid = grid_ids[i].item()
            key = (gid, xs[i].item(), ys[i].item())
            input_lookup[key] = colors[i].item()

    # For each demo output position, find corresponding input color
    for idx in demo_out_indices:
        idx_val = idx.item()
        gid = grid_ids[idx_val].item()
        input_gid = gid - 1  # input grid_id = output grid_id - 1
        key = (input_gid, xs[idx_val].item(), ys[idx_val].item())
        ttt_input_colors[0, idx_val] = input_lookup.get(key, PAD_COLOR)

    # For test output positions: use existing test input colors from standard meta
    test_out_indices = test_out_mask_raw.nonzero(as_tuple=True)[0]
    for idx in test_out_indices:
        idx_val = idx.item()
        ttt_input_colors[0, idx_val] = meta["target_input_colors"][0, idx_val]

    # Context mask: everything not in target_mask (for real token positions)
    ttt_context = torch.zeros(1, max_len, dtype=torch.bool, device=device)
    real_mask = torch.zeros(1, max_len, dtype=torch.bool, device=device)
    real_mask[0, :N] = True
    ttt_context = real_mask & ~ttt_target_mask

    ttt_meta["target_mask"] = ttt_target_mask
    ttt_meta["target_labels"] = ttt_labels
    ttt_meta["target_input_colors"] = ttt_input_colors
    ttt_meta["context_mask"] = ttt_context

    return ttt_meta


def _forward_model(model, meta, device, n_steps=None, geo_phase=0):
    """Run model forward pass with metadata dict."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                             enabled=(device.type == "cuda")):
        return model(
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
            geo_phase=geo_phase,
        )


def test_time_adapt(
    base_model: LiquidARCModel,
    task: dict,
    config: LiquidARCConfig,
    device: torch.device,
    ttt_steps: Optional[int] = None,
    ttt_lr: Optional[float] = None,
    ttt_curvature_lambda: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[Dict, Optional[LiquidARCModel]]:
    """Clone-Sculpt-Execute: adapt model to a single ARC task via TTT.

    Args:
        base_model: pre-trained LiquidARCModel (not modified)
        task: ARC task dict with "train" (demos) and "test" keys
        config: model config
        device: torch device
        ttt_steps: override config.ttt_steps
        ttt_lr: override config.ttt_lr
        ttt_curvature_lambda: override config.ttt_curvature_lambda
        verbose: print per-step loss

    Returns:
        (result_dict, adapted_model) where result_dict has per-test-pair results
    """
    steps = ttt_steps or config.ttt_steps
    lr = ttt_lr or config.ttt_lr
    curv_lambda = ttt_curvature_lambda if ttt_curvature_lambda is not None else config.ttt_curvature_lambda
    early_stop = config.ttt_early_stop_threshold

    # Build sequence(s) from task
    # Experiment B: D4 augmentation — build 8 D4 variants for round-robin TTT
    if config.ttt_d4_augment:
        ttt_metas = []
        n_d4_variants = 0
        for d4_i in range(8):
            cperm = random_color_perm() if config.ttt_d4_color_perm else None
            seq_i = build_sequence(task, d4_idx=d4_i, color_perm=cperm, test_idx=0,
                                   max_seq_len=config.max_seq_len)
            if seq_i is None:
                continue  # rotated grid exceeds max_seq_len — skip this variant
            meta_i = pad_single_to_batch(seq_i, config.max_seq_len, device)
            ttt_meta_i = make_ttt_training_meta(meta_i, seq_i)
            n_targets_i = (ttt_meta_i["target_labels"] != -100).sum().item()
            if n_targets_i > 0:
                ttt_metas.append(ttt_meta_i)
                n_d4_variants += 1
        if not ttt_metas:
            return {"skipped": True, "reason": "no_d4_variants_fit"}, None
        if verbose:
            print(f"    D4 augmentation: {n_d4_variants}/8 variants fit max_seq_len")
    else:
        # Standard single-sequence path
        seq = build_sequence(task, d4_idx=0, color_perm=None, test_idx=0,
                             max_seq_len=config.max_seq_len)
        if seq is None:
            return {"skipped": True, "reason": "seq_too_long"}, None
        meta = pad_single_to_batch(seq, config.max_seq_len, device)
        ttt_meta_single = make_ttt_training_meta(meta, seq)
        n_ttt_targets = (ttt_meta_single["target_labels"] != -100).sum().item()
        if n_ttt_targets == 0:
            return {"skipped": True, "reason": "no_ttt_targets"}, None
        ttt_metas = [ttt_meta_single]

    # Deep copy model — selective plasticity
    adapted = copy.deepcopy(base_model)
    adapted.to(device)
    adapted.train()

    # Freeze everything
    for p in adapted.parameters():
        p.requires_grad = False

    # Melt MetricNet + TauNet/GateNet + W_o (content transformation)
    # MetricNet/TauNet/GateNet: WHERE to route (geometry)
    # W_o: WHAT transformation to apply to routed values (content)
    melt_modules = [
        adapted.dynamics.metric_net_linear1,
        adapted.dynamics.metric_net_linear2,
        adapted.dynamics.W_o,
    ]
    if config.channel_gate_enabled:
        melt_modules.extend([
            adapted.dynamics.gate_net_linear1,
            adapted.dynamics.gate_net_linear2,
        ])
    else:
        melt_modules.extend([
            adapted.dynamics.tau_net_linear1,
            adapted.dynamics.tau_net_linear2,
        ])
    # Experiment A: FFN plasticity — unfreeze FFN output layer for nonlinear adaptation
    if config.ttt_unfreeze_ffn:
        melt_modules.append(adapted.dynamics.ffn[-1])  # Linear(d_ffn, d) = 131K params

    n_unfrozen = 0
    for mod in melt_modules:
        for p in mod.parameters():
            p.requires_grad = True
            n_unfrozen += p.numel()

    # Step embeds: melt during TTT for task-specific ODE heterogeneity
    if config.step_embed_enabled:
        adapted.dynamics.step_embeds.requires_grad = True
        n_unfrozen += adapted.dynamics.step_embeds.numel()

    optimizer = torch.optim.AdamW(
        [p for p in adapted.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,
    )

    # TTT inner loop — use xform_loss (CE on transform cells only)
    # ce_loss includes copy cells which teach MetricNet identity routing,
    # destroying the abstract transformation the base model learned.
    # With D4 augmentation: round-robin through ttt_metas variants.
    ce_history = []
    for step in range(steps):
        optimizer.zero_grad()

        ttt_meta = ttt_metas[step % len(ttt_metas)]
        result = _forward_model(adapted, ttt_meta, device,
                                n_steps=config.n_ode_steps, geo_phase=0)

        # Focus TTT on transformation pattern, not copy structure
        xf = result["xform_loss"]
        # Fall back to ce_loss if no transform cells in demos (rare)
        ttt_ce = xf if xf.item() > 0 else result["ce_loss"]
        curv = curv_lambda * result["avg_kappa"]
        loss = ttt_ce + curv

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in adapted.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        ce_val = ttt_ce.item()
        ce_history.append(ce_val)

        if verbose and step % 5 == 0:
            print(f"    TTT step {step}: xf_ce={ce_val:.4f}, |k|={result['avg_kappa'].item():.4f}")

        # Early stop
        if ce_val < early_stop:
            if verbose:
                print(f"    TTT early stop at step {step}: xf_ce={ce_val:.4f}")
            break

    # Evaluate on all test pairs with adapted model
    adapted.eval()
    results = []

    for test_idx in range(len(task["test"])):
        seq_eval = build_sequence(task, d4_idx=0, color_perm=None,
                                  test_idx=test_idx, max_seq_len=config.max_seq_len)
        if seq_eval is None:
            results.append({"skipped": True, "test_idx": test_idx})
            continue

        meta_eval = pad_single_to_batch(seq_eval, config.max_seq_len, device)

        with torch.no_grad():
            eval_result = _forward_model(adapted, meta_eval, device,
                                         n_steps=config.n_ode_steps)

        cell_acc = eval_result.get("cell_accuracy", torch.tensor(0.0))
        if isinstance(cell_acc, torch.Tensor):
            cell_acc = cell_acc.item()
        xform_acc = eval_result.get("transform_accuracy", torch.tensor(0.0))
        if isinstance(xform_acc, torch.Tensor):
            xform_acc = xform_acc.item()
        ce_loss = eval_result["ce_loss"].item()

        results.append({
            "test_idx": test_idx,
            "cell_acc": cell_acc,
            "xform_acc": xform_acc,
            "ce_loss": ce_loss,
            "ttt_steps_used": len(ce_history),
            "ttt_ce_start": ce_history[0] if ce_history else 0.0,
            "ttt_ce_end": ce_history[-1] if ce_history else 0.0,
        })

    task_id = task.get("task_id", "unknown")
    return {
        "task_id": task_id,
        "skipped": False,
        "test_results": results,
        "ttt_steps_used": len(ce_history),
        "ttt_ce_history": ce_history,
        "n_unfrozen_params": n_unfrozen,
    }, adapted


def test_time_adapt_amortized(
    base_model: LiquidARCModel,
    task: dict,
    config: LiquidARCConfig,
    device: torch.device,
    hypernet: "torch.nn.Module",
    verbose: bool = False,
) -> Tuple[Dict, Optional[LiquidARCModel]]:
    """Amortized TTT: predict weight deltas in one forward pass via hypernetwork.

    Instead of 100-step gradient descent, the hypernetwork predicts ΔW
    from demo pair embeddings. Much faster inference (~1 forward pass).

    Args:
        base_model: pre-trained LiquidARCModel (not modified)
        task: ARC task dict with "train" (demos) and "test" keys
        config: model config
        device: torch device
        hypernet: trained HyperNetwork instance
        verbose: print diagnostics

    Returns:
        (result_dict, adapted_model) same format as test_time_adapt()
    """
    # Build sequence (original orientation for encoding)
    seq = build_sequence(task, d4_idx=0, color_perm=None, test_idx=0,
                         max_seq_len=config.max_seq_len)
    if seq is None:
        return {"skipped": True, "reason": "seq_too_long"}, None

    meta = pad_single_to_batch(seq, config.max_seq_len, device)

    # Encode task via hypernetwork (no_grad — inference only)
    with torch.no_grad():
        task_embed = hypernet.encode_task(
            base_model, meta, device,
            role_input_demo=ROLE_INPUT_DEMO,
            role_output_demo=ROLE_OUTPUT_DEMO,
        )

        # Predict weight deltas
        deltas = hypernet(task_embed)

    if verbose:
        total_delta_norm = sum(
            d["weight"].norm().item() for d in deltas.values() if "weight" in d
        )
        print(f"    Amortized TTT: delta_norm={total_delta_norm:.4f}")

    # Apply deltas to a copy of the base model
    adapted = copy.deepcopy(base_model)
    adapted.to(device)
    hypernet.apply_deltas(adapted, deltas)

    # Evaluate on all test pairs
    adapted.eval()
    results = []

    for test_idx in range(len(task["test"])):
        seq_eval = build_sequence(task, d4_idx=0, color_perm=None,
                                  test_idx=test_idx, max_seq_len=config.max_seq_len)
        if seq_eval is None:
            results.append({"skipped": True, "test_idx": test_idx})
            continue

        meta_eval = pad_single_to_batch(seq_eval, config.max_seq_len, device)

        with torch.no_grad():
            eval_result = _forward_model(adapted, meta_eval, device,
                                         n_steps=config.n_ode_steps)

        cell_acc = eval_result.get("cell_accuracy", torch.tensor(0.0))
        if isinstance(cell_acc, torch.Tensor):
            cell_acc = cell_acc.item()
        xform_acc = eval_result.get("transform_accuracy", torch.tensor(0.0))
        if isinstance(xform_acc, torch.Tensor):
            xform_acc = xform_acc.item()
        ce_loss = eval_result["ce_loss"].item()

        results.append({
            "test_idx": test_idx,
            "cell_acc": cell_acc,
            "xform_acc": xform_acc,
            "ce_loss": ce_loss,
            "ttt_steps_used": 0,  # amortized — no gradient steps
            "ttt_ce_start": 0.0,
            "ttt_ce_end": 0.0,
        })

    task_id = task.get("task_id", "unknown")
    return {
        "task_id": task_id,
        "skipped": False,
        "test_results": results,
        "ttt_steps_used": 0,
        "ttt_ce_history": [],
        "n_unfrozen_params": 0,
        "amortized": True,
    }, adapted


def evaluate_ttt(
    base_model: LiquidARCModel,
    data_dir: str,
    config: LiquidARCConfig,
    device: torch.device,
    n_tasks: Optional[int] = None,
    verbose: bool = False,
    ttt_steps: Optional[int] = None,
    ttt_lr: Optional[float] = None,
) -> Tuple[float, float]:
    """Full TTT evaluation on ARC eval tasks.

    Args:
        base_model: pre-trained model
        data_dir: path to ARC data directory
        config: model config
        device: torch device
        n_tasks: limit number of tasks (None = all)
        verbose: print per-task results
        ttt_steps: override config.ttt_steps
        ttt_lr: override config.ttt_lr

    Returns:
        (cell_accuracy, transform_accuracy) aggregated over all test pairs
    """
    all_tasks = load_arc_tasks(data_dir)
    eval_tasks = all_tasks.get("eval", [])
    if not eval_tasks:
        print("  WARNING: No eval tasks found")
        return 0.0, 0.0

    if n_tasks is not None:
        eval_tasks = eval_tasks[:n_tasks]

    total_cell_correct = 0
    total_cell_count = 0
    total_xform_correct = 0
    total_xform_count = 0
    skipped = 0
    t0 = time.time()

    base_model.eval()

    for i, task in enumerate(eval_tasks):
        task_id = task.get("task_id", f"task_{i}")

        result, adapted = test_time_adapt(
            base_model, task, config, device,
            ttt_steps=ttt_steps, ttt_lr=ttt_lr,
            verbose=verbose,
        )

        # Free adapted model immediately
        del adapted

        if result.get("skipped", False):
            skipped += 1
            if verbose:
                print(f"  [{i+1}/{len(eval_tasks)}] {task_id}: SKIPPED ({result.get('reason', '')})")
            continue

        for tr in result["test_results"]:
            if tr.get("skipped", False):
                continue
            # Accumulate cell accuracy
            # We need actual cell counts — estimate from accuracy * n_targets
            # For precise counting, re-run with tracking. Here use the accuracy directly.
            total_cell_correct += tr["cell_acc"]
            total_cell_count += 1
            total_xform_correct += tr["xform_acc"]
            total_xform_count += 1

        if verbose:
            test_accs = [tr["xform_acc"] for tr in result["test_results"]
                         if not tr.get("skipped", False)]
            avg_xf = sum(test_accs) / max(len(test_accs), 1)
            print(f"  [{i+1}/{len(eval_tasks)}] {task_id}: "
                  f"xform_acc={avg_xf:.4f}, ttt_steps={result['ttt_steps_used']}, "
                  f"ce: {result['ttt_ce_history'][0]:.3f}→{result['ttt_ce_history'][-1]:.3f}")

    elapsed = time.time() - t0

    cell_acc = total_cell_correct / max(total_cell_count, 1)
    xform_acc = total_xform_correct / max(total_xform_count, 1)

    print(f"  TTT eval: {len(eval_tasks)} tasks ({skipped} skipped), "
          f"{elapsed:.1f}s")
    print(f"  Cell acc: {cell_acc:.4f}, Xform acc: {xform_acc:.4f}")

    return cell_acc, xform_acc


def evaluate_ttt_amortized(
    base_model: LiquidARCModel,
    data_dir: str,
    config: LiquidARCConfig,
    device: torch.device,
    hypernet: "torch.nn.Module",
    n_tasks: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[float, float]:
    """Amortized TTT evaluation via hypernetwork on ARC eval tasks.

    Same interface as evaluate_ttt() but uses hypernetwork for single-pass adaptation.
    """
    all_tasks = load_arc_tasks(data_dir)
    eval_tasks = all_tasks.get("eval", [])
    if not eval_tasks:
        print("  WARNING: No eval tasks found")
        return 0.0, 0.0

    if n_tasks is not None:
        eval_tasks = eval_tasks[:n_tasks]

    total_cell_correct = 0
    total_cell_count = 0
    total_xform_correct = 0
    total_xform_count = 0
    skipped = 0
    t0 = time.time()

    base_model.eval()
    hypernet.eval()

    for i, task in enumerate(eval_tasks):
        task_id = task.get("task_id", f"task_{i}")

        result, adapted = test_time_adapt_amortized(
            base_model, task, config, device, hypernet,
            verbose=verbose,
        )

        del adapted

        if result.get("skipped", False):
            skipped += 1
            if verbose:
                print(f"  [{i+1}/{len(eval_tasks)}] {task_id}: SKIPPED ({result.get('reason', '')})")
            continue

        for tr in result["test_results"]:
            if tr.get("skipped", False):
                continue
            total_cell_correct += tr["cell_acc"]
            total_cell_count += 1
            total_xform_correct += tr["xform_acc"]
            total_xform_count += 1

        if verbose:
            test_accs = [tr["xform_acc"] for tr in result["test_results"]
                         if not tr.get("skipped", False)]
            avg_xf = sum(test_accs) / max(len(test_accs), 1)
            print(f"  [{i+1}/{len(eval_tasks)}] {task_id}: xform_acc={avg_xf:.4f} (amortized)")

    elapsed = time.time() - t0

    cell_acc = total_cell_correct / max(total_cell_count, 1)
    xform_acc = total_xform_correct / max(total_xform_count, 1)

    print(f"  Amortized TTT eval: {len(eval_tasks)} tasks ({skipped} skipped), "
          f"{elapsed:.1f}s")
    print(f"  Cell acc: {cell_acc:.4f}, Xform acc: {xform_acc:.4f}")

    return cell_acc, xform_acc


def evaluate_baseline(
    model: torch.nn.Module,
    data_dir: str,
    config: LiquidARCConfig,
    device: torch.device,
    n_tasks: Optional[int] = None,
) -> Tuple[float, float]:
    """Baseline evaluation (no TTT) on ARC eval tasks, one task at a time.

    Same task-by-task evaluation as TTT but without adaptation,
    for fair A/B comparison.
    """
    all_tasks = load_arc_tasks(data_dir)
    eval_tasks = all_tasks.get("eval", [])
    if not eval_tasks:
        return 0.0, 0.0

    if n_tasks is not None:
        eval_tasks = eval_tasks[:n_tasks]

    model.eval()
    total_cell = 0.0
    total_xform = 0.0
    count = 0
    skipped = 0

    for task in eval_tasks:
        for test_idx in range(len(task["test"])):
            seq = build_sequence(task, d4_idx=0, color_perm=None,
                                 test_idx=test_idx, max_seq_len=config.max_seq_len)
            if seq is None:
                skipped += 1
                continue

            meta = pad_single_to_batch(seq, config.max_seq_len, device)

            with torch.no_grad():
                result = _forward_model(model, meta, device,
                                        n_steps=config.n_ode_steps)

            cell_acc = result.get("cell_accuracy", torch.tensor(0.0))
            if isinstance(cell_acc, torch.Tensor):
                cell_acc = cell_acc.item()
            xform_acc = result.get("transform_accuracy", torch.tensor(0.0))
            if isinstance(xform_acc, torch.Tensor):
                xform_acc = xform_acc.item()

            total_cell += cell_acc
            total_xform += xform_acc
            count += 1

    cell_acc = total_cell / max(count, 1)
    xform_acc = total_xform / max(count, 1)

    print(f"  Baseline eval: {count} test pairs ({skipped} skipped)")
    print(f"  Cell acc: {cell_acc:.4f}, Xform acc: {xform_acc:.4f}")

    return cell_acc, xform_acc
