"""Reptile meta-learning for LiquidARC TTT plasticity.

First-order MAML: interleave meta-steps that optimize melt params
(MetricNet, TauNet, W_o, FFN[-1]) to remain TTT-sensitive,
while CE training continues normally on all params.

Algorithm per meta-step:
1. Snapshot melt params
2. For K tasks: clone model via state_dict, run TTT inner loop, accumulate deltas
3. Meta-update: base_params += meta_lr * avg_delta

Uses state_dict cloning instead of copy.deepcopy to avoid torch.compile
non-leaf tensor issues.
"""

import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Import ARC data utilities from fgn-v3
FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if FGN_ROOT not in sys.path:
    sys.path.insert(0, FGN_ROOT)

from fgn.tasks.arc import build_sequence, pad_single_to_batch

from .config import LiquidARCConfig
from .model import LiquidARCModel
from .ttt import make_ttt_training_meta, _forward_model


def get_melt_modules(model: LiquidARCModel, include_ffn: bool) -> List[Tuple[str, torch.nn.Module]]:
    """Return [(name, module)] list matching ttt.py's melt set."""
    modules = [
        ("metric_net_linear1", model.dynamics.metric_net_linear1),
        ("metric_net_linear2", model.dynamics.metric_net_linear2),
        ("tau_net_linear1", model.dynamics.tau_net_linear1),
        ("tau_net_linear2", model.dynamics.tau_net_linear2),
        ("W_o", model.dynamics.W_o),
    ]
    if include_ffn:
        modules.append(("ffn_last", model.dynamics.ffn[-1]))
    return modules


def snapshot_melt_params(
    model: LiquidARCModel, include_ffn: bool,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Snapshot melt param values (cloned, detached)."""
    snap = {}
    for name, mod in get_melt_modules(model, include_ffn):
        entry = {"weight": mod.weight.data.clone()}
        if mod.bias is not None:
            entry["bias"] = mod.bias.data.clone()
        snap[name] = entry
    return snap


def extract_adapted_params(
    adapted: LiquidARCModel, include_ffn: bool,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Extract adapted param values (detached, no clone — model will be deleted)."""
    params = {}
    for name, mod in get_melt_modules(adapted, include_ffn):
        entry = {"weight": mod.weight.data.detach()}
        if mod.bias is not None:
            entry["bias"] = mod.bias.data.detach()
        params[name] = entry
    return params


def compute_reptile_meta_lr(step: int, config: LiquidARCConfig) -> float:
    """Compute effective meta_lr with linear warmup."""
    steps_since_start = step - config.reptile_start_step
    if steps_since_start < 0:
        return 0.0
    if config.reptile_warmup_steps <= 0:
        return config.reptile_meta_lr
    warmup_progress = min(1.0, steps_since_start / config.reptile_warmup_steps)
    return config.reptile_meta_lr * warmup_progress


def _clone_model(base_model: LiquidARCModel, config: LiquidARCConfig,
                 device: torch.device) -> LiquidARCModel:
    """Create a fresh model and load base_model's state_dict.

    Avoids copy.deepcopy which fails with torch.compile artifacts.
    Strips '_orig_mod.' prefix from keys that torch.compile inserts.
    """
    clone = LiquidARCModel(config).to(device)
    raw = base_model._orig_mod if hasattr(base_model, '_orig_mod') else base_model
    sd = {}
    for k, v in raw.state_dict().items():
        # torch.compile wraps modules in OptimizedModule, adding _orig_mod. prefix
        clean_k = k.replace("._orig_mod.", ".")
        sd[clean_k] = v
    clone.load_state_dict(sd)
    return clone


def _run_ttt_inner(
    adapted: LiquidARCModel,
    task: dict,
    config: LiquidARCConfig,
    device: torch.device,
    ttt_steps: int,
    ttt_lr: float,
) -> Tuple[Dict, bool]:
    """Minimal TTT inner loop for Reptile (no eval, no D4, no amortization).

    Returns (metrics_dict, success). On success, adapted model has been modified in-place.
    """
    seq = build_sequence(task, d4_idx=0, color_perm=None, test_idx=0,
                         max_seq_len=config.max_seq_len)
    if seq is None:
        return {"skipped": True, "reason": "seq_too_long"}, False

    meta = pad_single_to_batch(seq, config.max_seq_len, device)
    ttt_meta = make_ttt_training_meta(meta, seq)
    n_ttt_targets = (ttt_meta["target_labels"] != -100).sum().item()
    if n_ttt_targets == 0:
        return {"skipped": True, "reason": "no_ttt_targets"}, False

    adapted.train()

    # Freeze everything
    for p in adapted.parameters():
        p.requires_grad = False

    # Melt MetricNet + TauNet + W_o (+ optional FFN[-1])
    melt_modules = [mod for _, mod in get_melt_modules(adapted, config.reptile_include_ffn)]
    n_unfrozen = 0
    for mod in melt_modules:
        for p in mod.parameters():
            p.requires_grad = True
            n_unfrozen += p.numel()

    optimizer = torch.optim.AdamW(
        [p for p in adapted.parameters() if p.requires_grad],
        lr=ttt_lr, weight_decay=0.0,
    )

    curv_lambda = config.ttt_curvature_lambda
    early_stop = config.ttt_early_stop_threshold
    steps_run = 0

    for s in range(ttt_steps):
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
        steps_run += 1

        if ttt_ce.item() < early_stop:
            break

    return {"ttt_steps_used": steps_run, "skipped": False}, True


def reptile_step(
    base_model: LiquidARCModel,
    arc_tasks: list,
    config: LiquidARCConfig,
    device: torch.device,
    step: int,
) -> Dict:
    """Execute one Reptile meta-step.

    1. Compute effective meta_lr (warmup). Skip if <= 0.
    2. Sample K tasks randomly from arc_tasks.
    3. Snapshot base melt params. Init zero accumulator.
    4. For each task: clone via state_dict, run TTT, accumulate deltas.
    5. Apply: module.weight.data += meta_lr * (accum / K) for each melt module.

    Returns metrics dict.
    """
    t0 = time.time()
    meta_lr = compute_reptile_meta_lr(step, config)

    if meta_lr <= 0 or not arc_tasks:
        return {"skipped": True, "reason": "warmup_or_no_tasks", "meta_lr": meta_lr}

    K = min(config.reptile_n_tasks, len(arc_tasks))
    sampled_tasks = random.sample(arc_tasks, K)

    # Snapshot base melt params (works through compiled dynamics via _orig_mod)
    raw_model = base_model._orig_mod if hasattr(base_model, '_orig_mod') else base_model
    # If dynamics is compiled, access the uncompiled original for named submodules
    real_dynamics = raw_model.dynamics
    if hasattr(real_dynamics, '_orig_mod'):
        raw_model.dynamics = real_dynamics._orig_mod

    base_snap = snapshot_melt_params(raw_model, config.reptile_include_ffn)

    # Restore compiled dynamics for snapshot only
    if hasattr(real_dynamics, '_orig_mod'):
        raw_model.dynamics = real_dynamics

    # Init zero accumulator
    accum = {}
    for name, entry in base_snap.items():
        accum[name] = {"weight": torch.zeros_like(entry["weight"])}
        if "bias" in entry:
            accum[name]["bias"] = torch.zeros_like(entry["bias"])

    n_tasks_used = 0
    total_ttt_steps = 0
    delta_norms = []

    for task in sampled_tasks:
        # Clone model via state_dict (avoids deepcopy + torch.compile issues)
        adapted = _clone_model(raw_model, config, device)

        metrics, success = _run_ttt_inner(
            adapted, task, config, device,
            ttt_steps=config.reptile_ttt_steps,
            ttt_lr=config.reptile_ttt_lr,
        )

        if not success:
            del adapted
            continue

        # Extract adapted params and accumulate delta
        adapted_params = extract_adapted_params(adapted, config.reptile_include_ffn)

        task_delta_norm = 0.0
        for name in accum:
            if name in adapted_params:
                delta_w = adapted_params[name]["weight"] - base_snap[name]["weight"]
                accum[name]["weight"] += delta_w
                task_delta_norm += delta_w.norm().item() ** 2
                if "bias" in accum[name] and "bias" in adapted_params[name]:
                    delta_b = adapted_params[name]["bias"] - base_snap[name]["bias"]
                    accum[name]["bias"] += delta_b
                    task_delta_norm += delta_b.norm().item() ** 2

        delta_norms.append(task_delta_norm ** 0.5)
        n_tasks_used += 1
        total_ttt_steps += metrics.get("ttt_steps_used", 0)

        del adapted

    if n_tasks_used == 0:
        return {"skipped": True, "reason": "all_tasks_skipped", "meta_lr": meta_lr}

    # Apply meta-update to base model's melt params
    # Access uncompiled dynamics for named submodules
    if hasattr(real_dynamics, '_orig_mod'):
        raw_model.dynamics = real_dynamics._orig_mod

    melt_modules = dict(get_melt_modules(raw_model, config.reptile_include_ffn))
    for name, mod in melt_modules.items():
        if name in accum:
            mod.weight.data += meta_lr * (accum[name]["weight"] / n_tasks_used)
            if mod.bias is not None and "bias" in accum[name]:
                mod.bias.data += meta_lr * (accum[name]["bias"] / n_tasks_used)

    # Restore compiled dynamics
    if hasattr(real_dynamics, '_orig_mod'):
        raw_model.dynamics = real_dynamics

    elapsed = time.time() - t0
    avg_delta = sum(delta_norms) / len(delta_norms) if delta_norms else 0.0

    return {
        "skipped": False,
        "n_tasks_used": n_tasks_used,
        "avg_delta_norm": avg_delta,
        "meta_lr": meta_lr,
        "avg_ttt_steps": total_ttt_steps / max(n_tasks_used, 1),
        "time": elapsed,
    }
