"""train_memory.py — Train WorkingMemory module on a frozen LiquidARCModel base.

The base model (embedding, dynamics, output_head, etc.) is loaded from a
checkpoint and fully frozen. Only the WorkingMemory module is trained.

The forward pass is performed manually to insert memory between ODE steps:
    embedding → context_pool → dynamics.set_context →
    memory.reset → euler_solve_with_memory → norm_out → output_head → loss

Loss is the same curriculum-weighted CE used in train.py (5x transform cells,
0.05x copy cells). Eval runs both with and without memory for direct comparison.

Usage:
    python scripts/train_memory.py \\
        --base_checkpoint output_ttt_v2/checkpoints/best.pt \\
        --config configs/liquid_arc.yaml \\
        --data_dir data/arc \\
        --output_dir output_memory \\
        --max_steps 10000 \\
        --lr 1e-3 \\
        --batch_size 16
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel, create_model
from liquid_arc.working_memory import WorkingMemory
from liquid_arc.solver import euler_solve_with_observer
from liquid_arc.tasks.procedural import ProceduralARCTask, CurriculumStage

# Real ARC data (same location resolution as train.py)
FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if not Path(FGN_ROOT).exists():
    FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, FGN_ROOT)
from fgn.tasks.arc import ARCTask


# ARC constants (match model.py)
N_COLORS = 10
PAD_COLOR = 10


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss_and_accuracy(
    logits: torch.Tensor,
    target_labels: torch.Tensor,
    target_mask: torch.Tensor,
    target_input_colors: Optional[torch.Tensor],
    transform_weight: float,
    copy_weight: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Curriculum-weighted CE loss + accuracy metrics.

    Matches LiquidARCModel._compute_loss() from model.py exactly so results
    are directly comparable.

    Args:
        logits: [B, N, N_COLORS]
        target_labels: [B, N] with -100 for non-target cells
        target_mask: [B, N] bool — which cells are target
        target_input_colors: [B, N] input color for target cells (for transform detection)
        transform_weight: Weight for cells that changed (default 5.0).
        copy_weight: Weight for cells that stayed the same (default 0.05).
        device: Compute device.

    Returns:
        dict with: loss, ce_loss, cell_accuracy, transform_accuracy, xform_loss, n_transform
    """
    B = logits.shape[0]
    preds = logits.argmax(dim=-1)

    per_grid_loss = []
    per_grid_acc = []
    for b in range(B):
        tgt = target_labels[b]
        valid_b = tgt != -100
        if valid_b.sum() == 0:
            continue
        per_cell_ce = F.cross_entropy(logits[b][valid_b], tgt[valid_b], reduction="none")
        if target_input_colors is not None:
            inp_b = target_input_colors[b][valid_b]
            changed = tgt[valid_b] != inp_b
            cell_w = torch.where(changed, transform_weight, copy_weight)
        else:
            cell_w = torch.ones_like(per_cell_ce)
        loss_b = (per_cell_ce * cell_w).sum() / cell_w.sum()
        acc_b = (preds[b][valid_b] == tgt[valid_b]).float().mean()
        per_grid_loss.append(loss_b)
        per_grid_acc.append(acc_b.detach())

    if per_grid_loss:
        grid_losses = torch.stack(per_grid_loss)
        grid_accs = torch.stack(per_grid_acc)
        # Upweight grids the model is bad at (matches train.py)
        weights = 3.0 - 2.0 * grid_accs
        ce_loss = (grid_losses * weights).sum() / weights.sum()
    else:
        ce_loss = torch.tensor(0.0, device=device)

    # Flat accuracy
    flat_preds = preds.reshape(-1)
    flat_labels = target_labels.reshape(-1)
    valid = flat_labels != -100
    n_valid = valid.sum().clamp(min=1)
    correct_all = (flat_preds[valid] == flat_labels[valid]).sum()
    cell_accuracy = correct_all.float() / n_valid.float()

    # Transform accuracy
    flat_input = (target_input_colors.reshape(-1) if target_input_colors is not None else None)
    transform = valid & (flat_labels != flat_input) if flat_input is not None else valid
    n_transform = transform.sum().clamp(min=1)
    correct_transform = (flat_preds[transform] == flat_labels[transform]).sum()
    transform_accuracy = correct_transform.float() / n_transform.float()

    # Unweighted xform CE (for monitoring)
    flat_logits = logits.reshape(-1, logits.size(-1))
    if transform.sum() > 0:
        xform_loss = F.cross_entropy(flat_logits[transform], flat_labels[transform])
    else:
        xform_loss = torch.tensor(0.0, device=device)

    return {
        "loss": ce_loss,
        "ce_loss": ce_loss,
        "cell_accuracy": cell_accuracy,
        "transform_accuracy": transform_accuracy,
        "xform_loss": xform_loss,
        "n_transform": n_transform,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass with memory (replicates LiquidARCModel.forward manually)
# ─────────────────────────────────────────────────────────────────────────────

def forward_with_memory(
    base_model: LiquidARCModel,
    memory: WorkingMemory,
    meta: Dict,
    device: torch.device,
    n_steps: int,
) -> torch.Tensor:
    """Run the base model forward pass with memory inserted into the ODE loop.

    This replicates LiquidARCModel.forward() but calls euler_solve_with_memory
    instead of euler_solve. The base model parameters are frozen; only the
    memory module is in-graph for gradient computation.

    Steps:
        1. Mask target colors (same as model.forward)
        2. embedding(...)  → h0
        3. context_pool(h0) → context; dynamics.set_context(context)
        4. dynamics.set_n_steps(n_steps)
        5. memory.reset(B, device)
        6. euler_solve_with_memory(dynamics, memory, h0, (0, 1), n_steps)  → h
        7. norm_out(h) → output_head  → logits

    Args:
        base_model: Frozen LiquidARCModel.
        memory: Trainable WorkingMemory module.
        meta: Batch dict from task.generate_batch().
        device: Compute device.
        n_steps: Number of ODE steps to run.

    Returns:
        logits: [B, N, N_COLORS]
    """
    colors = meta["colors"]
    xs = meta["xs"]
    ys = meta["ys"]
    roles = meta["roles"]
    sep_mask = meta["sep_mask"]
    sep_types = meta["sep_types"]
    target_mask = meta["target_mask"]
    target_input_colors = meta.get("target_input_colors")
    context_mask = meta.get("context_mask")
    grid_ids = meta.get("grid_ids")

    # Step 1: mask test output colors (identical to model.forward)
    colors_masked = colors.clone()
    if target_input_colors is not None:
        colors_masked[target_mask] = target_input_colors[target_mask]
    else:
        colors_masked[target_mask] = PAD_COLOR

    # Step 2: embed (frozen)
    with torch.no_grad():
        h0 = base_model.embedding(colors_masked, xs, ys, roles, sep_mask, sep_types,
                                   grid_ids=grid_ids)
        # Step 3: context pool (frozen)
        context = base_model.context_pool(h0, context_mask)
        base_model.dynamics.set_context(context, mask=None)
        # Step 4: inform dynamics of step count
        base_model.dynamics.set_n_steps(n_steps)

    B = h0.shape[0]

    # Step 5: reset memory for this batch
    memory.reset(B, device)

    # Step 6: ODE integration with passive observation
    # Memory observes but NEVER modifies h — dynamics runs identically to base
    h0_for_ode = h0.detach()
    with torch.no_grad():
        h = euler_solve_with_observer(
            base_model.dynamics, memory, h0_for_ode,
            t_span=(0.0, 1.0), n_steps=n_steps,
        )

    # Step 7: base logits (frozen, no grad)
    with torch.no_grad():
        base_logits = base_model.output_head(base_model.norm_out(h))

    # Step 8: memory correction (trainable — only gradient path)
    correction = memory.get_output_correction(h.detach())
    logits = base_logits + correction

    return logits


def forward_base_only(
    base_model: LiquidARCModel,
    meta: Dict,
    device: torch.device,
    n_steps: int,
) -> torch.Tensor:
    """Run the base model forward pass without memory (for baseline eval comparison).

    Args:
        base_model: LiquidARCModel.
        meta: Batch dict from task.generate_batch().
        device: Compute device.
        n_steps: Number of ODE steps.

    Returns:
        logits: [B, N, N_COLORS]
    """
    from liquid_arc.solver import euler_solve

    colors = meta["colors"]
    xs = meta["xs"]
    ys = meta["ys"]
    roles = meta["roles"]
    sep_mask = meta["sep_mask"]
    sep_types = meta["sep_types"]
    target_mask = meta["target_mask"]
    target_input_colors = meta.get("target_input_colors")
    context_mask = meta.get("context_mask")
    grid_ids = meta.get("grid_ids")

    colors_masked = colors.clone()
    if target_input_colors is not None:
        colors_masked[target_mask] = target_input_colors[target_mask]
    else:
        colors_masked[target_mask] = PAD_COLOR

    h0 = base_model.embedding(colors_masked, xs, ys, roles, sep_mask, sep_types,
                               grid_ids=grid_ids)
    context = base_model.context_pool(h0, context_mask)
    base_model.dynamics.set_context(context, mask=None)
    base_model.dynamics.set_n_steps(n_steps)

    h = euler_solve(base_model.dynamics, h0, t_span=(0.0, 1.0), n_steps=n_steps)
    logits = base_model.output_head(base_model.norm_out(h))
    return logits


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_quick(
    base_model: LiquidARCModel,
    memory: Optional[WorkingMemory],
    eval_task,
    device: torch.device,
    config: LiquidARCConfig,
    n_batches: int = 10,
    batch_size: int = 8,
    n_steps: int = 16,
) -> Dict[str, float]:
    """Evaluate cell accuracy and xform accuracy.

    Runs with memory if memory is provided, without memory otherwise.

    Args:
        base_model: Frozen LiquidARCModel.
        memory: WorkingMemory to evaluate (None = base-only eval).
        eval_task: ARCTask or ProceduralARCTask instance.
        device: Compute device.
        config: LiquidARCConfig for loss weights.
        n_batches: Number of eval batches.
        batch_size: Eval batch size.
        n_steps: ODE steps.

    Returns:
        dict with cell_acc, xform_acc, ce_loss, xform_loss.
    """
    base_model.eval()
    if memory is not None:
        memory.eval()

    total_correct = 0
    total_cells = 0
    total_xform_correct = 0
    total_xform_cells = 0
    total_ce = 0.0
    total_xf = 0.0
    n_valid = 0

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for _ in range(n_batches):
            _, _, meta = eval_task.generate_batch(batch_size, device=device)
            target_labels = meta["target_labels"]
            target_mask = meta["target_mask"]
            target_input_colors = meta.get("target_input_colors")

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                if memory is not None:
                    logits = forward_with_memory(base_model, memory, meta, device, n_steps)
                else:
                    logits = forward_base_only(base_model, meta, device, n_steps)

            result = compute_loss_and_accuracy(
                logits, target_labels, target_mask, target_input_colors,
                config.transform_weight, config.copy_weight, device,
            )

            n_tgt = (target_labels != -100).sum().item()
            total_correct += int(result["cell_accuracy"].item() * n_tgt)
            total_cells += n_tgt

            n_xform = result["n_transform"].item()
            total_xform_correct += int(result["transform_accuracy"].item() * n_xform)
            total_xform_cells += n_xform

            total_ce += result["ce_loss"].item()
            total_xf += result["xform_loss"].item()
            n_valid += 1

    if memory is not None:
        memory.train()
    base_model.train()  # Restore train mode (even though frozen, for consistency)

    return {
        "cell_acc": total_correct / max(total_cells, 1),
        "xform_acc": total_xform_correct / max(total_xform_cells, 1),
        "ce_loss": total_ce / max(n_valid, 1),
        "xform_loss": total_xf / max(n_valid, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_memory_checkpoint(
    memory: WorkingMemory,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
    extra: Optional[Dict] = None,
) -> None:
    """Save memory module checkpoint (does NOT include base model weights).

    Args:
        memory: WorkingMemory module.
        optimizer: Memory optimizer (for resuming).
        step: Current training step.
        path: Output .pt file path.
        extra: Optional extra dict to merge into checkpoint.
    """
    ckpt = {
        "step": step,
        "memory": memory.state_dict(),
        "optimizer": optimizer.state_dict(),
        "memory_config": {
            "d_model": memory.d_model,
            "n_slots": memory.n_slots,
            "d_memory": memory.d_memory,
        },
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)
    print(f"  Saved memory checkpoint: {path}")


def load_base_model(
    checkpoint_path: str,
    config: LiquidARCConfig,
    device: torch.device,
) -> LiquidARCModel:
    """Load LiquidARCModel from checkpoint and freeze all parameters.

    Args:
        checkpoint_path: Path to base model .pt checkpoint.
        config: LiquidARCConfig for model construction.
        device: Device to load onto.

    Returns:
        Frozen LiquidARCModel in eval mode.
    """
    print(f"  Loading base model from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = LiquidARCModel(config).to(device)

    # Strip _orig_mod. prefix from torch.compile'd checkpoints (matches train.py)
    state = ckpt["model"]
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("._orig_mod.", ".")] = v
    model.load_state_dict(cleaned)

    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Base model loaded: {n_params:,} params (all frozen)")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args, config: LiquidARCConfig, device: torch.device) -> None:
    """Main training loop — trains WorkingMemory on frozen base model."""

    print(f"\n{'='*70}")
    print(f"WorkingMemory Training (frozen base)")
    print(f"{'='*70}")
    print(f"  Base checkpoint: {args.base_checkpoint}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Max steps: {args.max_steps}, LR: {args.lr}, Batch: {args.batch_size}")

    # ── Output directory ──────────────────────────────────────────────────────
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # ── Logging setup ─────────────────────────────────────────────────────────
    log_path = os.path.join(args.output_dir, "train_memory.log")
    logger = logging.getLogger("train_memory")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)

    import builtins
    _orig_print = builtins.print
    def _log_print(*a, **kw):
        logger.info(" ".join(str(x) for x in a))
    builtins.print = _log_print

    # ── Load frozen base model ────────────────────────────────────────────────
    base_model = load_base_model(args.base_checkpoint, config, device)

    # ── Working memory module (trainable) ─────────────────────────────────────
    memory = WorkingMemory(
        d_model=config.d_model,
        n_slots=args.n_slots,
        d_memory=args.d_memory,
    ).to(device)
    n_mem_params = sum(p.numel() for p in memory.parameters())
    print(f"  WorkingMemory params: {n_mem_params:,}")
    print(f"    n_slots={args.n_slots}, d_memory={args.d_memory}")

    # ── Data ──────────────────────────────────────────────────────────────────
    # Procedural training stream
    if config.use_procedural:
        train_task = ProceduralARCTask(
            seq_len=config.max_seq_len,
            stage=CurriculumStage.COMPOSITION,  # Use richest stage for fine-tuning
            include_lower=True,
            augment=True,
        )
        print(f"  Train data: ProceduralARCTask (COMPOSITION stage, {len(train_task.rules)} rules)")
    else:
        train_task = ARCTask(
            seq_len=config.max_seq_len,
            data_dir=args.data_dir,
            split="train",
            augment=True,
        )
        print(f"  Train data: ARCTask (real ARC train split)")

    # Real ARC mixed in at 50% (match V2 config)
    real_arc_train = None
    if config.real_arc_mix_ratio > 0 and args.data_dir:
        try:
            real_arc_train = ARCTask(
                seq_len=config.max_seq_len,
                data_dir=args.data_dir,
                split="train",
                augment=True,
            )
            print(f"  Real ARC mix: {config.real_arc_mix_ratio:.0%}")
        except Exception as e:
            print(f"  WARNING: Could not load real ARC data: {e}")

    # Eval always uses real ARC
    eval_task = None
    if args.data_dir:
        try:
            eval_task = ARCTask(
                seq_len=config.max_seq_len,
                data_dir=args.data_dir,
                split="eval",
                augment=False,
            )
            print(f"  Eval data: ARCTask (real ARC eval split)")
        except Exception as e:
            print(f"  WARNING: Could not load eval ARC data: {e}")

    if eval_task is None:
        # Fall back to procedural for eval
        eval_task = train_task
        print(f"  Eval data: ProceduralARCTask (no real ARC available)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(memory.parameters(), lr=args.lr, weight_decay=0.01)

    # Cosine schedule with warmup
    warmup_steps = min(500, args.max_steps // 20)
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Fixed ODE steps for memory training ───────────────────────────────────
    # No randomization — memory's step gate indexes are tied to specific steps.
    # Use the config default (typically 16).
    n_steps = config.n_ode_steps

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Training loop ─────────────────────────────────────────────────────────
    memory.train()
    base_model.eval()  # Always eval — it is frozen

    t0 = time.time()
    running_loss = 0.0
    running_xform_acc = 0.0
    running_cell_acc = 0.0

    print(f"\n  Starting training for {args.max_steps} steps...")

    for step in range(args.max_steps):
        optimizer.zero_grad()

        # Sample batch (50% real ARC if available)
        use_real = (real_arc_train is not None
                    and random.random() < config.real_arc_mix_ratio)
        if use_real:
            _, _, meta = real_arc_train.generate_batch(args.batch_size, device=device)
        else:
            _, _, meta = train_task.generate_batch(args.batch_size, device=device)

        target_labels = meta["target_labels"]
        target_mask = meta["target_mask"]
        target_input_colors = meta.get("target_input_colors")

        # Forward pass (base frozen, memory trainable)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_with_memory(base_model, memory, meta, device, n_steps)
                result = compute_loss_and_accuracy(
                    logits, target_labels, target_mask, target_input_colors,
                    config.transform_weight, config.copy_weight, device,
                )
                # Use xform_loss only — prevents copy bias from dominating memory learning
                loss = result["xform_loss"] if result["xform_loss"].item() > 0 else result["loss"]
        else:
            logits = forward_with_memory(base_model, memory, meta, device, n_steps)
            result = compute_loss_and_accuracy(
                logits, target_labels, target_mask, target_input_colors,
                config.transform_weight, config.copy_weight, device,
            )
            loss = result["xform_loss"] if result["xform_loss"].item() > 0 else result["loss"]

        # Backward + step (only memory params have gradients)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(memory.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(memory.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Accumulate for logging
        running_loss += loss.item()
        running_xform_acc += result["transform_accuracy"].item()
        running_cell_acc += result["cell_accuracy"].item()

        # ── Periodic logging ──────────────────────────────────────────────────
        if step % args.log_every == 0 or step == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / max(1, args.log_every if step > 0 else 1)
            avg_xform = running_xform_acc / max(1, args.log_every if step > 0 else 1)
            avg_cell = running_cell_acc / max(1, args.log_every if step > 0 else 1)
            tok_s = args.batch_size * config.max_seq_len * (step + 1) / max(elapsed, 1e-6)

            diag = memory.get_diagnostics()
            slot_norm = diag["mem_slot_norm"].item()
            slot_var = diag["mem_slot_var"].item()

            # Step gate values (if available)
            if hasattr(memory, 'step_gate_bias'):
                gate_vals = torch.sigmoid(memory.step_gate_bias.detach())
                gate_str = ", ".join(f"{v:.2f}" for v in gate_vals.tolist())
            else:
                gate_str = "N/A (v4 observe-only)"

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"  [step={step:5d}] loss={avg_loss:.4f}, "
                f"xform_acc={avg_xform:.4f}, cell_acc={avg_cell:.4f}, "
                f"slot_norm={slot_norm:.4f}, slot_var={slot_var:.4f}, "
                f"lr={current_lr:.2e}, tok/s={tok_s:.0f}"
            )
            print(f"           gates=[{gate_str}]")

            running_loss = 0.0
            running_xform_acc = 0.0
            running_cell_acc = 0.0

        # ── Periodic eval ─────────────────────────────────────────────────────
        if step > 0 and step % args.eval_every == 0:
            print(f"\n  >> EVAL at step {step}")

            # With memory
            mem_results = evaluate_quick(
                base_model, memory, eval_task, device, config,
                n_batches=args.eval_batches, batch_size=args.batch_size, n_steps=n_steps,
            )

            # Base only (no memory)
            base_results = evaluate_quick(
                base_model, None, eval_task, device, config,
                n_batches=args.eval_batches, batch_size=args.batch_size, n_steps=n_steps,
            )

            print(
                f"     base:   cell={base_results['cell_acc']:.4f}, "
                f"xform={base_results['xform_acc']:.4f}, "
                f"ce={base_results['ce_loss']:.4f}, xf_loss={base_results['xform_loss']:.4f}"
            )
            print(
                f"     +mem:   cell={mem_results['cell_acc']:.4f}, "
                f"xform={mem_results['xform_acc']:.4f}, "
                f"ce={mem_results['ce_loss']:.4f}, xf_loss={mem_results['xform_loss']:.4f}"
            )
            xform_delta = mem_results["xform_acc"] - base_results["xform_acc"]
            cell_delta = mem_results["cell_acc"] - base_results["cell_acc"]
            sign = "+" if xform_delta >= 0 else ""
            print(
                f"     delta:  cell={sign}{cell_delta:.4f}, "
                f"xform={sign}{xform_delta:.4f}"
            )
            print()

            memory.train()

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, "checkpoints", f"memory_step_{step}.pt")
            save_memory_checkpoint(memory, optimizer, step, ckpt_path)

    # ── Final checkpoint ──────────────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "checkpoints", "memory_final.pt")
    save_memory_checkpoint(memory, optimizer, args.max_steps, final_path)

    # ── Final comparison table ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Final Comparison: Base vs +Memory")
    print(f"{'='*70}")

    n_final_batches = max(args.eval_batches, 20)

    base_results = evaluate_quick(
        base_model, None, eval_task, device, config,
        n_batches=n_final_batches, batch_size=args.batch_size, n_steps=n_steps,
    )
    mem_results = evaluate_quick(
        base_model, memory, eval_task, device, config,
        n_batches=n_final_batches, batch_size=args.batch_size, n_steps=n_steps,
    )

    print(f"  {'Metric':<20} {'Base':>10} {'Base+Mem':>10} {'Delta':>10}")
    print(f"  {'-'*52}")
    for key, label in [
        ("cell_acc",   "Cell Accuracy"),
        ("xform_acc",  "Xform Accuracy"),
        ("ce_loss",    "CE Loss"),
        ("xform_loss", "Xform Loss"),
    ]:
        b = base_results[key]
        m = mem_results[key]
        d = m - b
        sign = "+" if d >= 0 else ""
        print(f"  {label:<20} {b:>10.4f} {m:>10.4f} {sign+f'{d:.4f}':>10}")

    print(f"  {'-'*52}")
    print(f"\n  Memory checkpoint: {final_path}")
    print(f"  Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing + entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train WorkingMemory module on frozen LiquidARCModel base."
    )

    # Required
    parser.add_argument("--base_checkpoint", type=str, required=True,
                        help="Path to base LiquidARCModel checkpoint (.pt)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to LiquidARCConfig YAML file")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/arc",
                        help="ARC data directory (for real ARC training + eval)")

    # Output
    parser.add_argument("--output_dir", type=str, default="output_memory",
                        help="Directory to save checkpoints and logs")

    # Training
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Total training steps")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="AdamW learning rate for memory module")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")

    # Memory module architecture
    parser.add_argument("--n_slots", type=int, default=8,
                        help="Number of memory slots")
    parser.add_argument("--d_memory", type=int, default=64,
                        help="Memory slot dimension")

    # Logging and checkpointing
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log every N training steps")
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Evaluate every N training steps")
    parser.add_argument("--eval_batches", type=int, default=10,
                        help="Number of batches per eval run")
    parser.add_argument("--save_every", type=int, default=2500,
                        help="Save checkpoint every N steps")

    args = parser.parse_args()

    config = LiquidARCConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Honor CUDA memory fraction if set
    mem_frac = float(os.environ.get("CUDA_MEMORY_FRACTION", "0"))
    if mem_frac > 0 and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(mem_frac)
        print(f"CUDA memory fraction capped at {mem_frac:.0%}")

    print(f"Device: {device}")
    print(f"Config: {args.config}")

    train(args, config, device)


if __name__ == "__main__":
    main()
