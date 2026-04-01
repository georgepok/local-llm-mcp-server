"""LiquidARC standalone training — no external dependencies.

Reproduces the CV-driven phase transition using only procedural data.
No ARC dataset, no fgn-v3 dependency, no TTT, no Reptile.

Usage:
    python scripts/train_standalone.py \
        --config configs/reproduce_phase_transition.yaml \
        --output_dir output_reproduce \
        --max_steps 15000 \
        --seed 42
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel, FlatBaselineARC, create_model
from liquid_arc.tasks.procedural import ProceduralARCTask, CurriculumStage


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"  Random seed: {seed}")


def create_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def print_status(model, result, step, tok_per_sec):
    """Print compact status line."""
    extra = ""
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    if isinstance(raw, LiquidARCModel):
        cv_val = result["metric_cv"]
        if isinstance(cv_val, torch.Tensor):
            cv_val = cv_val.item()
        kappa_val = result["avg_kappa"]
        if isinstance(kappa_val, torch.Tensor):
            kappa_val = kappa_val.item()
        tau_val = result.get("tau_avg", torch.tensor(0.0))
        if isinstance(tau_val, torch.Tensor):
            tau_val = tau_val.item()
        tau_std = result.get("tau_std", torch.tensor(0.0))
        if isinstance(tau_std, torch.Tensor):
            tau_std = tau_std.item()
        tau_min = result.get("tau_min", torch.tensor(0.0))
        if isinstance(tau_min, torch.Tensor):
            tau_min = tau_min.item()
        tau_max = result.get("tau_max", torch.tensor(0.0))
        if isinstance(tau_max, torch.Tensor):
            tau_max = tau_max.item()
        extra = (f", cv={cv_val:.4f}, |k|={kappa_val:.4f}, "
                 f"tau={tau_val:.2f}[{tau_min:.2f}-{tau_max:.2f}]σ={tau_std:.3f}")

    cell_acc = result.get("cell_accuracy", torch.tensor(0.0))
    if isinstance(cell_acc, torch.Tensor):
        cell_acc = cell_acc.item()
    xform_acc = result.get("transform_accuracy", torch.tensor(0.0))
    if isinstance(xform_acc, torch.Tensor):
        xform_acc = xform_acc.item()
    xform_loss = result.get("xform_loss", torch.tensor(0.0))
    if isinstance(xform_loss, torch.Tensor):
        xform_loss = xform_loss.item()

    print(f"  [step={step}] loss={result['loss'].item():.4f}, "
          f"ce={result['ce_loss'].item():.4f}, xf_loss={xform_loss:.4f}, "
          f"cell_acc={cell_acc:.4f}, xform_acc={xform_acc:.4f}, "
          f"tok/s={tok_per_sec:.0f}{extra}")


def save_checkpoint(model, optimizer, config, step, path, extra=None):
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def evaluate_procedural(model, config, device, n_batches=20, batch_size=8):
    """Eval on held-out procedural tasks (different seed range)."""
    model.eval()
    eval_task = ProceduralARCTask(
        seq_len=config.max_seq_len,
        stage=CurriculumStage.GLOBAL,
        include_lower=True,
        augment=False,
    )
    # Use a fixed seed offset so eval tasks are consistent across runs
    eval_task._seed_counter = 10_000_000

    total_correct = 0
    total_cells = 0
    total_xform_correct = 0
    total_xform_cells = 0
    total_copy_correct = 0
    total_ce = 0.0
    n_valid = 0

    with torch.no_grad():
        for _ in range(n_batches):
            _, _, meta = eval_task.generate_batch(batch_size, device=device)
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
                )

            cell_acc = result.get("cell_accuracy", torch.tensor(0.0))
            if isinstance(cell_acc, torch.Tensor):
                cell_acc = cell_acc.item()
            n_tgt = (meta["target_labels"] != -100).sum().item()
            total_correct += int(cell_acc * n_tgt)
            total_cells += n_tgt

            xform_acc = result.get("transform_accuracy", torch.tensor(0.0))
            if isinstance(xform_acc, torch.Tensor):
                xform_acc = xform_acc.item()
            n_xform = result.get("n_transform", torch.tensor(0))
            if isinstance(n_xform, torch.Tensor):
                n_xform = n_xform.item()
            total_xform_correct += int(xform_acc * n_xform)
            total_xform_cells += n_xform

            tgt = meta["target_labels"]
            inp = meta.get("target_input_colors")
            if inp is not None:
                valid = tgt != -100
                total_copy_correct += int((tgt[valid] == inp[valid]).sum().item())

            total_ce += result["ce_loss"].item()
            n_valid += 1

    cell_acc = total_correct / max(total_cells, 1)
    xform_acc = total_xform_correct / max(total_xform_cells, 1)
    copy_bl = total_copy_correct / max(total_cells, 1)
    avg_ce = total_ce / max(n_valid, 1)
    return cell_acc, xform_acc, copy_bl, avg_ce


def train(args, config, device):
    """Training loop — procedural only, no external data."""
    print(f"\n{'='*70}")
    print(f"LiquidARC Phase Transition Reproduction")
    print(f"{'='*70}")

    set_seed(args.seed)

    model = create_model(config, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    if isinstance(model, LiquidARCModel):
        n_geo = sum(p.numel() for p in model.geo_parameters())
        n_other = sum(p.numel() for p in model.other_parameters())
        print(f"  Geo params: {n_geo:,}, Other: {n_other:,}")
        print(f"  ODE steps: {config.n_ode_steps}, d_metric: {config.d_metric}")

    out_dir = args.output_dir
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    # File + stdout logging
    log_path = os.path.join(out_dir, "train.log")
    logger = logging.getLogger("train")
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
        msg = " ".join(str(x) for x in a)
        logger.info(msg)
    builtins.print = _log_print

    writer = SummaryWriter(os.path.join(out_dir, "logs"))

    # Procedural data — infinite stream with curriculum
    proc_tasks = {
        CurriculumStage.GLOBAL: ProceduralARCTask(
            seq_len=config.max_seq_len, stage=CurriculumStage.GLOBAL,
            include_lower=False, augment=True,
        ),
        CurriculumStage.RELATIONAL: ProceduralARCTask(
            seq_len=config.max_seq_len, stage=CurriculumStage.RELATIONAL,
            include_lower=True, augment=True,
        ),
        CurriculumStage.COMPOSITION: ProceduralARCTask(
            seq_len=config.max_seq_len, stage=CurriculumStage.COMPOSITION,
            include_lower=True, augment=True,
        ),
    }
    task = proc_tasks[CurriculumStage.GLOBAL]
    print(f"  Data: Procedural (infinite stream, seed={args.seed})")
    print(f"    Stage 1 (GLOBAL): steps 0-{config.curriculum_stage1_end}")
    print(f"    Stage 2 (RELATIONAL): steps {config.curriculum_stage1_end}+")

    # Optimizer
    if isinstance(model, LiquidARCModel) and args.geo_lr_mult != 1.0:
        param_groups = [
            {"params": model.other_parameters(), "lr": args.lr},
            {"params": model.geo_parameters(), "lr": args.lr * args.geo_lr_mult},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        print(f"  Geo LR: {args.lr * args.geo_lr_mult:.2e} ({args.geo_lr_mult}x)")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)

    scheduler = create_scheduler(optimizer, args.warmup_steps, args.max_steps)

    # Resume
    start_step = 0
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = ckpt["model"]
        cleaned = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
        model.load_state_dict(cleaned)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        for _ in range(start_step):
            scheduler.step()
        print(f"  Resumed at step {start_step}")

    # torch.compile
    if config.use_torch_compile and device.type == "cuda" and isinstance(model, LiquidARCModel):
        model.dynamics = torch.compile(model.dynamics, mode="default", dynamic=True)
        print(f"  torch.compile: dynamics compiled (unrolled Euler SDPA)")

    model.train()
    t0 = time.time()
    current_stage = CurriculumStage.GLOBAL
    grad_accum = args.grad_accum_steps

    # Phase transition detection
    transition_detected = False
    prev_cv = 0.0

    for step in range(start_step, args.max_steps):
        optimizer.zero_grad()

        # Curriculum transitions
        if step < config.curriculum_stage1_end:
            new_stage = CurriculumStage.GLOBAL
        elif step < config.curriculum_stage2_end:
            new_stage = CurriculumStage.RELATIONAL
        else:
            new_stage = CurriculumStage.COMPOSITION

        if new_stage != current_stage:
            current_stage = new_stage
            task = proc_tasks[current_stage]
            print(f"\n  >> CURRICULUM: Stage {current_stage.name} "
                  f"({len(task.rules)} rules) at step {step}\n")

        # Tau freeze
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        if isinstance(raw_model, LiquidARCModel):
            should_freeze = step < config.tau_freeze_steps
            if should_freeze != raw_model.dynamics.freeze_tau:
                raw_model.dynamics.freeze_tau = should_freeze
                if not should_freeze:
                    print(f"\n  >> TAU UNFROZEN at step {step}\n")

        # ODE step randomization
        if isinstance(raw_model, LiquidARCModel):
            n_steps = random.randint(config.ode_steps_min, config.ode_steps_max)
        else:
            n_steps = None

        # Forward + backward
        for micro in range(grad_accum):
            _, _, meta = task.generate_batch(args.batch_size, device=device)

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

            (result["loss"] / grad_accum).backward()

        # Tau grad zeroing during freeze
        if isinstance(raw_model, LiquidARCModel) and raw_model.dynamics.freeze_tau:
            if hasattr(config, 'channel_gate_enabled') and config.channel_gate_enabled:
                freeze_mods = [raw_model.dynamics.gate_net_linear1,
                               raw_model.dynamics.gate_net_linear2]
            else:
                freeze_mods = [raw_model.dynamics.tau_net_linear1,
                               raw_model.dynamics.tau_net_linear2]
            for mod in freeze_mods:
                for p in mod.parameters():
                    if p.grad is not None:
                        p.grad.zero_()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # Logging
        if step % args.log_every == 0:
            dt = time.time() - t0
            avg_n = meta.get("lengths", torch.tensor(config.max_seq_len)).float().mean().item()
            tok_s = args.batch_size * avg_n * (step + 1) / max(dt, 1e-6)
            print_status(model, result, step, tok_s)

            # TensorBoard
            writer.add_scalar("loss/total", result["loss"].item(), step)
            writer.add_scalar("loss/ce", result["ce_loss"].item(), step)

            if isinstance(raw_model, LiquidARCModel):
                cv_val = result["metric_cv"]
                if isinstance(cv_val, torch.Tensor):
                    cv_val = cv_val.item()
                kappa_val = result["avg_kappa"]
                if isinstance(kappa_val, torch.Tensor):
                    kappa_val = kappa_val.item()
                tau_val = result.get("tau_avg", torch.tensor(0.0))
                if isinstance(tau_val, torch.Tensor):
                    tau_val = tau_val.item()

                writer.add_scalar("metric/cv", cv_val, step)
                writer.add_scalar("metric/kappa", kappa_val, step)
                writer.add_scalar("metric/tau", tau_val, step)

                cell_acc = result.get("cell_accuracy", torch.tensor(0.0))
                if isinstance(cell_acc, torch.Tensor):
                    cell_acc = cell_acc.item()
                xf_acc = result.get("transform_accuracy", torch.tensor(0.0))
                if isinstance(xf_acc, torch.Tensor):
                    xf_acc = xf_acc.item()
                writer.add_scalar("accuracy/cell_train", cell_acc, step)
                writer.add_scalar("accuracy/xform_train", xf_acc, step)

                cv_floor_l = result.get("cv_floor_loss", torch.tensor(0.0))
                if isinstance(cv_floor_l, torch.Tensor):
                    cv_floor_l = cv_floor_l.item()
                writer.add_scalar("loss/cv_floor", cv_floor_l, step)

                # Phase transition detection
                if not transition_detected and cv_val > 5.5 and prev_cv > 0:
                    loss_val = result["loss"].item()
                    if loss_val < 1.5:  # loss has started dropping
                        transition_detected = True
                        print(f"\n  *** PHASE TRANSITION DETECTED at step {step} ***")
                        print(f"      CV={cv_val:.2f}, loss={loss_val:.4f}")
                        print(f"      Saving transition checkpoint...\n")
                        save_checkpoint(model, optimizer, config, step,
                                      os.path.join(out_dir, "checkpoints", "transition.pt"))
                prev_cv = cv_val

        # Procedural eval (no ARC data needed)
        if step > 0 and step % args.eval_every == 0:
            eval_acc, eval_xform, eval_copy_bl, eval_ce = evaluate_procedural(
                model, config, device, n_batches=args.eval_batches,
                batch_size=args.batch_size)
            writer.add_scalar("accuracy/cell_eval", eval_acc, step)
            writer.add_scalar("accuracy/xform_eval", eval_xform, step)
            writer.add_scalar("accuracy/copy_baseline", eval_copy_bl, step)
            writer.add_scalar("loss/eval_ce", eval_ce, step)
            print(f"  >> EVAL [step={step}] cell_acc={eval_acc:.4f}, "
                  f"xform_acc={eval_xform:.4f}, copy_bl={eval_copy_bl:.4f}, "
                  f"ce={eval_ce:.4f}")
            model.train()

        # Checkpoints
        if step > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, config, step,
                          os.path.join(out_dir, "checkpoints", f"step_{step}.pt"))

    save_checkpoint(model, optimizer, config, args.max_steps,
                   os.path.join(out_dir, "checkpoints", "final.pt"))
    writer.close()
    print(f"\n  Training complete ({args.max_steps} steps).")
    if transition_detected:
        print(f"  Phase transition was detected during training.")
    else:
        print(f"  WARNING: No phase transition detected. Check CV trajectory.")


def main():
    parser = argparse.ArgumentParser(description="LiquidARC Phase Transition Reproduction")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_reproduce")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--geo_lr_mult", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=2500)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = LiquidARCConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")

    train(args, config, device)


if __name__ == "__main__":
    main()
