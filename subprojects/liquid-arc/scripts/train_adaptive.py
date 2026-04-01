"""LiquidARC training script with adaptive criticality controller.

Identical to train.py except the fixed real_arc_mix_ratio is replaced by
a CriticalityController that steers CV toward the critical zone (4.5–6.0)
by adjusting the real ARC mixing ratio each step.

Usage:
    python scripts/train_adaptive.py --config configs/liquid_arc.yaml --data_dir data/arc

The controller trajectory is saved to <output_dir>/controller_log.csv.
See liquid_arc/criticality_controller.py for tuning knobs.
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel, FlatBaselineARC, create_model
from liquid_arc.tasks.procedural import ProceduralARCTask, CurriculumStage
from liquid_arc.criticality_controller import CriticalityController

# Import ARC task from fgn-v3 (for eval on real ARC data)
# On Spark container: /workspace/fgn-v3/; locally: ../../fgn-v3/
FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if not Path(FGN_ROOT).exists():
    FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, FGN_ROOT)
from fgn.tasks.arc import ARCTask
from liquid_arc.ttt import evaluate_ttt
from liquid_arc.reptile import reptile_step
from fgn.tasks.arc import load_arc_tasks as load_arc_tasks_raw


def compute_geo_phase(step: int, config) -> int:
    """Determine geo supervision phase from step count.
    Phase 0: geo disabled (geo_loss_enabled=False or past cutoff)
    Phase 1: squared Manhattan only (steps 0 to geo_phase2_start)
    Phase 2: object boundaries (geo_phase2_start onwards)
    """
    if not getattr(config, 'geo_loss_enabled', False):
        return 0
    cutoff = getattr(config, 'geo_cutoff_step', 0)
    if cutoff > 0 and step >= cutoff:
        return 0  # hard cutoff — geo is dead
    if step < config.geo_phase2_start:
        return 1
    return 2


def compute_geo_lambda(step: int, config) -> float:
    """Compute geo loss weight."""
    if not getattr(config, 'geo_loss_enabled', False):
        return 0.0
    cutoff = getattr(config, 'geo_cutoff_step', 0)
    if cutoff > 0 and step >= cutoff:
        return 0.0
    return config.geo_lambda_init


def compute_boundary_alpha(step: int, config) -> float:
    """Compute interpolation alpha for Phase 2 boundary targets.
    0.0 = pure Manhattan², 1.0 = pure boundary (0/wall).
    Ramps linearly over geo_phase2_interp_steps starting at geo_phase2_start.
    """
    if not getattr(config, 'geo_loss_enabled', False):
        return 1.0
    if step < config.geo_phase2_start:
        return 0.0  # Phase 1, not applicable
    interp_steps = getattr(config, 'geo_phase2_interp_steps', 3000)
    elapsed = step - config.geo_phase2_start
    if elapsed >= interp_steps:
        return 1.0
    return elapsed / max(1, interp_steps)


def compute_ce_lambda(step: int, config) -> float:
    """Compute CE loss weight with ramp schedule."""
    if not getattr(config, 'geo_loss_enabled', False):
        return 1.0  # no geo → full CE always
    cutoff = getattr(config, 'geo_cutoff_step', 0)
    if cutoff > 0 and step >= cutoff:
        return 1.0  # past cutoff → full CE immediately
    if step < config.geo_ce_ramp_start:
        return 0.0
    if step >= config.geo_ce_ramp_end:
        return 1.0
    # Linear ramp
    progress = (step - config.geo_ce_ramp_start) / max(
        1, config.geo_ce_ramp_end - config.geo_ce_ramp_start
    )
    return progress


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
    if isinstance(model, LiquidARCModel) or (
        hasattr(model, '_orig_mod') and isinstance(model._orig_mod, LiquidARCModel)
    ):
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

    # Geo loss info
    geo_extra = ""
    geo_l = result.get("geo_loss", torch.tensor(0.0))
    if isinstance(geo_l, torch.Tensor):
        geo_l = geo_l.item()
    if geo_l > 0:
        geo_kl = result.get("geo_mse", torch.tensor(0.0))
        if isinstance(geo_kl, torch.Tensor):
            geo_kl = geo_kl.item()
        geo_extra = f", geo={geo_l:.4f}"

    print(f"  [step={step}] loss={result['loss'].item():.4f}, "
          f"ce={result['ce_loss'].item():.4f}, xf_loss={xform_loss:.4f}, "
          f"cell_acc={cell_acc:.4f}, xform_acc={xform_acc:.4f}, "
          f"tok/s={tok_per_sec:.0f}{extra}{geo_extra}")


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


def evaluate_quick(model, eval_task, device, n_batches=10, batch_size=8):
    """Quick eval: cell accuracy, xform accuracy, CE loss, and copy baseline."""
    model.eval()
    total_correct = 0
    total_cells = 0
    total_xform_correct = 0
    total_xform_cells = 0
    total_copy_correct = 0  # cells where copy would be right
    total_ce = 0.0
    total_xf_loss = 0.0
    n_valid_batches = 0

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

        # Transform accuracy
        xform_acc = result.get("transform_accuracy", torch.tensor(0.0))
        if isinstance(xform_acc, torch.Tensor):
            xform_acc = xform_acc.item()
        n_xform = result.get("n_transform", torch.tensor(0))
        if isinstance(n_xform, torch.Tensor):
            n_xform = n_xform.item()
        total_xform_correct += int(xform_acc * n_xform)
        total_xform_cells += n_xform

        # Copy baseline: how many target cells == input cells
        tgt = meta["target_labels"]
        inp = meta.get("target_input_colors")
        if inp is not None:
            valid = tgt != -100
            total_copy_correct += int((tgt[valid] == inp[valid]).sum().item())

        total_ce += result["ce_loss"].item()
        xf_l = result.get("xform_loss", torch.tensor(0.0))
        if isinstance(xf_l, torch.Tensor):
            xf_l = xf_l.item()
        total_xf_loss += xf_l
        n_valid_batches += 1

    cell_acc = total_correct / max(total_cells, 1)
    xform_acc = total_xform_correct / max(total_xform_cells, 1)
    copy_baseline = total_copy_correct / max(total_cells, 1)
    avg_ce = total_ce / max(n_valid_batches, 1)
    avg_xf_loss = total_xf_loss / max(n_valid_batches, 1)
    return cell_acc, xform_acc, copy_baseline, avg_ce, avg_xf_loss


def train(args, config, device):
    """Training loop with adaptive criticality control."""
    print(f"\n{'='*70}")
    print(f"LiquidARC Adaptive Training — {config.model_type}")
    print(f"{'='*70}")

    model = create_model(config, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    if isinstance(model, LiquidARCModel):
        print(f"  Architecture: LiquidARC (continuous-time geometric ODE)")
        print(f"  ODE steps: {config.n_ode_steps}, d_metric: {config.d_metric}")
        n_geo = sum(p.numel() for p in model.geo_parameters())
        n_other = sum(p.numel() for p in model.other_parameters())
        print(f"  Geo params: {n_geo:,}, Other: {n_other:,}")
    else:
        print(f"  Architecture: Flat baseline transformer")

    # Output directory + file logging
    out_dir = args.output_dir
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    # Set up logging to file + stdout
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

    # Redirect print → logger
    import builtins
    _orig_print = builtins.print
    def _log_print(*a, **kw):
        msg = " ".join(str(x) for x in a)
        logger.info(msg)
    builtins.print = _log_print

    writer = SummaryWriter(os.path.join(out_dir, "logs"))

    # --- Adaptive criticality controller ---
    # Replaces the fixed config.real_arc_mix_ratio coin-flip with a
    # model-based controller that tracks CV and adjusts the ratio dynamically.
    controller_log_path = os.path.join(out_dir, "controller_log.csv")
    controller = CriticalityController(
        initial_ratio=config.real_arc_mix_ratio if config.real_arc_mix_ratio > 0 else 0.30,
        log_path=controller_log_path,
    )
    print(f"  Criticality controller: enabled")
    print(f"    initial_ratio={controller.arc_ratio:.2f}, "
          f"cv_zone=[{controller.cv_zone_low}, {controller.cv_zone_high}]")
    print(f"    alpha={controller.alpha}, alpha_fast={controller.alpha * 2.0:.3f}, "
          f"update_every={controller.update_every}")
    print(f"    log: {controller_log_path}")

    # State carried across steps for controller.update()
    last_cv: float = 0.0
    last_xform: float = 0.0
    # current_ratio is read from controller each step
    current_ratio: float = controller.arc_ratio

    # Data — procedural infinite stream with curriculum, or static ARC
    if config.use_procedural:
        # Create one task per curriculum stage; switch during training
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
        task = proc_tasks[CurriculumStage.GLOBAL]  # initial stage
        print(f"  Data: Procedural (infinite stream)")
        print(f"    Stage 1 (GLOBAL): steps 0-{config.curriculum_stage1_end}")
        print(f"    Stage 2 (RELATIONAL): steps {config.curriculum_stage1_end}-{config.curriculum_stage2_end}")
        print(f"    Stage 3 (COMPOSITION): steps {config.curriculum_stage2_end}+")
    else:
        task = ARCTask(
            seq_len=config.max_seq_len,
            data_dir=args.data_dir,
            split="train",
            augment=True,
            n_color_perms=args.n_color_perms,
        )
        proc_tasks = None

    # Eval always uses real ARC data
    eval_task = ARCTask(
        seq_len=config.max_seq_len,
        data_dir=args.data_dir,
        split="eval",
        augment=False,
    )

    # Real ARC training data mixed in (controller dynamically adjusts the ratio)
    real_arc_train = None
    if config.real_arc_mix_ratio > 0:
        real_arc_train = ARCTask(
            seq_len=config.max_seq_len,
            data_dir=args.data_dir,
            split="train",
            augment=True,
            n_color_perms=args.n_color_perms,
        )
        print(f"  Real ARC mix: adaptive (base={config.real_arc_mix_ratio:.0%})")

    # Reptile meta-learning: load ARC tasks for inner TTT loop
    reptile_arc_tasks = None
    if config.reptile_enabled:
        all_arc = load_arc_tasks_raw(args.data_dir)
        split = "train" if config.reptile_use_train_split else "eval"
        reptile_arc_tasks = all_arc.get(split, [])
        print(f"  Reptile: {len(reptile_arc_tasks)} {split} tasks, "
              f"every {config.reptile_every} steps, K={config.reptile_n_tasks}")

    # Optimizer with param groups
    if isinstance(model, LiquidARCModel) and args.geo_lr_mult != 1.0:
        geo_params = model.geo_parameters()
        other_params = model.other_parameters()
        param_groups = [
            {"params": other_params, "lr": args.lr},
            {"params": geo_params, "lr": args.lr * args.geo_lr_mult},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        print(f"  Geo LR: {args.lr * args.geo_lr_mult:.2e} ({args.geo_lr_mult}x)")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)

    scheduler = create_scheduler(optimizer, args.warmup_steps, args.max_steps)

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # Strip _orig_mod. prefix from torch.compile'd checkpoints
        state = ckpt["model"]
        cleaned = {}
        for k, v in state.items():
            cleaned[k.replace("._orig_mod.", ".")] = v
        model.load_state_dict(cleaned)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        # Advance scheduler to match resumed step
        for _ in range(start_step):
            scheduler.step()
        print(f"  Resumed at step {start_step}")

    # Torch compile — compile the dynamics module only (the hot path).
    # DEQ calls compiled dynamics inside autograd.Function.backward() — must
    # disable donated buffers to avoid conflict with nested backward passes.
    if config.use_torch_compile and device.type == "cuda" and isinstance(model, LiquidARCModel):
        if config.deq_solver:
            torch._functorch.config.donated_buffer = False
        model.dynamics = torch.compile(model.dynamics, mode="default", dynamic=True)
        solver_name = "DEQ" if config.deq_solver else "unrolled Euler (SDPA)"
        print(f"  torch.compile: dynamics compiled ({solver_name} solver)")
    compiled_model = model

    compiled_model.train()
    t0 = time.time()
    best_eval_acc = 0.0
    grad_accum = args.grad_accum_steps
    current_stage = CurriculumStage.GLOBAL if config.use_procedural else None

    for step in range(start_step, args.max_steps):
        optimizer.zero_grad()

        # --- Controller: update ratio using last step's CV + xform ---
        # On the very first step last_cv=0.0 — controller returns initial_ratio
        # until it accumulates enough history (30 steps).
        current_ratio = controller.update(last_cv, last_xform, step)

        # Curriculum stage transitions
        if proc_tasks is not None:
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

        # Fix B: Tau freeze for first N steps
        if isinstance(model, LiquidARCModel) or (
            hasattr(model, '_orig_mod') and isinstance(model._orig_mod, LiquidARCModel)
        ):
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            should_freeze = step < config.tau_freeze_steps
            if should_freeze != raw_model.dynamics.freeze_tau:
                raw_model.dynamics.freeze_tau = should_freeze
                if not should_freeze:
                    print(f"\n  >> TAU UNFROZEN at step {step}\n")

        # Temporal invariance: randomize ODE step count per batch
        if isinstance(model, LiquidARCModel) or (
            hasattr(model, '_orig_mod') and isinstance(model._orig_mod, LiquidARCModel)
        ):
            n_steps = random.randint(config.ode_steps_min, config.ode_steps_max)
        else:
            n_steps = None

        # Geo phase scheduling
        geo_phase = compute_geo_phase(step, config)
        lambda_ce = compute_ce_lambda(step, config)
        lambda_geo = compute_geo_lambda(step, config)
        boundary_alpha = compute_boundary_alpha(step, config)

        # Print phase transitions
        if step == 0 and geo_phase > 0:
            phase_labels = {1: "PURE GEOMETRY (MetricNet only, CE=0)",
                            2: "OBJECT ISLANDS (boundaries, CE ramp, permanent scaffold)"}
            print(f"\n  >> GEO PHASE {geo_phase}: {phase_labels.get(geo_phase, '')} "
                  f"(λ_ce={lambda_ce:.2f}, λ_geo={lambda_geo:.2f})\n")
        elif step > 0 and geo_phase != compute_geo_phase(step - 1, config):
            phase_labels = {0: "GEO CUTOFF — pure CE + curvature from here",
                            1: "PURE GEOMETRY (MetricNet only, CE=0)",
                            2: "OBJECT ISLANDS (boundaries, CE ramp)"}
            print(f"\n  >> GEO PHASE {geo_phase}: {phase_labels.get(geo_phase, '')} "
                  f"at step {step} (λ_ce={lambda_ce:.2f}, λ_geo={lambda_geo:.2f})\n")

        # Gradient accumulation: multiple micro-batches per optimizer step
        for micro in range(grad_accum):
            # Real ARC mixing: use controller ratio instead of fixed config ratio
            use_real = (real_arc_train is not None
                        and random.random() < current_ratio)
            if use_real:
                _, _, meta = real_arc_train.generate_batch(args.batch_size, device=device)
            else:
                _, _, meta = task.generate_batch(args.batch_size, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = compiled_model(
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
                    boundary_alpha=boundary_alpha,
                )

            # Assemble final loss: train.py controls the weighting
            if geo_phase > 0:
                final_loss = (
                    lambda_ce * result["ce_loss"]
                    + lambda_geo * result["geo_loss"]
                    + result["curv_loss"]
                    + result["tau_var_loss"]
                    + result.get("cv_floor_loss", torch.tensor(0.0, device=device))
                )
                # Store assembled loss for logging
                result["loss"] = final_loss
            # else: result["loss"] already = ce + curv + tau_var + cv_floor from model

            (result["loss"] / grad_accum).backward()

        # --- Extract CV and xform for next step's controller update ---
        cv_val = result.get("metric_cv", torch.tensor(0.0))
        if isinstance(cv_val, torch.Tensor):
            cv_val = cv_val.item()
        last_cv = cv_val

        xf_val = result.get("transform_accuracy", torch.tensor(0.0))
        if isinstance(xf_val, torch.Tensor):
            xf_val = xf_val.item()
        last_xform = xf_val

        # Phase 1 (Pure Geometry): only MetricNet trains.
        # Zero grads on everything except geo params so embedding, FFN, W_o, output
        # head don't drift from geo_loss gradients leaking through h_normed.
        if isinstance(model, LiquidARCModel) or (
            hasattr(model, '_orig_mod') and isinstance(model._orig_mod, LiquidARCModel)
        ):
            raw_m = model._orig_mod if hasattr(model, '_orig_mod') else model
            if geo_phase == 1:
                geo_ids = {id(p) for p in raw_m.geo_parameters()}
                for p in raw_m.parameters():
                    if id(p) not in geo_ids and p.grad is not None:
                        p.grad.zero_()

        # Fix B: Zero tau/gate grads during freeze (prevent stale gradient accumulation)
        if isinstance(model, LiquidARCModel) or (
            hasattr(model, '_orig_mod') and isinstance(model._orig_mod, LiquidARCModel)
        ):
            raw_m = model._orig_mod if hasattr(model, '_orig_mod') else model
            if raw_m.dynamics.freeze_tau:
                if config.channel_gate_enabled:
                    freeze_mods = [raw_m.dynamics.gate_net_linear1,
                                   raw_m.dynamics.gate_net_linear2]
                else:
                    freeze_mods = [raw_m.dynamics.tau_net_linear1,
                                   raw_m.dynamics.tau_net_linear2]
                for mod in freeze_mods:
                    for p in mod.parameters():
                        if p.grad is not None:
                            p.grad.zero_()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # Reptile meta-step (after optimizer, before logging)
        if (config.reptile_enabled and reptile_arc_tasks
            and step >= config.reptile_start_step
            and step % config.reptile_every == 0):
            raw_m = model._orig_mod if hasattr(model, '_orig_mod') else model
            if isinstance(raw_m, LiquidARCModel):
                rep = reptile_step(raw_m, reptile_arc_tasks, config, device, step)
                if not rep.get("skipped"):
                    print(f"  [REPTILE] step={step}, K={rep['n_tasks_used']}, "
                          f"delta={rep['avg_delta_norm']:.6f}, "
                          f"meta_lr={rep['meta_lr']:.4f}, "
                          f"ttt_steps={rep['avg_ttt_steps']:.0f}, "
                          f"time={rep['time']:.1f}s")
                    writer.add_scalar("reptile/delta_norm", rep["avg_delta_norm"], step)
                    writer.add_scalar("reptile/meta_lr", rep["meta_lr"], step)
                    writer.add_scalar("reptile/time", rep["time"], step)

        # Logging
        if step % args.log_every == 0:
            dt = time.time() - t0
            avg_n = meta.get("lengths", torch.tensor(config.max_seq_len)).float().mean().item()
            tok_s = args.batch_size * avg_n * (step + 1) / max(dt, 1e-6)
            print_status(model, result, step, tok_s)

            writer.add_scalar("loss/total", result["loss"].item(), step)
            writer.add_scalar("loss/ce", result["ce_loss"].item(), step)
            xf_loss = result.get("xform_loss", torch.tensor(0.0))
            if isinstance(xf_loss, torch.Tensor):
                xf_loss = xf_loss.item()
            writer.add_scalar("loss/xform", xf_loss, step)
            if n_steps is not None:
                writer.add_scalar("ode/n_steps", n_steps, step)

            cell_acc = result.get("cell_accuracy", torch.tensor(0.0))
            if isinstance(cell_acc, torch.Tensor):
                cell_acc = cell_acc.item()
            writer.add_scalar("accuracy/cell_train", cell_acc, step)
            xf_acc = result.get("transform_accuracy", torch.tensor(0.0))
            if isinstance(xf_acc, torch.Tensor):
                xf_acc = xf_acc.item()
            writer.add_scalar("accuracy/xform_train", xf_acc, step)

            m = model._orig_mod if hasattr(model, '_orig_mod') else model
            if isinstance(m, LiquidARCModel):
                cv_val = result["metric_cv"]
                if isinstance(cv_val, torch.Tensor):
                    cv_val = cv_val.item()
                writer.add_scalar("metric/cv", cv_val, step)
                kappa_val = result["avg_kappa"]
                if isinstance(kappa_val, torch.Tensor):
                    kappa_val = kappa_val.item()
                writer.add_scalar("metric/kappa", kappa_val, step)
                tau_val = result.get("tau_avg", torch.tensor(0.0))
                if isinstance(tau_val, torch.Tensor):
                    tau_val = tau_val.item()
                writer.add_scalar("metric/tau", tau_val, step)
                tau_std = result.get("tau_std", torch.tensor(0.0))
                if isinstance(tau_std, torch.Tensor):
                    tau_std = tau_std.item()
                writer.add_scalar("metric/tau_std", tau_std, step)
                tau_min_v = result.get("tau_min", torch.tensor(0.0))
                if isinstance(tau_min_v, torch.Tensor):
                    tau_min_v = tau_min_v.item()
                writer.add_scalar("metric/tau_min", tau_min_v, step)
                tau_max_v = result.get("tau_max", torch.tensor(0.0))
                if isinstance(tau_max_v, torch.Tensor):
                    tau_max_v = tau_max_v.item()
                writer.add_scalar("metric/tau_max", tau_max_v, step)
                tau_var_l = result.get("tau_var_loss", torch.tensor(0.0))
                if isinstance(tau_var_l, torch.Tensor):
                    tau_var_l = tau_var_l.item()
                writer.add_scalar("loss/tau_var", tau_var_l, step)

                cv_floor_l = result.get("cv_floor_loss", torch.tensor(0.0))
                if isinstance(cv_floor_l, torch.Tensor):
                    cv_floor_l = cv_floor_l.item()
                writer.add_scalar("loss/cv_floor", cv_floor_l, step)

                # Working memory diagnostics
                gate_ds = result.get("gate_dim_std")
                if gate_ds is not None:
                    if isinstance(gate_ds, torch.Tensor):
                        gate_ds = gate_ds.item()
                    writer.add_scalar("metric/gate_dim_std", gate_ds, step)
                se_norm = result.get("step_embed_norm")
                if se_norm is not None:
                    if isinstance(se_norm, torch.Tensor):
                        se_norm = se_norm.item()
                    writer.add_scalar("metric/step_embed_norm", se_norm, step)

                # Geo loss logging
                if geo_phase > 0:
                    geo_l = result.get("geo_loss", torch.tensor(0.0))
                    if isinstance(geo_l, torch.Tensor):
                        geo_l = geo_l.item()
                    geo_kl = result.get("geo_mse", torch.tensor(0.0))
                    if isinstance(geo_kl, torch.Tensor):
                        geo_kl = geo_kl.item()
                    writer.add_scalar("loss/geo", geo_l, step)
                    writer.add_scalar("geo/mse", geo_kl, step)
                    writer.add_scalar("geo/lambda_ce", lambda_ce, step)
                    writer.add_scalar("geo/lambda_geo", lambda_geo, step)
                    writer.add_scalar("geo/phase", geo_phase, step)

            # --- Controller diagnostics ---
            diag = controller.diagnostics()
            writer.add_scalar("ctrl/ratio", diag["arc_ratio"], step)
            writer.add_scalar("ctrl/cv_smoothed", diag["smooth_cv"], step)
            writer.add_scalar("ctrl/cv_rate", diag["cv_rate"], step)
            # Log zone as integer: 0=warmup, 1=sub_critical, 2=critical, 3=crystallizing
            zone_int = {"warmup": 0, "sub_critical": 1, "critical": 2, "crystallizing": 3}
            writer.add_scalar("ctrl/zone", zone_int.get(diag["zone"], 0), step)
            print(f"  [ctrl] ratio={diag['arc_ratio']:.3f}, "
                  f"cv_smooth={diag['smooth_cv']:.3f}, "
                  f"cv_rate={diag['cv_rate']:.3f}, zone={diag['zone']}")

        # Eval
        if step > 0 and step % args.eval_every == 0:
            eval_acc, eval_xform, eval_copy_bl, eval_loss, eval_xf_loss = evaluate_quick(
                model, eval_task, device, n_batches=args.eval_batches,
                batch_size=args.batch_size)
            writer.add_scalar("accuracy/cell_eval", eval_acc, step)
            writer.add_scalar("accuracy/xform_eval", eval_xform, step)
            writer.add_scalar("accuracy/copy_baseline", eval_copy_bl, step)
            writer.add_scalar("loss/eval_ce", eval_loss, step)
            writer.add_scalar("loss/eval_xform", eval_xf_loss, step)
            print(f"  >> EVAL [step={step}] cell_acc={eval_acc:.4f}, "
                  f"xform_acc={eval_xform:.4f}, copy_bl={eval_copy_bl:.4f}, "
                  f"ce={eval_loss:.4f}, xf_loss={eval_xf_loss:.4f}")

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                save_checkpoint(model, optimizer, config, step,
                              os.path.join(out_dir, "checkpoints", "best.pt"),
                              extra={"eval_acc": eval_acc})
                print(f"  >> New best eval accuracy: {eval_acc:.4f}")

            compiled_model.train()

        # Checkpointing
        if step > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, config, step,
                          os.path.join(out_dir, "checkpoints", f"step_{step}.pt"))

            # Optional TTT eval at periodic checkpoints
            if config.ttt_enabled and isinstance(raw_model, LiquidARCModel):
                print(f"\n  >> TTT eval at step {step} (n_tasks=50)...")
                ttt_cell, ttt_xform = evaluate_ttt(
                    raw_model, args.data_dir, config, device, n_tasks=50)
                writer.add_scalar("accuracy/ttt_cell_eval", ttt_cell, step)
                writer.add_scalar("accuracy/ttt_xform_eval", ttt_xform, step)
                compiled_model.train()

    # Final checkpoint
    save_checkpoint(model, optimizer, config, args.max_steps,
                   os.path.join(out_dir, "checkpoints", "final.pt"))

    # Save full controller trajectory (also available incrementally in controller_log.csv)
    controller.close()
    zone_stats = controller.get_zone_stats()
    if zone_stats:
        print(f"\n  Controller zone stats:")
        print(f"    sub_critical: {zone_stats['sub_critical_pct']:.1f}%")
        print(f"    critical:     {zone_stats['critical_pct']:.1f}%")
        print(f"    crystallizing:{zone_stats['crystallizing_pct']:.1f}%")
        print(f"    avg_ratio:    {zone_stats['avg_ratio']:.3f}")

    writer.close()
    print(f"\n  Training complete. Best eval acc: {best_eval_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="LiquidARC Adaptive Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/arc")
    parser.add_argument("--output_dir", type=str, default="output_liquid_arc")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--geo_lr_mult", type=float, default=1.0,
                        help="LR multiplier for geometric params")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--n_color_perms", type=int, default=10)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * accum)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume training from")
    args = parser.parse_args()

    config = LiquidARCConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mem_frac = float(os.environ.get("CUDA_MEMORY_FRACTION", "0"))
    if mem_frac > 0 and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(mem_frac)
        print(f"CUDA memory fraction capped at {mem_frac:.0%}")

    print(f"Device: {device}")
    print(f"Model type: {config.model_type}")
    print(f"Data dir: {args.data_dir}")
    print(f"Total steps: {args.max_steps}")

    train(args, config, device)


if __name__ == "__main__":
    main()
