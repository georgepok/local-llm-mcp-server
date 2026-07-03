"""FGN v3 Phase 1b — Multi-task training script.

Trains FGN or flat transformer baseline on 4 synthetic tasks:
  A: Temporal reasoning
  B: Pattern search
  C: Interleaved instruction-data
  D: Multi-hop retrieval

Stages:
  1: Single-task baselines (train on each task independently)
  2: Mixed training (round-robin across all selected tasks)
  3: Task-switching speed (fine-tune from Stage 2 checkpoint on single task)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch._inductor.config as _inductor_config

# Enable TensorFloat32 for matmul on Ampere+ GPUs (GB10 supports it).
# ~2-4× speedup with negligible accuracy loss. Suppresses the inductor warning.
torch.set_float32_matmul_precision("high")

# Kernel fusion control for LiquidARC at d≥768.
# Inspecting the failing kernel revealed num_load=19 (19 input tensors fused
# into one kernel: LN params, FiLM gammas/betas, tau paths, step embeddings,
# LN backward intermediates). R0_BLOCK=1024 × 19 tensors × bf16 easily
# exceeds 101KB SRAM. Limiting max_fusion_size forces Inductor to emit more,
# smaller kernels rather than one mega-kernel.
_inductor_config.max_fusion_size = 4
from torch.utils.tensorboard import SummaryWriter

# Add parent to path for fgn package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model import FGNModel
from fgn.flat_model import FlatTransformerModel
try:
    from fgn.liquid_model import LiquidSequenceModel
except Exception:
    LiquidSequenceModel = None
from fgn.tasks import get_task


def create_model(config: FGNConfig, device: torch.device):
    """Create model based on config.model_type."""
    if config.model_type == "flat":
        model = FlatTransformerModel(config).to(device)
    elif config.model_type == "liquid":
        assert LiquidSequenceModel is not None, (
            "liquid_model.py requires liquid_arc package on PYTHONPATH")
        model = LiquidSequenceModel(config).to(device)
    else:
        model = FGNModel(config).to(device)
    return model


def load_checkpoint(model, path: str, device: torch.device, max_seq_len: int):
    """Load checkpoint with pos_embed truncation for shorter seq_len."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

    # Truncate pos_embed if checkpoint has longer sequence length
    pe_key = "pos_embed.weight"
    if pe_key in state and state[pe_key].shape[0] > max_seq_len:
        print(f"  Truncating pos_embed: {state[pe_key].shape[0]} → {max_seq_len}")
        state[pe_key] = state[pe_key][:max_seq_len]

    model.load_state_dict(state)
    print(f"  Loaded checkpoint from {path}")


def create_optimizer(model, lr: float, weight_decay: float,
                     metric_lr_mult: float = 0.1):
    """Create AdamW with separate parameter groups."""
    slow_params = model.slow_parameters()
    fast_params = model.fast_parameters()

    if slow_params:
        param_groups = [
            {"params": fast_params, "lr": lr, "weight_decay": weight_decay},
            {"params": slow_params, "lr": lr * metric_lr_mult, "weight_decay": weight_decay},
        ]
    else:
        # Flat model: single group
        param_groups = [
            {"params": fast_params, "lr": lr, "weight_decay": weight_decay},
        ]
    return torch.optim.AdamW(param_groups)


def create_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def log_metrics(result: dict, writer: SummaryWriter, step: int,
                prefix: str = ""):
    """Log training metrics to tensorboard."""
    p = f"{prefix}/" if prefix else ""
    writer.add_scalar(f"{p}loss/total", result["loss"].item(), step)
    writer.add_scalar(f"{p}loss/ce", result["ce_loss"].item(), step)
    writer.add_scalar(f"{p}loss/curvature", result["curv_loss"].item(), step)
    writer.add_scalar(f"{p}loss/scale_entropy", result["scale_loss"].item(), step)
    writer.add_scalar(f"{p}metric/avg_cv", result["metric_cv"].item(), step)
    writer.add_scalar(f"{p}metric/avg_kappa", result["avg_kappa"].item(), step)


def save_checkpoint(model, optimizer, config, step, path, extra=None):
    """Save checkpoint."""
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def train_stage1(args, config, device):
    """Stage 1: Single-task baselines."""
    tokenizer = _get_tokenizer()
    task_names = args.tasks.split(",")

    for task_name in task_names:
        print(f"\n{'='*60}")
        print(f"STAGE 1 — Task {task_name} ({config.model_type})")
        print(f"{'='*60}")

        task_kwargs = json.loads(args.task_kwargs)
        task = get_task(task_name, tokenizer, seq_len=config.max_seq_len, **task_kwargs)

        # Fresh model per task
        model = create_model(config, device)
        if args.resume and config.model_type == "fgn":
            load_checkpoint(model, args.resume, device, config.max_seq_len)

        if config.use_torch_compile and device.type == "cuda" and not getattr(model, "_skip_top_compile", False):
            model = torch.compile(model, mode="default")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        optimizer = create_optimizer(model, args.lr, args.weight_decay,
                                     config.metric_lr_mult)
        scheduler = create_scheduler(optimizer, args.warmup_steps, args.max_steps)

        out_dir = os.path.join(args.output_dir, f"stage1_task{task_name}")
        os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
        writer = SummaryWriter(os.path.join(out_dir, "logs"))

        model.train()
        t0 = time.time()
        last_loss = float("nan")

        for step in range(args.max_steps):
            input_ids, labels, _ = task.generate_batch(args.batch_size, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = model(input_ids, labels=labels)
                loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            last_loss = loss.item()

            if step % args.log_every == 0:
                dt = time.time() - t0
                tokens_per_sec = args.batch_size * config.max_seq_len * (step + 1) / dt
                msg = (f"  task={task_name} step={step}, loss={last_loss:.4f}, "
                       f"ce={result['ce_loss'].item():.4f}, "
                       f"cv={result['metric_cv'].item():.4f}, "
                       f"tok/s={tokens_per_sec:.0f}")
                # Tier 3 halting stats if present
                if 'steps_mean' in result:
                    gate_str = ""
                    if 'ponder_gate' in result:
                        gate_str = f" gate={result['ponder_gate'].item():.2f}"
                    msg += (f" | steps[mean={result['steps_mean'].item():.1f} "
                            f"min={result['steps_min'].item():.1f} "
                            f"max={result['steps_max'].item():.1f}] "
                            f"halt_frac={result['halt_frac'].item():.2f} "
                            f"ponder={result['ponder_cost'].item():.3f}"
                            f"{gate_str}")
                print(msg)
                log_metrics(result, writer, step, prefix=f"task_{task_name}")

            if step > 0 and step % args.save_every == 0:
                save_checkpoint(model, optimizer, config, step,
                              os.path.join(out_dir, "checkpoints", f"step_{step}.pt"))

        # Final save
        save_checkpoint(model, optimizer, config, args.max_steps,
                       os.path.join(out_dir, "checkpoints", "final.pt"),
                       extra={"task": task_name, "stage": 1})
        writer.close()
        print(f"  Task {task_name} complete. Final loss: {last_loss:.4f}")


def train_stage2(args, config, device):
    """Stage 2: Mixed training (round-robin)."""
    print(f"\n{'='*60}")
    print(f"STAGE 2 — Mixed training ({config.model_type})")
    print(f"{'='*60}")

    tokenizer = _get_tokenizer()
    task_names = args.tasks.split(",")
    task_kwargs = json.loads(args.task_kwargs)
    # Support per-task kwargs: if the dict has keys matching task names,
    # use those per-task; otherwise apply the dict uniformly to all tasks.
    def _resolve_kwargs(name: str) -> dict:
        if name in task_kwargs and isinstance(task_kwargs[name], dict):
            return task_kwargs[name]
        return task_kwargs if not any(
            isinstance(v, dict) for v in task_kwargs.values()) else {}
    tasks = {name: get_task(name, tokenizer, seq_len=config.max_seq_len,
                             **_resolve_kwargs(name))
              for name in task_names}

    model = create_model(config, device)
    if args.resume and config.model_type == "fgn":
        load_checkpoint(model, args.resume, device, config.max_seq_len)

    if config.use_torch_compile and device.type == "cuda" and not getattr(model, "_skip_top_compile", False):
        model = torch.compile(model, mode="default")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Tasks: {task_names}")

    optimizer = create_optimizer(model, args.lr, args.weight_decay,
                                 config.metric_lr_mult)
    scheduler = create_scheduler(optimizer, args.warmup_steps, args.max_steps)

    out_dir = os.path.join(args.output_dir, "stage2_mixed")
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, "logs"))

    # Per-task loss tracking (running averages)
    task_losses = {name: 0.0 for name in task_names}
    task_counts = {name: 0 for name in task_names}

    model.train()
    t0 = time.time()

    for step in range(args.max_steps):
        # Round-robin task selection
        task_name = task_names[step % len(task_names)]
        task = tasks[task_name]

        input_ids, labels, _ = task.generate_batch(args.batch_size, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            result = model(input_ids, labels=labels)
            loss = result["loss"]

        # LiquidSequenceModel: compute aux losses OUTSIDE the compiled
        # forward (mirrors LiquidARC's canonical training pattern —
        # fusing aux ops into the forward's backward kernel exceeds
        # Triton shared-memory limits at coupled routing / rank>0).
        if hasattr(model, "compute_aux_losses"):
            aux = model.compute_aux_losses(input_ids)
            loss = loss + 0.01 * aux["crit_loss"] + 0.05 * aux["tq_loss"]
            # Surface diagnostics onto result dict for logging
            result["metric_cv"] = aux["cv"]
            result["avg_kappa"] = aux["d_over_4t"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # Track per-task losses
        task_losses[task_name] += loss.item()
        task_counts[task_name] += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            tokens_per_sec = args.batch_size * config.max_seq_len * (step + 1) / dt

            # Print per-task averages
            parts = []
            for tn in task_names:
                if task_counts[tn] > 0:
                    avg = task_losses[tn] / task_counts[tn]
                    parts.append(f"{tn}={avg:.4f}")
            task_str = ", ".join(parts)

            print(f"  step={step}, current={task_name}, loss={loss.item():.4f}, "
                  f"ce={result['ce_loss'].item():.4f}, "
                  f"cv={result['metric_cv'].item():.4f}, "
                  f"tok/s={tokens_per_sec:.0f} [{task_str}]")

            # Log current step metrics
            log_metrics(result, writer, step, prefix=f"task_{task_name}")
            writer.add_scalar("mixed/current_loss", loss.item(), step)

            # Log per-task running averages
            for tn in task_names:
                if task_counts[tn] > 0:
                    writer.add_scalar(f"mixed/avg_loss_{tn}",
                                    task_losses[tn] / task_counts[tn], step)

            # Reset running averages periodically
            if step > 0 and step % (args.log_every * 10) == 0:
                task_losses = {name: 0.0 for name in task_names}
                task_counts = {name: 0 for name in task_names}

        if step > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, config, step,
                          os.path.join(out_dir, "checkpoints", f"step_{step}.pt"))

    # Final save
    save_checkpoint(model, optimizer, config, args.max_steps,
                   os.path.join(out_dir, "checkpoints", "final.pt"),
                   extra={"stage": 2, "tasks": task_names})
    writer.close()
    print(f"  Mixed training complete.")


def train_stage3(args, config, device):
    """Stage 3: Task-switching speed test."""
    tokenizer = _get_tokenizer()
    task_names = args.tasks.split(",")

    # Load Stage 2 checkpoint
    stage2_ckpt = os.path.join(args.output_dir, "stage2_mixed", "checkpoints", "final.pt")
    if args.resume:
        stage2_ckpt = args.resume
    if not os.path.exists(stage2_ckpt):
        print(f"ERROR: Stage 2 checkpoint not found at {stage2_ckpt}")
        print("  Run stage 2 first, or specify --resume with the checkpoint path.")
        sys.exit(1)

    for task_name in task_names:
        print(f"\n{'='*60}")
        print(f"STAGE 3 — Task-switching to {task_name} ({config.model_type})")
        print(f"{'='*60}")

        task_kwargs = json.loads(args.task_kwargs)
        task = get_task(task_name, tokenizer, seq_len=config.max_seq_len, **task_kwargs)

        # Load from Stage 2 checkpoint
        model = create_model(config, device)
        load_checkpoint(model, stage2_ckpt, device, config.max_seq_len)

        if config.use_torch_compile and device.type == "cuda" and not getattr(model, "_skip_top_compile", False):
            model = torch.compile(model, mode="default")

        # Smaller LR for fine-tuning
        ft_lr = args.lr * 0.1
        optimizer = create_optimizer(model, ft_lr, args.weight_decay,
                                     config.metric_lr_mult)
        scheduler = create_scheduler(optimizer, 50, args.max_steps)

        out_dir = os.path.join(args.output_dir, f"stage3_switch_{task_name}")
        os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
        writer = SummaryWriter(os.path.join(out_dir, "logs"))

        model.train()
        t0 = time.time()
        last_loss = float("nan")

        for step in range(args.max_steps):
            input_ids, labels, _ = task.generate_batch(args.batch_size, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = model(input_ids, labels=labels)
                loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            last_loss = loss.item()

            if step % args.log_every == 0:
                dt = time.time() - t0
                tokens_per_sec = args.batch_size * config.max_seq_len * (step + 1) / dt
                print(f"  task={task_name} step={step}, loss={last_loss:.4f}, "
                      f"ce={result['ce_loss'].item():.4f}, "
                      f"cv={result['metric_cv'].item():.4f}, "
                      f"tok/s={tokens_per_sec:.0f}")
                log_metrics(result, writer, step, prefix=f"switch_{task_name}")

            if step > 0 and step % args.save_every == 0:
                save_checkpoint(model, optimizer, config, step,
                              os.path.join(out_dir, "checkpoints", f"step_{step}.pt"))

        save_checkpoint(model, optimizer, config, args.max_steps,
                       os.path.join(out_dir, "checkpoints", "final.pt"),
                       extra={"task": task_name, "stage": 3})
        writer.close()
        print(f"  Task-switch to {task_name} complete. Final loss: {last_loss:.4f}")


def _get_tokenizer():
    """Load GPT-2 tokenizer."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="FGN v3 Multi-task Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--model_type", type=str, default=None,
                        help="Override config.model_type (fgn or flat)")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3],
                        help="Training stage: 1=single-task, 2=mixed, 3=task-switch")
    parser.add_argument("--tasks", type=str, default="A,B,C,D",
                        help="Comma-separated task names (A,B,C,D)")
    parser.add_argument("--output_dir", type=str, default="output_multitask",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (model weights only)")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--task_kwargs", type=str, default="{}",
                        help="JSON dict of extra kwargs for task constructor")
    args = parser.parse_args()

    # Config
    config = FGNConfig.from_yaml(args.config)
    if args.model_type:
        config.model_type = args.model_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # On unified memory systems (e.g. DGX Spark), cap GPU allocation to leave
    # headroom for the OS and kernel.  Controlled via env var so the default
    # (no cap) is preserved on discrete-GPU machines.
    mem_frac = float(os.environ.get("CUDA_MEMORY_FRACTION", "0"))
    if mem_frac > 0 and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(mem_frac)
        print(f"CUDA memory fraction capped at {mem_frac:.0%}")

    print(f"Device: {device}")
    print(f"Model type: {config.model_type}")
    print(f"Stage: {args.stage}")
    print(f"Tasks: {args.tasks}")

    if args.stage == 1:
        train_stage1(args, config, device)
    elif args.stage == 2:
        train_stage2(args, config, device)
    elif args.stage == 3:
        train_stage3(args, config, device)


if __name__ == "__main__":
    main()
