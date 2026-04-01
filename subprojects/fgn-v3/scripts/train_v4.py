"""FGN v4 — Geometry-First Architecture training script.

Two-phase training:
  Phase 0: Geometric pre-training (GeoRoute only, attention frozen, gate forced to 1.0)
  Phase 1: Joint training (both pathways, gate learned)

Monitoring: gate values, geometric contribution norms, metric CV, |kappa|, log_t_geo.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model_v4 import FGNv4Model
from fgn.flat_model import FlatTransformerModel
from fgn.tasks import get_task


def create_model(config: FGNConfig, device: torch.device):
    """Create model based on config."""
    if config.model_type == "flat":
        return FlatTransformerModel(config).to(device)
    else:
        return FGNv4Model(config).to(device)


def create_optimizer_phase0(model: FGNv4Model, lr: float, weight_decay: float):
    """Create optimizer for Phase 0: only geo + shared params get gradients."""
    # In Phase 0, attention is frozen, so we only optimize trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)


def create_optimizer_phase1(model: FGNv4Model, lr: float, weight_decay: float):
    """Create optimizer for Phase 1: all parameters, single LR group.

    The spec says no slow multiplier for v4 — all params at base LR.
    """
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def log_v4_metrics(model, result, writer, step, prefix=""):
    """Log v4-specific metrics."""
    p = f"{prefix}/" if prefix else ""
    writer.add_scalar(f"{p}loss/total", result["loss"].item(), step)
    writer.add_scalar(f"{p}loss/ce", result["ce_loss"].item(), step)
    writer.add_scalar(f"{p}loss/curvature", result["curv_loss"].item(), step)
    writer.add_scalar(f"{p}metric/avg_cv", result["metric_cv"].item(), step)
    writer.add_scalar(f"{p}metric/avg_kappa", result["avg_kappa"].item(), step)

    # Unwrap compiled model if needed
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    if hasattr(m, 'layers') and hasattr(m.layers[0], 'gate_geo_raw'):
        for i, layer in enumerate(m.layers):
            gate_val = torch.sigmoid(layer.gate_geo_raw).item()
            writer.add_scalar(f"{p}gate/layer_{i}", gate_val, step)
            writer.add_scalar(f"{p}log_t/layer_{i}",
                              layer.geo_route.log_t.item(), step)


def print_v4_status(model, result, step, tok_per_sec, phase):
    """Print compact v4 status line."""
    m = model._orig_mod if hasattr(model, '_orig_mod') else model

    gate_str = ""
    log_t_str = ""
    if hasattr(m, 'layers') and hasattr(m.layers[0], 'gate_geo_raw'):
        gates = [torch.sigmoid(l.gate_geo_raw).item() for l in m.layers]
        log_ts = [l.geo_route.log_t.item() for l in m.layers]
        gate_str = f" gates=[{','.join(f'{g:.2f}' for g in gates)}]"
        log_t_str = f" log_t=[{','.join(f'{t:.2f}' for t in log_ts)}]"

    print(f"  [{phase}] step={step}, loss={result['loss'].item():.4f}, "
          f"ce={result['ce_loss'].item():.4f}, "
          f"cv={result['metric_cv'].item():.4f}, "
          f"|k|={result['avg_kappa'].item():.4f}, "
          f"tok/s={tok_per_sec:.0f}"
          f"{gate_str}{log_t_str}")


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


def train(args, config, device):
    """Main training loop with Phase 0→1 transition."""
    tokenizer = _get_tokenizer()
    task_kwargs = json.loads(args.task_kwargs)

    print(f"\n{'='*70}")
    print(f"FGN v4 Training — {config.model_type}")
    print(f"{'='*70}")

    model = create_model(config, device)
    is_v4 = isinstance(model, FGNv4Model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Architecture: {'v4 (GeoRoute + StandardAttention)' if is_v4 else 'flat baseline'}")

    out_dir = args.output_dir
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, "logs"))

    total_steps = args.max_steps
    phase0_steps = config.phase0_steps if is_v4 else 0

    # --- Phase 0: Geometric Pre-Training ---
    if is_v4 and phase0_steps > 0:
        print(f"\n  --- Phase 0: Geometric Pre-Training ({phase0_steps} steps) ---")
        model.freeze_attention()
        model.force_gate(10.0)

        task = get_task(args.task, tokenizer, seq_len=config.max_seq_len,
                        **_phase0_task_kwargs(task_kwargs))

        optimizer = create_optimizer_phase0(model, args.lr, args.weight_decay)
        scheduler = create_scheduler(optimizer, args.warmup_steps, phase0_steps)

        if config.use_torch_compile and device.type == "cuda":
            compiled_model = torch.compile(model, mode="default")
        else:
            compiled_model = model

        compiled_model.train()
        t0 = time.time()

        for step in range(phase0_steps):
            input_ids, labels, _ = task.generate_batch(args.batch_size, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = compiled_model(input_ids, labels=labels)
                loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            if step % args.log_every == 0:
                dt = time.time() - t0
                tok_s = args.batch_size * config.max_seq_len * (step + 1) / max(dt, 1e-6)
                print_v4_status(model, result, step, tok_s, "P0")
                log_v4_metrics(model, result, writer, step, prefix="phase0")

        # Save Phase 0 checkpoint
        save_checkpoint(model, optimizer, config, phase0_steps,
                       os.path.join(out_dir, "checkpoints", "phase0_final.pt"),
                       extra={"phase": 0})

        # Check Phase 0 success: metric CV > 0.05
        cv = result["metric_cv"].item()
        if cv < 0.05:
            print(f"  WARNING: Phase 0 metric CV={cv:.4f} < 0.05. "
                  f"Geometric step may not have learned.")

        print(f"  Phase 0 complete. Final CE={result['ce_loss'].item():.4f}, CV={cv:.4f}")

    # --- Phase 1: Joint Training ---
    phase1_steps = total_steps - phase0_steps
    print(f"\n  --- Phase 1: Joint Training ({phase1_steps} steps) ---")

    if is_v4:
        model.unfreeze_attention()
        model.init_gate(config.gate_init)

    task = get_task(args.task, tokenizer, seq_len=config.max_seq_len, **task_kwargs)

    if is_v4:
        optimizer = create_optimizer_phase1(model, args.lr, args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args.warmup_steps, phase1_steps)

    if config.use_torch_compile and device.type == "cuda":
        compiled_model = torch.compile(model, mode="default")
    else:
        compiled_model = model

    compiled_model.train()
    t0 = time.time()
    last_loss = float("nan")

    for step_offset in range(phase1_steps):
        global_step = phase0_steps + step_offset
        input_ids, labels, _ = task.generate_batch(args.batch_size, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            result = compiled_model(input_ids, labels=labels)
            loss = result["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        last_loss = loss.item()

        if step_offset % args.log_every == 0:
            dt = time.time() - t0
            tok_s = args.batch_size * config.max_seq_len * (step_offset + 1) / max(dt, 1e-6)
            print_v4_status(model, result, global_step, tok_s, "P1")
            log_v4_metrics(model, result, writer, global_step, prefix="phase1")

            # Gate collapse warning (v4 only)
            if is_v4 and step_offset > 1000:
                m = model._orig_mod if hasattr(model, '_orig_mod') else model
                if hasattr(m.layers[0], 'gate_geo_raw'):
                    for i, layer in enumerate(m.layers):
                        g = torch.sigmoid(layer.gate_geo_raw).item()
                        if g < 0.05:
                            print(f"  WARNING: Gate collapsed in layer {i}: {g:.4f}")

        if step_offset > 0 and step_offset % args.save_every == 0:
            save_checkpoint(model, optimizer, config, global_step,
                          os.path.join(out_dir, "checkpoints", f"step_{global_step}.pt"))

    # Final save
    save_checkpoint(model, optimizer, config, total_steps,
                   os.path.join(out_dir, "checkpoints", "final.pt"),
                   extra={"phase": 1, "task": args.task})
    writer.close()

    print(f"\n  Training complete. Final loss: {last_loss:.4f}")

    # Print final gate summary
    if is_v4:
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        print(f"\n  Final gate values:")
        for i, layer in enumerate(m.layers):
            g = torch.sigmoid(layer.gate_geo_raw).item()
            t = layer.geo_route.log_t.item()
            print(f"    Layer {i}: gate={g:.4f}, log_t={t:.4f}")


def _phase0_task_kwargs(task_kwargs):
    """Generate easier task kwargs for Phase 0 (short chains)."""
    p0_kwargs = dict(task_kwargs)
    # Use short chains for Phase 0 geometric pre-training
    if "min_ops" in p0_kwargs:
        p0_kwargs["min_ops"] = min(p0_kwargs.get("min_ops", 10), 10)
    if "max_ops" in p0_kwargs:
        p0_kwargs["max_ops"] = min(p0_kwargs.get("max_ops", 20), 20)
    return p0_kwargs


def _get_tokenizer():
    """Load GPT-2 tokenizer."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


def main():
    parser = argparse.ArgumentParser(description="FGN v4 Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default="H",
                        help="Task name (default: H = affine group)")
    parser.add_argument("--output_dir", type=str, default="output_v4")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--task_kwargs", type=str, default="{}",
                        help="JSON dict of extra kwargs for task constructor")
    args = parser.parse_args()

    config = FGNConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mem_frac = float(os.environ.get("CUDA_MEMORY_FRACTION", "0"))
    if mem_frac > 0 and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(mem_frac)
        print(f"CUDA memory fraction capped at {mem_frac:.0%}")

    print(f"Device: {device}")
    print(f"Model: {config.model_type}, arch: {config.architecture_version}")
    print(f"Task: {args.task}, kwargs: {args.task_kwargs}")
    print(f"Phase 0 steps: {config.phase0_steps}")
    print(f"Total steps: {args.max_steps}")

    train(args, config, device)


if __name__ == "__main__":
    main()
