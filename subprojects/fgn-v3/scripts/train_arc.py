"""ARC-AGI training script for FluidNet and flat transformer.

Trains on ARC-AGI tasks using cell-as-token representation.
Supports FluidNetARC (geometric diffusion) and FlatTransformerARC (baseline).
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
from fgn.model_arc import FluidNetARC, FlatTransformerARC
from fgn.model_arc_sandwich import SandwichARC, create_arc_model
from fgn.tasks.arc import ARCTask


def create_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _has_geometry(model):
    """Check if model has geometric layers."""
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    return isinstance(m, (FluidNetARC, SandwichARC))


def print_status(model, result, step, tok_per_sec):
    """Print compact status line."""
    extra = ""
    if _has_geometry(model):
        t_local = result.get("avg_t_local", torch.tensor(0.0)).item()
        t_medium = result.get("avg_t_medium", torch.tensor(0.0)).item()
        t_global = result.get("avg_t_global", torch.tensor(0.0)).item()
        cv_val = result["metric_cv"]
        if isinstance(cv_val, torch.Tensor):
            cv_val = cv_val.item()
        kappa_val = result["avg_kappa"]
        if isinstance(kappa_val, torch.Tensor):
            kappa_val = kappa_val.item()
        extra = (f", cv={cv_val:.4f}, |k|={kappa_val:.4f}, "
                 f"t=[{t_local:.2f},{t_medium:.2f},{t_global:.2f}]")

        # Sandwich-specific: per-stage stats
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        if isinstance(m, SandwichARC):
            bot_cv = result.get("bot_metric_cv", torch.tensor(0.0))
            top_cv = result.get("top_metric_cv", torch.tensor(0.0))
            bot_k = result.get("bot_avg_kappa", torch.tensor(0.0))
            top_k = result.get("top_avg_kappa", torch.tensor(0.0))
            if isinstance(bot_cv, torch.Tensor): bot_cv = bot_cv.item()
            if isinstance(top_cv, torch.Tensor): top_cv = top_cv.item()
            if isinstance(bot_k, torch.Tensor): bot_k = bot_k.item()
            if isinstance(top_k, torch.Tensor): top_k = top_k.item()
            extra += f", bot_cv={bot_cv:.3f}, top_cv={top_cv:.3f}"
            extra += f", bot_|k|={bot_k:.2f}, top_|k|={top_k:.2f}"

    cell_acc = result.get("cell_accuracy", torch.tensor(0.0))
    if isinstance(cell_acc, torch.Tensor):
        cell_acc = cell_acc.item()
    xform_acc = result.get("transform_accuracy", torch.tensor(0.0))
    if isinstance(xform_acc, torch.Tensor):
        xform_acc = xform_acc.item()

    print(f"  [step={step}] loss={result['loss'].item():.4f}, "
          f"ce={result['ce_loss'].item():.4f}, "
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


def train(args, config, device):
    """Training loop."""
    print(f"\n{'='*70}")
    print(f"ARC-AGI Training — {config.architecture_version}")
    print(f"{'='*70}")

    # Create model
    model = create_arc_model(config, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    if isinstance(model, SandwichARC):
        print(f"  Architecture: Sandwich (geo→attn→geo)")
        print(f"  Bottom geo: {config.sandwich_bottom_geo_layers}, "
              f"Middle attn: {config.sandwich_middle_attn_layers}, "
              f"Top geo: {config.sandwich_top_geo_layers}")
        print(f"  Scales: {config.n_scales}, d_metric: {config.d_metric}")
    elif isinstance(model, FluidNetARC):
        print(f"  Architecture: FluidNet (geometric diffusion)")
        print(f"  Scales: {config.n_scales}, d_metric: {config.d_metric}")
        print(f"  Structural energy lambda: {config.structural_energy_lambda}")
        print(f"  Diffusion iters: {config.n_diffusion_iters}")
    else:
        print(f"  Architecture: flat baseline")

    # Output directory
    out_dir = args.output_dir
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, "logs"))

    # Data
    task = ARCTask(
        seq_len=config.max_seq_len,
        data_dir=args.data_dir,
        split="train",
        augment=True,
        n_color_perms=args.n_color_perms,
    )

    # Eval task (no augmentation for consistent evaluation)
    eval_task = ARCTask(
        seq_len=config.max_seq_len,
        data_dir=args.data_dir,
        split="eval",
        augment=False,
    )

    # Optimizer with param groups
    has_geo = _has_geometry(model)
    if has_geo and config.metric_lr_mult != 1.0:
        geo_params = model.geo_parameters()
        other_params = model.other_parameters()
        param_groups = [
            {"params": other_params, "lr": args.lr},
            {"params": geo_params, "lr": args.lr * config.metric_lr_mult},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        print(f"  Geo LR: {args.lr * config.metric_lr_mult:.2e} "
              f"({config.metric_lr_mult}x)")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)

    scheduler = create_scheduler(optimizer, args.warmup_steps, args.max_steps)

    # Torch compile — use dynamic=True for variable-length ARC sequences
    if config.use_torch_compile and device.type == "cuda":
        compiled_model = torch.compile(model, mode="default", dynamic=True)
    else:
        compiled_model = model

    compiled_model.train()
    t0 = time.time()
    best_eval_acc = 0.0
    if config.n_refine_iters > 1:
        print(f"  Refinement: {config.n_refine_iters} latent-space iterations (inside model)")

    for step in range(args.max_steps):
        _, _, meta = task.generate_batch(args.batch_size, device=device)

        optimizer.zero_grad()

        # Single forward pass — refinement happens inside the model in latent space
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
                grid_ids=meta["grid_ids"],
                lengths=meta["lengths"],
                target_input_colors=meta["target_input_colors"],
            )

        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # Logging
        if step % args.log_every == 0:
            dt = time.time() - t0
            avg_n = meta["lengths"].float().mean().item() if "lengths" in meta else config.max_seq_len
            tok_s = args.batch_size * avg_n * (step + 1) / max(dt, 1e-6)
            print_status(model, result, step, tok_s)

            writer.add_scalar("loss/total", result["loss"].item(), step)
            writer.add_scalar("loss/ce", result["ce_loss"].item(), step)

            cell_acc = result.get("cell_accuracy", torch.tensor(0.0))
            if isinstance(cell_acc, torch.Tensor):
                cell_acc = cell_acc.item()
            writer.add_scalar("accuracy/cell_train", cell_acc, step)

            if has_geo:
                cv_val = result["metric_cv"]
                if isinstance(cv_val, torch.Tensor):
                    cv_val = cv_val.item()
                writer.add_scalar("metric/cv", cv_val, step)
                writer.add_scalar("metric/kappa",
                                  result["avg_kappa"].item(), step)
                writer.add_scalar("loss/structural_energy",
                                  result["structural_energy"].item(), step)
                for key in ("avg_t_local", "avg_t_medium", "avg_t_global"):
                    if key in result:
                        writer.add_scalar(f"timescale/{key.split('_')[-1]}",
                                          result[key].item(), step)

        # Eval
        if step > 0 and step % args.eval_every == 0:
            eval_acc, eval_loss = evaluate_quick(
                model, eval_task, device, n_batches=args.eval_batches,
                batch_size=args.batch_size)
            writer.add_scalar("accuracy/cell_eval", eval_acc, step)
            writer.add_scalar("loss/eval_ce", eval_loss, step)
            print(f"  >> EVAL [step={step}] cell_acc={eval_acc:.4f}, "
                  f"ce_loss={eval_loss:.4f}")

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

    # Final checkpoint
    save_checkpoint(model, optimizer, config, args.max_steps,
                   os.path.join(out_dir, "checkpoints", "final.pt"),
                   extra={"architecture_version": config.architecture_version})
    writer.close()

    print(f"\n  Training complete. Best eval acc: {best_eval_acc:.4f}")


def evaluate_quick(model, eval_task, device, n_batches=10, batch_size=8):
    """Quick eval: cell accuracy + CE loss. Refinement is inside the model."""
    model.eval()
    total_correct = 0
    total_cells = 0
    total_ce = 0.0
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
                    grid_ids=meta["grid_ids"],
                    lengths=meta["lengths"],
                    target_input_colors=meta["target_input_colors"],
                )

            cell_acc = result.get("cell_accuracy", torch.tensor(0.0))
            if isinstance(cell_acc, torch.Tensor):
                cell_acc = cell_acc.item()
            target_labels = meta["target_labels"]
            n_tgt = (target_labels != -100).sum().item()
            total_correct += int(cell_acc * n_tgt)
            total_cells += n_tgt

            total_ce += result["ce_loss"].item()
            n_valid_batches += 1

    cell_acc = total_correct / max(total_cells, 1)
    avg_ce = total_ce / max(n_valid_batches, 1)
    return cell_acc, avg_ce


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI Training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, default="data/arc",
                        help="Path to ARC-AGI data directory")
    parser.add_argument("--output_dir", type=str, default="output_arc",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--n_color_perms", type=int, default=10,
                        help="Number of color permutations per task per epoch")
    args = parser.parse_args()

    config = FGNConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mem_frac = float(os.environ.get("CUDA_MEMORY_FRACTION", "0"))
    if mem_frac > 0 and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(mem_frac)
        print(f"CUDA memory fraction capped at {mem_frac:.0%}")

    print(f"Device: {device}")
    print(f"Model: {config.model_type}, arch: {config.architecture_version}")
    print(f"Data dir: {args.data_dir}")
    print(f"Total steps: {args.max_steps}")

    train(args, config, device)


if __name__ == "__main__":
    main()
