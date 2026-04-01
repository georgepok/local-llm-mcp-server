"""FGN v3 training script — single-GPU training on DGX Spark.

Phase 1a.1 additions:
  - Curriculum warm-up: synthetic copy-pattern → WikiText transition
  - LR schedule inversion: metric uses high LR during curriculum, then 0.1x
  - Optional Q/K/V freezing during curriculum phase
  - Curvature reward mode: -mu*mean(|kappa|) instead of Var(kappa) penalty
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent to path for fgn package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model import FGNModel


def get_dataset(tokenizer, max_seq_len: int, split: str = "train"):
    """Load WikiText-103 and tokenize."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    # Concatenate all text and chunk into sequences
    all_ids = []
    for example in ds:
        text = example["text"].strip()
        if text:
            ids = tokenizer.encode(text)
            all_ids.extend(ids)

    # Chunk into fixed-length sequences
    n_chunks = len(all_ids) // max_seq_len
    all_ids = all_ids[:n_chunks * max_seq_len]
    chunks = torch.tensor(all_ids).view(n_chunks, max_seq_len)
    return chunks


def generate_curriculum_batch(batch_size: int, seq_len: int, vocab_size: int,
                               device: torch.device):
    """Generate synthetic copy-pattern batch for curriculum warm-up.

    Format: [random_tokens..., SEP, random_tokens_copy...]
    This task REQUIRES geometric structure at the SEP boundary,
    giving the metric network direct gradient signal to develop curvature.
    """
    SEP = vocab_size - 1
    half = seq_len // 2

    # Random content (excluding SEP token)
    content = torch.randint(0, vocab_size - 1, (batch_size, half), device=device)

    # Build: [content, SEP, content[:-1], pad]
    sep = torch.full((batch_size, 1), SEP, device=device, dtype=torch.long)

    # For sequences longer than 2*half+1, pad with random tokens at start
    prefix_len = seq_len - (2 * half + 1)
    if prefix_len > 0:
        prefix = torch.randint(0, vocab_size - 1, (batch_size, prefix_len), device=device)
        input_ids = torch.cat([prefix, content, sep, content[:, :-1]], dim=1)[:, :seq_len]
        # Labels: ignore prefix + original content + SEP, predict copy region
        labels = torch.cat([
            torch.full((batch_size, prefix_len + half + 1), -100, device=device),
            content,
        ], dim=1)[:, :seq_len]
    else:
        input_ids = torch.cat([content, sep, content[:, :-1]], dim=1)[:, :seq_len]
        labels = torch.cat([
            torch.full((batch_size, half + 1), -100, device=device),
            content,
        ], dim=1)[:, :seq_len]

    return input_ids, labels


def create_optimizer(model: FGNModel, lr: float, weight_decay: float,
                     metric_lr_mult: float = 0.1):
    """Create AdamW with separate parameter groups for slow/fast params.

    Args:
        model: FGN model
        lr: base learning rate
        weight_decay: weight decay
        metric_lr_mult: multiplier for metric/diffusion params (0.1 = 10x slower)
    """
    slow_params = model.slow_parameters()
    fast_params = model.fast_parameters()

    param_groups = [
        {"params": fast_params, "lr": lr, "weight_decay": weight_decay},
        {"params": slow_params, "lr": lr * metric_lr_mult, "weight_decay": weight_decay},
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


def set_qkv_frozen(model, frozen: bool):
    """Freeze or unfreeze Q/K/V projection weights across all layers."""
    for layer in model.layers:
        for proj in [layer.attention.W_q, layer.attention.W_k, layer.attention.W_v]:
            for p in proj.parameters():
                p.requires_grad = not frozen
    state = "frozen" if frozen else "unfrozen"
    print(f"Q/K/V projections: {state}")


def log_gradient_norms(model: FGNModel, writer: SummaryWriter, step: int):
    """Log per-module gradient norms to tensorboard."""
    for name, p in model.named_parameters():
        if p.grad is not None:
            writer.add_scalar(f"grad_norm/{name}", p.grad.norm().item(), step)


def log_scale_weights(model: FGNModel, writer: SummaryWriter, step: int):
    """Log scale weight statistics."""
    for i, layer in enumerate(model.layers):
        log_t = layer.attention.log_t
        t = log_t.exp()
        for s in range(len(t)):
            writer.add_scalar(f"diffusion_time/layer{i}_scale{s}", t[s].item(), step)


def log_curvature_stats(model: FGNModel, writer: SummaryWriter, step: int):
    """Log curvature statistics per layer."""
    for i, layer in enumerate(model.layers):
        if layer.last_curvature is not None:
            k = layer.last_curvature
            writer.add_scalar(f"curvature/layer{i}_mean", k.mean().item(), step)
            writer.add_scalar(f"curvature/layer{i}_std", k.std().item(), step)
    lengths = model.curv_reg.correlation_lengths()
    for i, l in enumerate(lengths):
        writer.add_scalar(f"correlation_length/layer{i}", l, step)


def log_metric_stats(model: FGNModel, writer: SummaryWriter, step: int):
    """Log metric statistics per layer."""
    for i, layer in enumerate(model.layers):
        if layer.last_metric is not None:
            g = layer.last_metric
            writer.add_scalar(f"metric/layer{i}_mean", g.mean().item(), step)
            writer.add_scalar(f"metric/layer{i}_std", g.std().item(), step)
            writer.add_scalar(f"metric/layer{i}_cv", (g.std() / g.mean()).item(), step)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Config
    config = FGNConfig.from_yaml(args.config)
    print(f"Config: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"n_heads={config.n_heads}, n_scales={config.n_scales}")
    print(f"Curriculum: {config.curriculum_steps} steps, "
          f"metric_lr_mult={config.curriculum_metric_lr_mult}, "
          f"freeze_qkv={config.curriculum_freeze_qkv}")
    print(f"Curvature reward mu: {config.curvature_reward_mu}")

    # Model
    model = FGNModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Resume from checkpoint (loads model weights only, not optimizer)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state)
        print(f"Resumed model weights from {args.resume}")

    if config.use_torch_compile and device.type == "cuda":
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="default")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    assert tokenizer.vocab_size <= config.vocab_size

    # Dataset (loaded eagerly — needed after curriculum phase)
    print("Loading dataset...")
    train_data = get_dataset(tokenizer, config.max_seq_len, split="train")
    print(f"Training sequences: {len(train_data):,}")

    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    # Total steps = curriculum + main training
    curriculum_steps = config.curriculum_steps
    main_steps = args.max_steps if args.max_steps > 0 else len(loader) * args.epochs
    total_steps = curriculum_steps + main_steps

    # ===== PHASE 1: Curriculum Warm-Up =====
    if curriculum_steps > 0:
        print(f"\n{'='*60}")
        print(f"CURRICULUM PHASE: {curriculum_steps} steps on synthetic copy-pattern")
        print(f"{'='*60}")

        # During curriculum: metric gets HIGH LR to develop geometry first
        curric_optimizer = create_optimizer(
            model, args.lr, args.weight_decay,
            metric_lr_mult=config.curriculum_metric_lr_mult,
        )
        curric_scheduler = create_scheduler(curric_optimizer, warmup_steps=200,
                                             total_steps=curriculum_steps)

        # Optionally freeze Q/K/V to force metric-only learning
        if config.curriculum_freeze_qkv:
            set_qkv_frozen(model, frozen=True)

        model.train()
        t0 = time.time()

        for step in range(curriculum_steps):
            input_ids, labels = generate_curriculum_batch(
                args.batch_size, config.max_seq_len, config.vocab_size, device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = model(input_ids, labels=labels)
                loss = result["loss"]

            curric_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            curric_optimizer.step()
            curric_scheduler.step()

            if step % args.log_every == 0:
                avg_cv = result["metric_cv"].item()
                avg_kappa = result["avg_kappa"].item()

                print(f"  curric step={step}, loss={loss.item():.4f}, "
                      f"ce={result['ce_loss'].item():.4f}, "
                      f"curv_reg={result['curv_loss'].item():.6f}, "
                      f"metric_cv={avg_cv:.4f}, |kappa|={avg_kappa:.6f}")

        # Unfreeze Q/K/V for main training
        if config.curriculum_freeze_qkv:
            set_qkv_frozen(model, frozen=False)

        # Save post-curriculum checkpoint
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({
            "step": curriculum_steps,
            "model": model.state_dict(),
            "config": config,
            "phase": "post_curriculum",
        }, os.path.join(ckpt_dir, "post_curriculum.pt"))

        print(f"\nCurriculum complete. Metric CV: {avg_cv:.4f}, |kappa|: {avg_kappa:.6f}")

    # ===== PHASE 2: Main WikiText Training =====
    print(f"\n{'='*60}")
    print(f"MAIN TRAINING: {main_steps} steps on WikiText-103")
    print(f"{'='*60}")

    # Main training: metric uses configured LR multiplier (default 0.1x for stability)
    optimizer = create_optimizer(model, args.lr, args.weight_decay,
                                 metric_lr_mult=config.metric_lr_mult)
    scheduler = create_scheduler(optimizer, args.warmup_steps, main_steps)

    # Logging
    log_dir = os.path.join(args.output_dir, "logs")
    writer = SummaryWriter(log_dir)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # Training loop
    model.train()
    step = 0
    best_loss = float("inf")
    t0 = time.time()
    loss = None

    for _epoch in range(args.epochs):
        for batch in loader:
            if args.max_steps > 0 and step >= args.max_steps:
                break

            input_ids = batch.to(device)
            labels = input_ids.clone()
            # Shift: predict next token
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Ignore last position

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                result = model(input_ids, labels=labels)
                loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            # Logging
            global_step = curriculum_steps + step
            if step % args.log_every == 0:
                dt = time.time() - t0
                tokens_per_sec = args.batch_size * config.max_seq_len * (step + 1) / dt
                avg_cv = result["metric_cv"].item()

                print(f"step={step}, loss={loss.item():.4f}, "
                      f"ce={result['ce_loss'].item():.4f}, "
                      f"curv={result['curv_loss'].item():.6f}, "
                      f"scale={result['scale_loss'].item():.4f}, "
                      f"metric_cv={avg_cv:.4f}, "
                      f"tok/s={tokens_per_sec:.0f}")

                writer.add_scalar("loss/total", loss.item(), global_step)
                writer.add_scalar("loss/ce", result["ce_loss"].item(), global_step)
                writer.add_scalar("loss/curvature", result["curv_loss"].item(), global_step)
                writer.add_scalar("loss/scale_entropy", result["scale_loss"].item(), global_step)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("metric/avg_cv", avg_cv, global_step)

                log_scale_weights(model, writer, global_step)
                log_curvature_stats(model, writer, global_step)
                log_metric_stats(model, writer, global_step)

            if step % args.grad_log_every == 0:
                log_gradient_norms(model, writer, global_step)

            # Checkpointing
            if step > 0 and step % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, "checkpoints", f"step_{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                }, ckpt_path)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_path = os.path.join(args.output_dir, "checkpoints", "best.pt")
                    torch.save({
                        "step": global_step,
                        "model": model.state_dict(),
                        "config": config,
                    }, best_path)
                    print(f"  New best: {best_loss:.4f}")

            step += 1

    # Final save
    final_path = os.path.join(args.output_dir, "checkpoints", "final.pt")
    torch.save({
        "step": curriculum_steps + step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }, final_path)
    last_loss = loss.item() if loss is not None else float("nan")
    print(f"Training complete. Final loss: {last_loss:.4f}")

    writer.close()


def main():
    parser = argparse.ArgumentParser(description="FGN v3 Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=-1, help="-1 for full epochs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume model weights from checkpoint (no optimizer state)")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--grad_log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
