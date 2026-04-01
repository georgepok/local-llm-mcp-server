"""FGN v6 — Budget-based attention escalation training script.

No sharpness annealing. Fixed per-layer attention budgets.
Shared context pooling from world-description prefix.
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
from fgn.model_v6 import FGNv6Model
from fgn.flat_model import FlatTransformerModel
from fgn.tasks import get_task


def create_model(config: FGNConfig, device: torch.device):
    """Create model based on config."""
    if config.model_type == "flat":
        return FlatTransformerModel(config).to(device)
    else:
        return FGNv6Model(config).to(device)


def create_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def print_v6_status(model, result, step, tok_per_sec):
    """Print compact v6 status line."""
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    is_v6 = isinstance(m, FGNv6Model)

    esc_str = ""
    ent_str = ""

    if is_v6:
        esc_rates = result.get("esc_rates_per_layer", [])
        entropies = result.get("entropies_per_layer", [])

        if esc_rates:
            esc_str = (f"\n  esc_rate=[{','.join(f'{r.item():.2f}' for r in esc_rates)}]")
        if entropies:
            ent_str = (f"\n  entropy=[{','.join(f'{e.item():.2f}' for e in entropies)}]")

    print(f"  [step={step}] loss={result['loss'].item():.4f}, "
          f"ce={result['ce_loss'].item():.4f}, "
          f"cv={result['metric_cv'].item():.4f}, "
          f"|k|={result['avg_kappa'].item():.4f}, "
          f"tok/s={tok_per_sec:.0f}"
          f"{esc_str}{ent_str}")


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
    """Training loop with fixed budgets (no annealing)."""
    tokenizer = _get_tokenizer()
    task_kwargs = json.loads(args.task_kwargs)

    print(f"\n{'='*70}")
    print(f"FGN v6 Training — {config.model_type}")
    print(f"{'='*70}")

    model = create_model(config, device)
    is_v6 = isinstance(model, FGNv6Model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    if is_v6:
        print(f"  Architecture: v6 budget-based (GeoRoute → top-k entropy → Attention)")
        print(f"  Metric type: {config.geo_metric_type}")
        budgets = config.attention_budgets
        print(f"  Budgets: [{', '.join(f'{b:.2f}' for b in budgets)}]")
    else:
        print(f"  Architecture: flat baseline")

    out_dir = args.output_dir
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, "logs"))

    total_steps = args.max_steps

    # Create task
    task = get_task(args.task, tokenizer, seq_len=config.max_seq_len, **task_kwargs)

    # Single optimizer for all params
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args.warmup_steps, total_steps)

    if config.use_torch_compile and device.type == "cuda":
        compiled_model = torch.compile(model, mode="default")
    else:
        compiled_model = model

    compiled_model.train()
    t0 = time.time()
    loss = torch.tensor(0.0)

    for step in range(total_steps):
        # No sharpness annealing in v6

        input_ids, labels, meta = task.generate_batch(args.batch_size, device=device)

        # Extract context_mask from task metadata (v6)
        context_mask = meta.get("context_mask")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            if is_v6:
                result = compiled_model(input_ids, labels=labels,
                                        context_mask=context_mask)
            else:
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
            print_v6_status(model, result, step, tok_s)

            # TensorBoard logging
            writer.add_scalar("loss/total", result["loss"].item(), step)
            writer.add_scalar("loss/ce", result["ce_loss"].item(), step)
            writer.add_scalar("metric/cv", result["metric_cv"].item(), step)
            writer.add_scalar("metric/kappa", result["avg_kappa"].item(), step)

            if is_v6:
                writer.add_scalar("escalation/rate",
                                  result["escalation_rate"].item(), step)
                writer.add_scalar("escalation/entropy",
                                  result["avg_entropy"].item(), step)
                esc_rates = result.get("esc_rates_per_layer", [])
                for i, r in enumerate(esc_rates):
                    writer.add_scalar(f"escalation/rate_layer_{i}", r.item(), step)

        if step > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, config, step,
                          os.path.join(out_dir, "checkpoints", f"step_{step}.pt"))

    # Final save
    save_checkpoint(model, optimizer, config, total_steps,
                   os.path.join(out_dir, "checkpoints", "final.pt"),
                   extra={"task": args.task})
    writer.close()

    print(f"\n  Training complete. Final loss: {loss.item():.4f}")

    # Print final summary
    if is_v6:
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        print(f"\n  Final per-layer summary:")
        for i, layer in enumerate(m.layers):
            parts = [f"Layer {i}: budget={layer.budget:.2f}"]
            if hasattr(layer, 'geo_route'):
                log_t = layer.geo_route.log_t
                if log_t.numel() > 1:
                    parts.append(f"log_t=[{','.join(f'{t:.2f}' for t in log_t.tolist())}]")
                else:
                    parts.append(f"log_t={log_t.item():.4f}")
            print(f"    {', '.join(parts)}")


def _get_tokenizer():
    """Load GPT-2 tokenizer."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


def main():
    parser = argparse.ArgumentParser(description="FGN v6 Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default="W",
                        help="Task name (default: W = gridworld)")
    parser.add_argument("--output_dir", type=str, default="output_v6")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=10000)
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
    if config.model_type == "fgn":
        print(f"Metric type: {config.geo_metric_type}")
        print(f"Budgets: {config.attention_budgets}")
    print(f"Task: {args.task}, kwargs: {args.task_kwargs}")
    print(f"Total steps: {args.max_steps}")

    train(args, config, device)


if __name__ == "__main__":
    main()
