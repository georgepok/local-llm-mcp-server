"""FluidNet v1 — Training script for pure geometric computation.

Supports FluidNet, v6-metric, and flat transformer models.
Works with both discrete gridworld (W) and continuous gridworld (CW) tasks.
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
from fgn.model_fluid import FluidNetModel
from fgn.model_v6 import FGNv6Model
from fgn.flat_model import FlatTransformerModel
from fgn.tasks import get_task


def create_model(config: FGNConfig, device: torch.device):
    """Create model based on config."""
    if config.model_type == "flat":
        return FlatTransformerModel(config).to(device)
    elif config.architecture_version == "fluid":
        return FluidNetModel(config).to(device)
    elif config.architecture_version == "v6":
        return FGNv6Model(config).to(device)
    else:
        return FlatTransformerModel(config).to(device)


def create_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def print_fluid_status(model, result, step, tok_per_sec):
    """Print compact status line with FluidNet-specific metrics."""
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    is_fluid = isinstance(m, FluidNetModel)

    extra = ""
    if is_fluid:
        t_local = result.get("avg_t_local", torch.tensor(0.0)).item()
        t_medium = result.get("avg_t_medium", torch.tensor(0.0)).item()
        t_global = result.get("avg_t_global", torch.tensor(0.0)).item()
        extra = f", t=[{t_local:.2f},{t_medium:.2f},{t_global:.2f}]"

    print(f"  [step={step}] loss={result['loss'].item():.4f}, "
          f"ce={result['ce_loss'].item():.4f}, "
          f"cv={result['metric_cv']:.4f}, "
          f"|k|={result['avg_kappa'].item():.4f}, "
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
    tokenizer = _get_tokenizer()
    task_kwargs = json.loads(args.task_kwargs)

    print(f"\n{'='*70}")
    print(f"FluidNet Training — {config.architecture_version}")
    print(f"{'='*70}")

    model = create_model(config, device)
    is_fluid = isinstance(model, FluidNetModel)
    is_v6 = isinstance(model, FGNv6Model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    if is_fluid:
        print(f"  Architecture: FluidNet (pure geometric diffusion)")
        print(f"  Scales: {config.n_scales}, d_metric: {config.d_metric}")
        print(f"  d_ffn_fluid: {config.d_ffn_fluid}")
    elif is_v6:
        print(f"  Architecture: v6 budget-based")
        print(f"  Metric type: {config.geo_metric_type}")
        print(f"  Budgets: [{', '.join(f'{b:.2f}' for b in config.attention_budgets)}]")
    else:
        print(f"  Architecture: flat baseline")

    out_dir = args.output_dir
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, "logs"))

    total_steps = args.max_steps
    task = get_task(args.task, tokenizer, seq_len=config.max_seq_len, **task_kwargs)

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
        input_ids, labels, meta = task.generate_batch(args.batch_size, device=device)
        context_mask = meta.get("context_mask")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            if is_fluid or is_v6:
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
            print_fluid_status(model, result, step, tok_s)

            writer.add_scalar("loss/total", result["loss"].item(), step)
            writer.add_scalar("loss/ce", result["ce_loss"].item(), step)

            cv_val = result["metric_cv"]
            if isinstance(cv_val, torch.Tensor):
                cv_val = cv_val.item()
            writer.add_scalar("metric/cv", cv_val, step)
            writer.add_scalar("metric/kappa", result["avg_kappa"].item(), step)

            if is_fluid:
                writer.add_scalar("timescale/local",
                                  result.get("avg_t_local", torch.tensor(0.0)).item(), step)
                writer.add_scalar("timescale/medium",
                                  result.get("avg_t_medium", torch.tensor(0.0)).item(), step)
                writer.add_scalar("timescale/global",
                                  result.get("avg_t_global", torch.tensor(0.0)).item(), step)

        if step > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, config, step,
                          os.path.join(out_dir, "checkpoints", f"step_{step}.pt"))

    save_checkpoint(model, optimizer, config, total_steps,
                   os.path.join(out_dir, "checkpoints", "final.pt"),
                   extra={"task": args.task,
                          "architecture_version": config.architecture_version})
    writer.close()

    print(f"\n  Training complete. Final loss: {loss.item():.4f}")

    if is_fluid:
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        print(f"\n  Final timescales per layer:")
        for i, layer in enumerate(m.layers):
            import torch.nn.functional as F
            t_bias = F.softplus(layer.time_net_linear2.bias)
            print(f"    Layer {i}: t_init=[{','.join(f'{t:.3f}' for t in t_bias.tolist())}]")


def _get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


def main():
    parser = argparse.ArgumentParser(description="FluidNet Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default="CW",
                        help="Task: CW=continuous gridworld, W=discrete gridworld")
    parser.add_argument("--output_dir", type=str, default="output_fluid")
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
    print(f"Task: {args.task}, kwargs: {args.task_kwargs}")
    print(f"Total steps: {args.max_steps}")

    train(args, config, device)


if __name__ == "__main__":
    main()
