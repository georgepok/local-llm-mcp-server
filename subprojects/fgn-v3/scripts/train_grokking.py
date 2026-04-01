"""Grokking — fixed dataset, train past memorization, watch for phase shifts.

Generate N episodes once, then train on them repeatedly.
Log curvature and metric stats densely to catch phase transitions.
No mutations, no perturb, no dynamic decay — just pure overtraining.
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
from fgn.flat_model import FlatTransformerModel
from fgn.tasks import get_task


def create_model(config, device):
    if config.model_type == "flat":
        return FlatTransformerModel(config).to(device)
    elif config.architecture_version == "fluid":
        return FluidNetModel(config).to(device)
    else:
        return FlatTransformerModel(config).to(device)


def generate_fixed_dataset(task, n_episodes, seq_len, device):
    """Generate a fixed dataset of episodes. Returns list of (input_ids, labels, context_mask)."""
    pad_id = task.tokenizer.eos_token_id or 0
    dataset = []

    for i in range(n_episodes):
        for _retry in range(200):
            ep_result = task._generate_valid_episode()
            if ep_result is None:
                continue
            episode_text, _actions, n_steps, optimal_cost, step_costs, world = ep_result
            input_ids, labels, context_end_pos, action_spans, room_token_pos = \
                task._tokenize_episode(episode_text)

            if len(input_ids) > seq_len:
                input_ids = input_ids[:seq_len]
                labels = labels[:seq_len]
            else:
                pad_len = seq_len - len(input_ids)
                input_ids += [pad_id] * pad_len
                labels += [-100] * pad_len

            n_supervised = sum(1 for l in labels if l != -100)
            if n_supervised >= 5:
                break
        else:
            continue

        context_mask = [False] * seq_len
        for j in range(min(context_end_pos, seq_len)):
            context_mask[j] = True

        dataset.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "labels": torch.tensor(labels, dtype=torch.long, device=device),
            "context_mask": torch.tensor(context_mask, dtype=torch.bool, device=device),
        })

    print(f"  Generated {len(dataset)} episodes (requested {n_episodes})")
    return dataset


def make_batch(dataset, batch_size, step, device):
    """Cycle through fixed dataset deterministically."""
    n = len(dataset)
    batch_items = []
    for i in range(batch_size):
        idx = (step * batch_size + i) % n
        batch_items.append(dataset[idx])

    return (
        torch.stack([b["input_ids"] for b in batch_items]),
        torch.stack([b["labels"] for b in batch_items]),
        torch.stack([b["context_mask"] for b in batch_items]),
    )


def train(args, config, device):
    tokenizer = _get_tokenizer()
    task_kwargs = json.loads(args.task_kwargs)

    print(f"\n{'='*60}")
    print(f"  Grokking Run — Fixed Dataset Phase Shift Detection")
    print(f"{'='*60}")

    model = create_model(config, device)
    is_fluid = isinstance(model, FluidNetModel)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {'FluidNet' if is_fluid else 'flat'}, {n_params:,} params")
    print(f"  seq_len={config.max_seq_len}, d_model={config.d_model}, "
          f"n_layers={config.n_layers}")

    # Generate fixed dataset
    task = get_task("CW", tokenizer, seq_len=config.max_seq_len, **task_kwargs)
    print(f"  Generating {args.n_episodes} fixed episodes...")
    dataset = generate_fixed_dataset(task, args.n_episodes, config.max_seq_len, device)

    if len(dataset) < args.batch_size:
        print(f"  ERROR: Only generated {len(dataset)} episodes, need at least {args.batch_size}")
        return

    # Setup
    out_dir = args.output_dir
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, "logs"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model_raw = model
    if config.use_torch_compile and device.type == "cuda":
        compiled_model = torch.compile(model, mode="default")
    else:
        compiled_model = model

    compiled_model.train()
    t0 = time.time()

    # How many full passes through the dataset
    steps_per_epoch = max(1, len(dataset) // args.batch_size)
    print(f"  Dataset: {len(dataset)} episodes, {steps_per_epoch} steps/epoch")
    print(f"  Training for {args.max_steps} steps "
          f"({args.max_steps / steps_per_epoch:.0f} epochs)")
    print(f"  Logging every {args.log_every} steps")
    print()

    for step in range(args.max_steps):
        input_ids, labels, context_mask = make_batch(
            dataset, args.batch_size, step, device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            if is_fluid:
                result = compiled_model(input_ids, labels=labels,
                                        context_mask=context_mask)
            else:
                result = compiled_model(input_ids, labels=labels)
            loss = result["loss"]

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [step={step}] WARNING: NaN/Inf loss, skipping")
            optimizer.zero_grad()
            scheduler.step()
            continue

        ce_val = result["ce_loss"].item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # Dense logging
        if step % args.log_every == 0:
            dt = time.time() - t0
            tok_s = args.batch_size * config.max_seq_len * (step + 1) / max(dt, 1e-6)
            epoch = step / steps_per_epoch

            cv_val = result["metric_cv"]
            if isinstance(cv_val, torch.Tensor):
                cv_val = cv_val.item()
            kappa = result["avg_kappa"].item()

            extra = ""
            if is_fluid:
                t_local = result.get("avg_t_local", torch.tensor(0.0)).item()
                t_medium = result.get("avg_t_medium", torch.tensor(0.0)).item()
                t_global = result.get("avg_t_global", torch.tensor(0.0)).item()
                extra = f", t=[{t_local:.2f},{t_medium:.2f},{t_global:.2f}]"

            print(f"  [step={step}] ep={epoch:.1f} loss={loss.item():.4f} "
                  f"ce={ce_val:.4f} cv={cv_val:.4f} "
                  f"|k|={kappa:.4f} tok/s={tok_s:.0f}{extra}")

            writer.add_scalar("loss/total", loss.item(), step)
            writer.add_scalar("loss/ce", ce_val, step)
            writer.add_scalar("metric/cv", cv_val, step)
            writer.add_scalar("metric/kappa", kappa, step)
            writer.add_scalar("train/epoch", epoch, step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

        if step > 0 and step % args.save_every == 0:
            ckpt = {"step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "config": config}
            torch.save(ckpt, os.path.join(out_dir, "checkpoints", f"step_{step}.pt"))

    # Final save
    ckpt = {"step": args.max_steps, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "config": config}
    torch.save(ckpt, os.path.join(out_dir, "checkpoints", "final.pt"))
    writer.close()
    print(f"\n  Done. Final loss: {loss.item():.4f}")


def _get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


def main():
    parser = argparse.ArgumentParser(description="Grokking Phase Shift Detection")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_grokking")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--n_episodes", type=int, default=64,
                        help="Number of fixed episodes in dataset")
    parser.add_argument("--task_kwargs", type=str, default="{}",
                        help="JSON dict of CW task kwargs")

    args = parser.parse_args()
    config = FGNConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mem_frac = float(os.environ.get("CUDA_MEMORY_FRACTION", "0"))
    if mem_frac > 0 and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(mem_frac)

    train(args, config, device)


if __name__ == "__main__":
    main()
