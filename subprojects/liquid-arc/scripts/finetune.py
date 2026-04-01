"""Fine-tune LiquidARC from checkpoint on real ARC data.

Loads a pre-trained checkpoint (e.g. from procedural training) and
continues training on real ARC tasks to transfer transform abilities.

Usage:
    python scripts/finetune.py --checkpoint output_v2/checkpoints/final.pt \
        --data_dir /workspace/fgn-v3/data/arc --output_dir output_v2_ft
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if not Path(FGN_ROOT).exists():
    FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, FGN_ROOT)

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel, create_model
from fgn.tasks.arc import ARCTask


def create_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, eval_task, device, n_batches=20, batch_size=8):
    model.eval()
    total_cells = 0
    total_correct = 0
    total_xform_cells = 0
    total_xform_correct = 0
    total_copy_correct = 0
    total_ce = 0.0
    total_xf_loss = 0.0
    n_valid = 0

    with torch.no_grad():
        for _ in range(n_batches):
            _, _, meta = eval_task.generate_batch(batch_size, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = model(
                    colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                    roles=meta["roles"], sep_mask=meta["sep_mask"],
                    sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                    target_labels=meta["target_labels"],
                    context_mask=meta["context_mask"],
                    grid_ids=meta.get("grid_ids"),
                    target_input_colors=meta.get("target_input_colors"),
                )

            preds = result["logits"].argmax(dim=-1)
            tgt = meta["target_labels"]
            inp = meta.get("target_input_colors")

            valid = tgt != -100
            nv = valid.sum().item()
            total_cells += nv
            total_correct += (preds[valid] == tgt[valid]).sum().item()

            if inp is not None:
                total_copy_correct += (tgt[valid] == inp[valid]).sum().item()
                transform = valid & (tgt != inp)
                nxf = transform.sum().item()
                total_xform_cells += nxf
                total_xform_correct += (preds[transform] == tgt[transform]).sum().item()

            total_ce += result["ce_loss"].item()
            xfl = result.get("xform_loss", torch.tensor(0.0))
            if isinstance(xfl, torch.Tensor):
                xfl = xfl.item()
            total_xf_loss += xfl
            n_valid += 1

    return {
        "cell_acc": total_correct / max(total_cells, 1),
        "xform_acc": total_xform_correct / max(total_xform_cells, 1),
        "copy_bl": total_copy_correct / max(total_cells, 1),
        "ce": total_ce / max(n_valid, 1),
        "xf_loss": total_xf_loss / max(n_valid, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LiquidARC on real ARC")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/workspace/fgn-v3/data/arc")
    parser.add_argument("--output_dir", type=str, default="output_ft")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=30)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--n_color_perms", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    start_step = ckpt["step"]
    print(f"Loaded checkpoint: {args.checkpoint}, step={start_step}")
    print(f"  tau_max={config.tau_max}, copy_weight={config.copy_weight}, "
          f"transform_weight={config.transform_weight}")

    model = create_model(config, device)
    model.load_state_dict(ckpt["model"], strict=False)

    # Ensure tau is unfrozen for fine-tuning
    raw_m = model._orig_mod if hasattr(model, '_orig_mod') else model
    if isinstance(raw_m, LiquidARCModel):
        raw_m.dynamics.freeze_tau = False
        print("  Tau: unfrozen")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "logs"))

    # Train on REAL ARC data with heavy augmentation
    train_task = ARCTask(
        seq_len=config.max_seq_len,
        data_dir=args.data_dir,
        split="train",
        augment=True,
        n_color_perms=args.n_color_perms,
    )
    eval_task = ARCTask(
        seq_len=config.max_seq_len,
        data_dir=args.data_dir,
        split="eval",
        augment=False,
    )
    print(f"  Train: real ARC ({len(train_task.tasks)} tasks, {args.n_color_perms} color perms)")
    print(f"  Eval: real ARC ({len(eval_task.tasks)} tasks)")

    # Optimizer — lower LR for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args.warmup_steps, args.max_steps)

    # Compile dynamics
    if config.use_torch_compile and device.type == "cuda" and isinstance(model, LiquidARCModel):
        model.dynamics = torch.compile(model.dynamics, mode="default", dynamic=True)
        print("  torch.compile: dynamics compiled")

    # Initial eval before any fine-tuning
    print("\n  Pre-finetune eval:")
    ev = evaluate(model, eval_task, device, n_batches=args.eval_batches,
                  batch_size=args.batch_size)
    print(f"    cell_acc={ev['cell_acc']:.4f}, xform_acc={ev['xform_acc']:.4f}, "
          f"copy_bl={ev['copy_bl']:.4f}, ce={ev['ce']:.4f}, xf_loss={ev['xf_loss']:.4f}")

    print(f"\n{'='*70}")
    print(f"Fine-tuning on real ARC — LR={args.lr}, {args.max_steps} steps")
    print(f"{'='*70}")

    model.train()
    t0 = time.time()
    best_xform_acc = 0.0

    for step in range(args.max_steps):
        optimizer.zero_grad()

        # Temporal invariance
        n_steps = random.randint(config.ode_steps_min, config.ode_steps_max)

        _, _, meta = train_task.generate_batch(args.batch_size, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            result = model(
                colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                roles=meta["roles"], sep_mask=meta["sep_mask"],
                sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                target_labels=meta["target_labels"],
                context_mask=meta["context_mask"],
                grid_ids=meta.get("grid_ids"),
                target_input_colors=meta.get("target_input_colors"),
                n_steps=n_steps,
            )

        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # Log
        if step % args.log_every == 0:
            dt = time.time() - t0
            avg_n = meta.get("lengths", torch.tensor(config.max_seq_len)).float().mean().item()
            tok_s = args.batch_size * avg_n * (step + 1) / max(dt, 1e-6)

            cell_acc = result.get("cell_accuracy", torch.tensor(0.0))
            if isinstance(cell_acc, torch.Tensor):
                cell_acc = cell_acc.item()
            xform_acc = result.get("transform_accuracy", torch.tensor(0.0))
            if isinstance(xform_acc, torch.Tensor):
                xform_acc = xform_acc.item()
            xf_loss = result.get("xform_loss", torch.tensor(0.0))
            if isinstance(xf_loss, torch.Tensor):
                xf_loss = xf_loss.item()

            extra = ""
            m = model._orig_mod if hasattr(model, '_orig_mod') else model
            if isinstance(m, LiquidARCModel):
                cv = result["metric_cv"]
                if isinstance(cv, torch.Tensor):
                    cv = cv.item()
                tau_val = result.get("tau_avg", torch.tensor(0.0))
                if isinstance(tau_val, torch.Tensor):
                    tau_val = tau_val.item()
                extra = f", cv={cv:.3f}, tau={tau_val:.2f}"

            print(f"  [ft step={step}] loss={result['loss'].item():.4f}, "
                  f"xf_loss={xf_loss:.4f}, "
                  f"cell={cell_acc:.4f}, xform={xform_acc:.4f}, "
                  f"tok/s={tok_s:.0f}{extra}")

            writer.add_scalar("loss/total", result["loss"].item(), step)
            writer.add_scalar("loss/xform", xf_loss, step)
            writer.add_scalar("accuracy/cell_train", cell_acc, step)
            writer.add_scalar("accuracy/xform_train", xform_acc, step)

        # Eval
        if step > 0 and step % args.eval_every == 0:
            ev = evaluate(model, eval_task, device, n_batches=args.eval_batches,
                          batch_size=args.batch_size)
            print(f"  >> EVAL [ft step={step}] cell={ev['cell_acc']:.4f}, "
                  f"xform={ev['xform_acc']:.4f}, copy_bl={ev['copy_bl']:.4f}, "
                  f"ce={ev['ce']:.4f}, xf_loss={ev['xf_loss']:.4f}")

            writer.add_scalar("accuracy/cell_eval", ev["cell_acc"], step)
            writer.add_scalar("accuracy/xform_eval", ev["xform_acc"], step)
            writer.add_scalar("loss/eval_ce", ev["ce"], step)
            writer.add_scalar("loss/eval_xform", ev["xf_loss"], step)

            if ev["xform_acc"] > best_xform_acc:
                best_xform_acc = ev["xform_acc"]
                torch.save({
                    "step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "config": config,
                    "eval": ev,
                }, os.path.join(args.output_dir, "checkpoints", "best_xform.pt"))
                print(f"  >> New best eval xform_acc: {ev['xform_acc']:.4f}")

            model.train()

        # Save
        if step > 0 and step % args.save_every == 0:
            torch.save({
                "step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "config": config,
            }, os.path.join(args.output_dir, "checkpoints", f"ft_step_{step}.pt"))

    # Final save
    torch.save({
        "step": args.max_steps, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "config": config,
    }, os.path.join(args.output_dir, "checkpoints", "ft_final.pt"))
    writer.close()
    print(f"\n  Fine-tuning complete. Best eval xform_acc: {best_xform_acc:.4f}")


if __name__ == "__main__":
    main()
