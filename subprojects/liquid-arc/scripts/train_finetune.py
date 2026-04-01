"""Post-transition fine-tuning: freeze routing, train computation on 100% ARC.

Resumes from post-transition checkpoint. Freezes MetricNet + TauNet (preserves
the phase transition's routing structure). Unfreezes FFN + W_v + W_o + output
head + LayerNorms. Trains on 100% real ARC data with low LR.

Usage:
    python scripts/train_finetune.py \
        --checkpoint output_reproduce/checkpoints/step_7500.pt \
        --config configs/finetune_arc.yaml \
        --data_dir /workspace/fgn-v3/data/arc-repo/data \
        --output_dir output_finetune_arc \
        --max_steps 5000 --lr 3e-5
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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel, create_model

FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if not Path(FGN_ROOT).exists():
    FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, FGN_ROOT)
from fgn.tasks.arc import ARCTask


def create_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, eval_task, device, config, n_batches=20):
    model.eval()
    total_xform_correct = 0
    total_xform = 0
    total_cell_correct = 0
    total_cells = 0
    total_ce = 0.0
    total_solved = 0
    total_tasks = 0

    with torch.no_grad():
        for _ in range(n_batches):
            _, _, meta = eval_task.generate_batch(8, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = model(
                    colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                    roles=meta["roles"], sep_mask=meta["sep_mask"],
                    sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                    target_labels=meta["target_labels"],
                    context_mask=meta["context_mask"],
                    grid_ids=meta.get("grid_ids"), lengths=meta.get("lengths"),
                    target_input_colors=meta.get("target_input_colors"),
                )

            tgt = meta["target_labels"]
            inp = meta.get("target_input_colors")
            valid = tgt != -100
            preds = result["logits"].argmax(dim=-1)

            n_valid = valid.sum().item()
            total_cell_correct += (preds[valid] == tgt[valid]).sum().item()
            total_cells += n_valid

            if inp is not None:
                xform = valid & (tgt != inp)
                n_xf = xform.sum().item()
                total_xform_correct += (preds[xform] == tgt[xform]).sum().item()
                total_xform += n_xf

            total_ce += result["ce_loss"].item()

            B = tgt.shape[0]
            for b in range(B):
                v = valid[b]
                if v.sum() == 0:
                    continue
                total_tasks += 1
                if (preds[b][v] == tgt[b][v]).all():
                    total_solved += 1

    return {
        "cell_acc": total_cell_correct / max(total_cells, 1),
        "xform_acc": total_xform_correct / max(total_xform, 1),
        "ce": total_ce / max(n_batches, 1),
        "solved": total_solved,
        "total_tasks": total_tasks,
    }


def main():
    parser = argparse.ArgumentParser(description="Post-transition fine-tuning")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_finetune_arc")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LiquidARCConfig.from_yaml(args.config)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # Load model from post-transition checkpoint
    print(f"Loading from {args.checkpoint}...")
    model = create_model(config, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = {k.replace("._orig_mod.", "."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    start_step = ckpt.get("step", 0)
    print(f"  Loaded at step {start_step}")

    # Freeze routing (MetricNet + TauNet + context_pool + t_diffusion)
    geo_params = set(id(p) for p in model.geo_parameters())
    n_frozen = 0
    n_trainable = 0
    for p in model.parameters():
        if id(p) in geo_params:
            p.requires_grad = False
            n_frozen += p.numel()
        else:
            p.requires_grad = True
            n_trainable += p.numel()

    print(f"  Frozen (routing): {n_frozen:,} params")
    print(f"  Trainable (computation): {n_trainable:,} params")
    print(f"  Trainable modules: FFN, W_v, W_o, LayerNorms, embedding, output_head")

    # Optimizer on trainable params only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args.warmup_steps, args.max_steps)

    # 100% real ARC data
    train_task = ARCTask(seq_len=config.max_seq_len, data_dir=args.data_dir,
                         split="train", augment=True, n_color_perms=10)
    eval_task = ARCTask(seq_len=config.max_seq_len, data_dir=args.data_dir,
                        split="eval", augment=False)
    print(f"  Data: 100% real ARC (400 train, 400 eval)")
    print(f"  LR: {args.lr}, warmup: {args.warmup_steps}, max_steps: {args.max_steps}")

    # Logging
    log_path = os.path.join(args.output_dir, "train.log")
    logger = logging.getLogger("finetune")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    import builtins
    builtins.print = lambda *a, **kw: logger.info(" ".join(str(x) for x in a))

    writer = SummaryWriter(os.path.join(args.output_dir, "logs"))

    # Initial eval
    print("\n  Initial eval (before fine-tuning)...")
    init_eval = evaluate(model, eval_task, device, config)
    print(f"  Base: xform={init_eval['xform_acc']*100:.1f}%, "
          f"cell={init_eval['cell_acc']*100:.1f}%, "
          f"ce={init_eval['ce']:.4f}, "
          f"solved={init_eval['solved']}/{init_eval['total_tasks']}")

    # Training loop
    model.train()
    best_xform = init_eval['xform_acc']
    t0 = time.time()

    for step in range(args.max_steps):
        optimizer.zero_grad()

        _, _, meta = train_task.generate_batch(args.batch_size, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            result = model(
                colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                roles=meta["roles"], sep_mask=meta["sep_mask"],
                sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                target_labels=meta["target_labels"],
                context_mask=meta["context_mask"],
                grid_ids=meta.get("grid_ids"), lengths=meta.get("lengths"),
                target_input_colors=meta.get("target_input_colors"),
                n_steps=config.n_ode_steps,
            )

        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        # Logging
        if step % args.log_every == 0:
            dt = time.time() - t0
            loss = result["loss"].item()
            ce = result["ce_loss"].item()
            xf = result.get("xform_loss", torch.tensor(0.0))
            if isinstance(xf, torch.Tensor):
                xf = xf.item()
            cell_acc = result.get("cell_accuracy", torch.tensor(0.0))
            if isinstance(cell_acc, torch.Tensor):
                cell_acc = cell_acc.item()
            xform_acc = result.get("transform_accuracy", torch.tensor(0.0))
            if isinstance(xform_acc, torch.Tensor):
                xform_acc = xform_acc.item()
            cv = result.get("metric_cv", torch.tensor(0.0))
            if isinstance(cv, torch.Tensor):
                cv = cv.item()

            lr = optimizer.param_groups[0]["lr"]
            print(f"  [step={step}] loss={loss:.4f}, ce={ce:.4f}, "
                  f"xf_loss={xf:.4f}, cell={cell_acc:.4f}, xform={xform_acc:.4f}, "
                  f"cv={cv:.4f}, lr={lr:.2e}")

            writer.add_scalar("loss/total", loss, step)
            writer.add_scalar("loss/ce", ce, step)
            writer.add_scalar("accuracy/xform_train", xform_acc, step)
            writer.add_scalar("metric/cv", cv, step)

        # Eval
        if step > 0 and step % args.eval_every == 0:
            ev = evaluate(model, eval_task, device, config)
            delta = ev['xform_acc'] - init_eval['xform_acc']
            print(f"  >> EVAL [step={step}] xform={ev['xform_acc']*100:.1f}%, "
                  f"cell={ev['cell_acc']*100:.1f}%, ce={ev['ce']:.4f}, "
                  f"solved={ev['solved']}/{ev['total_tasks']}, "
                  f"Δxform={'+' if delta>=0 else ''}{delta*100:.1f}%")

            writer.add_scalar("accuracy/xform_eval", ev['xform_acc'], step)
            writer.add_scalar("accuracy/cell_eval", ev['cell_acc'], step)
            writer.add_scalar("loss/eval_ce", ev['ce'], step)

            if ev['xform_acc'] > best_xform:
                best_xform = ev['xform_acc']
                torch.save({"step": step, "model": model.state_dict(), "config": config},
                           os.path.join(args.output_dir, "checkpoints", "best.pt"))
                print(f"  >> New best: {best_xform*100:.1f}%")

            model.train()

        # Checkpoints
        if step > 0 and step % args.save_every == 0:
            torch.save({"step": step, "model": model.state_dict(), "config": config},
                       os.path.join(args.output_dir, "checkpoints", f"step_{step}.pt"))

    # Final eval
    final_eval = evaluate(model, eval_task, device, config, n_batches=40)
    delta = final_eval['xform_acc'] - init_eval['xform_acc']
    print(f"\n{'='*60}")
    print(f"Fine-tuning Results ({args.max_steps} steps)")
    print(f"{'='*60}")
    print(f"Before: xform={init_eval['xform_acc']*100:.1f}%, "
          f"solved={init_eval['solved']}/{init_eval['total_tasks']}")
    print(f"After:  xform={final_eval['xform_acc']*100:.1f}%, "
          f"solved={final_eval['solved']}/{final_eval['total_tasks']}")
    print(f"Delta:  {'+' if delta>=0 else ''}{delta*100:.1f}% xform, "
          f"{'+' if final_eval['solved']>=init_eval['solved'] else ''}"
          f"{final_eval['solved']-init_eval['solved']} tasks solved")
    print(f"Best:   {best_xform*100:.1f}%")

    torch.save({"step": args.max_steps, "model": model.state_dict(), "config": config},
               os.path.join(args.output_dir, "checkpoints", "final.pt"))
    writer.close()


if __name__ == "__main__":
    main()
