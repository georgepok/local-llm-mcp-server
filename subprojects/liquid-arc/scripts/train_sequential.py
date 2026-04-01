"""Sequential episode training for persistent state experiments.

Trains on multi-turn stateful episodes where each turn is one forward pass.
Compares persistent state (α<1.0) vs non-persistent (α=1.0).
Logs per-turn accuracy to measure temporal depth benefit.

Usage:
    python scripts/train_sequential.py \
        --config configs/agentic_persistent.yaml \
        --resume output_30m/checkpoints/step_10000.pt \
        --output_dir output_persistent/with_state \
        --persist_alpha 0.7 \
        --max_steps 3000
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel, create_model
from liquid_arc.tasks.sequential_agentic import SequentialAgenticDataset


def evaluate_sequential(model, device, n_episodes=50, n_vars=4, total_ops=8,
                        ops_per_turn=2, seq_len=2048, persist_active=True):
    """Evaluate on sequential episodes, reporting per-turn accuracy."""
    model.eval()
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    n_turns = total_ops // ops_per_turn
    turn_correct = [0] * n_turns
    turn_total = [0] * n_turns
    turn_xform_correct = [0] * n_turns
    turn_xform_total = [0] * n_turns

    dataset = SequentialAgenticDataset(
        batch_size=1, n_vars=n_vars, total_ops=total_ops,
        ops_per_turn=ops_per_turn, seq_len=seq_len,
    )

    for _ in range(n_episodes):
        dataset.reset_episodes()
        raw_model.persistent.reset()

        for turn in range(n_turns):
            (_, _, meta), is_start, is_end, turn_idx = dataset.get_next_turn_batch(device)

            with torch.no_grad():
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

            ca = result.get("cell_accuracy", torch.tensor(0.0))
            if isinstance(ca, torch.Tensor):
                ca = ca.item()
            xa = result.get("transform_accuracy", torch.tensor(0.0))
            if isinstance(xa, torch.Tensor):
                xa = xa.item()
            nx = result.get("n_transform", torch.tensor(0))
            if isinstance(nx, torch.Tensor):
                nx = nx.item()
            n_tgt = (meta["target_labels"] != -100).sum().item()

            turn_correct[turn] += int(ca * n_tgt)
            turn_total[turn] += n_tgt
            turn_xform_correct[turn] += int(xa * nx)
            turn_xform_total[turn] += nx

    turn_accs = [turn_correct[t] / max(turn_total[t], 1) for t in range(n_turns)]
    turn_xforms = [turn_xform_correct[t] / max(turn_xform_total[t], 1) for t in range(n_turns)]

    return turn_accs, turn_xforms


def main():
    parser = argparse.ArgumentParser(description="Sequential Episode Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--persist_alpha", type=float, default=0.7)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_episodes", type=int, default=50)
    args = parser.parse_args()

    config = LiquidARCConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        mem_frac = float(os.environ.get("CUDA_MEMORY_FRACTION", "0"))
        if mem_frac > 0:
            torch.cuda.set_per_process_memory_fraction(mem_frac)
            print(f"CUDA memory fraction capped at {mem_frac*100:.0f}%")

    # Override persist_alpha from command line
    config.persist_alpha = args.persist_alpha

    print(f"Device: {device}")
    print(f"Persist alpha: {args.persist_alpha}")
    persist_label = "PERSISTENT" if args.persist_alpha < 1.0 else "NON-PERSISTENT"
    print(f"Mode: {persist_label}")

    # Create model
    model = create_model(config, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Load checkpoint
    print(f"Resuming from {args.resume}")
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    state = ckpt["model"]
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
    # Allow missing persistent_state keys (new module)
    model.load_state_dict(cleaned, strict=False)
    start_step = ckpt["step"]
    print(f"Resumed at step {start_step}")

    # Set persistence mode
    raw_model = model
    if hasattr(model, '_orig_mod'):
        raw_model = model._orig_mod
    raw_model.persistent.set_active(args.persist_alpha < 1.0)
    print(f"Persistent state active: {raw_model.persistent._active}")

    # Setup
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "logs"))

    # Episode data
    total_ops = getattr(config, 'episode_total_ops', 8)
    ops_per_turn = getattr(config, 'episode_ops_per_turn', 2)
    n_vars = getattr(config, 'episode_n_vars', 4)
    n_turns = total_ops // ops_per_turn

    dataset = SequentialAgenticDataset(
        batch_size=args.batch_size,
        n_vars=n_vars, total_ops=total_ops, ops_per_turn=ops_per_turn,
        seq_len=config.max_seq_len,
    )

    print(f"Episodes: {total_ops} ops, {ops_per_turn}/turn, {n_turns} turns")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Torch compile
    if config.use_torch_compile and device.type == "cuda" and isinstance(model, LiquidARCModel):
        model.dynamics = torch.compile(model.dynamics, mode="default", dynamic=True)
        print("torch.compile: dynamics compiled")
    compiled_model = model

    compiled_model.train()
    t0 = time.time()

    # Per-turn stats
    turn_stats = {t: {"loss": 0.0, "xform": 0.0, "count": 0} for t in range(n_turns)}

    for step in range(start_step, start_step + args.max_steps):
        optimizer.zero_grad()

        # Get next turn (auto-resets episodes at boundaries)
        (_, _, meta), is_start, is_end, turn_idx = dataset.get_next_turn_batch(device)

        # Reset persistent state at episode start
        if is_start:
            raw_m = model._orig_mod if hasattr(model, '_orig_mod') else model
            raw_m.persistent.reset()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            result = compiled_model(
                colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                roles=meta["roles"], sep_mask=meta["sep_mask"],
                sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                target_labels=meta["target_labels"],
                context_mask=meta["context_mask"],
                grid_ids=meta.get("grid_ids"), lengths=meta.get("lengths"),
                target_input_colors=meta.get("target_input_colors"),
            )

        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Track per-turn stats
        xf = result.get("transform_accuracy", torch.tensor(0.0))
        if isinstance(xf, torch.Tensor):
            xf = xf.item()
        turn_stats[turn_idx]["loss"] += result["loss"].item()
        turn_stats[turn_idx]["xform"] += xf
        turn_stats[turn_idx]["count"] += 1

        # Logging
        if step % args.log_every == 0:
            cv = result.get("metric_cv", torch.tensor(0.0))
            if isinstance(cv, torch.Tensor):
                cv = cv.item()

            persist_diag = raw_model.persistent.get_diagnostics() if hasattr(raw_model, 'persistent') else {}
            h_norm = persist_diag.get('persist_h_norm', 0.0)

            print(f"  [step={step}] turn={turn_idx} loss={result['loss'].item():.4f} "
                  f"xform={xf:.4f} cv={cv:.4f} h_norm={h_norm:.1f}")

            writer.add_scalar("loss/total", result["loss"].item(), step)
            writer.add_scalar("accuracy/xform", xf, step)
            writer.add_scalar("meta/turn_idx", turn_idx, step)
            writer.add_scalar("persist/h_norm", h_norm, step)

            # Per-turn breakdown
            if any(ts["count"] > 0 for ts in turn_stats.values()):
                print(f"  [per-turn] (last {args.log_every} steps)")
                for t in range(n_turns):
                    ts = turn_stats[t]
                    if ts["count"] > 0:
                        avg_xf = ts["xform"] / ts["count"]
                        avg_loss = ts["loss"] / ts["count"]
                        print(f"    turn {t}: xform={avg_xf*100:5.1f}%  loss={avg_loss:.4f}  n={ts['count']}")
                        writer.add_scalar(f"turn/xform_{t}", avg_xf, step)
                        writer.add_scalar(f"turn/loss_{t}", avg_loss, step)
                # Reset
                turn_stats = {t: {"loss": 0.0, "xform": 0.0, "count": 0} for t in range(n_turns)}

        # Eval
        if step > start_step and step % args.eval_every == 0:
            turn_accs, turn_xforms = evaluate_sequential(
                compiled_model, device, n_episodes=args.eval_episodes,
                n_vars=n_vars, total_ops=total_ops, ops_per_turn=ops_per_turn,
                seq_len=config.max_seq_len,
            )
            print(f"  >> EVAL [step={step}] per-turn xform:")
            for t in range(n_turns):
                print(f"    turn {t}: cell={turn_accs[t]:.4f}  xform={turn_xforms[t]:.4f}")
                writer.add_scalar(f"eval/turn_{t}_cell", turn_accs[t], step)
                writer.add_scalar(f"eval/turn_{t}_xform", turn_xforms[t], step)
            avg_xform = sum(turn_xforms) / len(turn_xforms)
            print(f"    AVG:    xform={avg_xform:.4f}")
            writer.add_scalar("eval/avg_xform", avg_xform, step)

            compiled_model.train()

        # Save
        if step > start_step and step % args.save_every == 0:
            path = os.path.join(args.output_dir, "checkpoints", f"step_{step}.pt")
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "persist_alpha": args.persist_alpha,
            }, path)

    print(f"\n  Training complete ({persist_label}). {args.max_steps} steps.")
    writer.close()


if __name__ == "__main__":
    main()
