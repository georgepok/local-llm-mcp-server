"""Train HyperNetwork via distillation from gradient-based TTT.

For each ARC training task:
  1. Run full gradient TTT → get adapted model weights
  2. Compute target deltas: adapted.weight - base.weight
  3. HyperNetwork predicts deltas from demo embeddings
  4. Loss = MSE(predicted, target) per module
  5. Backprop through hypernetwork only

Usage:
    python scripts/train_hypernet.py \
        --base_checkpoint output_ttt_v2/checkpoints/best.pt \
        --config configs/liquid_arc_ttt_v3c.yaml \
        --data_dir data/arc \
        --output_dir output_hypernet
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import fgn-v3 for ARC data
FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if FGN_ROOT not in sys.path:
    sys.path.insert(0, FGN_ROOT)

from fgn.tasks.arc import (
    load_arc_tasks, build_sequence, pad_single_to_batch,
    ROLE_INPUT_DEMO, ROLE_OUTPUT_DEMO,
)

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import create_model, LiquidARCModel
from liquid_arc.hypernet import HyperNetwork, _get_melt_module_specs
from liquid_arc.ttt import test_time_adapt


def compute_target_deltas(
    base_model: LiquidARCModel,
    adapted_model: LiquidARCModel,
    include_ffn: bool = False,
) -> dict:
    """Compute weight deltas between adapted and base model.

    Returns:
        Dict mapping module_name → {"weight": ΔW, "bias": Δb}
    """
    base_specs = dict(_get_melt_module_specs(base_model, include_ffn=include_ffn))
    adapted_specs = dict(_get_melt_module_specs(adapted_model, include_ffn=include_ffn))

    deltas = {}
    for name in base_specs:
        d = {}
        d["weight"] = adapted_specs[name].weight.data - base_specs[name].weight.data
        if (base_specs[name].bias is not None and adapted_specs[name].bias is not None):
            d["bias"] = adapted_specs[name].bias.data - base_specs[name].bias.data
        deltas[name] = d
    return deltas


def main():
    parser = argparse.ArgumentParser(description="Train HyperNetwork via distillation")
    parser.add_argument("--base_checkpoint", type=str, required=True,
                        help="Path to pre-trained base model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (overrides checkpoint config)")
    parser.add_argument("--data_dir", type=str, default="data/arc",
                        help="Path to ARC data directory")
    parser.add_argument("--output_dir", type=str, default="output_hypernet",
                        help="Output directory for checkpoints")
    parser.add_argument("--n_tasks", type=int, default=None,
                        help="Limit number of training tasks (default: all)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override hypernet learning rate")
    parser.add_argument("--n_epochs", type=int, default=3,
                        help="Number of epochs over training tasks")
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save checkpoint every N tasks")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load base model checkpoint
    ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    if isinstance(config, dict):
        config = LiquidARCConfig(**config)

    if args.config:
        config = LiquidARCConfig.from_yaml(args.config)

    # No torch.compile for deepcopy
    config.use_torch_compile = False

    base_model = create_model(config, device)
    state_dict = ckpt["model"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    base_model.load_state_dict(cleaned)
    base_model.eval()

    if not isinstance(base_model, LiquidARCModel):
        print("ERROR: HyperNetwork requires LiquidARCModel")
        sys.exit(1)

    # Create hypernetwork
    hypernet = HyperNetwork(config, base_model).to(device)
    n_hyper_params = sum(p.numel() for p in hypernet.parameters())
    print(f"HyperNetwork params: {n_hyper_params:,}")

    lr = args.lr or config.hypernet_distill_lr
    optimizer = torch.optim.AdamW(hypernet.parameters(), lr=lr, weight_decay=1e-4)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Load training tasks
    all_tasks = load_arc_tasks(args.data_dir)
    train_tasks = all_tasks.get("train", [])
    if not train_tasks:
        print("ERROR: No training tasks found")
        sys.exit(1)

    if args.n_tasks is not None:
        train_tasks = train_tasks[:args.n_tasks]
    print(f"Training tasks: {len(train_tasks)}")

    include_ffn = config.hypernet_include_ffn

    # Distillation training loop
    global_step = 0
    log_entries = []
    t0 = time.time()

    for epoch in range(args.n_epochs):
        epoch_loss = 0.0
        epoch_count = 0
        skipped = 0

        for task_i, task in enumerate(train_tasks):
            task_id = task.get("task_id", f"task_{task_i}")

            # Step 1: Run gradient-based TTT to get adapted model
            result, adapted = test_time_adapt(
                base_model, task, config, device,
                ttt_steps=config.ttt_steps,
                ttt_lr=config.ttt_lr,
                verbose=False,
            )

            if result.get("skipped", False) or adapted is None:
                skipped += 1
                continue

            # Step 2: Compute target deltas
            target_deltas = compute_target_deltas(
                base_model, adapted, include_ffn=include_ffn,
            )

            # Free adapted model (keep only deltas)
            del adapted

            # Step 3: Hypernetwork forward — predict deltas from demo embeddings
            seq = build_sequence(task, d4_idx=0, color_perm=None, test_idx=0,
                                 max_seq_len=config.max_seq_len)
            if seq is None:
                skipped += 1
                continue

            meta = pad_single_to_batch(seq, config.max_seq_len, device)

            task_embed = hypernet.encode_task(
                base_model, meta, device,
                role_input_demo=ROLE_INPUT_DEMO,
                role_output_demo=ROLE_OUTPUT_DEMO,
            )
            pred_deltas = hypernet(task_embed)

            # Step 4: MSE loss per module
            loss = torch.tensor(0.0, device=device)
            n_modules = 0
            for name in pred_deltas:
                if name not in target_deltas:
                    continue
                if "weight" in pred_deltas[name] and "weight" in target_deltas[name]:
                    target_w = target_deltas[name]["weight"].to(device)
                    pred_w = pred_deltas[name]["weight"]
                    loss = loss + torch.nn.functional.mse_loss(pred_w, target_w)
                    n_modules += 1
                if "bias" in pred_deltas[name] and "bias" in target_deltas[name]:
                    target_b = target_deltas[name]["bias"].to(device)
                    pred_b = pred_deltas[name]["bias"]
                    loss = loss + torch.nn.functional.mse_loss(pred_b, target_b)

            if n_modules == 0:
                skipped += 1
                continue
            loss = loss / n_modules  # average across modules

            # Step 5: Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hypernet.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_count += 1
            global_step += 1

            if args.verbose and global_step % 10 == 0:
                print(f"  [{epoch+1}/{args.n_epochs}] step {global_step} "
                      f"task={task_id}: loss={loss_val:.6f}")

            log_entries.append({
                "step": global_step,
                "epoch": epoch + 1,
                "task_id": task_id,
                "loss": loss_val,
            })

            # Save checkpoint periodically
            if global_step % args.save_every == 0:
                ckpt_path = output_dir / "checkpoints" / f"step_{global_step}.pt"
                torch.save({
                    "hypernet": hypernet.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "config": config.__dict__,
                }, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

        avg_loss = epoch_loss / max(epoch_count, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.n_epochs}: avg_loss={avg_loss:.6f}, "
              f"tasks={epoch_count} ({skipped} skipped), {elapsed:.1f}s total")

    # Save final checkpoint
    final_path = output_dir / "checkpoints" / "final.pt"
    torch.save({
        "hypernet": hypernet.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": global_step,
        "config": config.__dict__,
    }, final_path)
    print(f"Final checkpoint: {final_path}")

    # Save training log
    log_path = output_dir / "distill_log.json"
    with open(log_path, "w") as f:
        json.dump(log_entries, f, indent=2)
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
