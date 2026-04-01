"""Train output correction network on frozen base model features.

The base model is frozen. A small correction net (~50K params) learns to
refine the base predictions by adding correction logits.

Usage:
    python scripts/train_correction.py \
        --base_checkpoint output_30to50/checkpoints/best.pt \
        --config configs/liquid_arc_zero_scaffold.yaml \
        --data_dir /workspace/fgn-v3/data/arc-repo/data
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import create_model
from liquid_arc.correction_net import OutputCorrectionNet

FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if not Path(FGN_ROOT).exists():
    FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, FGN_ROOT)
from fgn.tasks.arc import ARCTask


def evaluate(base_model, corr_net, eval_task, device, config, n_batches=20):
    """Evaluate base + correction on eval set."""
    total_xform_correct_base = 0
    total_xform_correct_corr = 0
    total_xform = 0
    total_solved_base = 0
    total_solved_corr = 0
    total_tasks = 0

    with torch.no_grad():
        for _ in range(n_batches):
            _, _, meta = eval_task.generate_batch(8, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = base_model(
                    colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                    roles=meta["roles"], sep_mask=meta["sep_mask"],
                    sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                    target_labels=meta["target_labels"],
                    context_mask=meta["context_mask"],
                    grid_ids=meta.get("grid_ids"), lengths=meta.get("lengths"),
                    target_input_colors=meta.get("target_input_colors"),
                )

            hidden = result["h_final"].detach().float()
            base_logits = result["logits"].detach().float()
            base_pred = base_logits.argmax(dim=-1)

            correction = corr_net(hidden, base_pred)
            final_logits = base_logits + correction
            final_pred = final_logits.argmax(dim=-1)

            tgt = meta["target_labels"]
            inp = meta.get("target_input_colors")
            valid = tgt != -100

            if inp is not None:
                xform = valid & (tgt != inp)
                n_xf = xform.sum().item()
                total_xform_correct_base += (base_pred[xform] == tgt[xform]).sum().item()
                total_xform_correct_corr += (final_pred[xform] == tgt[xform]).sum().item()
                total_xform += n_xf

            # Per-sample solve check
            B = tgt.shape[0]
            for b in range(B):
                valid_b = valid[b]
                if valid_b.sum() == 0:
                    continue
                total_tasks += 1
                base_correct = (base_pred[b][valid_b] == tgt[b][valid_b]).all().item()
                corr_correct = (final_pred[b][valid_b] == tgt[b][valid_b]).all().item()
                total_solved_base += int(base_correct)
                total_solved_corr += int(corr_correct)

    base_acc = total_xform_correct_base / max(total_xform, 1)
    corr_acc = total_xform_correct_corr / max(total_xform, 1)
    return {
        "base_xform": base_acc,
        "corr_xform": corr_acc,
        "base_solved": total_solved_base,
        "corr_solved": total_solved_corr,
        "total_tasks": total_tasks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="output_correction_net")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--batches_per_epoch", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LiquidARCConfig.from_yaml(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load frozen base model
    print(f"Loading base model from {args.base_checkpoint}...")
    base_model = create_model(config, device)
    ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    state = {k.replace("._orig_mod.", "."): v for k, v in ckpt["model"].items()}
    base_model.load_state_dict(state)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    print(f"  Loaded at step {ckpt.get('step', '?')}")

    # Create correction net
    corr_net = OutputCorrectionNet(d_model=config.d_model).to(device)
    n_params = sum(p.numel() for p in corr_net.parameters())
    print(f"  Correction net: {n_params:,} params")

    optimizer = torch.optim.AdamW(corr_net.parameters(), lr=args.lr)

    # ARC data
    train_task = ARCTask(seq_len=config.max_seq_len, data_dir=args.data_dir,
                         split="train", augment=True, n_color_perms=10)
    eval_task = ARCTask(seq_len=config.max_seq_len, data_dir=args.data_dir,
                        split="eval", augment=False)

    print(f"  Training: {args.n_epochs} epochs × {args.batches_per_epoch} batches")
    print(f"  Batch size: {args.batch_size}, LR: {args.lr}")

    best_eval = 0.0
    t0 = time.time()

    for epoch in range(args.n_epochs):
        corr_net.train()
        total_loss = 0
        n_batches = 0

        for _ in range(args.batches_per_epoch):
            _, _, meta = train_task.generate_batch(args.batch_size, device=device)

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=(device.type == "cuda")):
                    result = base_model(
                        colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                        roles=meta["roles"], sep_mask=meta["sep_mask"],
                        sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                        target_labels=meta["target_labels"],
                        context_mask=meta["context_mask"],
                        grid_ids=meta.get("grid_ids"), lengths=meta.get("lengths"),
                        target_input_colors=meta.get("target_input_colors"),
                    )
                hidden = result["h_final"].detach().float()
                base_logits = result["logits"].detach().float()
                base_pred = base_logits.argmax(dim=-1)

            # Correction forward (trainable)
            correction = corr_net(hidden, base_pred)
            final_logits = base_logits + correction

            # Loss on target positions with transform weighting
            tgt = meta["target_labels"]
            valid = tgt != -100
            if valid.sum() == 0:
                continue

            inp = meta.get("target_input_colors")
            if inp is not None:
                changed = valid & (tgt != inp)
                flat_valid = valid.view(-1)
                flat_changed = changed.view(-1)[flat_valid]
                weights = torch.where(flat_changed, 5.0, 0.05)
                per_cell = F.cross_entropy(
                    final_logits.view(-1, 10)[flat_valid],
                    tgt.view(-1)[flat_valid],
                    reduction='none')
                loss = (per_cell * weights).sum() / weights.sum()
            else:
                loss = F.cross_entropy(
                    final_logits.view(-1, 10)[valid.view(-1)],
                    tgt.view(-1)[valid.view(-1)])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(corr_net.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Eval every 10 epochs
        if epoch % 10 == 0 or epoch == args.n_epochs - 1:
            corr_net.eval()
            ev = evaluate(base_model, corr_net, eval_task, device, config)
            if ev["corr_xform"] > best_eval:
                best_eval = ev["corr_xform"]
                torch.save(corr_net.state_dict(),
                           os.path.join(args.output_dir, "best.pt"))
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}: loss={total_loss/max(n_batches,1):.4f}, "
                  f"base_xf={ev['base_xform']*100:.1f}%, "
                  f"corr_xf={ev['corr_xform']*100:.1f}% "
                  f"(Δ={'+' if ev['corr_xform']>=ev['base_xform'] else ''}"
                  f"{(ev['corr_xform']-ev['base_xform'])*100:.1f}%), "
                  f"solved: {ev['base_solved']}/{ev['corr_solved']}/{ev['total_tasks']}, "
                  f"best={best_eval*100:.1f}%, "
                  f"{elapsed:.0f}s")

    # Final report
    corr_net.eval()
    final = evaluate(base_model, corr_net, eval_task, device, config, n_batches=40)
    print(f"\n{'='*60}")
    print(f"Final Results (40 eval batches)")
    print(f"{'='*60}")
    print(f"Base model:  xform={final['base_xform']*100:.1f}%, "
          f"solved={final['base_solved']}/{final['total_tasks']}")
    print(f"Corrected:   xform={final['corr_xform']*100:.1f}%, "
          f"solved={final['corr_solved']}/{final['total_tasks']}")
    delta = final['corr_xform'] - final['base_xform']
    print(f"Delta:       {'+' if delta>=0 else ''}{delta*100:.1f}% xform, "
          f"{'+' if final['corr_solved']>=final['base_solved'] else ''}"
          f"{final['corr_solved']-final['base_solved']} tasks solved")

    # Save report
    report = (
        f"# Correction Net Results\n\n"
        f"Base checkpoint: {args.base_checkpoint}\n"
        f"Correction params: {n_params:,}\n"
        f"Training: {args.n_epochs} epochs × {args.batches_per_epoch} batches\n\n"
        f"| Metric | Base | Corrected | Delta |\n"
        f"|--------|------|-----------|-------|\n"
        f"| Xform acc | {final['base_xform']*100:.1f}% | {final['corr_xform']*100:.1f}% | "
        f"{'+' if delta>=0 else ''}{delta*100:.1f}% |\n"
        f"| Tasks solved | {final['base_solved']}/{final['total_tasks']} | "
        f"{final['corr_solved']}/{final['total_tasks']} | "
        f"{'+' if final['corr_solved']>=final['base_solved'] else ''}"
        f"{final['corr_solved']-final['base_solved']} |\n"
    )
    with open(os.path.join(args.output_dir, "CORRECTION_NET_REPORT.md"), "w") as f:
        f.write(report)
    print(f"\nReport saved to {args.output_dir}/CORRECTION_NET_REPORT.md")


if __name__ == "__main__":
    main()
