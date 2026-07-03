"""Train a learned gripper-phase predictor from expert demos.

Goal: predict P(approach_phase) per state, where approach=1 if the expert's
gripper at this state is still -1 (open) before the first sign-flip. Trained
on memory_bank_v11.npz features (DINOv2(agent) || DINOv2(wrist) || state8).

At inference, gates the gripper clamp: when P(approach) > threshold, override
gripper to -1; else trust the model's prediction. This replaces the fixed
step<N clamp with a learned conditional one — fixes libero_10 regression while
preserving object/goal lifts.

Training data construction:
  For each sample in memory_bank_v11.npz:
    1. Look up its source suite from suite_idx, episode from (ep), step from t
    2. Find first_flip_step for that (suite, ep) from teacher_chunks
    3. label = 1 if t < first_flip_step else 0
  Roughly 50/50 balanced because expert demos spend ~half their time approaching.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SUITE_IDX_TO_NAME = {0: "libero_10", 1: "libero_goal", 2: "libero_object", 3: "libero_spatial"}


def compute_first_flip_per_episode(suite_dir: Path):
    """Returns dict: (suite_name, ep_idx) → first_flip_step.

    Loads teacher_chunks.dat + labels_index.npz. For each episode, finds the
    first step at which gripper flips sign from -1 to +1 (or never).
    """
    idx = np.load(suite_dir / "index.npz")
    starts = idx["episode_starts"]
    lengths = idx["episode_lengths"]
    task_indices = idx["task_indices"]
    n_episodes = len(lengths)
    labels = np.load(suite_dir / "labels_index.npz")
    sample_idx = labels["sample_idx"]
    n_samples = int(labels["n_samples"])

    K, A = 16, 7
    chunks = np.memmap(suite_dir / "teacher_chunks.dat", dtype=np.float32, mode="r",
                       shape=(n_samples, K, A))
    grip0 = chunks[:, 0, -1]  # [n_samples]

    first_flip = {}
    for ep_i in range(n_episodes):
        ep_mask = sample_idx[:, 0] == ep_i
        if not ep_mask.any():
            first_flip[ep_i] = -1
            continue
        ep_samples = np.where(ep_mask)[0]
        order = np.argsort(sample_idx[ep_samples, 1])
        ep_samples_sorted = ep_samples[order]
        ep_grips = grip0[ep_samples_sorted]
        ep_steps = sample_idx[ep_samples_sorted, 1]
        # First step at which grip != initial sign
        start_sign = next((s for s in np.sign(ep_grips) if abs(s) > 0), 0)
        ff = -1
        for i in range(1, len(ep_grips)):
            s_i = np.sign(ep_grips[i])
            if abs(s_i) > 0 and s_i != start_sign:
                ff = int(ep_steps[i])
                break
        first_flip[ep_i] = ff
    return first_flip


def build_dataset(memory_bank_path: Path, dataset_root: Path):
    print(f"[phase] loading memory_bank {memory_bank_path}")
    bank = np.load(memory_bank_path, allow_pickle=True)
    feats = bank["features"]              # [N, 776]
    suite_idx = bank["suite_idx"]         # [N]
    task = bank["task"]                   # [N] — task within suite
    ep = bank["ep"]                       # [N] — GLOBAL episode index (not within-task)
    t = bank["t"]                         # [N] — step in episode
    N = len(feats)
    print(f"[phase] bank: N={N}, d={feats.shape[1]}")

    # Per-suite: compute first_flip for each global episode in that suite
    per_suite_first_flip = {}
    for s_int, s_name in SUITE_IDX_TO_NAME.items():
        sd = dataset_root / f"libero-{s_name.split('_')[-1]}-expert-v1"
        if not sd.exists():
            print(f"  [skip] {s_name}: {sd} not found")
            continue
        t0 = time.time()
        ff = compute_first_flip_per_episode(sd)
        per_suite_first_flip[s_int] = ff
        print(f"  [{s_name}] computed first_flip for {len(ff)} episodes ({time.time()-t0:.1f}s)")

    # Build labels: phase=1 (approach) if t < first_flip[ep], else 0 (grasp/transport)
    labels = np.zeros(N, dtype=np.int8)
    for i in range(N):
        s_int = int(suite_idx[i])
        ep_i = int(ep[i])
        t_i = int(t[i])
        ff = per_suite_first_flip.get(s_int, {}).get(ep_i, -1)
        if ff < 0:
            # No flip detected — treat whole episode as approach
            labels[i] = 1
        else:
            labels[i] = 1 if t_i < ff else 0

    print(f"[phase] labels: approach={int((labels==1).sum())} grasp={int((labels==0).sum())}")
    return feats.astype(np.float32), labels.astype(np.int64)


class PhaseHead(nn.Module):
    def __init__(self, d_in=776, d_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--memory_bank", default="/home/pokazge/datasets/memory_bank_v11.npz")
    p.add_argument("--dataset_root", default="/home/pokazge/datasets")
    p.add_argument("--output", default="/tmp/phase_classifier.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.1)
    args = p.parse_args()

    feats, labels = build_dataset(Path(args.memory_bank), Path(args.dataset_root))
    N = len(feats)
    # Random split
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    val_n = int(N * args.val_frac)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    train_feats = torch.from_numpy(feats[train_idx])
    train_labels = torch.from_numpy(labels[train_idx]).float()
    val_feats = torch.from_numpy(feats[val_idx])
    val_labels = torch.from_numpy(labels[val_idx]).float()
    print(f"[phase] train: {len(train_idx)}  val: {len(val_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseHead(d_in=feats.shape[1], d_hidden=128).to(device)
    print(f"[phase] params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    val_feats = val_feats.to(device)
    val_labels = val_labels.to(device)

    n_train = len(train_idx)
    for epoch in range(args.epochs):
        # Shuffle
        perm = torch.randperm(n_train, device=device)
        train_feats_e = train_feats[perm]
        train_labels_e = train_labels[perm]
        loss_sum = 0.0; n_batches = 0
        for start in range(0, n_train, args.batch_size):
            end = min(start + args.batch_size, n_train)
            x = train_feats_e[start:end]
            y = train_labels_e[start:end]
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.detach())
            n_batches += 1
        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(val_feats)
            val_preds = (val_logits > 0).float()
            val_acc = (val_preds == val_labels).float().mean().item()
            val_loss = F.binary_cross_entropy_with_logits(val_logits, val_labels).item()
            # Per-class accuracy
            approach_mask = val_labels == 1
            grasp_mask = val_labels == 0
            acc_approach = (val_preds[approach_mask] == 1).float().mean().item() if approach_mask.any() else 0
            acc_grasp = (val_preds[grasp_mask] == 0).float().mean().item() if grasp_mask.any() else 0
        model.train()
        print(f"epoch {epoch:>2}  train_loss={loss_sum/max(n_batches,1):.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
              f"acc_approach={acc_approach:.3f}  acc_grasp={acc_grasp:.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "d_in": feats.shape[1], "d_hidden": 128,
        "n_train": n_train, "epochs": args.epochs,
    }, out_path)
    print(f"\n[phase] saved → {out_path}")


if __name__ == "__main__":
    main()
