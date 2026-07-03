"""V11: train GR00T-side complexity head — predicts struggle-horizon from z_vl.

Given the current scene+language fusion (z_vl), predict how many chunks from
now until Liquid will signal struggle (forward-model prediction error > τ).
Provides anticipatory cadence advice that complements V10's reactive struggle
trigger.

Pipeline:
1. Run V9-offline checkpoint forward over the dataset, collect cond_t + chunk_t
2. Compute predicted cond_{t+1} via forward_pred_head + actual cond_{t+1}
3. Per-chunk prediction error → struggle indicator at threshold τ
4. For each chunk t: target_horizon[t] = first chunk j > t in same episode
   where struggle indicator true; clipped to [1, K_max]
5. Train small MLP: z_vl → log(1 + horizon) regression
6. Save head weights for groot_server to load

Run on Spark in main venv:
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
  python train_complexity_head.py \\
    --base_ckpt /tmp/distill_s1s2_v9_offline/step_008000.pt \\
    --data_dir /home/pokazge/datasets/groot-sim-tempquery-v1 \\
    --out_path /tmp/complexity_head_v11.pt \\
    --struggle_threshold 0.05 --k_max 16 --max_steps 3000
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot import TeacherLabelDataset, collate_batch
from distill_groot_flow import LiquidFlowPolicy
from train_forward_head import load_v7b_with_forward_head, precompute_conds

print = functools.partial(print, flush=True)
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
except Exception:
    pass


def compute_prediction_errors(model, conds, chunks, dataset, device):
    """For each sample i: error_i = ‖predict(cond_i, chunk_i.mean) - cond_{i+1}‖²
    if i+1 in same episode, else 0 (no struggle indicator at episode end).
    """
    n = len(dataset)
    errors = np.zeros(n, dtype=np.float32)
    same_episode = np.zeros(n, dtype=bool)

    si = dataset.sample_idx
    for i in range(n - 1):
        same_episode[i] = (int(si[i + 1][0]) == int(si[i][0]))

    head = model.encoder.forward_pred_head
    head.eval()
    conds_t = torch.from_numpy(conds).to(device)
    chunks_mean_t = torch.from_numpy(chunks.mean(axis=1)).to(device)

    batch_size = 1024
    with torch.no_grad():
        for start in range(0, n - 1, batch_size):
            end = min(start + batch_size, n - 1)
            same = same_episode[start:end]
            valid_idx = np.where(same)[0] + start
            if len(valid_idx) == 0:
                continue
            cond_t = conds_t[valid_idx]
            chunk_mean = chunks_mean_t[valid_idx]
            x = torch.cat([cond_t, chunk_mean], dim=-1)
            pred = head(x)
            cond_t1 = conds_t[valid_idx + 1]
            err = ((pred - cond_t1) ** 2).mean(dim=-1).cpu().numpy()
            errors[valid_idx] = err

    print(f"[errors] computed {n - 1} pair errors. "
          f"mean={errors[same_episode].mean():.4f}  "
          f"p50={np.percentile(errors[same_episode], 50):.4f}  "
          f"p90={np.percentile(errors[same_episode], 90):.4f}  "
          f"p99={np.percentile(errors[same_episode], 99):.4f}")
    return errors, same_episode


def compute_horizons(dataset, errors, same_episode, threshold, k_max):
    """For each chunk i: horizon_i = number of chunks until next struggle, or k_max."""
    n = len(dataset)
    horizons = np.full(n, k_max, dtype=np.int64)
    si = dataset.sample_idx

    for ep_i in range(len(dataset.starts)):
        start = int(dataset.starts[ep_i])
        length = int(dataset.lengths[ep_i])
        # Find struggle indices within this episode
        struggle_local = np.array(
            [j for j in range(length)
             if start + j < n and same_episode[start + j] and errors[start + j] > threshold],
            dtype=np.int64,
        )
        for j in range(length):
            # First struggle at local idx > j
            future = struggle_local[struggle_local > j]
            if len(future) > 0:
                h = int(future[0]) - j
            else:
                h = k_max
            horizons[start + j] = min(h, k_max)

    print(f"[horizons] mean={horizons.mean():.2f}  "
          f"p50={np.percentile(horizons, 50):.0f}  "
          f"min={horizons.min()}  max={horizons.max()}")
    return horizons


def train_complexity_head(z_vl, horizons, device, max_steps=3000,
                          batch_size=256, lr=1e-3, hidden_dim=512):
    """Train small MLP: z_vl → log(1 + horizon)."""
    z_vl_dim = z_vl.shape[1]
    head = nn.Sequential(
        nn.Linear(z_vl_dim, hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim, 1),
    ).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=lr * 0.05,
    )

    z_vl_t = torch.from_numpy(z_vl).to(device)
    target_t = torch.log1p(torch.from_numpy(horizons).float().to(device))

    n = z_vl.shape[0]
    print(f"[train head] {max_steps} steps, batch={batch_size}, lr={lr}, "
          f"head params={sum(p.numel() for p in head.parameters()):,}")
    t0 = time.time()
    losses = []
    for step in range(max_steps):
        sel = torch.randint(0, n, (batch_size,), device=device)
        x = z_vl_t[sel]
        y = target_t[sel]
        pred = head(x).squeeze(-1)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if step % 200 == 0:
            recent = float(np.mean(losses[-100:]))
            print(f"  step {step:5d}/{max_steps}  loss={loss.item():.5f}  "
                  f"avg100={recent:.5f}  ({time.time()-t0:.1f}s)")
    print(f"[train head] done. final loss = {np.mean(losses[-100:]):.5f}")

    # Sanity: test on a fresh batch
    with torch.no_grad():
        sel = torch.randint(0, n, (1024,), device=device)
        pred_horizons = (torch.expm1(head(z_vl_t[sel]).squeeze(-1)).clamp(1, 16)).cpu().numpy()
        actual_horizons = horizons[sel.cpu().numpy()]
        corr = float(np.corrcoef(pred_horizons, actual_horizons)[0, 1])
    print(f"[train head] sanity: pred-actual correlation = {corr:.3f}, "
          f"mean pred horizon = {pred_horizons.mean():.2f}, "
          f"mean actual horizon = {actual_horizons.mean():.2f}")
    return head


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", required=True, type=str,
                   help="V9-offline checkpoint with calibrated forward_pred_head")
    p.add_argument("--data_dir", required=True, type=str)
    p.add_argument("--out_path", required=True, type=str)
    p.add_argument("--struggle_threshold", type=float, default=0.05)
    p.add_argument("--k_max", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # Load V9-offline (V7b + trained forward_pred_head)
    model, ckpt = load_v7b_with_forward_head(Path(args.base_ckpt), device)
    sa = ckpt["args"]

    ds = TeacherLabelDataset(
        Path(args.data_dir), Path(args.data_dir),
        action_horizon=sa["action_horizon"],
        return_task_id=False,
        target_img_size=sa["img_size"],
        return_z_groot=sa.get("use_z_groot", "z_vl"),
        cadence_dropout=None,
        use_query_bank=sa.get("use_query_bank", False),
    )
    n_samples = ds.n_samples
    print(f"[dataset] {n_samples} samples")

    # Phase 1: precompute V9 conds
    conds = precompute_conds(model, ds, device, batch_size=256, num_workers=8)

    # Load chunks via memmap
    chunks = np.array(np.memmap(
        Path(args.data_dir) / "teacher_chunks.dat",
        dtype=np.float32, mode="r",
        shape=(n_samples, sa["action_horizon"], 7),
    ))
    # Load z_vl (the GR00T-side input feature) — used as complexity head input
    z_vl = np.array(np.memmap(
        Path(args.data_dir) / "z_vl.dat",
        dtype=np.float32, mode="r",
        shape=(n_samples, 2048),
    ))

    # Phase 2: prediction errors
    errors, same_episode = compute_prediction_errors(
        model, conds, chunks, ds, device,
    )

    # Phase 3: horizons
    horizons = compute_horizons(
        ds, errors, same_episode, args.struggle_threshold, args.k_max,
    )

    # Phase 4: train complexity head
    head = train_complexity_head(
        z_vl, horizons, device,
        max_steps=args.max_steps, batch_size=args.batch_size,
    )

    # Save
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state_dict": head.state_dict(),
        "z_vl_dim": int(z_vl.shape[1]),
        "k_max": args.k_max,
        "struggle_threshold": args.struggle_threshold,
    }, out)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
