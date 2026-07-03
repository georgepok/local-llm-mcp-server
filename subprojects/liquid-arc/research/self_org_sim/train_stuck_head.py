"""Train a stuck-detection head on h_goal trajectories collected from chained
LIBERO rollouts. Input: h_goal at chunk t [K, d] + delta h_goal_t - h_goal_{t-W}.
Output: P(sub-task will fail). BCE loss.

Substrate body is frozen. Only the new head_stuck is trained (small MLP).

Usage:
  python train_stuck_head.py \
    --traj_files /tmp/traj_libero10_s10.pt,/tmp/traj_libero10_s20.pt,... \
    --substrate_ckpt /tmp/substrate_dynamics_corr.pt \
    --output /tmp/substrate_stuck.pt
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from liquid_goal_tracker_proprio import JEPA_LGT_Proprio  # type: ignore

torch.set_float32_matmul_precision("high")


class StuckHead(nn.Module):
    """Small MLP: (h_goal [K,d] + delta_h_goal [K,d]) -> P(stuck) logit.

    Predicts whether the current sub-task will fail, from substrate belief state
    + change over the last window. Bounded ±5 logits via tanh*5 to keep gradients
    well-behaved.
    """
    def __init__(self, K: int, d: int, hidden: int = 64):
        super().__init__()
        in_dim = 2 * K * d
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, h_goal: torch.Tensor, h_goal_delta: torch.Tensor) -> torch.Tensor:
        """h_goal: [B, K, d], h_goal_delta: [B, K, d] -> [B, 1] logit"""
        x = torch.cat([h_goal.flatten(1), h_goal_delta.flatten(1)], dim=-1)
        return self.net(x).squeeze(-1)


def build_samples(records: list, window: int, min_chunks: int):
    """Per record, emit (h_goal[t], h_goal[t]-h_goal[t-W], label, traj_id) samples.
    traj_id is the index of the source trajectory — used downstream to split
    train/val at trajectory level (avoid leakage).
    label = 1 if sub-task failed (succ=False), 0 if succeeded.
    """
    feats_h = []
    feats_delta = []
    labels = []
    traj_ids = []
    for ri, r in enumerate(records):
        traj = r["h_goal_traj"]  # [T, K, d]
        T = traj.shape[0]
        if T < max(window + 1, min_chunks):
            continue
        succ = bool(r["succ"])
        label = 0.0 if succ else 1.0
        # For failed sub-tasks, only count the SECOND HALF as stuck (early chunks
        # could still be making progress). For succeeded ones, all windows count.
        if succ:
            ts = list(range(window, T))
        else:
            half = max(window, T // 2)
            ts = list(range(half, T))
        for t in ts:
            h_t = traj[t - 1].float()
            h_tw = traj[t - 1 - window].float()
            feats_h.append(h_t)
            feats_delta.append(h_t - h_tw)
            labels.append(label)
            traj_ids.append(ri)
    if not feats_h:
        return None
    return (torch.stack(feats_h, dim=0),
            torch.stack(feats_delta, dim=0),
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(traj_ids, dtype=torch.long))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_files", required=True,
                   help="Comma-separated paths to /tmp/traj_*.pt collection files")
    p.add_argument("--substrate_ckpt", required=True)
    p.add_argument("--output", default="/tmp/substrate_stuck.pt")
    p.add_argument("--window", type=int, default=10,
                   help="Number of chunks back to compute delta h_goal")
    p.add_argument("--min_chunks", type=int, default=15,
                   help="Skip sub-tasks shorter than this many chunks")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--pos_weight", type=float, default=2.0,
                   help="BCE positive (stuck) class weight (failed sub-tasks "
                        "typically fewer than successful ones; upweight to balance)")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stuck] device={device}, output={args.output}", flush=True)

    # Load substrate to get K, d, and base config
    ck = torch.load(args.substrate_ckpt, map_location=device, weights_only=False)
    sa = ck["args"]
    d_sub = sa.get("d_substrate", 64)
    K_bel = sa.get("K_belief", 4)
    substrate = JEPA_LGT_Proprio(
        z_vl_dim=ck["z_vl_dim"], action_dim=ck["action_dim"],
        horizon=ck["horizon"], state_dim=ck["state_dim"],
        d=d_sub, K=K_bel, n_tok_per_k=sa.get("n_tok_per_k", 1),
    ).to(device)
    substrate.load_state_dict(ck["substrate_state_dict"], strict=False)
    for pp in substrate.parameters():
        pp.requires_grad = False
    print(f"[stuck] substrate K={K_bel} d={d_sub}", flush=True)

    # Load and combine trajectory files
    all_records = []
    for fp in [x.strip() for x in args.traj_files.split(",") if x.strip()]:
        ck_t = torch.load(fp, map_location="cpu", weights_only=False)
        all_records.extend(ck_t["records"])
        print(f"  {fp}: {len(ck_t['records'])} records", flush=True)
    n_total = len(all_records)
    n_succ = sum(1 for r in all_records if r["succ"])
    print(f"[stuck] {n_total} sub-task records ({n_succ} succ, "
          f"{n_total - n_succ} fail) — base rate {n_succ/n_total:.2f}", flush=True)

    # Build (feat, label, traj_id) samples
    samples = build_samples(all_records, args.window, args.min_chunks)
    if samples is None:
        print("[stuck] ERROR: no usable samples after filtering", flush=True)
        return
    h_feats, d_feats, labels, traj_ids = samples
    n_samples = h_feats.shape[0]
    pos_frac = float(labels.mean())
    print(f"[stuck] {n_samples} (window-end, label) samples; "
          f"positive (stuck) frac {pos_frac:.3f}", flush=True)

    # Train/val split AT TRAJECTORY LEVEL (samples from same sub-task stay together).
    # Otherwise multiple samples from one sub-task can leak across train/val.
    rng = np.random.default_rng(42)
    unique_trajs = traj_ids.unique().numpy()
    rng.shuffle(unique_trajs)
    n_val_traj = max(1, int(args.val_frac * len(unique_trajs)))
    val_traj_set = set(unique_trajs[:n_val_traj].tolist())
    train_traj_set = set(unique_trajs[n_val_traj:].tolist())
    traj_np = traj_ids.numpy()
    val_idx = np.where(np.isin(traj_np, list(val_traj_set)))[0]
    train_idx = np.where(np.isin(traj_np, list(train_traj_set)))[0]
    print(f"[stuck] trajectory-level split: {len(train_traj_set)} train traj "
          f"({len(train_idx)} samples), {len(val_traj_set)} val traj "
          f"({len(val_idx)} samples)", flush=True)

    # To device
    h_feats = h_feats.to(device)
    d_feats = d_feats.to(device)
    labels = labels.to(device)

    head = StuckHead(K_bel, d_sub, hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[stuck] StuckHead with {n_params:,} params", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    pos_w = torch.tensor([args.pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    t_start = time.time()
    best_val_auc = 0.0
    rolling_loss = []
    for step in range(args.max_steps):
        b = train_idx[rng.choice(len(train_idx), args.batch_size, replace=False)]
        h_b = h_feats[b]; d_b = d_feats[b]; y_b = labels[b]
        logit = head(h_b, d_b)
        loss = loss_fn(logit, y_b)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
        opt.step()
        rolling_loss.append(float(loss))
        rolling_loss = rolling_loss[-100:]

        if step % args.log_every == 0:
            with torch.no_grad():
                vh = h_feats[val_idx]; vd = d_feats[val_idx]; vy = labels[val_idx]
                vlogit = head(vh, vd)
                vloss = loss_fn(vlogit, vy).item()
                vprob = torch.sigmoid(vlogit)
                # AUC via numpy
                vy_np = vy.cpu().numpy(); vp_np = vprob.cpu().numpy()
                # Simple AUC computation (rank-based)
                order = np.argsort(-vp_np)
                ys = vy_np[order]
                tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
                tpr = tp / max(ys.sum(), 1); fpr = fp / max((1 - ys).sum(), 1)
                # Trapezoidal AUC
                auc = float(np.trapz(tpr, fpr))
                if auc > best_val_auc:
                    best_val_auc = auc
                # Threshold 0.5 accuracy
                pred05 = (vp_np > 0.5).astype(np.float32)
                acc = float((pred05 == vy_np).mean())
                # Precision/Recall at default threshold
                tp05 = int(((pred05 == 1) & (vy_np == 1)).sum())
                fp05 = int(((pred05 == 1) & (vy_np == 0)).sum())
                fn05 = int(((pred05 == 0) & (vy_np == 1)).sum())
                prec = tp05 / max(tp05 + fp05, 1)
                rec = tp05 / max(tp05 + fn05, 1)
            print(f"step {step:>4}  train_loss={np.mean(rolling_loss):.4f}  "
                  f"val_loss={vloss:.4f}  val_AUC={auc:.3f} (best {best_val_auc:.3f})  "
                  f"acc@0.5={acc:.3f}  P={prec:.3f}  R={rec:.3f}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)

    # Final save
    torch.save({
        "substrate_state_dict": substrate.state_dict(),
        "stuck_head_state_dict": head.state_dict(),
        "stuck_head_config": {"K": K_bel, "d": d_sub, "hidden": args.hidden,
                              "window": args.window},
        "args": vars(args), "step": args.max_steps,
        "z_vl_dim": ck["z_vl_dim"], "action_dim": ck["action_dim"],
        "horizon": ck["horizon"], "state_dim": ck["state_dim"],
        "dist_mean": ck["dist_mean"], "dist_std": ck["dist_std"],
        "sd_mean": ck["sd_mean"], "sd_std": ck["sd_std"],
        "val_AUC": best_val_auc,
    }, args.output)
    print(f"\n[stuck] saved → {args.output}  best_val_AUC={best_val_auc:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
