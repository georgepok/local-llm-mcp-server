"""Train a task-end-proximity head: predict P(sub-task ends within K_END chunks)
from substrate's h_goal at chunk t + h_goal_delta over window W.

Identified by probe_task_signals.py as the strongest learnable signal:
  (h_goal_t [K,d] + h_delta_W10 [K,d]) → success_within_5_chunks
  val_AUC = 0.956 on held-out trajectories

This is what Liquid OPTIMALLY tracks across transformer turns: it knows when the
current sub-task is approaching completion (success OR failure). The signal is
substrate's belief EVOLUTION over a chunk window, not its instantaneous state.

Substrate body frozen. Only the new head_end_proximity is trained.

Usage on Spark:
  python train_end_proximity_head.py \
    --traj_files /tmp/traj_libero10_s10.pt,/tmp/traj_libero10_s20.pt,... \
    --substrate_ckpt /tmp/substrate_projection.pt \
    --output /tmp/substrate_end_proximity.pt
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


class EndProximityHead(nn.Module):
    """MLP head: (h_goal_t [K,d] + h_delta_W [K,d]) → P(ends within K_END chunks).

    Output: single sigmoid logit. Architecture identical to stuck-head; differs
    only in training target (per probe: success_within_5 is much more learnable
    than success_binary).
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
        x = torch.cat([h_goal.flatten(1), h_goal_delta.flatten(1)], dim=-1)
        return self.net(x).squeeze(-1)


def build_samples(records, window: int, k_end: int, min_chunks: int):
    """Per record, emit per-timestep (h_goal[t], h_goal_delta_W, label, traj_id).

    label = 1 if sub-task ends successfully within k_end chunks of t, else 0.
    Includes both successful and failed sub-tasks — the head learns to flag
    "approaching success", not just "stuck-or-not".

    Skip records too short for window. For each t >= window, compute features +
    label. Even failed sub-tasks contribute negative samples (label=0 everywhere).
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
        for t in range(window, T):
            h_t = traj[t - 1].float()
            h_tw = traj[t - 1 - window].float()
            chunks_remaining = T - t
            label = 1.0 if (succ and chunks_remaining <= k_end) else 0.0
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


def auc_from_probs(y, p):
    """Trapezoidal AUC."""
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    tpr = tp / max(ys.sum(), 1); fpr = fp / max((1 - ys).sum(), 1)
    return float(np.trapezoid(tpr, fpr))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_files", required=True)
    p.add_argument("--substrate_ckpt", required=True)
    p.add_argument("--output", default="/tmp/substrate_end_proximity.pt")
    p.add_argument("--window", type=int, default=10,
                   help="Chunks back to compute h_goal_delta (W=10 per probe)")
    p.add_argument("--k_end", type=int, default=5,
                   help="Chunks-from-end threshold for positive label")
    p.add_argument("--min_chunks", type=int, default=15)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--pos_weight", type=float, default=4.0,
                   help="BCE positive class weight (base rate ~12-15%)")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--early_stop_patience", type=int, default=8,
                   help="Stop if no val_AUC improvement for N log intervals")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[end-prox] device={device}, output={args.output}, "
          f"W={args.window}, K_END={args.k_end}", flush=True)

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
    print(f"[end-prox] substrate frozen, K={K_bel} d={d_sub}", flush=True)

    all_records = []
    for fp in [x.strip() for x in args.traj_files.split(",") if x.strip()]:
        ck_t = torch.load(fp, map_location="cpu", weights_only=False)
        all_records.extend(ck_t["records"])
        print(f"  {fp}: {len(ck_t['records'])} records", flush=True)
    n_total = len(all_records)
    n_succ = sum(1 for r in all_records if r["succ"])
    print(f"[end-prox] {n_total} records ({n_succ} succ, "
          f"{n_total - n_succ} fail)", flush=True)

    samples = build_samples(all_records, args.window, args.k_end, args.min_chunks)
    if samples is None:
        print("[end-prox] no samples"); return
    h_feats, d_feats, labels, traj_ids = samples
    n_samples = h_feats.shape[0]
    pos_frac = float(labels.mean())
    print(f"[end-prox] {n_samples} per-timestep samples; "
          f"positive (within {args.k_end} chunks of success) frac {pos_frac:.3f}",
          flush=True)

    # TRAJECTORY-level split (no in-trajectory leakage)
    rng = np.random.default_rng(42)
    unique_trajs = traj_ids.unique().numpy()
    rng.shuffle(unique_trajs)
    n_val_traj = max(1, int(args.val_frac * len(unique_trajs)))
    val_traj_set = set(unique_trajs[:n_val_traj].tolist())
    train_traj_set = set(unique_trajs[n_val_traj:].tolist())
    traj_np = traj_ids.numpy()
    val_idx = np.where(np.isin(traj_np, list(val_traj_set)))[0]
    train_idx = np.where(np.isin(traj_np, list(train_traj_set)))[0]
    print(f"[end-prox] trajectory split: {len(train_traj_set)} train traj "
          f"({len(train_idx)} samples), {len(val_traj_set)} val traj "
          f"({len(val_idx)} samples)", flush=True)

    h_feats = h_feats.to(device)
    d_feats = d_feats.to(device)
    labels = labels.to(device)

    head = EndProximityHead(K_bel, d_sub, hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[end-prox] EndProximityHead {n_params:,} params", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    pos_w = torch.tensor([args.pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    t_start = time.time()
    best_val_auc = 0.0
    best_state = None
    last_improvement = 0
    rolling = []
    for step in range(args.max_steps):
        b = train_idx[rng.choice(len(train_idx), args.batch_size, replace=False)]
        logit = head(h_feats[b], d_feats[b])
        loss = loss_fn(logit, labels[b])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
        opt.step()
        rolling.append(float(loss))
        rolling = rolling[-100:]

        if step % args.log_every == 0:
            with torch.no_grad():
                vh = h_feats[val_idx]; vd = d_feats[val_idx]; vy = labels[val_idx]
                vlogit = head(vh, vd)
                vloss = loss_fn(vlogit, vy).item()
                vprob = torch.sigmoid(vlogit).cpu().numpy()
                vy_np = vy.cpu().numpy()
                auc = auc_from_probs(vy_np, vprob)
                # Precision/recall at 0.5 (calibration sanity)
                pred05 = (vprob > 0.5).astype(np.float32)
                acc = float((pred05 == vy_np).mean())
                tp = float(((pred05 == 1) & (vy_np == 1)).sum())
                fp = float(((pred05 == 1) & (vy_np == 0)).sum())
                fn = float(((pred05 == 0) & (vy_np == 1)).sum())
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
            if auc > best_val_auc:
                best_val_auc = auc
                best_state = {k: v.clone().cpu() for k, v in head.state_dict().items()}
                last_improvement = step
            stale = (step - last_improvement) // args.log_every
            print(f"step {step:>4}  train_loss={np.mean(rolling):.4f}  "
                  f"vL={vloss:.4f}  val_AUC={auc:.3f} (best {best_val_auc:.3f})  "
                  f"acc@0.5={acc:.3f}  P={prec:.3f}  R={rec:.3f}  "
                  f"stale={stale}  wall={time.time()-t_start:.0f}s", flush=True)
            if stale >= args.early_stop_patience:
                print(f"[end-prox] early stop at step {step} "
                      f"(best val_AUC={best_val_auc:.3f} @ step {last_improvement})",
                      flush=True)
                break

    if best_state is not None:
        head.load_state_dict(best_state)
        print(f"[end-prox] restored best ckpt (val_AUC={best_val_auc:.3f})",
              flush=True)

    torch.save({
        "substrate_state_dict": substrate.state_dict(),
        "end_proximity_head_state_dict": head.state_dict(),
        "end_proximity_config": {"K": K_bel, "d": d_sub, "hidden": args.hidden,
                                   "window": args.window, "k_end": args.k_end},
        "args": vars(args), "step": args.max_steps,
        "z_vl_dim": ck["z_vl_dim"], "action_dim": ck["action_dim"],
        "horizon": ck["horizon"], "state_dim": ck["state_dim"],
        "dist_mean": ck["dist_mean"], "dist_std": ck["dist_std"],
        "sd_mean": ck["sd_mean"], "sd_std": ck["sd_std"],
        "val_AUC": best_val_auc,
    }, args.output)
    print(f"\n[end-prox] saved → {args.output}  best_val_AUC={best_val_auc:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
