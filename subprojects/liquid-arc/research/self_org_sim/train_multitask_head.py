"""Train a single multi-output task-state head from (h_goal_t, h_delta_W10).

Combined per-chunk objectives (all from same shared MLP trunk):
  Multi-horizon success forecasting (binary, 5 horizons):
    - p_succ_within_1   (~16 env steps ahead — imminent)
    - p_succ_within_3   (~48 steps)
    - p_succ_within_5   (~80 steps — sweet spot per probe)
    - p_succ_within_10  (~160 steps)
    - p_succ_within_15  (~240 steps)
  Task phase classification (binary):
    - p_early_phase     (first 1/3)
    - p_late_phase      (last 1/3)
  Continuous progress regression:
    - progress_t_over_T (linear progress 0..1)

8 outputs from one shared representation. Per-task loss + class weights.

Substrate body frozen; only the multi-task head trains.

Usage on Spark:
  python train_multitask_head.py \
    --traj_files /tmp/traj_libero10_s10.pt,/tmp/traj_libero10_s20.pt,... \
    --substrate_ckpt /tmp/substrate_projection.pt \
    --output /tmp/substrate_multitask.pt
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


# Output index → (name, type) — keep in sync everywhere.
# Task-INVARIANT targets only (confirmed via task-id-stratified probe):
# success_within_K was inflated by task-id memorization in trajectory-split eval.
# These targets survive task-id holdout → genuinely encode task-state structure.
OUTPUTS = [
    ("early_phase",          "binary"),
    ("late_phase",           "binary"),
    ("chunks_remaining_norm", "regression"),  # task-invariant: R²=0.46 on novel tasks
    ("progress",             "regression"),   # mostly task-invariant: R²=0.39
]
N_OUT = len(OUTPUTS)


class MultiTaskHead(nn.Module):
    """Shared MLP trunk → 8 outputs (7 logits + 1 regression scalar).
    Forces single representation to encode multi-horizon success forecasting,
    phase classification, and continuous progress.
    """
    def __init__(self, K: int, d: int, hidden: int = 96, trunk_layers: int = 2):
        super().__init__()
        in_dim = 2 * K * d
        layers = [nn.Linear(in_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden)]
        for _ in range(trunk_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden)]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, N_OUT)
        with torch.no_grad():
            self.head.weight.mul_(0.01)
            self.head.bias.zero_()

    def forward(self, h_goal: torch.Tensor, h_goal_delta: torch.Tensor) -> torch.Tensor:
        """Returns [B, N_OUT] — logits for binary outputs, raw for regression."""
        x = torch.cat([h_goal.flatten(1), h_goal_delta.flatten(1)], dim=-1)
        return self.head(self.trunk(x))


def compute_labels(record, t, window):
    """Return [N_OUT] label vector for record at timestep t."""
    T = record["h_goal_traj"].shape[0]
    lbls = np.array([
        float(t < T // 3),                       # early_phase
        float(t >= 2 * T // 3),                  # late_phase
        float((T - t) / max(T, 1)),              # chunks_remaining_norm (0..1)
        float(t / max(T - 1, 1)),                # progress 0..1
    ], dtype=np.float32)
    return lbls


def build_samples(records, window: int, min_chunks: int):
    feats_h, feats_d, labels, traj_ids = [], [], [], []
    for ri, r in enumerate(records):
        traj = r["h_goal_traj"]
        T = traj.shape[0]
        if T < max(window + 1, min_chunks):
            continue
        for t in range(window, T):
            h_t = traj[t - 1].float()
            h_tw = traj[t - 1 - window].float()
            lbl = compute_labels(r, t, window)
            feats_h.append(h_t)
            feats_d.append(h_t - h_tw)
            labels.append(torch.from_numpy(lbl))
            traj_ids.append(ri)
    if not feats_h:
        return None
    return (torch.stack(feats_h),
            torch.stack(feats_d),
            torch.stack(labels),
            torch.tensor(traj_ids, dtype=torch.long))


def auc_from_probs(y, p):
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    tpr = tp / max(ys.sum(), 1); fpr = fp / max((1 - ys).sum(), 1)
    return float(np.trapezoid(tpr, fpr))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj_files", required=True)
    p.add_argument("--substrate_ckpt", required=True)
    p.add_argument("--output", default="/tmp/substrate_multitask.pt")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--min_chunks", type=int, default=15)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--trunk_layers", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--split_mode", choices=["trajectory", "task_id"],
                    default="task_id",
                    help="task_id holds out novel sub-tasks (honest); trajectory "
                         "holds out random trajectories (inflated by task-id memorization).")
    p.add_argument("--reg_loss_weight", type=float, default=2.0,
                   help="Loss weight on regression (progress) head")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mt] device={device}, output={args.output}, W={args.window}", flush=True)

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
    print(f"[mt] substrate frozen, K={K_bel} d={d_sub}", flush=True)

    all_records = []
    for fp in [x.strip() for x in args.traj_files.split(",") if x.strip()]:
        ck_t = torch.load(fp, map_location="cpu", weights_only=False)
        all_records.extend(ck_t["records"])
        print(f"  {fp}: {len(ck_t['records'])} records", flush=True)
    n_total = len(all_records)
    print(f"[mt] {n_total} records", flush=True)

    samples = build_samples(all_records, args.window, args.min_chunks)
    if samples is None:
        print("[mt] no samples"); return
    h_feats, d_feats, labels, traj_ids = samples
    n_samples = h_feats.shape[0]
    # Per-output base rates (for pos_weight calibration)
    base_rates = labels.float().mean(dim=0).numpy()
    print(f"[mt] {n_samples} samples; base rates: " +
          ", ".join(f"{OUTPUTS[i][0]}={base_rates[i]:.3f}" for i in range(N_OUT)),
          flush=True)

    rng = np.random.default_rng(42)
    if args.split_mode == "trajectory":
        unique_trajs = traj_ids.unique().numpy()
        rng.shuffle(unique_trajs)
        n_val_traj = max(1, int(args.val_frac * len(unique_trajs)))
        val_traj_set = set(unique_trajs[:n_val_traj].tolist())
        train_traj_set = set(unique_trajs[n_val_traj:].tolist())
        traj_np = traj_ids.numpy()
        val_idx = np.where(np.isin(traj_np, list(val_traj_set)))[0]
        train_idx = np.where(np.isin(traj_np, list(train_traj_set)))[0]
        print(f"[mt] traj split: {len(train_traj_set)} train ({len(train_idx)} samples) / "
              f"{len(val_traj_set)} val ({len(val_idx)} samples)", flush=True)
    else:  # task_id split — holds out novel sub-task IDs
        # Map each traj_id index back to its sub_id via records
        sub_id_per_traj = np.array([int(all_records[ti]["sub_id"])
                                       for ti in traj_ids.numpy().tolist()],
                                      dtype=np.int64)
        all_sub_ids = sorted(set(sub_id_per_traj.tolist()))
        rng.shuffle(all_sub_ids)
        n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
        val_ids = set(all_sub_ids[:n_val_ids])
        train_ids = set(all_sub_ids[n_val_ids:])
        val_idx = np.where(np.isin(sub_id_per_traj, list(val_ids)))[0]
        train_idx = np.where(np.isin(sub_id_per_traj, list(train_ids)))[0]
        print(f"[mt] task_id split: train ids {sorted(train_ids)} "
              f"({len(train_idx)} samples) / val ids {sorted(val_ids)} "
              f"({len(val_idx)} samples)", flush=True)

    h_feats = h_feats.to(device)
    d_feats = d_feats.to(device)
    labels = labels.to(device)

    head = MultiTaskHead(K_bel, d_sub, hidden=args.hidden,
                          trunk_layers=args.trunk_layers).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[mt] MultiTaskHead {n_params:,} params (trunk + 8-dim output)", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    # Per-output pos_weight: 1/base_rate clipped (only for binary outputs)
    pos_weights = []
    for i, (name, kind) in enumerate(OUTPUTS):
        if kind == "binary":
            br = max(base_rates[i], 0.05)  # clip to avoid extreme weights
            pos_weights.append(min(float((1 - br) / br), 10.0))
        else:
            pos_weights.append(1.0)
    pos_weights_t = torch.tensor(pos_weights, device=device)
    print(f"[mt] pos_weights: " + ", ".join(f"{OUTPUTS[i][0]}={pos_weights[i]:.2f}"
                                                for i in range(N_OUT)), flush=True)

    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss(reduction="none")

    def per_output_loss(pred, target):
        """Returns [B, N_OUT] per-sample per-output loss."""
        out = torch.zeros_like(pred)
        for i, (_, kind) in enumerate(OUTPUTS):
            if kind == "binary":
                out[:, i] = bce(pred[:, i], target[:, i]) * pos_weights_t[i]
            else:
                out[:, i] = mse(pred[:, i], target[:, i]) * args.reg_loss_weight
        return out

    t_start = time.time()
    best_val_avg_auc = 0.0  # average AUC across binary heads
    best_state = None
    last_improvement = 0
    for step in range(args.max_steps):
        b = train_idx[rng.choice(len(train_idx), args.batch_size, replace=False)]
        pred = head(h_feats[b], d_feats[b])
        loss = per_output_loss(pred, labels[b]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
        opt.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                vh = h_feats[val_idx]; vd = d_feats[val_idx]; vy = labels[val_idx]
                vp = head(vh, vd)
                # Per-output val metrics
                vp_np = vp.cpu().numpy(); vy_np = vy.cpu().numpy()
                aucs = []
                metric_strs = []
                for i, (name, kind) in enumerate(OUTPUTS):
                    if kind == "binary":
                        prob = 1.0 / (1.0 + np.exp(-vp_np[:, i]))
                        auc = auc_from_probs(vy_np[:, i], prob)
                        aucs.append(auc)
                        metric_strs.append(f"{name}:{auc:.3f}")
                    else:
                        mse_v = float(((vp_np[:, i] - vy_np[:, i]) ** 2).mean())
                        var = float(vy_np[:, i].var())
                        r2 = 1.0 - mse_v / max(var, 1e-9)
                        metric_strs.append(f"{name}:R²{r2:.3f}")
            avg_auc = float(np.mean(aucs)) if aucs else 0.0
            if avg_auc > best_val_avg_auc:
                best_val_avg_auc = avg_auc
                best_state = {k: v.clone().cpu() for k, v in head.state_dict().items()}
                last_improvement = step
            stale = (step - last_improvement) // args.log_every
            print(f"step {step:>4}  loss={float(loss):.4f}  avg_AUC={avg_auc:.3f} "
                  f"(best {best_val_avg_auc:.3f})  stale={stale}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)
            print(f"           " + " ".join(metric_strs), flush=True)
            if stale >= args.early_stop_patience:
                print(f"[mt] early stop at step {step} "
                      f"(best avg_AUC={best_val_avg_auc:.3f} @ step {last_improvement})",
                      flush=True)
                break

    if best_state is not None:
        head.load_state_dict(best_state)
        print(f"[mt] restored best ckpt (avg_AUC={best_val_avg_auc:.3f})", flush=True)

    # Final per-output report
    with torch.no_grad():
        vh = h_feats[val_idx]; vd = d_feats[val_idx]; vy = labels[val_idx]
        vp = head(vh, vd)
        vp_np = vp.cpu().numpy(); vy_np = vy.cpu().numpy()
    print(f"\n=== FINAL per-output (val set) ===", flush=True)
    for i, (name, kind) in enumerate(OUTPUTS):
        if kind == "binary":
            prob = 1.0 / (1.0 + np.exp(-vp_np[:, i]))
            auc = auc_from_probs(vy_np[:, i], prob)
            acc = float(((prob > 0.5).astype(np.float32) == vy_np[:, i]).mean())
            print(f"  {name:18s}  AUC={auc:.3f}  acc={acc:.3f}  base={base_rates[i]:.3f}",
                  flush=True)
        else:
            mse_v = float(((vp_np[:, i] - vy_np[:, i]) ** 2).mean())
            var = float(vy_np[:, i].var())
            r2 = 1.0 - mse_v / max(var, 1e-9)
            print(f"  {name:18s}  R²={r2:.3f}  MSE={mse_v:.4f}  var={var:.4f}",
                  flush=True)

    torch.save({
        "substrate_state_dict": substrate.state_dict(),
        "multitask_head_state_dict": head.state_dict(),
        "multitask_config": {"K": K_bel, "d": d_sub, "hidden": args.hidden,
                               "trunk_layers": args.trunk_layers,
                               "window": args.window, "outputs": OUTPUTS},
        "args": vars(args), "step": args.max_steps,
        "z_vl_dim": ck["z_vl_dim"], "action_dim": ck["action_dim"],
        "horizon": ck["horizon"], "state_dim": ck["state_dim"],
        "dist_mean": ck["dist_mean"], "dist_std": ck["dist_std"],
        "sd_mean": ck["sd_mean"], "sd_std": ck["sd_std"],
        "val_avg_AUC": best_val_avg_auc,
    }, args.output)
    print(f"\n[mt] saved → {args.output}  best_avg_AUC={best_val_avg_auc:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
