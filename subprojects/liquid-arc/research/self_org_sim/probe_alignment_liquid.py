"""Probe alignment-Liquid: per-turn aggregate drift detection.

For each conversation:
  - Forward through full trajectory
  - Compute per-chunk V(h_fast[t], z_goal[t]) and per-chunk alignment cos
  - Per turn: aggregate V (mean, last, min) and cos (mean, last)
  - AUC predicting per-turn drift

Compare against cosine baseline aggregates.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from alignment_liquid import AlignmentLiquid, forward_trajectory


def roc_auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rs = scores[labels == 1]
    rn = scores[labels == 0]
    wins = (rs[:, None] > rn[None, :]).sum()
    ties = (rs[:, None] == rn[None, :]).sum()
    return (wins + 0.5 * ties) / (n_pos * n_neg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--text_traj", required=True)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--use_all", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sa = ck.get("args", {})
    model = AlignmentLiquid(
        z_goal_dim=ck["z_goal_dim"],
        d=sa.get("d", 32) if isinstance(sa, dict) else 32,
        K=sa.get("K", 4) if isinstance(sa, dict) else 4,
        value_hidden=sa.get("value_hidden", 128) if isinstance(sa, dict) else 128,
    ).to(device).eval()
    model.load_state_dict(ck["model_state_dict"], strict=False)
    print(f"[probe] loaded best_auc={ck.get('best_auc_drift', '?')}", flush=True)

    pack = torch.load(args.text_traj, map_location="cpu", weights_only=False)
    raw_records = pack["records"]
    rng = np.random.default_rng(42)
    if args.use_all:
        probe_records = raw_records
        print(f"[probe] probing ALL {len(probe_records)} conversations")
    else:
        all_sub_ids = sorted({int(r["sub_id"]) for r in raw_records})
        rng.shuffle(all_sub_ids)
        n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
        val_ids = set(all_sub_ids[:n_val_ids])
        probe_records = [r for r in raw_records if int(r["sub_id"]) in val_ids]
        print(f"[probe] probing val {len(probe_records)} conversations")

    rows = []
    with torch.no_grad():
        for r in probe_records:
            T = int(r["T"])
            if T < 2:
                continue
            z_t = r["z_t_traj"].float().to(device)
            z_g = r["z_lang_traj"].float().to(device)
            h_fast_traj, h_slow_traj, align_traj = forward_trajectory(
                model, z_t, z_g, device, target_t=None, training=False)
            # Per-chunk V
            V_traj = []
            for t in range(T):
                logit = model.value(h_fast_traj[t].unsqueeze(0),
                                      z_g[t].unsqueeze(0))
                V_traj.append(float(torch.sigmoid(logit).item()))
            V_traj = np.array(V_traj)
            cos_traj = align_traj[:, 0].cpu().numpy()  # first feature = cos(z_t, z_goal)

            turn_starts = list(r["turn_chunk_starts"])
            turn_followed = list(r["turn_followed"])
            n_turns = len(turn_followed)
            for ti in range(n_turns):
                start = turn_starts[ti]
                end = turn_starts[ti + 1] if ti + 1 < n_turns else T
                start = max(0, min(start, T))
                end = max(start, min(end, T))
                if end <= start:
                    continue
                seg_V = V_traj[start:end]
                seg_cos = cos_traj[start:end]
                rows.append({
                    "sub_id": int(r["sub_id"]),
                    "turn_idx": ti,
                    "followed": int(turn_followed[ti]),
                    "n_chunks": int(end - start),
                    "mean_V": float(seg_V.mean()),
                    "last_V": float(seg_V[-1]),
                    "min_V": float(seg_V.min()),
                    "delta_V": float(seg_V[-1] - seg_V[0]),
                    "mean_cos": float(seg_cos.mean()),
                    "last_cos": float(seg_cos[-1]),
                    "min_cos": float(seg_cos.min()),
                })

    labels = np.array([1 - r["followed"] for r in rows], dtype=int)
    n_drift = int(labels.sum())
    print(f"[probe] {len(rows)} per-turn rows, drifts={n_drift}/{len(rows)}")
    print()
    print("=" * 60)
    print("PER-TURN drift AUC (predict FAILURE)")
    print("=" * 60)

    # V-based (higher V means more followed; for drift use -V)
    print("\nALIGNMENT-LIQUID V-based:")
    for stat in ("mean_V", "last_V", "min_V"):
        scores = np.array([-r[stat] for r in rows])  # negate for drift score
        auc = roc_auc(scores, labels)
        print(f"  -{stat:<10s}  AUC={auc:.3f}")
    scores = np.array([-r["delta_V"] for r in rows])
    auc_dv = roc_auc(scores, labels)
    print(f"  -delta_V    AUC={auc_dv:.3f}  (V change over turn; negative = degrading)")

    # Cosine-baseline aggregates (compare apples-to-apples on same val set)
    print("\nCOSINE baseline (raw features):")
    for stat in ("mean_cos", "last_cos", "min_cos"):
        scores = np.array([-r[stat] for r in rows])
        auc = roc_auc(scores, labels)
        print(f"  -{stat:<10s}  AUC={auc:.3f}")

    # Per turn position
    print("\nPER-TURN-POSITION (mean_V):")
    for ti in sorted(set(r["turn_idx"] for r in rows)):
        ti_rows = [r for r in rows if r["turn_idx"] == ti]
        ti_labels = np.array([1 - r["followed"] for r in ti_rows], dtype=int)
        if ti_labels.sum() < 2 or len(ti_rows) - ti_labels.sum() < 2:
            continue
        scores = np.array([-r["mean_V"] for r in ti_rows])
        auc = roc_auc(scores, ti_labels)
        s_cos = np.array([-r["mean_cos"] for r in ti_rows])
        auc_cos = roc_auc(s_cos, ti_labels)
        print(f"  turn {ti}: n={len(ti_rows)} drifts={int(ti_labels.sum())}  "
              f"V_auc={auc:.3f}  cos_auc={auc_cos:.3f}")


if __name__ == "__main__":
    main()
