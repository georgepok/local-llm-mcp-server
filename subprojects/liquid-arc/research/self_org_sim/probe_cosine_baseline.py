"""Cosine-similarity baseline for goal-alignment drift detection.

For each per-turn chunk: compute cos(z_t, z_goal) directly.
If this correlates with turn_followed, we have a no-training baseline that
the substrate must beat to justify itself.

Predicts FAILURE (drift). Higher score = more likely to drift.
Score candidates:
  -cos(z_t, z_goal)    if drift means low similarity
  +cos(z_t, z_goal)    if drift means high similarity (could be inverted too)

Plus aggregates: mean/last/min cosine over the turn.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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
    p.add_argument("--rb_traj", required=True)
    p.add_argument("--raw_traj", required=True)
    p.add_argument("--use_all", action="store_true")
    p.add_argument("--val_frac", type=float, default=0.15)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_substrate_twoflow import load_records_with_inputs

    rb_records = load_records_with_inputs([args.rb_traj])
    raw_pack = torch.load(args.raw_traj, map_location="cpu", weights_only=False)
    raw_by_sub = {int(r["sub_id"]): r for r in raw_pack["records"]}
    rb_records = [r for r in rb_records if int(r["sub_id"]) in raw_by_sub]

    rng = np.random.default_rng(42)
    if args.use_all:
        probe = rb_records
        print(f"[cos] probing ALL {len(probe)} conversations")
    else:
        all_sub_ids = sorted({int(r["sub_id"]) for r in rb_records})
        rng.shuffle(all_sub_ids)
        n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
        val_ids = set(all_sub_ids[:n_val_ids])
        probe = [r for r in rb_records if int(r["sub_id"]) in val_ids]
        print(f"[cos] probing val {len(probe)} conversations")

    rows = []
    for r in probe:
        T = r["z_vl_traj"].shape[0]
        if T < 2:
            continue
        z_t = r["z_vl_traj"]      # [T, 384]
        z_goal = r["z_lang_traj"]  # [T, 384] — already the current goal per chunk

        # Per-chunk cosine
        cos_per = F.cosine_similarity(z_t, z_goal, dim=-1).numpy()  # [T]

        raw = raw_by_sub[int(r["sub_id"])]
        turn_starts = list(raw["turn_chunk_starts"])
        turn_followed = list(raw["turn_followed"])
        n_turns = len(turn_followed)

        for ti in range(n_turns):
            start = turn_starts[ti]
            end = turn_starts[ti + 1] if ti + 1 < n_turns else T
            start = max(0, min(start, T))
            end = max(start, min(end, T))
            if end <= start:
                continue
            seg = cos_per[start:end]
            rows.append({
                "sub_id": int(r["sub_id"]),
                "turn_idx": ti,
                "followed": int(turn_followed[ti]),
                "mean_cos": float(seg.mean()),
                "last_cos": float(seg[-1]),
                "min_cos": float(seg.min()),
                "max_cos": float(seg.max()),
                "n_chunks": int(end - start),
            })

    labels = np.array([1 - r["followed"] for r in rows], dtype=int)
    n_drift = int(labels.sum())
    print(f"[cos] {len(rows)} per-turn rows; drift events {n_drift}/{len(rows)}")
    print()
    print("=" * 60)
    print("COSINE-BASELINE per-turn drift AUC (predict FAILURE)")
    print("=" * 60)
    for direction in ("-cos (low sim = drift)", "+cos (high sim = drift)"):
        sign = -1 if direction.startswith("-") else 1
        print(f"\n{direction}:")
        for stat in ("mean_cos", "last_cos", "min_cos", "max_cos"):
            vals = np.array([sign * r[stat] for r in rows])
            auc = roc_auc(vals, labels)
            mF = vals[labels == 1].mean() if labels.sum() > 0 else float("nan")
            mS = vals[labels == 0].mean() if (1 - labels).sum() > 0 else float("nan")
            print(f"  {stat:<12s} AUC={auc:.3f}  mean(F)={mF:.4f}  mean(S)={mS:.4f}")

    # By turn position
    print()
    print("PER-TURN-POSITION AUC (using -mean_cos):")
    for ti in sorted(set(r["turn_idx"] for r in rows)):
        ti_rows = [r for r in rows if r["turn_idx"] == ti]
        ti_labels = np.array([1 - r["followed"] for r in ti_rows], dtype=int)
        if ti_labels.sum() < 2 or len(ti_rows) - ti_labels.sum() < 2:
            continue
        vals = np.array([-r["mean_cos"] for r in ti_rows])
        auc = roc_auc(vals, ti_labels)
        print(f"  turn {ti}: n={len(ti_rows)} drifts={int(ti_labels.sum())} AUC={auc:.3f}")


if __name__ == "__main__":
    main()
