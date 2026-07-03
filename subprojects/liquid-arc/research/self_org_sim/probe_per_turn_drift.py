"""Per-turn drift detection probe for multigoal substrate.

For each conversation:
  - Forward through entire trajectory
  - Compute per-chunk L_pred, L_coup, succ_logit
  - Split chunks by turn (using turn_chunk_starts from record metadata)
  - For each turn, compute aggregate stats (mean, last, max of pred/coup/residual)
  - Label: turn_followed (constraint compliance)
  - Compute ROC AUC per stat × turn position × overall

The framework's claim: drift = decoupling between slow goal-stream and fast generation.
A goal is linguistically present; when model fails to follow it, the slow goal state
and fast generation state should diverge in their predictions of next-chunk h_fast.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from liquid_goal_tracker_proprio import JEPA_LGT_Proprio
from train_substrate_twoflow import load_records_with_inputs, forward_two_flow_no_grad


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
    p.add_argument("--rb_traj", required=True,
                   help="robotics-format trajectories")
    p.add_argument("--raw_traj", required=True,
                   help="original multigoal traj file with per-turn metadata")
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--min_chunks", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--use_all", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    init_belief_shape = ck["substrate_state_dict"]["init_belief"].shape
    K_bel = int(init_belief_shape[0])
    d_sub = int(init_belief_shape[1])
    model = JEPA_LGT_Proprio(
        z_vl_dim=ck["z_vl_dim"], action_dim=ck["action_dim"],
        horizon=ck["horizon"], state_dim=ck["state_dim"],
        d=d_sub, K=K_bel, n_tok_per_k=1,
    )
    model.load_state_dict(ck["substrate_state_dict"], strict=False)
    model.use_evidence_layernorm = True
    model.h_input_clamp = 50.0
    model = model.to(device).eval()
    print(f"[turn] model loaded d={d_sub} K={K_bel}, best_val_score={ck.get('best_val_score','?')}",
          flush=True)

    # Load rb (substrate inputs) AND raw (per-turn metadata)
    rb_records = load_records_with_inputs([args.rb_traj])
    raw_pack = torch.load(args.raw_traj, map_location="cpu", weights_only=False)
    raw_by_sub = {int(r["sub_id"]): r for r in raw_pack["records"]}

    rng = np.random.default_rng(42)
    if args.use_all:
        probe = rb_records
    else:
        all_sub_ids = sorted({int(r["sub_id"]) for r in rb_records})
        rng.shuffle(all_sub_ids)
        n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
        val_ids = set(all_sub_ids[:n_val_ids])
        probe = [r for r in rb_records if int(r["sub_id"]) in val_ids]
    print(f"[turn] probing {len(probe)} conversations", flush=True)

    # Per-turn collected stats and labels
    rows = []
    n_skipped = 0

    with torch.no_grad():
        for r in probe:
            T = r["z_vl_traj"].shape[0]
            if T < args.window + 2:
                n_skipped += 1
                continue
            raw = raw_by_sub.get(int(r["sub_id"]))
            if raw is None:
                n_skipped += 1
                continue
            turn_starts = list(raw["turn_chunk_starts"])
            turn_followed = list(raw["turn_followed"])
            n_turns = len(turn_followed)

            # Forward through whole trajectory
            h_slow_traj, h_fast_traj = forward_two_flow_no_grad(model, r, device, T - 1)
            # Compute per-chunk predictions
            preds_pred, preds_coup, succ_logits = [], [], []
            for t in range(T - args.window):
                h_slow_now = h_slow_traj[t].unsqueeze(0)
                h_fast_now = h_fast_traj[t].unsqueeze(0)
                h_fast_future = h_fast_traj[t + args.window].unsqueeze(0)
                chunks = r["chunk_traj"][t].float().to(device).flatten().unsqueeze(0)
                pred = model.jepa_predict_future_h_goal(h_fast_now, chunks)
                L_pred = ((pred - h_fast_future) ** 2).mean().item()
                coup = model.coupling_predict(h_slow_now, chunks, h_fast_now)
                L_coup = ((coup - h_fast_future) ** 2).mean().item()
                sl = model.jepa_predict_success(h_fast_now).squeeze().item()
                preds_pred.append(L_pred)
                preds_coup.append(L_coup)
                succ_logits.append(sl)
            preds_pred = np.array(preds_pred)
            preds_coup = np.array(preds_coup)
            succ_logits = np.array(succ_logits)

            # For each turn, find which chunk indices belong to it
            T_eff = len(preds_pred)
            for ti in range(n_turns):
                start = turn_starts[ti]
                end = turn_starts[ti + 1] if ti + 1 < n_turns else T_eff
                start = max(0, min(start, T_eff))
                end = max(start, min(end, T_eff))
                if end <= start:
                    continue
                seg_pred = preds_pred[start:end]
                seg_coup = preds_coup[start:end]
                seg_succ = succ_logits[start:end]
                seg_resid = seg_coup - seg_pred

                rows.append({
                    "sub_id": int(r["sub_id"]),
                    "turn_idx": ti,
                    "followed": int(turn_followed[ti]),
                    "n_chunks": int(end - start),
                    "mean_pred": float(seg_pred.mean()),
                    "max_pred": float(seg_pred.max()),
                    "last_pred": float(seg_pred[-1]),
                    "mean_coup": float(seg_coup.mean()),
                    "max_coup": float(seg_coup.max()),
                    "last_coup": float(seg_coup[-1]),
                    "mean_resid": float(seg_resid.mean()),
                    "max_resid": float(seg_resid.max()),
                    "last_resid": float(seg_resid[-1]),
                    "mean_succ_neg": float(-seg_succ.mean()),  # negate for failure-prediction
                    "last_succ_neg": float(-seg_succ[-1]),
                })

    if n_skipped:
        print(f"[turn] skipped {n_skipped} (missing raw or too short)", flush=True)
    print(f"[turn] collected {len(rows)} per-turn rows from {len(probe)} conversations",
          flush=True)
    n_drift = sum(1 for x in rows if x["followed"] == 0)
    print(f"[turn] drift events (followed==0): {n_drift}/{len(rows)} = "
          f"{100*n_drift/max(1,len(rows)):.0f}%", flush=True)
    if n_drift < 5:
        print("[turn] WARN: very few drift events — AUC will be noisy", flush=True)

    # Compute AUC of each stat (failure_label = 1 - followed) per overall + per turn_idx
    labels = np.array([1 - x["followed"] for x in rows], dtype=int)
    stat_names = ["mean_pred", "max_pred", "last_pred", "mean_coup", "max_coup",
                  "last_coup", "mean_resid", "max_resid", "last_resid",
                  "mean_succ_neg", "last_succ_neg"]

    print()
    print("=" * 72)
    print(f"OVERALL per-turn drift AUC (predict FAILURE, n={len(rows)}, drifts={n_drift})")
    print("=" * 72)
    print(f"{'stat':<22s}{'AUC':>8s}{'mean(F)':>12s}{'mean(S)':>12s}{'delta':>10s}")
    print("-" * 72)
    aucs = {}
    for name in stat_names:
        vals = np.array([r[name] for r in rows])
        auc = roc_auc(vals, labels)
        mF = vals[labels == 1].mean() if labels.sum() > 0 else float("nan")
        mS = vals[labels == 0].mean() if (1 - labels).sum() > 0 else float("nan")
        print(f"{name:<22s}{auc:>8.3f}{mF:>12.4f}{mS:>12.4f}{(mF-mS):>+10.4f}")
        aucs[name] = auc

    # By turn position
    print()
    print("=" * 72)
    print("PER-TURN-POSITION AUC")
    print("=" * 72)
    for ti in sorted(set(r["turn_idx"] for r in rows)):
        ti_rows = [r for r in rows if r["turn_idx"] == ti]
        ti_labels = np.array([1 - r["followed"] for r in ti_rows], dtype=int)
        ti_drifts = ti_labels.sum()
        if ti_drifts < 2 or len(ti_rows) - ti_drifts < 2:
            print(f"turn {ti}: n={len(ti_rows)} drifts={ti_drifts} — skipped (insufficient)")
            continue
        line = f"turn {ti} n={len(ti_rows)} drifts={ti_drifts}:  "
        for name in ("mean_pred", "mean_coup", "last_resid", "max_resid",
                     "mean_succ_neg"):
            vals = np.array([r[name] for r in ti_rows])
            auc = roc_auc(vals, ti_labels)
            line += f" {name}={auc:.3f} "
        print(line)

    # Headline
    print()
    pred_stats = [s for s in stat_names if "pred" in s or "coup" in s]
    resid_stats = [s for s in stat_names if "resid" in s]
    succ_stats = [s for s in stat_names if "succ" in s]
    best_pred = max((aucs[s], s) for s in pred_stats if not np.isnan(aucs[s]))
    best_resid = max((aucs[s], s) for s in resid_stats if not np.isnan(aucs[s]))
    best_succ = max((aucs[s], s) for s in succ_stats if not np.isnan(aucs[s]))
    print(f"BEST pred/coup:  {best_pred[1]} AUC={best_pred[0]:.3f}")
    print(f"BEST residual:   {best_resid[1]} AUC={best_resid[0]:.3f}")
    print(f"BEST succ-head:  {best_succ[1]} AUC={best_succ[0]:.3f}")


if __name__ == "__main__":
    main()
