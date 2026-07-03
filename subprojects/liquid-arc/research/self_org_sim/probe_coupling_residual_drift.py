"""Trajectory-level coupling-residual drift probe.

Question: does the per-step coupling residual L_coup(t) - L_pred(t) carry a
drift signal that discriminates success vs failure trajectories better than
the substrate's own succ-prediction head on h_fast alone?

For each trajectory:
  - Compute L_coup(t), L_pred(t) at every valid t (where t+window < T).
  - residual(t) = L_coup(t) - L_pred(t)        (negative = coupling beats fast-only)
  - succ_logit(t) = head_success_predictor(h_fast(t))
  - Compute summary stats per trajectory:
      * mean_resid, max_resid, last_resid, slope_resid (over last K steps)
      * mean_succ_logit, last_succ_logit
  - Compute ROC AUC of each stat vs (1 - succ) label (predicting FAILURE).

If ANY coupling-residual stat gives AUC meaningfully higher than the succ-head
baseline, two-flow architecture is a real drift detector beyond what fast-only
provides — validates framework step 4 (contrastive deviation readout).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from liquid_goal_tracker_proprio import JEPA_LGT_Proprio
from train_substrate_twoflow import (
    load_records_with_inputs, forward_two_flow_no_grad,
)


def roc_auc(scores, labels):
    """Simple ROC AUC (positive label = 1 = failure). No sklearn dependency."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[labels == 1]
    auc = (len(scores) - pos_ranks.mean()) / max(1, n_neg) - 0.5 * (n_pos + 1) / n_neg + 0.5
    auc = (n_pos * n_neg - sum((labels[order[i]] == 0).sum() for i in range(len(order))
                                                            if labels[order[i]] == 1))
    rs = scores[labels == 1]
    rn = scores[labels == 0]
    wins = (rs[:, None] > rn[None, :]).sum()
    ties = (rs[:, None] == rn[None, :]).sum()
    return (wins + 0.5 * ties) / (n_pos * n_neg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--traj_files", required=True)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--min_chunks", type=int, default=8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--last_k", type=int, default=5,
                   help="window for last_resid and slope_resid stats")
    p.add_argument("--use_all", action="store_true",
                   help="probe ALL trajectories (not just task-id val split)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[drift] device={device}, ckpt={args.ckpt}", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    init_belief_shape = ck["substrate_state_dict"]["init_belief"].shape
    K_bel = int(init_belief_shape[0])
    d_sub = int(init_belief_shape[1])
    sa = ck.get("args", {})
    n_tok = sa.get("n_tok_per_k", 1) if isinstance(sa, dict) else 1

    model = JEPA_LGT_Proprio(
        z_vl_dim=ck["z_vl_dim"], action_dim=ck["action_dim"],
        horizon=ck["horizon"], state_dim=ck["state_dim"],
        d=d_sub, K=K_bel, n_tok_per_k=n_tok,
    )
    model.load_state_dict(ck["substrate_state_dict"], strict=False)
    model.use_evidence_layernorm = True
    model.h_input_clamp = 50.0
    model = model.to(device).eval()
    print(f"[drift] model loaded, best_val_score={ck.get('best_val_score','?')}", flush=True)

    records = load_records_with_inputs([s.strip() for s in args.traj_files.split(",")])
    records = [r for r in records
               if r["h_goal_traj"].shape[0] >= args.min_chunks + args.window]

    rng = np.random.default_rng(42)
    if args.use_all:
        probe_records = records
        print(f"[drift] probing ALL {len(probe_records)} records", flush=True)
    else:
        all_sub_ids = sorted({int(r["sub_id"]) for r in records})
        rng.shuffle(all_sub_ids)
        n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
        val_ids = set(all_sub_ids[:n_val_ids])
        probe_records = [r for r in records if int(r["sub_id"]) in val_ids]
        print(f"[drift] probing val task ids={sorted(val_ids)} ({len(probe_records)})",
              flush=True)

    stats = {
        "mean_resid": [], "max_resid": [], "last_resid": [], "slope_resid": [],
        "mean_succ_logit": [], "last_succ_logit": [],
        "mean_L_coup": [], "mean_L_pred": [],
    }
    labels = []  # 1 = FAILURE, 0 = success (we predict failure)

    with torch.no_grad():
        for r in probe_records:
            T = r["h_goal_traj"].shape[0]
            if T < args.window + 2:
                continue
            # Forward through whole trajectory once (final t = T-1 sweeps state)
            h_slow_traj, h_fast_traj = forward_two_flow_no_grad(
                model, r, device, T - 1
            )
            # h_slow_traj, h_fast_traj are [T, K, d]
            T_eff = h_slow_traj.shape[0]
            residuals, coups, preds, succ_logits = [], [], [], []
            for t in range(T_eff - args.window):
                h_slow_now = h_slow_traj[t].unsqueeze(0)
                h_fast_now = h_fast_traj[t].unsqueeze(0)
                # Target h_fast_future
                h_fast_future = h_fast_traj[t + args.window].unsqueeze(0)
                chunks = r["chunk_traj"][t].float().to(device).flatten().unsqueeze(0)

                pred = model.jepa_predict_future_h_goal(h_fast_now, chunks)
                L_pred = ((pred - h_fast_future) ** 2).mean().item()

                coup = model.coupling_predict(h_slow_now, chunks, h_fast_now)
                L_coup = ((coup - h_fast_future) ** 2).mean().item()

                succ_logit = model.jepa_predict_success(h_fast_now).squeeze().item()

                residuals.append(L_coup - L_pred)
                coups.append(L_coup)
                preds.append(L_pred)
                succ_logits.append(succ_logit)

            if not residuals:
                continue
            res = np.array(residuals)
            succ = np.array(succ_logits)
            stats["mean_resid"].append(res.mean())
            stats["max_resid"].append(res.max())
            stats["last_resid"].append(res[-min(args.last_k, len(res)):].mean())
            if len(res) >= args.last_k:
                xs = np.arange(args.last_k)
                ys = res[-args.last_k:]
                slope = float(np.polyfit(xs, ys, 1)[0])
            else:
                slope = 0.0
            stats["slope_resid"].append(slope)
            stats["mean_L_coup"].append(np.mean(coups))
            stats["mean_L_pred"].append(np.mean(preds))
            # succ_logit: higher = more confident SUCCESS; for failure prediction we negate
            stats["mean_succ_logit"].append(-succ.mean())
            stats["last_succ_logit"].append(-succ[-min(args.last_k, len(succ)):].mean())
            labels.append(1 - int(r["succ"]))

    labels = np.array(labels, dtype=int)
    print(f"[drift] n_trajectories={len(labels)}  n_fail={labels.sum()}  "
          f"n_succ={(1-labels).sum()}", flush=True)
    print()
    print("=" * 72)
    print("DRIFT-DETECTION AUC (higher score → predict FAILURE)")
    print("=" * 72)
    print(f"{'stat':<22s}{'AUC':>8s}{'mean(F)':>12s}{'mean(S)':>12s}{'delta':>10s}")
    print("-" * 72)
    for name, vals in stats.items():
        v = np.array(vals)
        auc = roc_auc(v, labels)
        mF = v[labels == 1].mean() if labels.sum() > 0 else float("nan")
        mS = v[labels == 0].mean() if (1 - labels).sum() > 0 else float("nan")
        print(f"{name:<22s}{auc:>8.3f}{mF:>12.4f}{mS:>12.4f}{(mF-mS):>+10.4f}")
    print()

    # Headline: coupling-residual vs succ-head
    best_resid = max((roc_auc(np.array(stats[k]), labels), k)
                     for k in ("mean_resid", "max_resid", "last_resid", "slope_resid"))
    best_succ = max((roc_auc(np.array(stats[k]), labels), k)
                    for k in ("mean_succ_logit", "last_succ_logit"))
    print(f"BEST coupling-residual stat: {best_resid[1]} AUC={best_resid[0]:.3f}")
    print(f"BEST succ-head stat:         {best_succ[1]} AUC={best_succ[0]:.3f}")
    if best_resid[0] > best_succ[0] + 0.02:
        print(f"VERDICT: coupling residual ADDS drift signal beyond succ-head (+{best_resid[0]-best_succ[0]:.3f})")
    elif best_resid[0] < best_succ[0] - 0.02:
        print(f"VERDICT: succ-head dominates; coupling residual does NOT add drift signal")
    else:
        print(f"VERDICT: coupling residual and succ-head provide comparable signal")


if __name__ == "__main__":
    main()
