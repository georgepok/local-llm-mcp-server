"""h_slow zero-ablation probe.

Question: does the slow channel actually carry the +5% lift seen in vL_coup vs vL_pred,
or did the coupling predictor learn to ignore h_slow and produce gain from action_ctx alone?

Three measurements on val_records:
  vL_pred                  — JEPA fast-only predictor (from training)
  vL_coup_full             — coupling predictor with REAL h_slow
  vL_coup_zerohslow        — coupling predictor with h_slow := 0

Interpretation:
  - If vL_coup_zerohslow ≈ vL_coup_full: slow channel uninformative; gain came from
    bigger predictor capacity processing chunks-only.
  - If vL_coup_zerohslow ≈ vL_pred (no gain): slow channel IS the source of the lift.
  - In between: slow contributes partially.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from liquid_goal_tracker_proprio import JEPA_LGT_Proprio

# Re-implement minimal forward (matches train_substrate_twoflow.py)
from train_substrate_twoflow import (
    load_records_with_inputs, forward_two_flow_no_grad,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--traj_files", required=True)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--min_chunks", type=int, default=8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--split_mode", default="task_id")
    p.add_argument("--n_samples", type=int, default=64,
                   help="resample window position N times per record for variance reduction")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device}, ckpt={args.ckpt}", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Infer d, K from state_dict shape — trainer doesn't always save d_substrate in args
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
    print(f"[probe] model loaded, best_val_score={ck.get('best_val_score','?')}", flush=True)

    # Build val split (same scheme as training)
    records = load_records_with_inputs([s.strip() for s in args.traj_files.split(",")])
    records = [r for r in records
               if r["h_goal_traj"].shape[0] >= args.min_chunks + args.window]

    rng = np.random.default_rng(42)
    all_sub_ids = sorted({int(r["sub_id"]) for r in records})
    rng.shuffle(all_sub_ids)
    n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
    val_ids = set(all_sub_ids[:n_val_ids])
    val_records = [r for r in records if int(r["sub_id"]) in val_ids]
    print(f"[probe] val task ids={sorted(val_ids)}  records={len(val_records)}", flush=True)

    L_pred_all, L_coup_full_all, L_coup_zero_all = [], [], []
    L_coup_zero_action_all = []  # extra: h_slow=0 AND chunks=0 (predictor bias only)

    with torch.no_grad():
        for r in val_records:
            T = r["h_goal_traj"].shape[0]
            if T < args.window + 2:
                continue
            for _ in range(args.n_samples):
                t = int(rng.integers(0, T - args.window))
                h_slow_traj, h_fast_traj = forward_two_flow_no_grad(model, r, device, t)
                h_slow_now = h_slow_traj[t].unsqueeze(0)
                h_fast_now = h_fast_traj[t].unsqueeze(0)

                # Target future from same model (EMA tied closely at convergence)
                _, h_fast_future_traj = forward_two_flow_no_grad(model, r, device, t + args.window)
                h_fast_future = h_fast_future_traj[t + args.window].unsqueeze(0)

                chunks = r["chunk_traj"][t].float().to(device).flatten().unsqueeze(0)

                pred = model.jepa_predict_future_h_goal(h_fast_now, chunks)
                L_pred_all.append(float(((pred - h_fast_future) ** 2).mean()))

                coup_full = model.coupling_predict(h_slow_now, chunks, h_fast_now)
                L_coup_full_all.append(float(((coup_full - h_fast_future) ** 2).mean()))

                # ABLATION 1: zero h_slow
                h_slow_zero = torch.zeros_like(h_slow_now)
                coup_zero = model.coupling_predict(h_slow_zero, chunks, h_fast_now)
                L_coup_zero_all.append(float(((coup_zero - h_fast_future) ** 2).mean()))

                # ABLATION 2: zero h_slow AND zero chunks (predictor bias only)
                chunks_zero = torch.zeros_like(chunks)
                coup_bias = model.coupling_predict(h_slow_zero, chunks_zero, h_fast_now)
                L_coup_zero_action_all.append(float(((coup_bias - h_fast_future) ** 2).mean()))

    def stats(x, name):
        arr = np.array(x)
        return f"{name:24s} mean={arr.mean():.4f}  std={arr.std():.4f}  n={len(arr)}"

    print()
    print("=" * 64)
    print("VAL PROBE RESULTS")
    print("=" * 64)
    print(stats(L_pred_all,           "L_pred (fast→fast)"))
    print(stats(L_coup_full_all,      "L_coup (slow+chunks)"))
    print(stats(L_coup_zero_all,      "L_coup (0+chunks)"))
    print(stats(L_coup_zero_action_all, "L_coup (0+0) [bias]"))
    print()
    lift_full = np.mean(L_pred_all) - np.mean(L_coup_full_all)
    lift_zero = np.mean(L_pred_all) - np.mean(L_coup_zero_all)
    print(f"lift_full (pred - coup_full)      = {lift_full:+.4f}")
    print(f"lift_zero (pred - coup_zerohslow) = {lift_zero:+.4f}")
    print(f"slow_contribution (full - zero)   = {lift_full - lift_zero:+.4f}")
    print()
    if abs(lift_full - lift_zero) < 0.1 * abs(lift_full):
        verdict = "FALSIFIED: slow channel uninformative — coupling head learned to ignore h_slow."
    elif lift_zero < 0.1 * lift_full:
        verdict = "VALIDATED: slow channel IS the source of the gain."
    else:
        verdict = f"PARTIAL: slow contributes ~{100*(lift_full-lift_zero)/lift_full:.0f}% of the gain."
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
