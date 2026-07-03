"""A1: Decompose scalar substrate's distance-prediction error.

Question: R²=0.34 — is it uniform, or 80% accurate in some regime + noise in others?

Stratifies prediction errors by:
  - Suite (4 levels)
  - Task ID within suite
  - Step-position quartile within episode (0-25%, 25-50%, 50-75%, 75-100%)
  - True-distance quartile (low/med/high distance to goal)
  - Joint (suite × step-position)

Output: per-strata R², MAE, mean predicted, mean true, count. Saved to JSON
and printed as table.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from liquid_goal_tracker_scalar import JEPA_LGT_Scalar  # type: ignore

torch.set_float32_matmul_precision("high")


def load_triples(path):
    d = np.load(path, allow_pickle=True)
    return {
        "z_t": d["z_t"], "chunks": d["chunks"], "z_next": d["z_next"],
        "z_goal": d["z_goal"], "ep_id": d["ep_id"], "suite": d["suite"],
        "task_id": d["task_id"],
    }


def build_episode_index(ep_id, suite):
    keys = [(str(s), int(e)) for s, e in zip(suite, ep_id)]
    eps: Dict[tuple, List[int]] = {}
    for i, k in enumerate(keys):
        eps.setdefault(k, []).append(i)
    return [sorted(idxs) for idxs in eps.values()]


def compute_per_turn(substrate, data, episodes, device, dist_mean, dist_std):
    """Per-turn (pred_dist, true_dist, suite, task_id, step_in_ep, ep_len, gripper)."""
    records = []
    with torch.no_grad():
        for ep_idxs in episodes:
            h_goal = substrate.init_state(1, device)
            L = len(ep_idxs)
            for pos, i in enumerate(ep_idxs):
                z_t = torch.from_numpy(data["z_t"][i]).to(device).unsqueeze(0)
                chunk_t = torch.from_numpy(data["chunks"][i]).to(device).unsqueeze(0)
                z_goal = torch.from_numpy(data["z_goal"][i]).to(device).unsqueeze(0)
                h_goal, _, aux, _ = substrate.step(h_goal, z_t, z_goal, chunk_t)
                p_dist = float(aux["pred_goaldist"]) * dist_std + dist_mean
                true_dist = float(np.linalg.norm(
                    data["z_goal"][i] - data["z_t"][i]))
                # Gripper state from the chunk's last column (mean across horizon)
                gripper = float(data["chunks"][i][:, -1].mean())
                records.append({
                    "suite": str(data["suite"][i]),
                    "task_id": int(data["task_id"][i]),
                    "step": pos,
                    "ep_len": L,
                    "frac_done": pos / max(L - 1, 1),
                    "pred": p_dist,
                    "true": true_dist,
                    "abs_err": abs(p_dist - true_dist),
                    "signed_err": p_dist - true_dist,
                    "gripper_cmd": gripper,
                })
    return records


def r2(true, pred):
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(true) < 2:
        return None
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-8)


def mae(true, pred):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def stratified_summary(records, group_fn, name):
    """For each unique group key, compute n / R² / MAE / mean(true) / mean(pred)."""
    groups: Dict = {}
    for r in records:
        k = group_fn(r)
        groups.setdefault(k, {"true": [], "pred": [], "signed_err": []})
        groups[k]["true"].append(r["true"])
        groups[k]["pred"].append(r["pred"])
        groups[k]["signed_err"].append(r["signed_err"])
    out = []
    for k, v in sorted(groups.items(), key=lambda x: str(x[0])):
        out.append({
            "group": k,
            "n": len(v["true"]),
            "r2": r2(v["true"], v["pred"]),
            "mae": mae(v["true"], v["pred"]),
            "mean_true": float(np.mean(v["true"])),
            "mean_pred": float(np.mean(v["pred"])),
            "mean_signed_err": float(np.mean(v["signed_err"])),
            "std_signed_err": float(np.std(v["signed_err"])),
        })
    return {"by": name, "groups": out}


def print_table(label, summary, top=12):
    print(f"\n--- {label} ---")
    print(f"{'group':<30s} {'n':>5s} {'R²':>8s} {'MAE':>7s} "
          f"{'true_avg':>9s} {'pred_avg':>9s} {'bias':>8s}")
    for g in summary["groups"][:top]:
        gname = str(g["group"])[:28]
        r2v = g["r2"]; r2s = f"{r2v:+.3f}" if r2v is not None else "  n/a"
        print(f"{gname:<30s} {g['n']:>5d} {r2s:>8s} "
              f"{g['mae']:>7.3f} {g['mean_true']:>9.3f} "
              f"{g['mean_pred']:>9.3f} {g['mean_signed_err']:>+8.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/tmp/lgt_scalar.pt")
    p.add_argument("--held_out_data", default="/tmp/libero_jepa_held_out_triples.npz")
    p.add_argument("--out_json", default="/tmp/a1_distance_analysis.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[a1] device={device}", flush=True)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sa = ck["args"]
    substrate = JEPA_LGT_Scalar(
        z_vl_dim=ck["z_vl_dim"], action_dim=ck["action_dim"],
        horizon=ck["horizon"], d=sa["d_substrate"], K=sa["K_belief"],
    ).to(device)
    substrate.load_state_dict(ck["substrate_state_dict"])
    substrate.eval()
    dist_mean = ck["dist_mean"]; dist_std = ck["dist_std"]
    print(f"[a1] substrate d={sa['d_substrate']}, dist_mean={dist_mean:.2f} "
          f"std={dist_std:.2f}", flush=True)

    data = load_triples(args.held_out_data)
    episodes = build_episode_index(data["ep_id"], data["suite"])
    print(f"[a1] {len(data['z_t'])} held-out triples, {len(episodes)} episodes",
          flush=True)

    records = compute_per_turn(substrate, data, episodes, device,
                                 dist_mean, dist_std)
    print(f"[a1] {len(records)} per-turn records computed", flush=True)

    # Overall
    overall = {
        "n": len(records),
        "r2": r2([r["true"] for r in records], [r["pred"] for r in records]),
        "mae": mae([r["true"] for r in records], [r["pred"] for r in records]),
        "mean_true": float(np.mean([r["true"] for r in records])),
        "mean_pred": float(np.mean([r["pred"] for r in records])),
        "mean_signed_err": float(np.mean([r["signed_err"] for r in records])),
        "std_signed_err": float(np.std([r["signed_err"] for r in records])),
    }
    print(f"\n=== OVERALL ===")
    print(f"n={overall['n']}  R²={overall['r2']:+.4f}  MAE={overall['mae']:.3f}  "
          f"mean_true={overall['mean_true']:.3f}  "
          f"mean_pred={overall['mean_pred']:.3f}  "
          f"bias={overall['mean_signed_err']:+.3f}")

    # Stratifications
    by_suite = stratified_summary(records, lambda r: r["suite"], "suite")
    by_task = stratified_summary(records,
                                   lambda r: f"{r['suite']}/t{r['task_id']}", "suite_task")
    # Frac-done quartiles
    by_phase = stratified_summary(
        records,
        lambda r: "Q1[0-25%]" if r["frac_done"] < 0.25
                  else "Q2[25-50%]" if r["frac_done"] < 0.5
                  else "Q3[50-75%]" if r["frac_done"] < 0.75
                  else "Q4[75-100%]",
        "phase_quartile")
    # True-distance quartiles (computed across full record set)
    true_dists = [r["true"] for r in records]
    q1, q2, q3 = np.percentile(true_dists, [25, 50, 75])
    by_dist = stratified_summary(
        records,
        lambda r: "low_dist" if r["true"] < q1
                  else "med_lo" if r["true"] < q2
                  else "med_hi" if r["true"] < q3
                  else "high_dist",
        "true_dist_quartile")
    # Gripper state (open vs closed)
    by_grip = stratified_summary(
        records,
        lambda r: "open(g<0)" if r["gripper_cmd"] < -0.1
                  else "closed(g>0)" if r["gripper_cmd"] > 0.1
                  else "transitioning",
        "gripper_state")
    # Joint suite × phase
    by_suite_phase = stratified_summary(
        records,
        lambda r: f"{r['suite']}|{('Q1' if r['frac_done']<0.25 else 'Q2' if r['frac_done']<0.5 else 'Q3' if r['frac_done']<0.75 else 'Q4')}",
        "suite_phase")

    print_table("BY SUITE", by_suite, top=10)
    print_table("BY PHASE QUARTILE", by_phase, top=10)
    print_table("BY TRUE-DISTANCE QUARTILE", by_dist, top=10)
    print_table("BY GRIPPER STATE", by_grip, top=10)
    print_table("BY SUITE × PHASE (top 20)", by_suite_phase, top=20)
    print_table("BY SUITE × TASK (top 25)", by_task, top=25)

    out = {
        "ckpt": args.ckpt, "overall": overall,
        "by_suite": by_suite, "by_phase": by_phase, "by_dist": by_dist,
        "by_grip": by_grip, "by_suite_phase": by_suite_phase, "by_task": by_task,
        "dist_quartiles": {"q1": float(q1), "q2": float(q2), "q3": float(q3)},
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[a1] saved → {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
