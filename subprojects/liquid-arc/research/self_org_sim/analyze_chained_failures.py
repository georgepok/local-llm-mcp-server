"""Analyze chained-LIBERO failure modes from diag JSON.

For each sub-task in each chain rollout:
  - Tag SUCCEEDED (used few steps + succ=True) vs FAILED (hit 720 cap + succ=False)
  - For failures: characterize what's happening in the last 200 steps
    - Is eef_xyz stuck (variance over last 200 steps tiny)?
    - Is gripper held closed (qpos always small)?
    - Is action repeating (action_xyz var small)?
    - How far is eef from initial-of-sub-task pose? (could indicate stuck near prev grasp)

Outputs a per-failure characterization table.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def analyze_failure(diag, lookback=20):
    """Given a list of per-step records (each every 10 env steps), characterize the
    last `lookback` records (~200 env steps)."""
    if len(diag) < 3:
        return {"insufficient_data": True}
    recent = diag[-lookback:] if len(diag) >= lookback else diag
    eef = np.array([r["eef_xyz"] for r in recent if r.get("eef_xyz") is not None])
    grip = np.array([r["grip_qpos"] for r in recent if r.get("grip_qpos") is not None])
    act_xyz = np.array([r["action_xyz"] for r in recent])
    act_grip = np.array([r["action_grip"] for r in recent])

    out = {
        "n_records": len(recent),
        "eef_var_total": float(eef.std(axis=0).sum()) if len(eef) else None,
        "eef_range_xyz": (eef.max(axis=0) - eef.min(axis=0)).tolist() if len(eef) else None,
        "grip_qpos_mean": float(grip.mean()) if len(grip) else None,
        "grip_qpos_var": float(grip.std()) if len(grip) else None,
        "action_xyz_var": float(act_xyz.std(axis=0).sum()),
        "action_grip_mean": float(act_grip.mean()),
        "action_grip_var": float(act_grip.std()),
    }
    # Classify
    stuck_pose = out["eef_var_total"] is not None and out["eef_var_total"] < 0.02
    grip_closed = out["grip_qpos_mean"] is not None and out["grip_qpos_mean"] < 0.025
    grip_open = out["grip_qpos_mean"] is not None and out["grip_qpos_mean"] > 0.03
    action_repeating = out["action_xyz_var"] < 0.05
    out["tag"] = []
    if stuck_pose: out["tag"].append("EEF_STUCK")
    if grip_closed: out["tag"].append("GRIP_CLOSED")
    if grip_open: out["tag"].append("GRIP_OPEN")
    if action_repeating: out["tag"].append("ACTION_REPEATING")
    if out["action_grip_mean"] > 0.5: out["tag"].append("ACT_GRIP_CLOSE")
    if out["action_grip_mean"] < -0.5: out["tag"].append("ACT_GRIP_OPEN")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", type=str)
    args = p.parse_args()
    data = json.loads(Path(args.path).read_text())
    print(f"Loaded {args.path}")
    print(f"  overall: {data.get('total_subtasks_completed')}/{data.get('total_subtasks_attempted')}"
          f" = {data.get('completion_rate', 0)*100:.0f}%")

    failure_tags = defaultdict(int)
    success_initial_pose_taken = defaultdict(int)

    print(f"\n{'chain':>5} {'r':>2} {'sub':>3} {'task':>5} {'steps':>5} {'succ':>5} {'tags':<40}")
    print("-" * 90)
    for entry in data.get("chains", []):
        chain_idx = entry["chain_idx"]
        r = entry["rollout"]
        for res in entry.get("results", []):
            sub_idx = res["sub_idx"]
            sub_id = res["sub_id"]
            steps = res.get("steps", "")
            succ = res.get("succeeded", False)
            diag = res.get("diag", [])
            if diag:
                a = analyze_failure(diag, lookback=20)
                tag_list = a.get("tag", []) if isinstance(a.get("tag"), list) else []
                tags = ",".join(tag_list)
                if not succ:
                    for t in tag_list:
                        failure_tags[t] += 1
                    failure_tags["TOTAL_FAILURES"] += 1
            else:
                tags = "(no diag)"
            print(f"  {chain_idx:>3} {r:>2} {sub_idx:>3} {sub_id:>5} {str(steps):>5} {str(succ):>5} {tags:<40}")

    print(f"\n=== FAILURE TAG SUMMARY ===")
    for t, c in sorted(failure_tags.items(), key=lambda x: -x[1]):
        print(f"  {t:<25} {c}")


if __name__ == "__main__":
    main()
