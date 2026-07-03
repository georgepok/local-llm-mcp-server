"""Per-task router: select the best checkpoint per sim_id and run rollouts."""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

print = functools.partial(print, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--router_json", required=True, type=str,
                   help="JSON file mapping sim_id (str) -> checkpoint path")
    p.add_argument("--rollouts_per_task", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=720)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--infer_steps", type=int, default=10)
    p.add_argument("--gripper_sign", type=float, default=1.0)
    p.add_argument("--out_json", type=str, default="")
    args = p.parse_args()

    router = {int(k): v for k, v in json.loads(Path(args.router_json).read_text()).items()}
    print(f"Router: {len(router)} task→ckpt mappings")
    for sid in sorted(router):
        print(f"  sim{sid} -> {Path(router[sid]).parent.name}")

    summary = {"tasks": []}
    overall_succ, overall_total = 0, 0
    for sim_id, ckpt in sorted(router.items()):
        print(f"\n=== sim{sim_id} via {Path(ckpt).parent.name} ===")
        out_json = f"/tmp/_router_sim{sim_id}.json"
        cmd = [
            "python", "rollout_libero_flow.py",
            "--student_ckpt", ckpt,
            "--task_suite", "libero_10",
            "--task_indices", str(sim_id),
            "--rollouts_per_task", str(args.rollouts_per_task),
            "--max_steps", str(args.max_steps),
            "--exec_horizon", str(args.exec_horizon),
            "--infer_steps", str(args.infer_steps),
            "--gripper_sign", str(args.gripper_sign),
            "--use_lang",
            "--out_json", out_json,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
        # Parse the printed OVERALL line
        for line in result.stdout.splitlines()[-30:]:
            print(f"  | {line}")
        try:
            data = json.loads(Path(out_json).read_text())
            t = data["tasks"][0]
            n_rollouts = t["n_rollouts"]
            n_succ = t["n_successes"]
            overall_succ += n_succ; overall_total += n_rollouts
            summary["tasks"].append({
                "sim_id": sim_id,
                "ckpt": ckpt,
                "task_name": t["task_name"],
                "n_rollouts": n_rollouts,
                "n_successes": n_succ,
                "success_rate": t["success_rate"],
            })
            print(f"  -> {n_succ}/{n_rollouts} = {n_succ/n_rollouts:.0%}")
        except Exception as e:
            print(f"  FAILED to parse: {e}")
            summary["tasks"].append({"sim_id": sim_id, "ckpt": ckpt, "error": str(e)})

    print("\n" + "=" * 80)
    print(f"ROUTER OVERALL: {overall_succ}/{overall_total} = "
          f"{overall_succ/max(overall_total,1):.0%}")
    print("=" * 80)
    summary["overall_successes"] = overall_succ
    summary["overall_total"] = overall_total
    summary["overall_success_rate"] = overall_succ / max(overall_total, 1)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"Saved router summary to {args.out_json}")


if __name__ == "__main__":
    main()
