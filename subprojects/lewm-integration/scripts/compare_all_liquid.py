"""Head-to-head comparison of Liquid checkpoints.

Runs rollout MSE + control success on the SAME fixed set of evaluation
episodes for each checkpoint, so variance from different starting states
is eliminated.

Checkpoints compared:
  - liquid_long      — 5K steps, low-rank metric, NO criticality
  - liquid_crit      — 5K steps, low-rank + criticality (prev best)
  - liquid_20k       — 20K steps, low-rank + criticality
  - ar_matched       — 5K steps, param-matched AR (reference)
  - ar_20k           — 20K steps, param-matched AR (reference)

Outputs a side-by-side table so we can see whether:
  (a) MSE and control are anti-correlated (spec hypothesis)
  (b) Criticality helps or hurts control
  (c) Training length monotonically improves or degrades control
"""
import argparse
import json
import sys
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False
torch.backends.cuda.enable_flash_sdp(True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "le-wm"))


def _parse_success(line: str) -> float:
    """Extract success_rate: N.N from a 'success_rate': N.N line."""
    import re
    m = re.search(r"'success_rate':\s*([\d.]+)", line)
    return float(m.group(1)) if m else -1.0


def main():
    import subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--rollout_batches", type=int, default=20)
    args = ap.parse_args()

    # (tag, kind, weights_path, obj_path, ar_depth, ar_mlp)
    checkpoints = [
        ("liquid_long",     "liquid", None, None, None, None),
        ("liquid_crit",     "liquid", None, None, None, None),
        ("liquid_20k",      "liquid", None, None, None, None),
        ("ar_matched",      "ar",     None, None, 2,   512),
        ("ar_20k",          "ar",     None, None, 2,   512),
    ]

    print(f"\n=== COMPARISON TABLE ({args.n_eval} control episodes, {args.rollout_batches} rollout batches) ===")
    print(f"{'Checkpoint':<20}  {'H=1':>10}  {'H=5':>10}  {'H=10':>10}  {'H=20':>10}  {'Success':>10}")
    print("-" * 78)

    results = {}
    for tag, kind, _, _, ar_depth, ar_mlp in checkpoints:
        weights = f"/workspace/models/stable-wm/{tag}_weights.ckpt"
        # Rollout — pair with a dummy second ckpt since script needs both
        rollout_cmd = [
            "python", "-u", "/workspace/lewm-integration/scripts/run_with_cudnn_compat.py",
            "/workspace/lewm-integration/scripts/eval_rollout.py",
            "--pretrained_ckpt", args.pretrained_ckpt,
            "--horizons", "1", "5", "10", "20",
            "--batch_size", "32", "--n_batches", str(args.rollout_batches),
        ]
        if kind == "ar":
            rollout_cmd.extend([
                "--ar_ckpt", weights,
                "--liquid_ckpt", "/workspace/models/stable-wm/liquid_crit_weights.ckpt",
                "--ar_depth", str(ar_depth), "--ar_heads", "4",
                "--ar_dim_head", "48", "--ar_mlp_dim", str(ar_mlp),
            ])
        else:
            rollout_cmd.extend([
                "--ar_ckpt", "/workspace/models/stable-wm/ar_matched_weights.ckpt",
                "--liquid_ckpt", weights,
                "--ar_depth", "2", "--ar_heads", "4",
                "--ar_dim_head", "48", "--ar_mlp_dim", "512",
            ])
        env = {"STABLEWM_HOME": "/workspace/models/stable-wm",
               "PYTHONPATH": "/workspace/liquid-arc:/workspace/lewm-integration/scripts:/workspace/lewm-integration/le-wm:/workspace/lewm-integration",
               "PATH": "/usr/local/bin:/usr/bin:/bin"}
        out = subprocess.run(rollout_cmd, capture_output=True, text=True, env=env).stdout

        # Parse the target model's rollout from output
        marker = "AR" if kind == "ar" else "LIQUID"
        lines = out.split("\n")
        start = next((i for i, l in enumerate(lines) if f"=== {marker} ===" in l), -1)
        rollout = {}
        if start > 0:
            for l in lines[start:start + 20]:
                for h in [1, 5, 10, 20]:
                    if f'"{h}"' in l:
                        rollout[h] = float(l.split(":")[1].strip().strip(","))

        # Control eval
        ctrl_cmd = [
            "python", "/workspace/lewm-integration/le-wm/eval.py",
            f"policy={tag}", f"eval.num_eval={args.n_eval}",
        ]
        env2 = {**env, "SDL_VIDEODRIVER": "dummy"}
        out2 = subprocess.run(
            ctrl_cmd, capture_output=True, text=True,
            cwd="/workspace/lewm-integration/le-wm", env=env2).stdout
        success_lines = [l for l in out2.split("\n") if "success_rate" in l]
        success = _parse_success(success_lines[-1]) if success_lines else -1.0

        results[tag] = {"rollout": rollout, "success": success}
        print(f"{tag:<20}  {rollout.get(1, -1):>10.6f}  {rollout.get(5, -1):>10.6f}  "
              f"{rollout.get(10, -1):>10.6f}  {rollout.get(20, -1):>10.6f}  {success:>9.1f}%")

    print()
    out_path = "/workspace/lewm-integration/runs/comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
