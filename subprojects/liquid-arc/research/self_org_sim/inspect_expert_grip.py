"""Stage 1 of v10-DEMO failure validation: do expert demos for libero_object
have gripper open (grip ≈ -1) during the early-approach phase?

Loads libero-{suite}-expert-v1/{teacher_chunks.dat, labels_index.npz, index.npz}.
For each task, aggregates gripper-by-episode-step across all episodes.
Reports: at step t, what's the mean/std/distribution of expert gripper values?

If libero_object expert demos start with grip ≈ -1 (open) for steps 0-50 then
transition to grip ≈ +1: the model SHOULD have learned this — its failure is
a learning failure, fixable architecturally.

If libero_object expert demos start with grip ≈ +1 (closed): the model is
correctly imitating expert — our diagnosis is wrong.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


def analyze_suite(suite_dir: Path, suite_label: str):
    print(f"\n{'='*70}")
    print(f"=== {suite_label}: {suite_dir} ===")
    print(f"{'='*70}")

    idx = np.load(suite_dir / "index.npz")
    starts = idx["episode_starts"]
    lengths = idx["episode_lengths"]
    task_indices = idx["task_indices"]
    n_episodes = len(lengths)

    labels = np.load(suite_dir / "labels_index.npz")
    sample_idx = labels["sample_idx"]  # [n_samples, 3] = (ep, t, task)
    n_samples = int(labels["n_samples"])

    # Action shape: teacher_chunks.dat is [n_samples, K=16, A=7]
    K, A = 16, 7
    chunks = np.memmap(suite_dir / "teacher_chunks.dat", dtype=np.float32, mode="r",
                       shape=(n_samples, K, A))

    print(f"n_episodes={n_episodes}  n_samples={n_samples}  unique tasks={len(np.unique(task_indices))}")

    # Episode-step grip: for each sample, the IMMEDIATE NEXT action's gripper.
    # That's chunks[s, 0, -1]. We index by (task, ep, t) → s.
    # But ep is per-episode index; t is step in episode.
    grip0 = chunks[:, 0, -1].copy()  # [n_samples]
    task = sample_idx[:, 2]
    t_in_ep = sample_idx[:, 1]

    # === Per-step gripper distribution across all tasks ===
    step_bins = [(0, 10), (10, 30), (30, 50), (50, 80), (80, 120),
                 (120, 200), (200, 300), (300, 500), (500, 800)]
    print(f"\n--- expert grip[0] (immediate next action) by episode step bin, ALL tasks ---")
    print(f"  step bin       n      mean    median   p10     p90   frac_neg  frac_pos")
    for lo, hi in step_bins:
        mask = (t_in_ep >= lo) & (t_in_ep < hi)
        if mask.sum() < 10:
            continue
        g = grip0[mask]
        print(f"  [{lo:>3},{hi:>3})  {mask.sum():>6}   {g.mean():+.3f}  {np.median(g):+.3f}  "
              f"{np.percentile(g, 10):+.3f}  {np.percentile(g, 90):+.3f}   "
              f"{(g < -0.1).mean():.3f}     {(g > 0.1).mean():.3f}")

    # === Per-task per-step at step 0 ===
    print(f"\n--- expert grip[0] at step 0-10 per TASK ---")
    print(f"  task   n_ep   grip_mean  grip_median  frac_open(<-0.1)  frac_close(>0.1)")
    for ti in sorted(np.unique(task)):
        mask = (task == ti) & (t_in_ep < 10)
        if mask.sum() < 1:
            continue
        g = grip0[mask]
        print(f"  {ti:>4}  {mask.sum():>5}  {g.mean():+.3f}     {np.median(g):+.3f}        "
              f"{(g < -0.1).mean():.3f}            {(g > 0.1).mean():.3f}")

    # === First grip-flip step per episode ===
    print(f"\n--- first step at which expert flips grip sign per episode (across all tasks) ---")
    flip_steps = []
    grip_start = []
    for ep_i in range(n_episodes):
        ep_mask = sample_idx[:, 0] == ep_i
        if not ep_mask.any():
            continue
        ep_samples = np.where(ep_mask)[0]
        # Sort by t within episode
        ep_t = sample_idx[ep_samples, 1]
        order = np.argsort(ep_t)
        ep_samples = ep_samples[order]
        ep_grips = grip0[ep_samples]
        ep_steps = ep_t[order]
        # Find first sign change
        signs = np.sign(ep_grips)
        # ignore zeros (use last non-zero sign)
        start_sign = next((s for s in signs if abs(s) > 0), 0)
        first_flip = -1
        for i in range(1, len(signs)):
            if abs(signs[i]) > 0 and signs[i] != start_sign:
                first_flip = int(ep_steps[i])
                break
        flip_steps.append(first_flip)
        grip_start.append(float(ep_grips[0]))

    flip_arr = np.array(flip_steps)
    grip_start_arr = np.array(grip_start)
    print(f"  episodes analyzed: {len(flip_arr)}")
    print(f"  grip[0] at step 0 mean:    {grip_start_arr.mean():+.3f}")
    print(f"  grip[0] at step 0 frac<-.1: {(grip_start_arr < -0.1).mean():.3f}")
    print(f"  grip[0] at step 0 frac>+.1: {(grip_start_arr > 0.1).mean():.3f}")
    print(f"  episodes that flip sign:   {(flip_arr > 0).sum()}/{len(flip_arr)}")
    if (flip_arr > 0).any():
        flips_pos = flip_arr[flip_arr > 0]
        print(f"  first-flip step (when it happens): mean={flips_pos.mean():.0f}  "
              f"median={np.median(flips_pos):.0f}  p10={np.percentile(flips_pos, 10):.0f}  "
              f"p90={np.percentile(flips_pos, 90):.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/pokazge/datasets")
    p.add_argument("--suites", default="object,spatial",
                   help="comma-separated suite suffixes (matches libero-{suffix}-expert-v1)")
    args = p.parse_args()

    for s in args.suites.split(","):
        s = s.strip()
        sd = Path(args.root) / f"libero-{s}-expert-v1"
        if not sd.exists():
            print(f"[skip] {sd} not found")
            continue
        analyze_suite(sd, f"libero_{s}")


if __name__ == "__main__":
    main()
