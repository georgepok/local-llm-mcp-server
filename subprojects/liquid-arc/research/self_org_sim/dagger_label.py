"""DAgger phase 2 — relabel collected obs with GR00T expert.

Reads (img, wrist, state) memmap produced by dagger_collect.py, runs GR00T
in batches, saves expert action_chunks in the same format as
gen_groot_labels.py so distill_groot_flow can mix it in via TeacherLabelDataset.

Run on Spark in the main 3.12 venv (CUDA torch + gr00t):
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
  HF_HOME=/home/pokazge/hf_cache HF_TOKEN=hf_... python dagger_label.py \\
    --collected_dir /home/pokazge/datasets/dagger-iter1-obs \\
    --teacher_path /home/pokazge/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_10 \\
    --raw_data_root /home/pokazge/datasets/libero-10-r-raw/libero-10-r \\
    --out_dir /home/pokazge/datasets/dagger-iter1-groot-labels \\
    --batch_size 8 --action_horizon 16
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collected_dir", required=True, type=str)
    p.add_argument("--teacher_path", required=True, type=str)
    p.add_argument("--raw_data_root", required=True, type=str,
                   help="Path to libero-r raw root (used only for tasks.jsonl language strings)")
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--max_per_traj", type=int, default=0,
                   help="If >0, sample at most N states per trajectory (default: all)")
    args = p.parse_args()

    coll_dir = Path(args.collected_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load collected obs
    idx = np.load(coll_dir / "index.npz")
    starts = idx["episode_starts"]
    lengths = idx["episode_lengths"]
    task_indices = idx["task_indices"]
    n_total = int(idx["n_total"])
    img_size = int(idx["img_size"])
    print(f"Collected: {len(lengths)} trajectories, {n_total:,} states, img_size={img_size}")

    imgs = np.memmap(coll_dir / "imgs.dat", dtype=np.uint8, mode="r",
                     shape=(n_total, img_size, img_size, 3))
    wrists = np.memmap(coll_dir / "wrists.dat", dtype=np.uint8, mode="r",
                       shape=(n_total, img_size, img_size, 3))
    states = np.memmap(coll_dir / "states.dat", dtype=np.float32, mode="r",
                       shape=(n_total, 8))

    # Build flat sample list (ep_i, t, ti)
    samples = []
    for ep_i in range(len(lengths)):
        n = int(lengths[ep_i])
        ti = int(task_indices[ep_i])
        if args.max_per_traj > 0 and n > args.max_per_traj:
            t_subset = np.linspace(0, n - 1, args.max_per_traj, dtype=int)
        else:
            t_subset = np.arange(n)
        for t in t_subset:
            samples.append((ep_i, int(t), ti))
    actual_n = len(samples)
    print(f"Will label {actual_n:,} states (max_per_traj={args.max_per_traj or 'all'})")

    # Load tasks.jsonl
    raw_root = Path(args.raw_data_root)
    tasks_map = {}
    with open(raw_root / "meta" / "tasks.jsonl") as f:
        for line in f:
            d = json.loads(line)
            tasks_map[d["task_index"]] = d["task"]

    # Load GR00T
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    print(f"Loading GR00T teacher...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path=str(args.teacher_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    modality_configs = policy.get_modality_config()
    state_keys = modality_configs["state"].modality_keys
    action_keys = modality_configs["action"].modality_keys
    video_keys = modality_configs["video"].modality_keys
    language_keys = modality_configs["language"].modality_keys

    state_slots = {
        "x": (0, 1), "y": (1, 2), "z": (2, 3),
        "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6),
        "gripper": (6, 8),
    }
    # In dagger collect, "image" was workspace cam (agentview) and
    # "wrist_image" was wrist cam — same convention as libero-r.
    video_orig_keys = {"image": "image", "wrist_image": "wrist_image"}

    # Allocate output
    teacher_chunks = np.memmap(out_dir / "teacher_chunks.dat", dtype=np.float32, mode="w+",
                                shape=(actual_n, args.action_horizon, 7))
    sample_idx_arr = np.zeros((actual_n, 3), dtype=np.int64)

    print(f"Running GR00T inference, batch_size={args.batch_size}...")
    t_start = time.time()
    for batch_start in range(0, actual_n, args.batch_size):
        batch_end = min(batch_start + args.batch_size, actual_n)
        B = batch_end - batch_start
        videos_batched = {k: [] for k in video_keys}
        states_batched = {k: [] for k in state_keys}
        languages_batched = {k: [] for k in language_keys}
        for j in range(batch_start, batch_end):
            ep_i, t, ti = samples[j]
            global_idx = int(starts[ep_i]) + t
            for k in video_keys:
                # Both video keys use raw 256x256 imgs; pick image vs wrist via field name
                if video_orig_keys[k] == "image":
                    arr = np.array(imgs[global_idx])
                else:
                    arr = np.array(wrists[global_idx])
                videos_batched[k].append(arr)
            state_full = np.array(states[global_idx], dtype=np.float32)
            for k in state_keys:
                lo, hi = state_slots[k]
                states_batched[k].append(state_full[lo:hi])
            for lk in language_keys:
                languages_batched[lk].append([tasks_map[ti]])
            sample_idx_arr[j] = [ep_i, t, ti]

        new_obs = {"video": {}, "state": {}, "language": {}}
        for k in video_keys:
            arr = np.stack(videos_batched[k])
            new_obs["video"][k] = arr[:, None, ...]
        for k in state_keys:
            arr = np.stack(states_batched[k])
            new_obs["state"][k] = arr[:, None, :]
        for lk in language_keys:
            new_obs["language"][lk] = languages_batched[lk]

        action_chunk_dict, _ = policy.get_action(new_obs)
        for j_local in range(B):
            chunk_rows = []
            for ak in action_keys:
                v = action_chunk_dict[ak][j_local]
                chunk_rows.append(np.atleast_2d(v.reshape(args.action_horizon, -1)))
            chunk = np.concatenate(chunk_rows, axis=1)  # [K, 7]
            chunk[:, -1] = 1.0 - 2.0 * chunk[:, -1]      # gripper [0,1] -> [-1,+1]
            teacher_chunks[batch_start + j_local] = chunk.astype(np.float32)

        if (batch_start + B) % (args.batch_size * 20) == 0 or batch_end == actual_n:
            elapsed = time.time() - t_start
            done = batch_start + B
            rate = done / elapsed if elapsed > 0 else 0
            eta = (actual_n - done) / rate / 60 if rate > 0 else 0
            print(f"  {done}/{actual_n} ({100*done/actual_n:.1f}%)  rate={rate:.1f}/s  eta={eta:.1f} min")

    teacher_chunks.flush()
    np.savez(out_dir / "labels_index.npz",
             sample_idx=sample_idx_arr,
             n_samples=np.int64(actual_n),
             action_horizon=np.int64(args.action_horizon))
    print(f"Done. {actual_n} dagger labels saved to {out_dir}")


if __name__ == "__main__":
    main()
