"""Generate teacher (GR00T-N1.7-LIBERO) action_chunk labels for libero-r samples.

Strategy:
  - Stratified-sample N points from the dataset (uniform across the 10 tasks)
  - For each sampled (episode, t), build GR00T observation from the raw parquet
    (256x256 images + state + language) and call policy.get_action() in batches
  - Save (sample_indices, teacher_chunks) to a memmap dataset that
    distill_groot.py can consume via --teacher_labels_dir

Run on Spark (uses our 3.12 venv with CUDA torch and gr00t):
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
  HF_HOME=/home/pokazge/hf_cache HF_TOKEN=hf_... python gen_groot_labels.py \\
    --raw_data_root /home/pokazge/datasets/libero-10-r-raw/libero-10-r \\
    --decoded_data_dir /home/pokazge/datasets/libero-10-r-decoded \\
    --teacher_path /home/pokazge/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_10 \\
    --out_dir /home/pokazge/datasets/libero-10-r-groot-labels \\
    --n_samples 20000 --batch_size 8 --action_horizon 16
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

print = functools.partial(print, flush=True)


def _decode_jpeg(b: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(b)), dtype=np.uint8)


def stratified_sample(decoded_dir: Path, train_split_episodes: int,
                      n_per_task: int, action_horizon: int, seed: int = 0,
                      mode: str = "stratified"):
    """Sample (episode_idx, frame_t) pairs from the train split.

    mode='stratified': sample n_per_task per task (with replacement).
    mode='all': enumerate every (ep, t) pair in train split (n_per_task ignored).
    """
    idx = np.load(decoded_dir / "index.npz")
    starts = idx["episode_starts"]
    lengths = idx["episode_lengths"]
    task_indices = idx["task_indices"]
    n_eps = len(lengths)
    train_eps = set(range(n_eps - max(10, n_eps // 10)))
    del train_split_episodes

    rng = np.random.default_rng(seed)
    samples = []

    if mode == "all":
        for ep_i in range(n_eps):
            if ep_i in train_eps:
                ti = int(task_indices[ep_i])
                n = int(lengths[ep_i])
                for t in range(n):
                    samples.append((ep_i, t, ti))
    else:
        by_task = {}
        for ep_i in range(n_eps):
            if ep_i in train_eps:
                ti = int(task_indices[ep_i])
                by_task.setdefault(ti, []).append(ep_i)
        for ti, eps in sorted(by_task.items()):
            for _ in range(n_per_task):
                ep_i = int(rng.choice(eps))
                n = int(lengths[ep_i])
                t = int(rng.integers(0, n))
                samples.append((ep_i, t, ti))

    rng.shuffle(samples)
    return samples, starts, lengths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_data_root", required=True, type=str)
    p.add_argument("--decoded_data_dir", required=True, type=str,
                   help="Used only for index.npz (episode lengths/starts/tasks)")
    p.add_argument("--teacher_path", required=True, type=str)
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument("--n_samples", type=int, default=20000,
                   help="Total samples (split evenly across 10 tasks). Ignored when --mode=all.")
    p.add_argument("--mode", choices=["stratified", "all"], default="stratified",
                   help="'stratified' samples n_per_task per task; 'all' enumerates every train state.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    raw_root = Path(args.raw_data_root)
    decoded_dir = Path(args.decoded_data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks.jsonl for language strings (indexed by task_index)
    tasks_map = {}
    with open(raw_root / "meta" / "tasks.jsonl") as f:
        for line in f:
            d = json.loads(line)
            tasks_map[d["task_index"]] = d["task"]

    # Load GR00T teacher
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    print(f"Loading GR00T teacher from {args.teacher_path} ...")
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
    print(f"state_keys={state_keys}, action_keys={action_keys}, video_keys={video_keys}")

    state_slots = {
        "x": (0, 1), "y": (1, 2), "z": (2, 3),
        "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6),
        "gripper": (6, 8),
    }
    video_orig_keys = {
        "image": "image",                # parquet column for workspace cam
        "wrist_image": "wrist_image",    # parquet column for wrist cam
    }

    # Sample selection
    n_per_task = args.n_samples // 10
    samples, starts, lengths = stratified_sample(
        decoded_dir, 0, n_per_task, args.action_horizon, seed=args.seed, mode=args.mode,
    )
    actual_n = len(samples)
    if args.mode == "all":
        print(f"Mode=all: enumerating {actual_n:,} train states (no sampling)")
    else:
        print(f"Mode=stratified: sampling {actual_n:,} points across 10 tasks ({n_per_task}/task)")

    # Allocate output arrays
    teacher_chunks = np.memmap(out_dir / "teacher_chunks.dat", dtype=np.float32, mode="w+",
                                shape=(actual_n, args.action_horizon, 7))
    sample_idx = np.zeros((actual_n, 3), dtype=np.int64)  # (episode_idx, frame_t, task_idx)

    # Cache parquet episodes (only those used)
    eps_used = sorted(set(s[0] for s in samples))
    print(f"Loading {len(eps_used)} unique parquet episodes...")
    eps_cache: dict[int, pd.DataFrame] = {}
    for ep_i in eps_used:
        ep_path = raw_root / "data" / "chunk-000" / f"episode_{ep_i:06d}.parquet"
        eps_cache[ep_i] = pd.read_parquet(ep_path)
    print(f"  cached {len(eps_cache)} episodes")

    def build_obs_for_sample(ep_i: int, t: int, ti: int):
        row = eps_cache[ep_i].iloc[t]
        state_full = np.asarray(row["state"], dtype=np.float32)
        video = {}
        for k in video_keys:
            field = "image" if video_orig_keys[k] == "image" else "wrist_image"
            img = _decode_jpeg(row[field]["bytes"])
            video[k] = img  # (H, W, 3)
        state = {k: state_full[state_slots[k][0]:state_slots[k][1]].astype(np.float32) for k in state_keys}
        return video, state, tasks_map[ti]

    # Process in batches
    print(f"Running GR00T inference in batches of {args.batch_size}...")
    t_start = time.time()
    n_done = 0
    for batch_start in range(0, actual_n, args.batch_size):
        batch_end = min(batch_start + args.batch_size, actual_n)
        B = batch_end - batch_start
        # Build batched observation
        videos_batched = {k: [] for k in video_keys}
        states_batched = {k: [] for k in state_keys}
        languages_batched = {k: [] for k in language_keys}
        for j in range(batch_start, batch_end):
            ep_i, t, ti = samples[j]
            video, state, lang = build_obs_for_sample(ep_i, t, ti)
            for k in video_keys:
                videos_batched[k].append(video[k])
            for k in state_keys:
                states_batched[k].append(state[k])
            for lk in language_keys:
                languages_batched[lk].append([lang])
            sample_idx[j] = [ep_i, t, ti]

        # Stack into arrays with batch dim
        new_obs = {"video": {}, "state": {}, "language": {}}
        for k in video_keys:
            arr = np.stack(videos_batched[k])    # (B, H, W, 3)
            new_obs["video"][k] = arr[:, None, ...]  # (B, 1, H, W, 3)
        for k in state_keys:
            arr = np.stack(states_batched[k])    # (B, D)
            new_obs["state"][k] = arr[:, None, :]    # (B, 1, D)
        for lk in language_keys:
            new_obs["language"][lk] = languages_batched[lk]  # list-of-list

        action_chunk_dict, _ = policy.get_action(new_obs)
        # action_chunk_dict[key] is list of length B; each element [K, slot_dim]
        for j_local in range(B):
            chunk_rows = []
            for ak in action_keys:
                v = action_chunk_dict[ak][j_local]  # array
                chunk_rows.append(np.atleast_2d(v.reshape(args.action_horizon, -1)))
            chunk = np.concatenate(chunk_rows, axis=1)  # (K, 7)
            # Convert gripper from GR00T's [0,1] convention to demo's [-1,+1]
            #   teacher_grip=0 -> demo_grip=+1 (open),
            #   teacher_grip=1 -> demo_grip=-1 (close)
            chunk[:, -1] = 1.0 - 2.0 * chunk[:, -1]
            teacher_chunks[batch_start + j_local] = chunk.astype(np.float32)

        n_done += B
        if n_done % (args.batch_size * 20) == 0 or batch_end == actual_n:
            elapsed = time.time() - t_start
            rate = n_done / elapsed
            eta_min = (actual_n - n_done) / rate / 60 if rate > 0 else 0
            print(f"  {n_done}/{actual_n} ({100*n_done/actual_n:.1f}%)  "
                  f"rate={rate:.1f} samples/s  eta={eta_min:.1f} min")

    teacher_chunks.flush()
    np.savez(out_dir / "labels_index.npz",
             sample_idx=sample_idx,
             n_samples=np.int64(actual_n),
             action_horizon=np.int64(args.action_horizon))
    print(f"Done. {actual_n} teacher labels saved to {out_dir}")
    print(f"  teacher_chunks: ({actual_n}, {args.action_horizon}, 7) float32 "
          f"= {teacher_chunks.nbytes/1e6:.1f} MB")


if __name__ == "__main__":
    main()
