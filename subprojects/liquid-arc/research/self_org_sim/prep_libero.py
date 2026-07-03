"""Pre-decode LIBERO-10-r parquet shards into a single memmap dataset.

One-time pass that strips JPEG decoding out of the training hot path.
Output layout (in --out_dir):
  imgs.dat          uint8  [N, img_size, img_size, 3]
  wrists.dat        uint8  [N, img_size, img_size, 3]
  states.dat        float32 [N, 8]
  actions.dat       float32 [N, 7]
  index.npz         (episode_starts, episode_lengths, task_indices, n_total, img_size)

Run on Spark (~5-10 min for full LIBERO-10-r):
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  python prep_libero.py \\
    --data_root /home/pokazge/datasets/libero-10-r-raw/libero-10-r \\
    --out_dir /home/pokazge/datasets/libero-10-r-decoded \\
    --img_size 96 --workers 16
"""

from __future__ import annotations

import argparse
import functools
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

print = functools.partial(print, flush=True)


def decode_episode(path: str, img_size: int):
    """Read parquet, decode JPEGs at target img_size, return arrays."""
    df = pd.read_parquet(path)
    n = len(df)
    states = np.stack(df["state"].values).astype(np.float32)
    actions = np.stack(df["actions"].values).astype(np.float32)
    task_idx = int(df["task_index"].iloc[0])

    imgs = np.empty((n, img_size, img_size, 3), dtype=np.uint8)
    wrists = np.empty((n, img_size, img_size, 3), dtype=np.uint8)
    for i in range(n):
        imgs[i] = np.array(
            Image.open(io.BytesIO(df["image"].iloc[i]["bytes"])).resize((img_size, img_size)),
            dtype=np.uint8,
        )
        wrists[i] = np.array(
            Image.open(io.BytesIO(df["wrist_image"].iloc[i]["bytes"])).resize((img_size, img_size)),
            dtype=np.uint8,
        )
    return imgs, wrists, states, actions, task_idx, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, type=str)
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    data_dir = Path(args.data_root) / "data" / "chunk-000"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps = sorted(data_dir.glob("episode_*.parquet"))
    print(f"Found {len(eps)} episodes")

    # Pass 1: count frames per episode to size memmaps (cheap parquet metadata read)
    print("Pass 1: counting frames...")
    lengths = []
    for ep in eps:
        df = pd.read_parquet(ep, columns=["state"])
        lengths.append(len(df))
    starts = np.cumsum([0] + lengths[:-1]).astype(np.int64)
    n_total = int(sum(lengths))
    print(f"Total frames: {n_total:,}")

    # Allocate memmaps
    img_path = out_dir / "imgs.dat"
    wrist_path = out_dir / "wrists.dat"
    state_path = out_dir / "states.dat"
    act_path = out_dir / "actions.dat"

    imgs_mm = np.memmap(img_path, dtype=np.uint8, mode="w+",
                        shape=(n_total, args.img_size, args.img_size, 3))
    wrists_mm = np.memmap(wrist_path, dtype=np.uint8, mode="w+",
                          shape=(n_total, args.img_size, args.img_size, 3))
    states_mm = np.memmap(state_path, dtype=np.float32, mode="w+", shape=(n_total, 8))
    actions_mm = np.memmap(act_path, dtype=np.float32, mode="w+", shape=(n_total, 7))
    task_idx_arr = np.zeros(len(eps), dtype=np.int64)

    # Pass 2: parallel decode
    print(f"Pass 2: decoding {n_total:,} frames using {args.workers} workers...")
    n_done = 0
    n_eps = len(eps)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(decode_episode, str(ep), args.img_size): i
                   for i, ep in enumerate(eps)}
        for fut in as_completed(futures):
            ep_idx = futures[fut]
            imgs, wrists, states, actions, task_idx, n = fut.result()
            start = int(starts[ep_idx])
            imgs_mm[start:start + n] = imgs
            wrists_mm[start:start + n] = wrists
            states_mm[start:start + n] = states
            actions_mm[start:start + n] = actions
            task_idx_arr[ep_idx] = task_idx
            n_done += 1
            if n_done % 25 == 0 or n_done == n_eps:
                print(f"  decoded {n_done}/{n_eps} episodes")

    imgs_mm.flush(); wrists_mm.flush(); states_mm.flush(); actions_mm.flush()

    np.savez(out_dir / "index.npz",
             episode_starts=starts.astype(np.int64),
             episode_lengths=np.array(lengths, dtype=np.int64),
             task_indices=task_idx_arr,
             n_total=np.int64(n_total),
             img_size=np.int64(args.img_size))

    total_gb = (n_total * args.img_size * args.img_size * 3 * 2 +
                n_total * 8 * 4 + n_total * 7 * 4) / 1e9
    print(f"Done. Output {total_gb:.2f} GB in {out_dir}")


if __name__ == "__main__":
    main()
