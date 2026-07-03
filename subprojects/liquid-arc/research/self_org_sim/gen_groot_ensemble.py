"""Multi-noise ensemble distillation labels.

For each obs in a collected dataset, query GR00T server K times (different
noise → different action_chunks) and save the full ensemble. Captures teacher's
stochasticity / uncertainty.

Output layout (extends gen_groot_labels.py format):
  teacher_chunks_ensemble.dat   f32  [N, K, 16, 7]
  labels_index.npz              sample_idx [N, 3], n_samples, ensemble_k, action_horizon

Run on Spark in the libero venv (so we can read the obs memmap that was
saved by groot_sim_collect.py at 256x256). The GR00T server must already be
running in main venv on the same port.

  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python gen_groot_ensemble.py \\
    --collected_dir /home/pokazge/datasets/groot-in-sim-iter2-720 \\
    --raw_data_root /home/pokazge/datasets/libero-10-r-raw/libero-10-r \\
    --out_dir /home/pokazge/datasets/groot-ensemble-iter1 \\
    --ensemble_k 8 --port 5555
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle
import time
from pathlib import Path

import numpy as np
import zmq

print = functools.partial(print, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collected_dir", required=True, type=str,
                   help="Source obs dir (memmap of imgs/wrists/states + index)")
    p.add_argument("--raw_data_root", required=True, type=str,
                   help="libero-r raw root, used only for tasks.jsonl strings")
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument("--ensemble_k", type=int, default=8,
                   help="Number of teacher samples per obs")
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--max_obs", type=int, default=0,
                   help="If >0, cap the number of obs to label (for smoke tests)")
    args = p.parse_args()

    coll_dir = Path(args.collected_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.raw_data_root)

    # Load collected obs metadata
    idx = np.load(coll_dir / "index.npz")
    starts = idx["episode_starts"]
    lengths = idx["episode_lengths"]
    task_indices = idx["task_indices"]
    n_total = int(idx["n_total"])
    img_size = int(idx["img_size"])
    print(f"Source: {len(lengths)} trajs, {n_total:,} states, img_size={img_size}")

    imgs = np.memmap(coll_dir / "imgs.dat", dtype=np.uint8, mode="r",
                     shape=(n_total, img_size, img_size, 3))
    wrists = np.memmap(coll_dir / "wrists.dat", dtype=np.uint8, mode="r",
                       shape=(n_total, img_size, img_size, 3))
    states = np.memmap(coll_dir / "states.dat", dtype=np.float32, mode="r",
                       shape=(n_total, 8))

    # Build flat sample list
    samples = []
    for ep_i in range(len(lengths)):
        n = int(lengths[ep_i]); ti = int(task_indices[ep_i])
        for t in range(n):
            samples.append((ep_i, t, ti))
    if args.max_obs > 0:
        samples = samples[:args.max_obs]
    actual_n = len(samples)
    print(f"Will label {actual_n:,} obs × {args.ensemble_k} samples = "
          f"{actual_n * args.ensemble_k:,} GR00T calls")

    # Tasks lang
    tasks_map = {}
    with open(raw_root / "meta" / "tasks.jsonl") as f:
        for line in f:
            d = json.loads(line)
            tasks_map[d["task_index"]] = d["task"]

    # GR00T modality (mirror server)
    state_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    video_keys = ["image", "wrist_image"]
    language_keys = ["annotation.human.action.task_description"]
    state_slots = {"x": (0, 1), "y": (1, 2), "z": (2, 3), "roll": (3, 4),
                   "pitch": (4, 5), "yaw": (5, 6), "gripper": (6, 8)}

    def build_obs(global_idx, ti):
        new_obs = {"video": {}, "state": {}, "language": {}}
        for k in video_keys:
            arr = imgs[global_idx] if k == "image" else wrists[global_idx]
            new_obs["video"][k] = np.asarray(arr)[None, None, ...]   # (1,1,H,W,3)
        state_full = np.asarray(states[global_idx], dtype=np.float32)
        for k in state_keys:
            lo, hi = state_slots[k]
            new_obs["state"][k] = state_full[lo:hi][None, None, :]
        for lk in language_keys:
            new_obs["language"][lk] = [[tasks_map[ti]]]
        return new_obs

    # Allocate output: [N, K, 16, 7]
    K = args.ensemble_k
    out_chunks = np.memmap(out_dir / "teacher_chunks_ensemble.dat",
                           dtype=np.float32, mode="w+",
                           shape=(actual_n, K, args.action_horizon, 7))
    sample_idx = np.zeros((actual_n, 3), dtype=np.int64)

    # ZMQ
    ctx = zmq.Context(); sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    sock.setsockopt(zmq.RCVTIMEO, 60000)
    print(f"Connected to GR00T on tcp://127.0.0.1:{args.port}")

    t_start = time.time()
    n_done = 0
    for j, (ep_i, t, ti) in enumerate(samples):
        global_idx = int(starts[ep_i]) + int(t)
        obs = build_obs(global_idx, ti)
        sample_idx[j] = [ep_i, t, ti]
        for k in range(K):
            sock.send(pickle.dumps(obs))
            r = pickle.loads(sock.recv())
            if "error" in r:
                raise RuntimeError(f"server error: {r['error']}")
            out_chunks[j, k] = r["chunk"]
        n_done += 1
        if n_done % 50 == 0 or n_done == actual_n:
            elapsed = time.time() - t_start
            rate = (n_done * K) / elapsed if elapsed > 0 else 0
            eta_min = ((actual_n - n_done) * K) / rate / 60 if rate > 0 else 0
            print(f"  {n_done}/{actual_n} obs ({100*n_done/actual_n:.1f}%)  "
                  f"rate={rate:.1f} samples/s  eta={eta_min:.1f} min")

    out_chunks.flush()
    np.savez(out_dir / "labels_index.npz",
             sample_idx=sample_idx, n_samples=np.int64(actual_n),
             ensemble_k=np.int64(K),
             action_horizon=np.int64(args.action_horizon))
    print(f"Done. {actual_n:,} obs × {K} ensemble = {actual_n*K:,} chunks saved to {out_dir}")
    print(f"Size: {out_chunks.nbytes/1e6:.1f} MB")


if __name__ == "__main__":
    main()
