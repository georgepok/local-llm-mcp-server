"""Pre-collect GR00T chunks on expert observations for residual substrate training.

For each (suite, episode, turn), send (img, wrist, state, language) to GR00T
server and save the predicted 16-step chunk. Substrate residual training will
use (expert_chunk, GR00T_chunk) pairs — substrate learns to predict the delta
the expert would have applied to GR00T's chunk.

Output: /tmp/groot_chunks_<suite>.npz with arrays:
  - groot_chunks: [n_samples, 16, 7] float32 — GR00T's predicted chunks
  - sample_idx: [n_samples, 3] int64 — (ep, t, task) matching teacher_chunks.dat indexing
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pickle
import zmq


SUITES = ["libero_10", "libero_object", "libero_goal", "libero_spatial"]
DATASET_ROOT = Path("/home/pokazge/datasets")
PORT_MAP = {"libero_10": 5555, "libero_spatial": 5556,
            "libero_object": 5557, "libero_goal": 5558}


def build_groot_obs(img_256, wrist_256, st8, task):
    """Same as rollout_libero_v11_client.build_groot_obs"""
    state_slots = {"x": (0, 1), "y": (1, 2), "z": (2, 3),
                   "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6),
                   "gripper": (6, 8)}
    obs = {"video": {}, "state": {}, "language": {}}
    for k, arr in [("image", img_256), ("wrist_image", wrist_256)]:
        obs["video"][k] = arr[None, None, ...]
    for k, (lo, hi) in state_slots.items():
        obs["state"][k] = st8[lo:hi].astype(np.float32)[None, None, :]
    obs["language"]["annotation.human.action.task_description"] = [[task]]
    return obs


def query_groot(sock, obs):
    sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs}))
    return pickle.loads(sock.recv())


def collect_suite(suite_name, port, max_samples_per_task=50, max_per_episode=8):
    """For each task in this suite, sample up to max_samples_per_task chunks
    across episodes, querying GR00T server."""
    suite_short = suite_name.replace("libero_", "")
    suite_dir = DATASET_ROOT / f"libero-{suite_short}-expert-v1"
    if not suite_dir.exists():
        print(f"  [SKIP] {suite_name} not at {suite_dir}")
        return None

    idx = np.load(suite_dir / "index.npz")
    starts = idx["episode_starts"]
    lengths = idx["episode_lengths"]
    task_indices = idx["task_indices"]
    success = idx.get("success_per_episode", np.ones(len(lengths), dtype=bool))
    n_total = int(idx["n_total"])
    img_size = int(idx["img_size"])

    labels = np.load(suite_dir / "labels_index.npz")
    sample_idx = labels["sample_idx"]
    n_samples = int(labels["n_samples"])

    imgs = np.memmap(suite_dir / "imgs.dat", dtype=np.uint8, mode="r",
                     shape=(n_total, img_size, img_size, 3))
    wrists = np.memmap(suite_dir / "wrists.dat", dtype=np.uint8, mode="r",
                       shape=(n_total, img_size, img_size, 3))
    states = np.memmap(suite_dir / "states.dat", dtype=np.float32, mode="r",
                       shape=(n_total, 8))

    task_lang = {}
    lang_path = suite_dir / "task_languages.json"
    if lang_path.exists():
        import json
        task_lang = json.loads(lang_path.read_text())

    # Build sample selection: per task, pick `max_samples_per_task` random samples
    # from successful episodes, with at most `max_per_episode` per episode
    rng = np.random.default_rng(42)
    selected_samples = []
    selected_global_idx = []
    selected_meta = []

    for task_id in sorted(set(int(t) for t in task_indices)):
        # All sample_idx rows where task==task_id and episode is successful
        task_samples_mask = sample_idx[:, 2] == task_id
        task_sample_indices = np.where(task_samples_mask)[0]
        # Filter by successful episodes
        ep_ids = sample_idx[task_sample_indices, 0]
        succ_mask = success[ep_ids]
        task_sample_indices = task_sample_indices[succ_mask]
        if len(task_sample_indices) == 0:
            continue
        # Group by episode, pick at most max_per_episode per ep
        ep_groups = {}
        for si in task_sample_indices:
            ep = int(sample_idx[si, 0])
            ep_groups.setdefault(ep, []).append(int(si))
        # Picked indices for this task
        picked = []
        for ep, samples in ep_groups.items():
            k = min(max_per_episode, len(samples))
            picked.extend(rng.choice(samples, size=k, replace=False).tolist())
        # Cap per task
        if len(picked) > max_samples_per_task:
            picked = rng.choice(picked, size=max_samples_per_task, replace=False).tolist()

        for si in picked:
            ep, t, _ = sample_idx[si]
            global_idx = int(starts[ep]) + int(t)
            selected_samples.append(int(si))
            selected_global_idx.append(global_idx)
            selected_meta.append((int(ep), int(t), int(task_id)))

    print(f"  {suite_name}: selected {len(selected_samples)} samples across tasks")

    # Connect to GR00T server
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 60000)
    sock.connect(f"tcp://localhost:{port}")
    print(f"  GR00T server: tcp://localhost:{port}")

    groot_chunks = np.zeros((len(selected_samples), 16, 7), dtype=np.float32)
    t_start = time.time()
    for i, (si, gi, meta) in enumerate(zip(selected_samples, selected_global_idx, selected_meta)):
        ep_id, _, task_id = meta
        lang = task_lang.get(str(task_id)) or task_lang.get(task_id) or "do the task"
        img = np.array(imgs[gi])
        wri = np.array(wrists[gi])
        st = np.array(states[gi])
        # Flip image to match GR00T's expected orientation (per LIBERO rollout)
        img_flipped = img[::-1, ::-1].copy()
        wri_flipped = wri[::-1, ::-1].copy()
        obs = build_groot_obs(img_flipped, wri_flipped, st, lang)
        resp = query_groot(sock, obs)
        groot_chunks[i] = np.asarray(resp["chunk"], dtype=np.float32)
        if (i + 1) % 20 == 0:
            rate = (i + 1) / (time.time() - t_start)
            eta = (len(selected_samples) - i - 1) / max(rate, 0.01)
            print(f"    [{i+1}/{len(selected_samples)}] {rate:.1f} samples/s, ETA {eta:.0f}s")
    sock.close()

    return {
        "groot_chunks": groot_chunks,
        "sample_idx": np.array(selected_samples, dtype=np.int64),
        "meta": np.array(selected_meta, dtype=np.int64),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="/tmp", type=str)
    p.add_argument("--max_samples_per_task", type=int, default=40,
                   help="Cap per task (10 tasks/suite × 40 = 400 per suite, 4 suites = 1600 total)")
    p.add_argument("--max_per_episode", type=int, default=6)
    p.add_argument("--suites", default=",".join(SUITES))
    args = p.parse_args()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    for suite in suites:
        port = PORT_MAP.get(suite)
        if port is None:
            print(f"[SKIP] no port for {suite}")
            continue
        print(f"\n=== {suite} (port {port}) ===")
        data = collect_suite(suite, port,
                              max_samples_per_task=args.max_samples_per_task,
                              max_per_episode=args.max_per_episode)
        if data is not None:
            out = Path(args.out_dir) / f"groot_chunks_{suite}.npz"
            np.savez(out, **data)
            print(f"  saved → {out} ({data['groot_chunks'].shape})")


if __name__ == "__main__":
    main()
