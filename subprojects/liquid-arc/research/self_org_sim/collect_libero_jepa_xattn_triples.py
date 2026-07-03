"""Collect (bb_t, action_chunk_t, z_goal, bb_{t+1}) for X-attn JEPA-LGT.

bb_* are FULL per-token [seq_len, 2048] backbone features.
z_goal is per-task pooled [2048] (kept compact — single goal anchor per task).

Storage:  1059 triples × 2 (current+next) × 139 tokens × 2048 floats × 4 bytes ≈ 2.4 GB
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pickle
import zmq


SUITE_PORTS = {"libero_10": 5555, "libero_spatial": 5556,
               "libero_object": 5557, "libero_goal": 5558}
DATASET_ROOT = Path("/home/pokazge/datasets")


def build_groot_obs(img_256, wrist_256, st8, task):
    state_slots = {"x": (0, 1), "y": (1, 2), "z": (2, 3),
                   "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6),
                   "gripper": (6, 8)}
    obs = {"video": {}, "state": {}, "language": {}}
    obs["video"]["image"] = img_256[None, None, ...]
    obs["video"]["wrist_image"] = wrist_256[None, None, ...]
    for k, (lo, hi) in state_slots.items():
        obs["state"][k] = st8[lo:hi].astype(np.float32)[None, None, :]
    obs["language"]["annotation.human.action.task_description"] = [[task]]
    return obs


def collect_suite(suite_name, port, max_episodes, stride=16, action_horizon=16):
    suite_short = suite_name.replace("libero_", "")
    sd = DATASET_ROOT / f"libero-{suite_short}-expert-v1"
    if not sd.exists():
        return []
    idx = np.load(sd / "index.npz")
    starts = idx["episode_starts"]; lengths = idx["episode_lengths"]
    task_indices = idx["task_indices"]
    success = idx.get("success_per_episode", np.ones(len(lengths), dtype=bool))
    n_total = int(idx["n_total"]); img_size = int(idx["img_size"])
    imgs = np.memmap(sd / "imgs.dat", dtype=np.uint8, mode="r",
                     shape=(n_total, img_size, img_size, 3))
    wrists = np.memmap(sd / "wrists.dat", dtype=np.uint8, mode="r",
                       shape=(n_total, img_size, img_size, 3))
    states = np.memmap(sd / "states.dat", dtype=np.float32, mode="r",
                       shape=(n_total, 8))
    task_lang = {}
    lp = sd / "task_languages.json"
    if lp.exists():
        import json
        task_lang = json.loads(lp.read_text())
    labels = np.load(sd / "labels_index.npz")
    sample_idx = labels["sample_idx"]
    n_samples = int(labels["n_samples"])
    chunks_mm = np.memmap(sd / "teacher_chunks.dat", dtype=np.float32, mode="r",
                          shape=(n_samples, action_horizon, 7))
    lookup = {(int(s[0]), int(s[1])): i for i, s in enumerate(sample_idx)}

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 60000)
    sock.connect(f"tcp://localhost:{port}")

    # Extract per-task z_goal (pooled vl-embed of the final expert frame)
    z_goal_per_task = {}
    unique_tasks = sorted(set(int(t) for t in task_indices))
    for task_id in unique_tasks:
        for ep in range(len(lengths)):
            if int(task_indices[ep]) != task_id or not bool(success[ep]):
                continue
            ep_start = int(starts[ep]); ep_len = int(lengths[ep])
            last_gi = ep_start + ep_len - 1
            img = np.array(imgs[last_gi])[::-1, ::-1].copy()
            wri = np.array(wrists[last_gi])[::-1, ::-1].copy()
            st = np.array(states[last_gi])
            lang = (task_lang.get(str(task_id)) or task_lang.get(task_id)
                    or "do the task")
            o = build_groot_obs(img, wri, st, lang)
            sock.send(pickle.dumps({"op": "get_action_with_state", "obs": o}))
            resp = pickle.loads(sock.recv())
            z_goal_per_task[task_id] = resp["z_vl"].astype(np.float32)
            break
    print(f"  {suite_name}: extracted z_goal for {len(z_goal_per_task)} tasks", flush=True)

    succ_eps = [i for i in range(len(lengths)) if bool(success[i])][:max_episodes]
    triples = []
    t_start = time.time()
    for ep_i, ep in enumerate(succ_eps):
        ep_start = int(starts[ep]); ep_len = int(lengths[ep])
        task_id = int(task_indices[ep])
        if task_id not in z_goal_per_task:
            continue
        z_goal = z_goal_per_task[task_id]
        lang = (task_lang.get(str(task_id)) or task_lang.get(task_id)
                or "do the task")
        prev_bb = None
        prev_chunk = None
        for t in range(0, ep_len - action_horizon, stride):
            key = (ep, t)
            if key not in lookup:
                continue
            gi = ep_start + t
            img = np.array(imgs[gi])[::-1, ::-1].copy()
            wri = np.array(wrists[gi])[::-1, ::-1].copy()
            st = np.array(states[gi])
            chunk = np.array(chunks_mm[lookup[key]])
            o = build_groot_obs(img, wri, st, lang)
            sock.send(pickle.dumps({"op": "get_full_bb_features", "obs": o}))
            resp = pickle.loads(sock.recv())
            bb = resp["bb_full"].astype(np.float32)  # [seq, 2048]
            if prev_bb is not None:
                triples.append({
                    "bb_t": prev_bb, "chunk_t": prev_chunk, "bb_next": bb,
                    "z_goal": z_goal, "ep_id": ep, "suite": suite_name,
                    "task_id": task_id,
                })
            prev_bb = bb
            prev_chunk = chunk
        if (ep_i + 1) % 5 == 0:
            rate = (ep_i + 1) / (time.time() - t_start)
            print(f"    [{ep_i+1}/{len(succ_eps)}] {rate:.2f} eps/s, "
                  f"triples so far={len(triples)}", flush=True)
    sock.close()
    return triples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suites",
                   default="libero_10,libero_spatial,libero_object,libero_goal")
    p.add_argument("--max_episodes_per_suite", type=int, default=20)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--out_path", default="/tmp/libero_jepa_xattn_triples.npz")
    args = p.parse_args()

    all_triples = []
    for suite in [s.strip() for s in args.suites.split(",") if s.strip()]:
        port = SUITE_PORTS.get(suite)
        if port is None:
            continue
        print(f"=== {suite} (port {port}) ===", flush=True)
        triples = collect_suite(suite, port, args.max_episodes_per_suite, args.stride)
        print(f"  → {len(triples)} triples from {suite}", flush=True)
        all_triples.extend(triples)

    print(f"\n[collect-xattn] {len(all_triples)} triples total", flush=True)
    bb_t = np.stack([t["bb_t"] for t in all_triples]).astype(np.float32)
    chunks = np.stack([t["chunk_t"] for t in all_triples]).astype(np.float32)
    bb_next = np.stack([t["bb_next"] for t in all_triples]).astype(np.float32)
    z_goal = np.stack([t["z_goal"] for t in all_triples]).astype(np.float32)
    ep_ids = np.array([t["ep_id"] for t in all_triples], dtype=np.int64)
    suites_arr = np.array([t["suite"] for t in all_triples])
    task_ids = np.array([t["task_id"] for t in all_triples], dtype=np.int64)
    print(f"[collect-xattn] bb_t {bb_t.shape} ({bb_t.nbytes/1e9:.2f} GB), "
          f"bb_next {bb_next.shape}", flush=True)
    np.savez(args.out_path, bb_t=bb_t, chunks=chunks, bb_next=bb_next,
             z_goal=z_goal, ep_id=ep_ids, suite=suites_arr, task_id=task_ids)
    print(f"[collect-xattn] saved → {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
