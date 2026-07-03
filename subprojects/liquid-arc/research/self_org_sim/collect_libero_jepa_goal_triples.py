"""Collect (z_t, action_chunk_t, z_goal, z_{t+1}) tuples for goal-conditioned JEPA-LGT.

z_goal is extracted ONCE per task_id from the FINAL frame of one successful expert demo;
held constant across all triples within episodes of that task.

Output: /tmp/libero_jepa_goal_triples.npz
  z_t      [N, 2048]
  chunks   [N, 16, 7]
  z_next   [N, 2048]
  z_goal   [N, 2048]   (broadcast from per-task z_goal)
  ep_id    [N]
  suite    [N]
  task_id  [N]
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


def collect_suite(suite_name, port, max_episodes, stride=16, action_horizon=16, episode_offset=0):
    suite_short = suite_name.replace("libero_", "")
    sd = DATASET_ROOT / f"libero-{suite_short}-expert-v1"
    if not sd.exists():
        print(f"  skip {suite_name}: dir not found"); return []
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
    lang_path = sd / "task_languages.json"
    if lang_path.exists():
        import json
        task_lang = json.loads(lang_path.read_text())
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

    # Step 1: extract z_goal per task_id (from first successful ep's last frame)
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
            obs = build_groot_obs(img, wri, st, lang)
            sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs}))
            resp = pickle.loads(sock.recv())
            z_goal_per_task[task_id] = resp["z_vl"].astype(np.float32)
            break
    print(f"  {suite_name}: extracted z_goal for {len(z_goal_per_task)} tasks", flush=True)

    # Step 2: triples with z_goal per sample (broadcast from per-task)
    all_succ = [i for i in range(len(lengths)) if bool(success[i])]
    # Task-balanced sampling: group successful eps by task_id, take a slice from each
    by_task: dict = {}
    for ep in all_succ:
        tid = int(task_indices[ep])
        by_task.setdefault(tid, []).append(ep)
    n_tasks = max(1, len(by_task))
    per_task_total = max(1, max_episodes // n_tasks)
    succ_eps = []
    for tid in sorted(by_task.keys()):
        task_eps = by_task[tid]
        succ_eps.extend(task_eps[episode_offset:episode_offset + per_task_total])
    succ_eps = succ_eps[:max_episodes]
    triples = []
    t_start = time.time()
    for ep_i, ep in enumerate(succ_eps):
        ep_start = int(starts[ep]); ep_len = int(lengths[ep])
        task_id = int(task_indices[ep])
        if task_id not in z_goal_per_task:
            continue
        z_goal = z_goal_per_task[task_id]
        lang = (task_lang.get(str(task_id)) or task_lang.get(task_id) or "do the task")
        prev_z = None
        prev_chunk = None
        prev_chunk_groot = None
        prev_state8 = None
        for t in range(0, ep_len - action_horizon, stride):
            key = (ep, t)
            if key not in lookup:
                continue
            gi = ep_start + t
            img = np.array(imgs[gi])[::-1, ::-1].copy()
            wri = np.array(wrists[gi])[::-1, ::-1].copy()
            st = np.array(states[gi]).astype(np.float32)
            chunk_expert = np.array(chunks_mm[lookup[key]])
            obs = build_groot_obs(img, wri, st, lang)
            sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs}))
            resp = pickle.loads(sock.recv())
            z = resp["z_vl"].astype(np.float32)
            # GR00T's chunk for this obs — what it WOULD execute
            chunk_groot = resp["chunk"].astype(np.float32)  # [16, 7]
            if prev_z is not None:
                triples.append({
                    "z_t": prev_z, "chunk_t": prev_chunk, "z_next": z,
                    "z_goal": z_goal, "ep_id": ep, "suite": suite_name,
                    "task_id": task_id,
                    "state8_t": prev_state8, "state8_next": st,
                    "chunk_groot_t": prev_chunk_groot,
                })
            prev_z = z
            prev_chunk = chunk_expert
            prev_chunk_groot = chunk_groot
            prev_state8 = st
        if (ep_i + 1) % 5 == 0:
            rate = (ep_i + 1) / (time.time() - t_start)
            print(f"    [{ep_i+1}/{len(succ_eps)}] {rate:.2f} eps/s, "
                  f"triples so far={len(triples)}", flush=True)
    sock.close()
    return triples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suites", default="libero_10,libero_spatial,libero_object,libero_goal")
    p.add_argument("--max_episodes_per_suite", type=int, default=30)
    p.add_argument("--episode_offset", type=int, default=0,
                   help="Skip first N successful episodes (for held-out splits).")
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--out_path", default="/tmp/libero_jepa_goal_triples.npz")
    args = p.parse_args()

    all_triples = []
    for suite in [s.strip() for s in args.suites.split(",") if s.strip()]:
        port = SUITE_PORTS.get(suite)
        if port is None:
            continue
        print(f"=== {suite} (port {port}) ===", flush=True)
        triples = collect_suite(suite, port, args.max_episodes_per_suite,
                                  args.stride, episode_offset=args.episode_offset)
        print(f"  → {len(triples)} triples from {suite}", flush=True)
        all_triples.extend(triples)

    print(f"\n[collect-goal] {len(all_triples)} triples total", flush=True)
    z_t = np.stack([t["z_t"] for t in all_triples]).astype(np.float32)
    chunks = np.stack([t["chunk_t"] for t in all_triples]).astype(np.float32)
    z_next = np.stack([t["z_next"] for t in all_triples]).astype(np.float32)
    z_goal = np.stack([t["z_goal"] for t in all_triples]).astype(np.float32)
    ep_ids = np.array([t["ep_id"] for t in all_triples], dtype=np.int64)
    suites_arr = np.array([t["suite"] for t in all_triples])
    task_ids = np.array([t["task_id"] for t in all_triples], dtype=np.int64)
    state8_t = np.stack([t["state8_t"] for t in all_triples]).astype(np.float32)
    state8_next = np.stack([t["state8_next"] for t in all_triples]).astype(np.float32)
    chunk_groot = np.stack(
        [t.get("chunk_groot_t", np.zeros_like(t["chunk_t"])) for t in all_triples]
    ).astype(np.float32)
    np.savez(args.out_path, z_t=z_t, chunks=chunks, z_next=z_next, z_goal=z_goal,
             ep_id=ep_ids, suite=suites_arr, task_id=task_ids,
             state8_t=state8_t, state8_next=state8_next,
             chunk_groot=chunk_groot)
    print(f"[collect-goal] z_t {z_t.shape}, z_goal {z_goal.shape}, "
          f"state8_t {state8_t.shape}, chunk_groot {chunk_groot.shape}",
          flush=True)
    print(f"[collect-goal] saved → {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
