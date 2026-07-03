"""Pre-collect z_vl sequences from LIBERO expert demos for substrate self-sup training.

For each LIBERO suite × N episodes:
  For each frame at stride exec_horizon=8 within episode:
    - Build groot_obs from saved (img, wrist, state8, lang)
    - Query GR00T server → z_vl[2048]
    - Save sequence per episode

Output: /tmp/libero_zvl_episodes.npz
  - z_vls: [total_turns, 2048]
  - episode_starts: [N_episodes]
  - episode_lengths: [N_episodes]
"""
from __future__ import annotations
import argparse
import sys
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
    for k, arr in [("image", img_256), ("wrist_image", wrist_256)]:
        obs["video"][k] = arr[None, None, ...]
    for k, (lo, hi) in state_slots.items():
        obs["state"][k] = st8[lo:hi].astype(np.float32)[None, None, :]
    obs["language"]["annotation.human.action.task_description"] = [[task]]
    return obs


def collect_suite(suite_name, port, max_episodes=50, stride=8):
    suite_short = suite_name.replace("libero_", "")
    sd = DATASET_ROOT / f"libero-{suite_short}-expert-v1"
    if not sd.exists():
        print(f"  skip {suite_name}: dir not found")
        return None
    idx = np.load(sd / "index.npz")
    starts = idx["episode_starts"]
    lengths = idx["episode_lengths"]
    task_indices = idx["task_indices"]
    success = idx.get("success_per_episode", np.ones(len(lengths), dtype=bool))
    n_total = int(idx["n_total"])
    img_size = int(idx["img_size"])

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

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 60000)
    sock.connect(f"tcp://localhost:{port}")
    print(f"  {suite_name}: connected port {port}")

    # Pick first N successful episodes
    succ_eps = [i for i in range(len(lengths)) if bool(success[i])][:max_episodes]
    z_vl_seqs = []  # list of [T_ep, 2048] arrays
    t_start = time.time()
    for ep_i, ep in enumerate(succ_eps):
        ep_start = int(starts[ep])
        ep_len = int(lengths[ep])
        task_id = int(task_indices[ep])
        lang = (task_lang.get(str(task_id)) or task_lang.get(task_id)
                 or "do the task")
        # Flip images per LIBERO/GR00T convention
        ep_z_vls = []
        for t in range(0, ep_len, stride):
            gi = ep_start + t
            img = np.array(imgs[gi])[::-1, ::-1].copy()
            wri = np.array(wrists[gi])[::-1, ::-1].copy()
            st = np.array(states[gi])
            obs = build_groot_obs(img, wri, st, lang)
            sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs}))
            resp = pickle.loads(sock.recv())
            ep_z_vls.append(resp["z_vl"].astype(np.float32))
        if len(ep_z_vls) >= 2:
            z_vl_seqs.append(np.stack(ep_z_vls))
        if (ep_i + 1) % 10 == 0:
            rate = (ep_i + 1) / (time.time() - t_start)
            print(f"    [{ep_i+1}/{len(succ_eps)}] {rate:.2f} eps/s")
    sock.close()
    return z_vl_seqs  # list of [T_ep, 2048]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suites", default="libero_10,libero_spatial,libero_object,libero_goal")
    p.add_argument("--max_episodes_per_suite", type=int, default=50)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--out_path", default="/tmp/libero_zvl_episodes.npz")
    args = p.parse_args()

    all_seqs = []  # list of [T_ep, 2048]
    for suite in [s.strip() for s in args.suites.split(",") if s.strip()]:
        port = SUITE_PORTS.get(suite)
        if port is None:
            continue
        print(f"=== {suite} (port {port}) ===")
        seqs = collect_suite(suite, port, args.max_episodes_per_suite, args.stride)
        if seqs:
            all_seqs.extend(seqs)

    print(f"\n[collect] {len(all_seqs)} episode sequences")
    # Save as flat array + episode_starts/lengths
    z_vls = np.concatenate(all_seqs, axis=0).astype(np.float32)
    episode_lengths = np.array([len(s) for s in all_seqs], dtype=np.int64)
    episode_starts = np.concatenate([[0], np.cumsum(episode_lengths)[:-1]]).astype(np.int64)
    np.savez(args.out_path, z_vls=z_vls,
             episode_starts=episode_starts, episode_lengths=episode_lengths)
    print(f"[collect] z_vls shape {z_vls.shape}, "
          f"avg episode len {z_vls.shape[0] / len(all_seqs):.1f}")
    print(f"[collect] saved → {args.out_path}")


if __name__ == "__main__":
    main()
