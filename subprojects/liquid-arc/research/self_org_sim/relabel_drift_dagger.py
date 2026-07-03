"""DAgger relabel: query GR00T live on Liquid-drifted obs.

Input: .npz with imgs, wrists, states, langs, suites, steps (from rollout_libero_v11_client.py --save_drift_obs).
Output: .npz with the same fields PLUS:
  - z_state [N, 1536]
  - z_vl_bank [N, K_bank=4, 1024]  (GR00T's traj_model_output, depth-subsampled)
  - teacher_chunks [N, 16, 7]      (GR00T's predicted action chunk)
  - z_vl [N, 2048]                  (kept for completeness)

Then a DAgger fine-tune script can mix this with libero-*-expert-v1 to retrain v15.
"""
from __future__ import annotations
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import zmq


def build_groot_obs(img_256, wrist_256, state8, task_lang: str):
    state_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    video_keys = ["image", "wrist_image"]
    state_slots = {
        "x": (0, 1), "y": (1, 2), "z": (2, 3),
        "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6),
        "gripper": (6, 8),
    }
    obs = {"video": {}, "state": {}, "language": {}}
    for k in video_keys:
        arr = img_256 if k == "image" else wrist_256
        obs["video"][k] = arr[None, None, ...]
    for k in state_keys:
        lo, hi = state_slots[k]
        obs["state"][k] = state8[lo:hi].astype(np.float32)[None, None, :]
    obs["language"]["annotation.human.action.task_description"] = [[task_lang]]
    return obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=str, help="drift obs npz from rollout client")
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--groot_port", type=int, default=5557)
    p.add_argument("--depth_indices", default="0,1,2,3", type=str)
    p.add_argument("--limit", type=int, default=0, help="If >0, only process first N entries (debug)")
    args = p.parse_args()

    depth_idxs = [int(x) for x in args.depth_indices.split(",") if x.strip()]
    print(f"[relabel] loading {args.input}")
    data = np.load(args.input, allow_pickle=True)
    imgs = data["imgs"]
    wrists = data["wrists"]
    states = data["states"]
    langs = data["langs"]
    suites = data["suites"]
    steps = data["steps"]
    N = len(imgs)
    if args.limit > 0:
        N = min(N, args.limit)
    print(f"[relabel] {N} entries, imgs shape {imgs.shape}, states shape {states.shape}")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 60000)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://localhost:{args.groot_port}")
    print(f"[relabel] connected to GR00T port {args.groot_port}")

    z_vls = []
    z_states = []
    z_vl_banks = []
    teacher_chunks = []
    t_start = time.time()
    for i in range(N):
        obs_dict = build_groot_obs(imgs[i], wrists[i], states[i], str(langs[i]))
        try:
            sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs_dict}))
            resp = pickle.loads(sock.recv())
        except zmq.ZMQError as e:
            print(f"[relabel] ZMQ error at i={i}: {e}; reconnecting")
            sock.close(linger=0)
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 60000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://localhost:{args.groot_port}")
            sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs_dict}))
            resp = pickle.loads(sock.recv())
        if "error" in resp:
            raise RuntimeError(f"groot error at i={i}: {resp['error']}")

        z_vls.append(resp["z_vl"].astype(np.float32))
        z_states.append(resp["z_state"].astype(np.float32))
        full_traj = resp["traj_model_output"]  # [N_steps, hidden_dim]
        z_vl_bank = np.stack([full_traj[d] for d in depth_idxs]).astype(np.float32)
        z_vl_banks.append(z_vl_bank)
        teacher_chunks.append(resp["chunk"].astype(np.float32))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"[relabel] {i+1}/{N}  rate={rate:.1f}/s  eta={eta:.0f}s")

    z_vls = np.stack(z_vls)
    z_states = np.stack(z_states)
    z_vl_banks = np.stack(z_vl_banks)
    teacher_chunks = np.stack(teacher_chunks)

    print(f"\n[relabel] done in {time.time()-t_start:.1f}s")
    print(f"  z_vl: {z_vls.shape}")
    print(f"  z_state: {z_states.shape}")
    print(f"  z_vl_bank: {z_vl_banks.shape}")
    print(f"  teacher_chunks: {teacher_chunks.shape}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        imgs=imgs[:N], wrists=wrists[:N], states=states[:N],
        langs=langs[:N], suites=suites[:N], steps=steps[:N],
        z_vl=z_vls, z_state=z_states, z_vl_bank=z_vl_banks,
        teacher_chunks=teacher_chunks,
    )
    print(f"[relabel] saved → {out}")


if __name__ == "__main__":
    main()
