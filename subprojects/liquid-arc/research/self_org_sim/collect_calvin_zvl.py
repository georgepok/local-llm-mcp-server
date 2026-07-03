"""Pre-collect (z_vl, expert_chunk) sequences from CALVIN validation episodes.

For each language-annotated segment in CALVIN validation:
  For each frame at t (stride exec_horizon=8):
    - Load obs (rgb_static, rgb_gripper, robot_obs, language)
    - Query GR00T → z_vl[2048]
    - Save expert chunk = actions[t:t+16] from the original CALVIN .npz files

Output: /tmp/calvin_zvl_episodes.npz
  - episode_starts: [N] start index into z_vls / chunks arrays
  - episode_lengths: [N] turns per episode
  - z_vls: [total_turns, 2048] float32
  - chunks: [total_turns, 16, 7] float32 — expert actions per turn
  - languages: [N] str (one per episode)
  - frame_idxs: [total_turns] absolute CALVIN frame indices (for debug)

Substrate trains via BPTT through z_vl sequences, modulating each z_vl
toward producing the expert chunk via GR00T's frozen action head.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pickle
import zmq

CALVIN_ROOT = Path("/home/pokazge/calvin")
SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SELF_DIR))

# Reuse adapter
from rollout_calvin_zeroshot import calvin_obs_to_groot  # type: ignore


def load_calvin_frame(frame_idx, split="validation"):
    """Load CALVIN frame from per-frame .npz (returns obs dict)."""
    paths = [
        CALVIN_ROOT / "dataset" / "calvin_debug_dataset" / split / f"episode_{frame_idx:07d}.npz",
        CALVIN_ROOT / "dataset" / "task_D_D" / split / f"episode_{frame_idx:07d}.npz",
    ]
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        return None
    d = np.load(path)
    return {
        "rgb_obs": {
            "rgb_static": np.asarray(d["rgb_static"]),
            "rgb_gripper": np.asarray(d["rgb_gripper"]),
        },
        "robot_obs": np.asarray(d["robot_obs"]),
        "actions": np.asarray(d.get("actions", d.get("rel_actions"))),  # 7-d action
        "rel_actions": np.asarray(d.get("rel_actions", d.get("actions"))),
    }


def load_segments(split="validation"):
    ann_path = (CALVIN_ROOT / "dataset" / "task_D_D" / split
                 / "lang_annotations" / "auto_lang_ann.npy")
    if not ann_path.exists():
        ann_path = (CALVIN_ROOT / "dataset" / "calvin_debug_dataset" / split
                     / "lang_annotations" / "auto_lang_ann.npy")
    d = np.load(ann_path, allow_pickle=True).item()
    out = []
    for i in range(len(d["language"]["ann"])):
        s, e = d["info"]["indx"][i]
        out.append({"start": int(s), "end": int(e),
                     "lang": d["language"]["ann"][i],
                     "task": d["language"]["task"][i]})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groot_port", type=int, default=5559)
    p.add_argument("--split", default="validation")
    p.add_argument("--max_segments", type=int, default=200)
    p.add_argument("--stride", type=int, default=8, help="Turn stride in env frames")
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--out_path", default="/tmp/calvin_zvl_episodes.npz")
    args = p.parse_args()

    segments = load_segments(args.split)
    if args.max_segments > 0:
        segments = segments[:args.max_segments]
    print(f"[collect] {len(segments)} segments from {args.split}")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 60000)
    sock.connect(f"tcp://localhost:{args.groot_port}")
    print(f"[collect] connected GR00T port {args.groot_port}")

    z_vls = []
    chunks = []
    frame_idxs = []
    episode_starts = []
    episode_lengths = []
    languages = []

    t_start = time.time()
    for seg_i, seg in enumerate(segments):
        ep_start = len(z_vls)
        # Turn frames at stride within segment
        turn_frames = list(range(seg["start"], seg["end"] - args.action_horizon + 1, args.stride))
        if len(turn_frames) < 2:
            continue
        for t_idx, fidx in enumerate(turn_frames):
            obs = load_calvin_frame(fidx, args.split)
            if obs is None:
                continue
            # Query GR00T for z_vl
            groot_obs, _ = calvin_obs_to_groot(obs, seg["lang"])
            sock.send(pickle.dumps({"op": "get_action_with_state", "obs": groot_obs}))
            resp = pickle.loads(sock.recv())
            z_vls.append(resp["z_vl"].astype(np.float32))
            # Expert chunk = ground-truth actions[fidx : fidx + action_horizon]
            chunk = np.zeros((args.action_horizon, 7), dtype=np.float32)
            for k in range(args.action_horizon):
                f = load_calvin_frame(fidx + k, args.split)
                if f is None:
                    break
                chunk[k] = f["rel_actions"].astype(np.float32)
            chunks.append(chunk)
            frame_idxs.append(fidx)
        ep_len = len(z_vls) - ep_start
        if ep_len > 0:
            episode_starts.append(ep_start)
            episode_lengths.append(ep_len)
            languages.append(seg["lang"])
        if (seg_i + 1) % 10 == 0:
            rate = (seg_i + 1) / (time.time() - t_start)
            print(f"  [{seg_i+1}/{len(segments)}] {rate:.2f} seg/s, "
                  f"ETA {(len(segments) - seg_i - 1) / max(rate, 0.01):.0f}s, "
                  f"total turns: {len(z_vls)}")

    print(f"\n[collect] done. {len(z_vls)} turns across {len(episode_starts)} episodes")
    np.savez(args.out_path,
             z_vls=np.stack(z_vls).astype(np.float32),
             chunks=np.stack(chunks).astype(np.float32),
             frame_idxs=np.array(frame_idxs, dtype=np.int64),
             episode_starts=np.array(episode_starts, dtype=np.int64),
             episode_lengths=np.array(episode_lengths, dtype=np.int64),
             languages=np.array(languages, dtype=object))
    print(f"[collect] saved → {args.out_path}")


if __name__ == "__main__":
    main()
