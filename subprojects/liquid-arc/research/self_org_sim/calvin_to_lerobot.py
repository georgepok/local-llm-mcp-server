"""Convert CALVIN dataset to LeRobot v2 format for GR00T finetune.

CALVIN structure:
  /home/pokazge/calvin/dataset/task_D_D/training/episode_NNNNNNN.npz
  /home/pokazge/calvin/dataset/task_D_D/validation/episode_NNNNNNN.npz
  Each .npz has: rgb_static (200,200,3) uint8, rgb_gripper (84,84,3) uint8,
                  depth_static, depth_gripper, robot_obs (15,) float, scene_obs,
                  actions (7,) float (relative deltas + binary gripper),
                  rel_actions (7,) ...
  Language annotations in {split}/lang_annotations/auto_lang_ann.npy as dict:
    {'language': {'ann': [str], 'task': [str], 'emb': [N, 384]},
     'info': {'indx': [(start_frame, end_frame), ...]}}

We extract language-annotated SEGMENTS (each segment = one task instance,
~64 frames typically) and convert each segment as one episode in LeRobot v2.

LIBERO-compatible state8: [eef_xyz(3), eef_rpy(3), gripper_l(1), gripper_r(1)]
(CALVIN gripper width is 1-d; we mirror it to grip_l/grip_r per LIBERO convention)
Action7: [dx, dy, dz, droll, dpitch, dyaw, gripper] — same shape as LIBERO.

Use libero_demo's modality config (state slots: x, y, z, roll, pitch, yaw, gripper)
since action+state shapes align.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


CALVIN_ROOT = Path("/home/pokazge/calvin/dataset/task_D_D")
TARGET_IMG_SIZE = 256  # GR00T expects 256x256 (LIBERO format)


def load_segments(split):
    """Return list of (start_frame, end_frame, language, task_name) for split."""
    ann_path = CALVIN_ROOT / split / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_path.exists():
        return []
    d = np.load(ann_path, allow_pickle=True).item()
    out = []
    for i in range(len(d["language"]["ann"])):
        lang = d["language"]["ann"][i]
        task = d["language"]["task"][i]
        start, end = d["info"]["indx"][i]
        out.append({"start": int(start), "end": int(end),
                     "lang": lang, "task": task, "split": split})
    return out


def load_frame_data(split, frame_idx):
    """Load one CALVIN frame from .npz."""
    path = CALVIN_ROOT / split / f"episode_{frame_idx:07d}.npz"
    d = np.load(path)
    rgb_static = np.asarray(d["rgb_static"], dtype=np.uint8)
    rgb_gripper = np.asarray(d["rgb_gripper"], dtype=np.uint8)
    robot_obs = np.asarray(d["robot_obs"], dtype=np.float32)
    # Prefer rel_actions over actions (relative deltas, matches LIBERO format)
    if "rel_actions" in d.files:
        action = np.asarray(d["rel_actions"], dtype=np.float32)
    else:
        action = np.asarray(d["actions"], dtype=np.float32)
    return rgb_static, rgb_gripper, robot_obs, action


def make_state8(robot_obs):
    """CALVIN robot_obs[15] → LIBERO-style state8.
    robot_obs: [eef_x, y, z, roll, pitch, yaw, gripper_width, joint x7]
    state8: [x, y, z, roll, pitch, yaw, grip_l, grip_r]
    """
    eef_xyz = robot_obs[:3]
    rpy = robot_obs[3:6]
    grip_w = float(robot_obs[6])
    return np.concatenate([eef_xyz, rpy, [grip_w / 2.0, grip_w / 2.0]]).astype(np.float32)


def resize(img, sz):
    if img.shape[0] == sz and img.shape[1] == sz:
        return img
    return cv2.resize(img, (sz, sz), interpolation=cv2.INTER_LINEAR)


def build_modality_config(out_dir):
    """Write meta/modality.json — same as libero_demo."""
    modality = {
        "state": {
            "x": {"start": 0, "end": 1},
            "y": {"start": 1, "end": 2},
            "z": {"start": 2, "end": 3},
            "roll": {"start": 3, "end": 4},
            "pitch": {"start": 4, "end": 5},
            "yaw": {"start": 5, "end": 6},
            "gripper": {"start": 6, "end": 8},
        },
        "action": {
            "x": {"start": 0, "end": 1},
            "y": {"start": 1, "end": 2},
            "z": {"start": 2, "end": 3},
            "roll": {"start": 3, "end": 4},
            "pitch": {"start": 4, "end": 5},
            "yaw": {"start": 5, "end": 6},
            "gripper": {"start": 6, "end": 7},
        },
        "video": {
            "image": {"original_key": "observation.images.image"},
            "wrist_image": {"original_key": "observation.images.wrist_image"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "modality.json").write_text(json.dumps(modality, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="/tmp/calvin_lerobot", type=str)
    p.add_argument("--split", default="validation", choices=["training", "validation"],
                   help="Which CALVIN split to convert (validation is smaller, ~1k segments)")
    p.add_argument("--max_segments", type=int, default=0,
                   help="0 = all segments")
    p.add_argument("--fps", type=int, default=15,
                   help="CALVIN runs at 30Hz nominal but data is decimated")
    p.add_argument("--target_img_size", type=int, default=TARGET_IMG_SIZE)
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"[conv] split={args.split}")
    segments = load_segments(args.split)
    if args.max_segments > 0:
        segments = segments[:args.max_segments]
    print(f"[conv] {len(segments)} language-annotated segments")

    # Build unique task list
    unique_tasks = sorted({s["task"] for s in segments})
    task_to_idx = {t: i for i, t in enumerate(unique_tasks)}
    print(f"[conv] {len(unique_tasks)} unique tasks")

    # LeRobot v2 features (shape as TUPLE — list comparison is buggy in lerobot 0.4.1)
    features = {
        "observation.images.image": {
            "dtype": "video", "shape": (args.target_img_size, args.target_img_size, 3),
            "names": ["height", "width", "rgb"],
        },
        "observation.images.wrist_image": {
            "dtype": "video", "shape": (args.target_img_size, args.target_img_size, 3),
            "names": ["height", "width", "rgb"],
        },
        "observation.state": {
            "dtype": "float32", "shape": (8,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_l", "gripper_r"],
        },
        "action": {
            "dtype": "float32", "shape": (7,),
            "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        },
    }

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    # NOTE: do NOT pre-create the dir — LeRobotDataset.create() requires it absent

    dataset = LeRobotDataset.create(
        repo_id="calvin_d_lerobot",
        fps=args.fps,
        features=features,
        root=out_dir,
        robot_type="franka",
        use_videos=True,
        image_writer_processes=2,
        image_writer_threads=4,
    )

    t_start = time.time()
    for seg_i, seg in enumerate(segments):
        start, end = seg["start"], seg["end"]
        task_idx = task_to_idx[seg["task"]]
        lang = seg["lang"]
        for frame_idx in range(start, end + 1):
            try:
                rgb_static, rgb_gripper, robot_obs, action = load_frame_data(seg["split"], frame_idx)
            except FileNotFoundError:
                # Episode boundary; skip
                continue
            img = resize(rgb_static, args.target_img_size)
            wri = resize(rgb_gripper, args.target_img_size)
            state8 = make_state8(robot_obs)
            # CALVIN action gripper is binary {-1, +1}; just pass through
            frame = {
                "observation.images.image": img,
                "observation.images.wrist_image": wri,
                "observation.state": state8.astype(np.float32),
                "action": action.astype(np.float32),
                "task": lang,
            }
            dataset.add_frame(frame)
        dataset.save_episode()
        if (seg_i + 1) % 20 == 0:
            rate = (seg_i + 1) / (time.time() - t_start)
            print(f"  [{seg_i+1}/{len(segments)}] {rate:.2f} seg/s, "
                  f"ETA {(len(segments) - seg_i - 1) / max(rate, 1e-6):.0f}s")

    print(f"\n[conv] done. {len(segments)} segments in {time.time() - t_start:.0f}s")
    print(f"[conv] dataset root: {out_dir}")

    # Write GR00T modality.json
    build_modality_config(out_dir / "meta")
    print(f"[conv] wrote modality.json")


if __name__ == "__main__":
    main()
