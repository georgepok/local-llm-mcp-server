"""Zero-shot CALVIN probe: GR00T-LIBERO checkpoint on CALVIN tasks.

Goal of this script: see if GR00T-LIBERO does ANYTHING meaningful on CALVIN
(reasonable arm motion, attempts to approach object, attempts grasp) vs pure
garbage. If even partial, we have a CALVIN→GR00T integration path and can
test the substrate-as-goal-tracker there. If garbage, we need GR00T-CALVIN
finetune (multi-day).

Embodiment notes:
- CALVIN action: [dx, dy, dz, droll, dpitch, dyaw, gripper] — SAME as LIBERO
- CALVIN robot_obs[15]: [eef_xyz(3), eef_rpy(3), gripper_width(1), arm_joints(7), grip(1)]
- CALVIN rgb_static: 200x200; rgb_gripper: 84x84 → resize to 256x256 for GR00T
- LIBERO state8 -> [x,y,z,roll,pitch,yaw,grip_l,grip_r] (2-d gripper)
  CALVIN gripper_width is scalar; we mirror it across grip_l, grip_r
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
sys.path.insert(0, str(CALVIN_ROOT / "calvin_env"))
sys.path.insert(0, str(SELF_DIR))

import hydra
from omegaconf import OmegaConf
try:
    import cv2
except ImportError:
    cv2 = None


def calvin_obs_to_groot(calvin_obs, language, target_img=256):
    """Convert CALVIN env obs dict → GR00T server expected dict."""
    rgb_static = np.asarray(calvin_obs["rgb_obs"]["rgb_static"]).astype(np.uint8)
    rgb_gripper = np.asarray(calvin_obs["rgb_obs"]["rgb_gripper"]).astype(np.uint8)
    rob = np.asarray(calvin_obs["robot_obs"]).astype(np.float32)
    # robot_obs: [eef_x, y, z, roll, pitch, yaw, gripper_width, joint x7]
    eef_xyz = rob[:3]
    rpy = rob[3:6]
    grip_w = float(rob[6])
    # LIBERO state8: [x, y, z, roll, pitch, yaw, gl, gr]
    state8 = np.concatenate([eef_xyz, rpy, [grip_w / 2.0, grip_w / 2.0]]).astype(np.float32)

    def resize(arr, sz):
        if arr.shape[0] == sz and arr.shape[1] == sz:
            return arr
        if cv2 is not None:
            return cv2.resize(arr, (sz, sz), interpolation=cv2.INTER_LINEAR)
        # Fallback to numpy nearest
        h, w = arr.shape[:2]
        ih = (np.arange(sz) * h / sz).astype(np.int64)
        iw = (np.arange(sz) * w / sz).astype(np.int64)
        return arr[ih][:, iw]

    img = resize(rgb_static, target_img)
    wri = resize(rgb_gripper, target_img)

    # GR00T expects [B, T, H, W, 3] and [B, T, D] shapes
    obs = {
        "video": {
            "image": img[None, None, ...],
            "wrist_image": wri[None, None, ...],
        },
        "state": {
            "x": state8[0:1][None, None, :],
            "y": state8[1:2][None, None, :],
            "z": state8[2:3][None, None, :],
            "roll": state8[3:4][None, None, :],
            "pitch": state8[4:5][None, None, :],
            "yaw": state8[5:6][None, None, :],
            "gripper": state8[6:8][None, None, :],
        },
        "language": {
            "annotation.human.action.task_description": [[language]],
        },
    }
    return obs, state8


def make_task_oracle():
    """Load CALVIN's task success oracle (PlayTableTaskMonitor)."""
    conf_dir = CALVIN_ROOT / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    return hydra.utils.instantiate(task_cfg)


def make_env():
    dataset = CALVIN_ROOT / "dataset" / "calvin_debug_dataset"
    hydra_cfg = dataset / "validation" / ".hydra" / "merged_config.yaml"
    if not hydra_cfg.exists():
        hydra_cfg = dataset / "training" / ".hydra" / "merged_config.yaml"
    cfg = OmegaConf.load(hydra_cfg)
    env_cfg = cfg.env
    env_cfg["scene_cfg"]["data_path"] = str(CALVIN_ROOT / "calvin_env" / "data")
    env_cfg["use_egl"] = True
    env_cfg["show_gui"] = False
    env_cfg["use_scene_info"] = True  # needed for task_oracle's scene_info lookup
    if "cameras" in env_cfg and "tactile" in env_cfg["cameras"]:
        del env_cfg["cameras"]["tactile"]
    return hydra.utils.instantiate(env_cfg)


def load_tasks(split="validation"):
    """Return list of (task_name, language, start_frame, end_frame) per annotation."""
    ann = np.load(
        CALVIN_ROOT / "dataset" / "calvin_debug_dataset" / split
        / "lang_annotations" / "auto_lang_ann.npy",
        allow_pickle=True,
    ).item()
    tasks = []
    for i in range(len(ann["language"]["ann"])):
        lang = ann["language"]["ann"][i]
        task = ann["language"]["task"][i]
        start, end = ann["info"]["indx"][i]
        tasks.append({"task": task, "lang": lang, "start": int(start), "end": int(end)})
    return tasks


def load_episode_state(start_frame):
    """Load scene/robot state from expert episode's start frame for env reset."""
    # Search across both debug + task_D_D, validation + training
    candidates = [
        CALVIN_ROOT / "dataset" / "calvin_debug_dataset" / "validation" / f"episode_{start_frame:07d}.npz",
        CALVIN_ROOT / "dataset" / "calvin_debug_dataset" / "training" / f"episode_{start_frame:07d}.npz",
        CALVIN_ROOT / "dataset" / "task_D_D" / "validation" / f"episode_{start_frame:07d}.npz",
        CALVIN_ROOT / "dataset" / "task_D_D" / "training" / f"episode_{start_frame:07d}.npz",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"episode_{start_frame:07d}.npz not found in any split")
    d = np.load(path)
    return {
        "robot_obs": np.asarray(d["robot_obs"]),
        "scene_obs": np.asarray(d["scene_obs"]),
    }


def run_episode(env, task, groot_socket, task_oracle=None, max_steps=120,
                  exec_horizon=8, diag=False):
    """Run one CALVIN task with GR00T-LIBERO zero-shot."""
    # Reset env to expert episode's starting scene
    start_state = load_episode_state(task["start"])
    try:
        env.reset(robot_obs=start_state["robot_obs"], scene_obs=start_state["scene_obs"])
    except TypeError:
        obs = env.reset()

    settle_action = np.zeros(7, dtype=np.float32)
    settle_action[-1] = 1.0  # CALVIN requires gripper ∈ {-1, +1}
    obs, _, _, _ = env.step(settle_action)
    start_info = env.get_info() if task_oracle is not None else None

    chunk = None
    chunk_idx = 0
    eef_first = None
    eef_last = None
    grip_first = None
    grip_last = None
    n_groot_calls = 0
    last_gripper = 1  # CALVIN requires ±1; default to open
    for step in range(max_steps):
        if chunk is None or chunk_idx >= exec_horizon or chunk_idx >= len(chunk):
            groot_obs, state8 = calvin_obs_to_groot(obs, task["lang"])
            groot_socket.send(pickle.dumps({"op": "get_action_with_state",
                                              "obs": groot_obs}))
            resp = pickle.loads(groot_socket.recv())
            chunk = np.asarray(resp["chunk"], dtype=np.float32)
            chunk_idx = 0
            n_groot_calls += 1
            if eef_first is None:
                eef_first = state8[:3].copy()
                grip_first = float(obs["robot_obs"][6])
        action7 = chunk[chunk_idx].copy()
        g = action7[-1]
        # CALVIN requires gripper ∈ {-1, +1}; persist last when ambiguous
        if abs(g) > 0.1:
            last_gripper = int(np.sign(g))
        action7[-1] = last_gripper
        result = env.step(action7.astype(np.float32))
        if isinstance(result, tuple):
            obs = result[0]
        chunk_idx += 1
        eef_last = np.asarray(obs["robot_obs"][:3]).copy()
        grip_last = float(obs["robot_obs"][6])
    # Check CALVIN task success
    succeeded = False
    if task_oracle is not None and start_info is not None:
        try:
            current_info = env.get_info()
            done_tasks = task_oracle.get_task_info_for_set(
                start_info, current_info, {task["task"]},
            )
            succeeded = len(done_tasks) > 0
        except Exception as e:
            print(f"  [warn] task_oracle check failed: {type(e).__name__}: {e}")
    eef_motion = float(np.linalg.norm(eef_last - eef_first)) if eef_first is not None else 0
    return {
        "n_groot_calls": n_groot_calls,
        "eef_first": eef_first.tolist() if eef_first is not None else None,
        "eef_last": eef_last.tolist() if eef_last is not None else None,
        "eef_motion_m": eef_motion,
        "grip_first": grip_first,
        "grip_last": grip_last,
        "succeeded": succeeded,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groot_port", type=int, default=5555)
    p.add_argument("--max_steps", type=int, default=120,
                   help="Env steps per task (CALVIN episodes are ~64 expert steps)")
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--n_tasks", type=int, default=8,
                   help="Number of distinct tasks to probe")
    args = p.parse_args()

    ctx = zmq.Context.instance()
    gs = ctx.socket(zmq.REQ)
    gs.setsockopt(zmq.RCVTIMEO, 60000)
    gs.connect(f"tcp://localhost:{args.groot_port}")
    print(f"[probe] connected to GR00T server on port {args.groot_port}")

    env = make_env()
    print(f"[probe] CALVIN env instantiated")
    task_oracle = make_task_oracle()
    print(f"[probe] task oracle: {type(task_oracle).__name__}")

    tasks = load_tasks("validation")[:args.n_tasks]
    print(f"[probe] running {len(tasks)} CALVIN tasks zero-shot:")
    for t in tasks:
        print(f"  - {t['task']:<30}  '{t['lang']}'")

    print(f"\n{'#':>2} {'task':<30} {'lang':<45} {'eef_m':>7} {'grip_Δ':>8} {'succ':>5}")
    print("-" * 100)
    n_succ = 0
    for i, t in enumerate(tasks):
        result = run_episode(env, t, gs, task_oracle=task_oracle,
                              max_steps=args.max_steps,
                              exec_horizon=args.exec_horizon)
        eef_m = result["eef_motion_m"]
        grip_d = (result["grip_last"] - result["grip_first"]
                   if result["grip_first"] is not None and result["grip_last"] is not None
                   else 0)
        succ = "YES" if result.get("succeeded") else " no"
        if result.get("succeeded"):
            n_succ += 1
        print(f"{i:>2} {t['task']:<30} {t['lang'][:45]:<45} {eef_m:>7.3f} {grip_d:>+8.3f} {succ:>5}")

    print(f"\n[probe] zero-shot results: {n_succ}/{len(tasks)} = {100*n_succ/len(tasks):.0f}% success")
    print(f"  Interpretation:")
    print(f"    succ ≥ 25%: GR00T-LIBERO works zero-shot, integration valid, can test substrate")
    print(f"    succ < 25% but eef_motion > 0.05m: model attempting tasks, finetune likely needed")
    print(f"    eef_motion < 0.05m on most: pure noise, finetune required")


if __name__ == "__main__":
    main()
