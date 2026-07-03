"""GR00T-in-sim apprenticeship: run GR00T (via ZMQ server) in LIBERO sim, record trajectories.

Output is a memmap directory in the same format as gen_groot_labels.py /
TeacherLabelDataset, but the obs come from GR00T's actual rollouts in libero_10
sim (not human demos). This is "real apprenticeship" data: state distribution
matches deployment, and labels are from teacher-acting-in-environment.

Layout:
  imgs.dat      uint8  [N, 256, 256, 3]
  wrists.dat    uint8  [N, 256, 256, 3]
  states.dat    f32    [N, 8]
  teacher_chunks.dat  f32  [N, 16, 7]
  index.npz     starts/lengths/task_indices/n_total/img_size, success_per_episode
  labels_index.npz    sample_idx [N, 3] (ep_i, t, ti) — for TeacherLabelDataset compat

Run inside the LIBERO sim venv (CPU torch ok). Make sure groot_server.py is
already running in main venv on the same port.

  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python groot_sim_collect.py --task_suite libero_10 --rollouts_per_task 10 \\
    --max_steps 250 --exec_horizon 8 --port 5555 \\
    --out_dir /home/pokazge/datasets/groot-in-sim-iter1
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import zmq

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

print = functools.partial(print, flush=True)


def query_groot(sock: zmq.Socket, obs_dict: dict) -> np.ndarray:
    sock.send(pickle.dumps(obs_dict))
    resp = pickle.loads(sock.recv())
    if "error" in resp:
        raise RuntimeError(f"server error: {resp['error']}")
    return resp["chunk"]


def get_raw_imgs(obs_raw):
    return (
        obs_raw["agentview_image"][::-1, ::-1].copy(),
        obs_raw["robot0_eye_in_hand_image"][::-1, ::-1].copy(),
    )


def quat2axisangle(quat):
    import math
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def build_state8(obs_raw):
    xyz = obs_raw["robot0_eef_pos"]
    rpy = quat2axisangle(obs_raw["robot0_eef_quat"])
    grip = obs_raw["robot0_gripper_qpos"]
    return np.concatenate([xyz, rpy, grip], axis=0).astype(np.float32)


def build_groot_obs(img_256, wrist_256, state8, task_lang: str,
                    state_keys, video_keys, language_keys, state_slots):
    """Build the (B=1) observation dict in the format groot_server expects."""
    new_obs = {"video": {}, "state": {}, "language": {}}
    for k in video_keys:
        arr = img_256 if k == "image" else wrist_256
        new_obs["video"][k] = arr[None, None, ...]            # (1, 1, H, W, 3)
    for k in state_keys:
        lo, hi = state_slots[k]
        new_obs["state"][k] = state8[lo:hi].astype(np.float32)[None, None, :]
    for lk in language_keys:
        new_obs["language"][lk] = [[task_lang]]
    return new_obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_suite", default="libero_10", type=str)
    p.add_argument("--rollouts_per_task", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=250)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--gripper_sign", type=float, default=1.0)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--out_dir", required=True, type=str)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Connect to GR00T server
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    sock.setsockopt(zmq.RCVTIMEO, 60000)  # 60s timeout per query
    print(f"Connected to GR00T server on tcp://127.0.0.1:{args.port}")

    # GR00T modality config (mirror what server uses)
    state_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    video_keys = ["image", "wrist_image"]
    language_keys = ["annotation.human.action.task_description"]
    state_slots = {
        "x": (0, 1), "y": (1, 2), "z": (2, 3),
        "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6),
        "gripper": (6, 8),
    }

    # libero-r task language list (for sim_id → train_id mapping if we want it later)
    libero_r_tasks = [
        "put the white mug on the left plate and put the yellow and white mug on the right plate",
        "turn on the stove and put the moka pot on it",
        "put both the cream cheese box and the butter in the basket",
        "put both moka pots on the stove",
        "put the black bowl in the bottom drawer of the cabinet and close it",
        "put both the alphabet soup and the cream cheese box in the basket",
        "put both the alphabet soup and the tomato sauce in the basket",
        "put the white mug on the plate and put the chocolate pudding to the right of the plate",
        "pick up the book and place it in the back compartment of the caddy",
        "put the yellow and white mug in the microwave and close it",
    ]

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.get_num_tasks()
    sim_to_r = {}
    for sim_id in range(n_tasks):
        lang = suite.get_task(sim_id).language.strip()
        for r_id, r_lang in enumerate(libero_r_tasks):
            if lang == r_lang.strip():
                sim_to_r[sim_id] = r_id
                break

    trajectories = []
    total_n = 0
    chunks_at_step = []   # list of np.array [16,7] aligned with each saved state
    print(f"Phase 1: GR00T-in-sim collection ({n_tasks} tasks × {args.rollouts_per_task} rollouts)")
    for sim_id in range(n_tasks):
        task = suite.get_task(sim_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = suite.get_task_init_states(sim_id)
        n_rollouts = min(args.rollouts_per_task, len(init_states))
        train_task_id = sim_to_r.get(sim_id, sim_id)
        task_lang = suite.get_task(sim_id).language
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        for r in range(n_rollouts):
            env.reset()
            env.set_init_state(init_states[r])
            obs = None
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

            traj_imgs, traj_wrists, traj_states, traj_chunks = [], [], [], []
            chunk = None
            chunk_idx = 0
            success = False
            n_steps = 0
            for step in range(args.max_steps):
                n_steps = step + 1
                img_raw, wrist_raw = get_raw_imgs(obs)
                state8 = build_state8(obs)

                if chunk is None or chunk_idx >= args.exec_horizon or chunk_idx >= len(chunk):
                    obs_dict = build_groot_obs(img_raw, wrist_raw, state8, task_lang,
                                                state_keys, video_keys, language_keys, state_slots)
                    chunk = query_groot(sock, obs_dict)
                    chunk_idx = 0
                    # Save state-chunk pairing AT THIS STEP (the obs that produced the chunk)
                    traj_imgs.append(img_raw)
                    traj_wrists.append(wrist_raw)
                    traj_states.append(state8)
                    traj_chunks.append(chunk.copy())

                action7 = chunk[chunk_idx].copy()
                g = action7[-1]
                action7[-1] = args.gripper_sign * np.sign(g) if abs(g) > 0.1 else 0.0
                obs, _, done, _ = env.step(action7.astype(np.float32))
                chunk_idx += 1
                if env.check_success():
                    success = True
                    break
                if done:
                    break

            trajectories.append({
                "task_idx": train_task_id,
                "sim_id": sim_id,
                "n_steps": len(traj_imgs),
                "success": success,
                "imgs": traj_imgs, "wrists": traj_wrists,
                "states": traj_states, "chunks": traj_chunks,
            })
            total_n += len(traj_imgs)
            print(f"  sim{sim_id} (train{train_task_id}) init{r}: "
                  f"{'SUCCESS' if success else 'fail'} {n_steps} steps  ({len(traj_imgs)} chunks saved, total {total_n})")
        env.close()

    # Phase 2: write memmaps
    print(f"\nPhase 2: writing {len(trajectories)} trajectories ({total_n} samples) to {out_dir}")
    img_mm = np.memmap(out_dir / "imgs.dat", dtype=np.uint8, mode="w+",
                       shape=(total_n, 256, 256, 3))
    wri_mm = np.memmap(out_dir / "wrists.dat", dtype=np.uint8, mode="w+",
                       shape=(total_n, 256, 256, 3))
    st_mm = np.memmap(out_dir / "states.dat", dtype=np.float32, mode="w+",
                      shape=(total_n, 8))
    chunks_mm = np.memmap(out_dir / "teacher_chunks.dat", dtype=np.float32, mode="w+",
                          shape=(total_n, 16, 7))
    starts = np.zeros(len(trajectories), dtype=np.int64)
    lengths = np.zeros(len(trajectories), dtype=np.int64)
    task_indices = np.zeros(len(trajectories), dtype=np.int64)
    success_arr = np.zeros(len(trajectories), dtype=np.int64)
    sample_idx_arr = np.zeros((total_n, 3), dtype=np.int64)

    cur = 0
    for ti, tr in enumerate(trajectories):
        n = tr["n_steps"]
        starts[ti] = cur
        lengths[ti] = n
        task_indices[ti] = tr["task_idx"]
        success_arr[ti] = int(tr["success"])
        for j in range(n):
            img_mm[cur + j] = tr["imgs"][j]
            wri_mm[cur + j] = tr["wrists"][j]
            st_mm[cur + j] = tr["states"][j]
            chunks_mm[cur + j] = tr["chunks"][j]
            sample_idx_arr[cur + j] = [ti, j, tr["task_idx"]]
        cur += n

    img_mm.flush(); wri_mm.flush(); st_mm.flush(); chunks_mm.flush()
    np.savez(out_dir / "index.npz",
             episode_starts=starts, episode_lengths=lengths,
             task_indices=task_indices, n_total=np.int64(total_n),
             img_size=np.int64(256), success_per_episode=success_arr)
    np.savez(out_dir / "labels_index.npz",
             sample_idx=sample_idx_arr, n_samples=np.int64(total_n),
             action_horizon=np.int64(16))

    success_rate = int(success_arr.sum()) / max(len(trajectories), 1)
    summary = {
        "n_trajectories": len(trajectories),
        "n_samples_collected": int(total_n),
        "success_rate": success_rate,
        "successes": int(success_arr.sum()),
        "size_gb": (total_n * 256 * 256 * 3 * 2 + total_n * 16 * 7 * 4) / 1e9,
    }
    Path(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. {total_n:,} samples from {len(trajectories)} trajs in {out_dir}")
    print(f"GR00T-in-sim success rate: {summary['successes']}/{summary['n_trajectories']} = {success_rate:.0%}")
    print(f"Size: {summary['size_gb']:.2f} GB")


if __name__ == "__main__":
    main()
