"""V7a: GR00T-in-sim with temporal-depth query bank.

V6a/V6c probe perception (state perturbation, image attention). Liquid already
perceives — those queries probe a redundant dimension. V7a probes the
*cognitive* axis: GR00T's denoising-depth trajectory carries different levels
of motor commitment from abstract scene+language fusion (early steps) toward
concrete motor plan (late steps).

Per chunk: ONE GR00T call. The server already captures model_output (mean-
pooled DiT hidden state) at each of the 16 denoising steps. We save K=4
stratified depths {0, 5, 10, 15} as the query bank. Liquid then learns to
emit a depth-attention vector picking which level of GR00T's reasoning to
consume.

Architecturally simpler than V6a/V6c: no input perturbation, no extra GR00T
calls beyond the one that produced the action.

Run inside LIBERO sim venv with groot_server.py active (must be V7a-enabled
server that exposes traj_model_output in get_action_with_state response):
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python gen_groot_with_temporal_queries.py --task_suite libero_10 --rollouts_per_task 10 \\
    --max_steps 720 --port 5555 --depth_indices 0,5,10,15 \\
    --out_dir /home/pokazge/datasets/groot-sim-tempquery-v1
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


def query_groot_with_state(sock: zmq.Socket, obs_dict: dict) -> dict:
    sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs_dict}))
    resp = pickle.loads(sock.recv())
    if "error" in resp:
        raise RuntimeError(f"server error: {resp['error']}")
    return resp


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
    new_obs = {"video": {}, "state": {}, "language": {}}
    for k in video_keys:
        arr = img_256 if k == "image" else wrist_256
        new_obs["video"][k] = arr[None, None, ...]
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
    p.add_argument("--max_steps", type=int, default=720)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--gripper_sign", type=float, default=1.0)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument("--depth_indices", default="0,5,10,15", type=str,
                   help="Comma-separated denoising-step indices to subsample.")
    args = p.parse_args()

    depth_idx = [int(x) for x in args.depth_indices.split(",")]
    K = len(depth_idx)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    sock.setsockopt(zmq.RCVTIMEO, 60000)
    print(f"Connected to GR00T server on tcp://127.0.0.1:{args.port}")
    print(f"depth-query mode: K={K} depths {depth_idx} from denoising trajectory")

    state_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    video_keys = ["image", "wrist_image"]
    language_keys = ["annotation.human.action.task_description"]
    state_slots = {
        "x": (0, 1), "y": (1, 2), "z": (2, 3),
        "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6),
        "gripper": (6, 8),
    }

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
    z_vl_dim = None
    z_state_dim = None
    hidden_dim = None
    print(f"Phase 1: GR00T-in-sim with K={K} depth-query bank "
          f"({n_tasks} tasks × {args.rollouts_per_task} rollouts)")
    t_phase1 = time.time()
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
            traj_zvl, traj_zstate, traj_zmotor = [], [], []
            traj_depth_bank = []
            chunk = None
            chunk_idx = 0
            success = False
            n_steps = 0
            for step in range(args.max_steps):
                n_steps = step + 1
                img_raw, wrist_raw = get_raw_imgs(obs)
                state8 = build_state8(obs)

                if chunk is None or chunk_idx >= args.exec_horizon or chunk_idx >= len(chunk):
                    obs_dict = build_groot_obs(
                        img_raw, wrist_raw, state8, task_lang,
                        state_keys, video_keys, language_keys, state_slots,
                    )
                    resp = query_groot_with_state(sock, obs_dict)
                    chunk = resp["chunk"]
                    z_vl = resp["z_vl"].copy()
                    z_state = resp["z_state"].copy()
                    z_motor = resp["z_motor"].copy()
                    # Subsample depth bank from full traj_model_output [N_steps, hidden]
                    full_traj = resp["traj_model_output"]  # [N_steps, hidden_dim]
                    depth_bank = np.stack([full_traj[d] for d in depth_idx]).astype(np.float32)

                    if z_vl_dim is None:
                        z_vl_dim = int(z_vl.shape[0])
                        z_state_dim = int(z_state.shape[0])
                        hidden_dim = int(depth_bank.shape[1])
                        print(f"  detected z_vl_dim={z_vl_dim}, z_state_dim={z_state_dim}, "
                              f"hidden_dim={hidden_dim}, full_traj_steps={full_traj.shape[0]}, "
                              f"K={K}")
                    chunk_idx = 0
                    traj_imgs.append(img_raw)
                    traj_wrists.append(wrist_raw)
                    traj_states.append(state8)
                    traj_chunks.append(chunk.copy())
                    traj_zvl.append(z_vl)
                    traj_zstate.append(z_state)
                    traj_zmotor.append(z_motor)
                    traj_depth_bank.append(depth_bank)

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
                "zvl": traj_zvl, "zstate": traj_zstate, "zmotor": traj_zmotor,
                "depth_bank": traj_depth_bank,
            })
            total_n += len(traj_imgs)
            elapsed = time.time() - t_phase1
            print(f"  sim{sim_id} (train{train_task_id}) init{r}: "
                  f"{'SUCCESS' if success else 'fail'} {n_steps} steps  "
                  f"({len(traj_imgs)} chunks, total {total_n}, {elapsed/60:.1f}min)")
        env.close()

    print(f"\nPhase 2: writing {len(trajectories)} trajectories ({total_n} samples) to {out_dir}")
    assert z_vl_dim is not None and z_state_dim is not None and hidden_dim is not None
    img_mm = np.memmap(out_dir / "imgs.dat", dtype=np.uint8, mode="w+",
                       shape=(total_n, 256, 256, 3))
    wri_mm = np.memmap(out_dir / "wrists.dat", dtype=np.uint8, mode="w+",
                       shape=(total_n, 256, 256, 3))
    st_mm = np.memmap(out_dir / "states.dat", dtype=np.float32, mode="w+",
                      shape=(total_n, 8))
    chunks_mm = np.memmap(out_dir / "teacher_chunks.dat", dtype=np.float32, mode="w+",
                          shape=(total_n, 16, 7))
    zvl_mm = np.memmap(out_dir / "z_vl.dat", dtype=np.float32, mode="w+",
                       shape=(total_n, z_vl_dim))
    zstate_mm = np.memmap(out_dir / "z_state.dat", dtype=np.float32, mode="w+",
                          shape=(total_n, z_state_dim))
    zmotor_mm = np.memmap(out_dir / "z_motor.dat", dtype=np.float32, mode="w+",
                          shape=(total_n, 7))
    # V7a depth bank: stored in z_vl_bank.dat slot for compatibility with the
    # existing dataset reader. The bank entries are model_output (DiT hidden
    # mean-pooled), with their own dim that may differ from z_vl_dim.
    bank_mm = np.memmap(out_dir / "z_vl_bank.dat", dtype=np.float32, mode="w+",
                        shape=(total_n, K, hidden_dim))
    # Query "delta" entries are one-hot K-d (identity matrix) — Liquid emits
    # K logits, softmax matches against e_k effectively becoming direct
    # attention over depths.
    delta_mm = np.memmap(out_dir / "delta_s_bank.dat", dtype=np.float32, mode="w+",
                          shape=(total_n, K, K))
    starts = np.zeros(len(trajectories), dtype=np.int64)
    lengths = np.zeros(len(trajectories), dtype=np.int64)
    task_indices = np.zeros(len(trajectories), dtype=np.int64)
    success_arr = np.zeros(len(trajectories), dtype=np.int64)
    sample_idx_arr = np.zeros((total_n, 3), dtype=np.int64)
    one_hot_K = np.eye(K, dtype=np.float32)

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
            zvl_mm[cur + j] = tr["zvl"][j]
            zstate_mm[cur + j] = tr["zstate"][j]
            zmotor_mm[cur + j] = tr["zmotor"][j]
            bank_mm[cur + j] = tr["depth_bank"][j]
            delta_mm[cur + j] = one_hot_K
            sample_idx_arr[cur + j] = [ti, j, tr["task_idx"]]
        cur += n

    for mm in (img_mm, wri_mm, st_mm, chunks_mm, zvl_mm, zstate_mm, zmotor_mm,
               bank_mm, delta_mm):
        mm.flush()
    np.savez(out_dir / "index.npz",
             episode_starts=starts, episode_lengths=lengths,
             task_indices=task_indices, n_total=np.int64(total_n),
             img_size=np.int64(256), success_per_episode=success_arr,
             z_vl_dim=np.int64(hidden_dim),  # bank dim, not actual z_vl
             z_state_dim=np.int64(z_state_dim),
             query_bank_K=np.int64(K), query_channel="depth",
             query_dim=np.int64(K),  # one-hot dim
             depth_indices=np.array(depth_idx, dtype=np.int64))
    np.savez(out_dir / "labels_index.npz",
             sample_idx=sample_idx_arr, n_samples=np.int64(total_n),
             action_horizon=np.int64(16))

    success_rate = int(success_arr.sum()) / max(len(trajectories), 1)
    summary = {
        "n_trajectories": len(trajectories),
        "n_samples_collected": int(total_n),
        "success_rate": success_rate,
        "successes": int(success_arr.sum()),
        "z_vl_dim": z_vl_dim,
        "z_state_dim": z_state_dim,
        "hidden_dim": hidden_dim,
        "query_bank_K": K,
        "query_channel": "depth",
        "depth_indices": depth_idx,
        "size_gb": (
            total_n * 256 * 256 * 3 * 2
            + total_n * 16 * 7 * 4
            + total_n * z_vl_dim * 4
            + total_n * z_state_dim * 4
            + total_n * 7 * 4
            + total_n * K * hidden_dim * 4
            + total_n * K * K * 4
        ) / 1e9,
    }
    Path(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. {total_n:,} samples, K={K} depths from {len(trajectories)} trajs in {out_dir}")
    print(f"GR00T-in-sim success: {summary['successes']}/{summary['n_trajectories']} = {success_rate:.0%}")
    print(f"hidden_dim={hidden_dim}, depths={depth_idx}")
    print(f"Size: {summary['size_gb']:.2f} GB")


if __name__ == "__main__":
    main()
