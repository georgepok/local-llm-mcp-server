"""V6c: GR00T-in-sim with image-attention query bank.

V6a's state-perturbation queries are broken because at deployment Liquid's Δs
becomes a real input perturbation that lies to GR00T about the body. V6c uses
image-space attention instead: Liquid emits an attention map that biases
*what GR00T sees about the scene* without changing what's actually in the scene.

Per chunk, run K=4 GR00T queries:
  query 0: image with uniform attention (neutral, used for actual rollout)
  queries 1-K: image multiplied by Gaussian-blob attention masks at random
               (cx, cy, σ) to give Liquid diverse "focus directions" to choose from.

Saves the same memmap layout as gen_groot_with_query_bank.py but with
attention_bank.dat (16×16 float maps) replacing delta_s_bank.dat (8-d Δs).
The dataset reader auto-detects channel via index.npz['query_channel'].

Run inside LIBERO sim venv with groot_server.py active on port 5555:
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python gen_groot_with_image_queries.py --task_suite libero_10 --rollouts_per_task 10 \\
    --max_steps 720 --port 5555 --query_bank_K 4 --attn_grid 16 --attn_alpha 0.4 \\
    --out_dir /home/pokazge/datasets/groot-sim-imgquery-v1
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


def random_gaussian_blob(grid: int, rng: np.random.Generator,
                         sigma_grid: float = 2.5) -> np.ndarray:
    """Return [grid, grid] attention map: Gaussian blob at random (cx, cy)."""
    cx = rng.uniform(0, grid - 1)
    cy = rng.uniform(0, grid - 1)
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma_grid ** 2))
    g = g / g.max()  # normalize peak to 1.0
    return g.astype(np.float32)


def upsample_attn_to_image(attn_grid: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Upsample a [G, G] attention map to [target_size, target_size] via bilinear."""
    from PIL import Image
    img = Image.fromarray((attn_grid * 255.0).astype(np.uint8))
    # PIL/Pillow compatibility: BILINEAR moved to Image.Resampling in Pillow 9.1+
    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    up = img.resize((target_size, target_size), bilinear)
    return np.array(up).astype(np.float32) / 255.0


def apply_attention_to_image(img: np.ndarray, attn_grid: np.ndarray,
                              alpha: float = 0.4) -> np.ndarray:
    """Modulate uint8 image by attention map.

    Each pixel scaled by (1 + alpha * (attn - 0.5)): regions with high attention
    are brightened, low attention dimmed. Attention has uniform mean 0.5 by
    convention (uniform map = 0.5 everywhere → no modulation). For a Gaussian
    blob centered at one location, the peak is +alpha boost, edges are -alpha
    suppression.
    """
    target = img.shape[0]
    attn_up = upsample_attn_to_image(attn_grid, target)  # [H, W] in [0,1]
    scale = 1.0 + alpha * (attn_up - 0.5)  # [H, W]
    modulated = img.astype(np.float32) * scale[:, :, None]
    return np.clip(modulated, 0.0, 255.0).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_suite", default="libero_10", type=str)
    p.add_argument("--rollouts_per_task", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=720)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--gripper_sign", type=float, default=1.0)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument("--query_bank_K", type=int, default=4,
                   help="Total queries per chunk: 1 uniform + (K-1) random blobs.")
    p.add_argument("--attn_grid", type=int, default=16,
                   help="Side length of attention grid; emitted as a [G,G] logit map.")
    p.add_argument("--attn_alpha", type=float, default=0.4,
                   help="Strength of image modulation. Pixel scale = 1 + alpha*(attn-0.5).")
    p.add_argument("--blob_sigma", type=float, default=2.5,
                   help="Std of random Gaussian blob attention masks (in grid cells).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    sock.setsockopt(zmq.RCVTIMEO, 60000)
    print(f"Connected to GR00T server on tcp://127.0.0.1:{args.port}")
    print(f"image-query mode: K={args.query_bank_K}, grid={args.attn_grid}, "
          f"alpha={args.attn_alpha}, blob_sigma={args.blob_sigma}")

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

    K = args.query_bank_K
    G = args.attn_grid
    trajectories = []
    total_n = 0
    z_vl_dim = None
    z_state_dim = None
    print(f"Phase 1: GR00T-in-sim with K={K} image-attention queries per chunk "
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
            traj_zvl_bank, traj_attn_bank = [], []
            chunk = None
            chunk_idx = 0
            success = False
            n_steps = 0
            for step in range(args.max_steps):
                n_steps = step + 1
                img_raw, wrist_raw = get_raw_imgs(obs)
                state8 = build_state8(obs)

                if chunk is None or chunk_idx >= args.exec_horizon or chunk_idx >= len(chunk):
                    # Build K attention maps:
                    #   index 0: uniform 0.5 (neutral, no modulation)
                    #   indices 1..K-1: random Gaussian blobs
                    attn_bank = np.zeros((K, G, G), dtype=np.float32)
                    attn_bank[0] = np.full((G, G), 0.5, dtype=np.float32)
                    for k in range(1, K):
                        attn_bank[k] = random_gaussian_blob(G, rng, sigma_grid=args.blob_sigma)

                    z_vl_bank = []
                    chunk_neutral = None
                    z_state_neutral = None
                    z_motor_neutral = None
                    for k in range(K):
                        img_modulated = apply_attention_to_image(
                            img_raw, attn_bank[k], alpha=args.attn_alpha
                        )
                        obs_dict = build_groot_obs(
                            img_modulated, wrist_raw, state8, task_lang,
                            state_keys, video_keys, language_keys, state_slots,
                        )
                        resp = query_groot_with_state(sock, obs_dict)
                        z_vl_bank.append(resp["z_vl"].copy())
                        if k == 0:
                            chunk_neutral = resp["chunk"]
                            z_state_neutral = resp["z_state"].copy()
                            z_motor_neutral = resp["z_motor"].copy()
                    chunk = chunk_neutral
                    z_vl_bank = np.stack(z_vl_bank).astype(np.float32)

                    if z_vl_dim is None:
                        z_vl_dim = int(z_vl_bank.shape[1])
                        z_state_dim = int(z_state_neutral.shape[0])
                        print(f"  detected z_vl_dim={z_vl_dim}, z_state_dim={z_state_dim}, "
                              f"K={K}, attn_dim={G*G}")
                    chunk_idx = 0
                    traj_imgs.append(img_raw)
                    traj_wrists.append(wrist_raw)
                    traj_states.append(state8)
                    traj_chunks.append(chunk.copy())
                    traj_zvl.append(z_vl_bank[0].copy())
                    traj_zstate.append(z_state_neutral)
                    traj_zmotor.append(z_motor_neutral)
                    traj_zvl_bank.append(z_vl_bank)
                    traj_attn_bank.append(attn_bank.reshape(K, -1))  # flatten to [K, G*G]

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
                "zvl_bank": traj_zvl_bank, "attn_bank": traj_attn_bank,
            })
            total_n += len(traj_imgs)
            elapsed = time.time() - t_phase1
            print(f"  sim{sim_id} (train{train_task_id}) init{r}: "
                  f"{'SUCCESS' if success else 'fail'} {n_steps} steps  "
                  f"({len(traj_imgs)} chunks, total {total_n}, {elapsed/60:.1f}min)")
        env.close()

    print(f"\nPhase 2: writing {len(trajectories)} trajectories ({total_n} samples) to {out_dir}")
    assert z_vl_dim is not None and z_state_dim is not None
    attn_dim = G * G
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
    zvl_bank_mm = np.memmap(out_dir / "z_vl_bank.dat", dtype=np.float32, mode="w+",
                             shape=(total_n, K, z_vl_dim))
    # For V6c we save attention_bank.dat (flat). Filename matches gen_with_query_bank
    # (delta_s_bank.dat) so the existing dataset code works — but we tag the channel
    # in index.npz so rollout knows how to apply it.
    attn_bank_mm = np.memmap(out_dir / "delta_s_bank.dat", dtype=np.float32, mode="w+",
                              shape=(total_n, K, attn_dim))
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
            zvl_mm[cur + j] = tr["zvl"][j]
            zstate_mm[cur + j] = tr["zstate"][j]
            zmotor_mm[cur + j] = tr["zmotor"][j]
            zvl_bank_mm[cur + j] = tr["zvl_bank"][j]
            attn_bank_mm[cur + j] = tr["attn_bank"][j]
            sample_idx_arr[cur + j] = [ti, j, tr["task_idx"]]
        cur += n

    for mm in (img_mm, wri_mm, st_mm, chunks_mm, zvl_mm, zstate_mm, zmotor_mm,
               zvl_bank_mm, attn_bank_mm):
        mm.flush()
    np.savez(out_dir / "index.npz",
             episode_starts=starts, episode_lengths=lengths,
             task_indices=task_indices, n_total=np.int64(total_n),
             img_size=np.int64(256), success_per_episode=success_arr,
             z_vl_dim=np.int64(z_vl_dim), z_state_dim=np.int64(z_state_dim),
             query_bank_K=np.int64(K), query_channel="image",
             attn_grid=np.int64(G), attn_alpha=np.float32(args.attn_alpha),
             query_dim=np.int64(attn_dim))
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
        "query_bank_K": K,
        "query_channel": "image",
        "attn_grid": G,
        "attn_alpha": args.attn_alpha,
        "blob_sigma": args.blob_sigma,
        "size_gb": (
            total_n * 256 * 256 * 3 * 2
            + total_n * 16 * 7 * 4
            + total_n * z_vl_dim * 4
            + total_n * z_state_dim * 4
            + total_n * 7 * 4
            + total_n * K * z_vl_dim * 4
            + total_n * K * attn_dim * 4
        ) / 1e9,
    }
    Path(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. {total_n:,} samples × K={K} image queries from {len(trajectories)} trajs in {out_dir}")
    print(f"GR00T-in-sim success: {summary['successes']}/{summary['n_trajectories']} = {success_rate:.0%}")
    print(f"z_vl={z_vl_dim}d, z_state={z_state_dim}d, attn={G}x{G}={attn_dim}d")
    print(f"Size: {summary['size_gb']:.2f} GB")


if __name__ == "__main__":
    main()
