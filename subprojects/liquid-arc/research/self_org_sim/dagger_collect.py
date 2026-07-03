"""DAgger phase 1 — collect (obs, executed_action) trajectories from student rollouts in sim.

Runs the trained flow policy in LIBERO sim, saves every visited state's
observation (256x256 workspace + wrist images, 8-d state, task_index) into a
memmap dataset that GR00T can later re-label.

Saved layout (matches libero-10-r-decoded format that TeacherLabelDataset reads):
  imgs.dat      uint8  [N, 256, 256, 3]   workspace cam (raw, not flipped)
  wrists.dat    uint8  [N, 256, 256, 3]   wrist cam     (raw, not flipped)
  states.dat    f32    [N, 8]
  index.npz     traj_starts, traj_lengths, task_indices_per_traj, n_total

NOTE: imgs/wrists are saved at full 256x256 (NOT resized) so GR00T can later
relabel them with full fidelity. Student can resize at inference.

Run inside LIBERO sim venv (CPU torch ok):
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python dagger_collect.py \\
    --student_ckpt /tmp/distill_groot_flow_v1/step_030000.pt \\
    --task_suite libero_10 --rollouts_per_task 5 --max_steps 200 \\
    --exec_horizon 8 --infer_steps 10 \\
    --out_dir /home/pokazge/datasets/dagger-iter1-obs
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot_flow import LiquidFlowPolicy

print = functools.partial(print, flush=True)
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
except Exception:
    pass


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


def get_raw_imgs(obs_raw):
    """Return (img_256, wrist_256) as raw uint8 — NO flip (saved as-is for GR00T)."""
    return obs_raw["agentview_image"][::-1, ::-1].copy(), \
           obs_raw["robot0_eye_in_hand_image"][::-1, ::-1].copy()


def load_flow_policy(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa = ckpt["args"]
    halt_mode = "learned" if sa["policy"] == "liquid_halt" else "none"
    model = LiquidFlowPolicy(
        state_dim=8, action_dim=7, action_horizon=sa["action_horizon"],
        d=sa["d"], d_vis=sa["d"], img_size=sa["img_size"], k_max=sa["k"],
        halt_mode=halt_mode, min_steps=sa["halting_min_steps"],
        n_tasks=sa["n_tasks"], d_task=sa["d_task"],
        head_d=sa["head_d"], head_layers=sa["head_layers"], head_heads=sa["head_heads"],
    ).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own = model.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
    model.eval()
    return model, sa


@torch.no_grad()
def sample_chunk(model, img_256, wrist_256, state8, device, n_steps, task_id, target_size):
    img = np.array(Image.fromarray(img_256).resize((target_size, target_size)), dtype=np.uint8)
    wri = np.array(Image.fromarray(wrist_256).resize((target_size, target_size)), dtype=np.uint8)
    img_t = torch.from_numpy(img).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    wri_t = torch.from_numpy(wri).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    st_t = torch.from_numpy(state8).to(device).float().unsqueeze(0)
    tid_t = None if task_id is None else torch.tensor([task_id], dtype=torch.long, device=device)
    chunk = model.sample(img_t, wri_t, st_t, task_id=tid_t, n_steps=n_steps)
    return chunk[0].cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", required=True, type=str)
    p.add_argument("--task_suite", default="libero_10", type=str)
    p.add_argument("--rollouts_per_task", type=int, default=5)
    p.add_argument("--rollouts_per_task_overrides", type=str, default="",
                   help="JSON dict 'sim_id:n_rollouts' overrides, e.g. '{\"0\":10,\"4\":10,\"8\":10}'. "
                        "Tasks not in the dict use --rollouts_per_task.")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--infer_steps", type=int, default=10)
    p.add_argument("--gripper_sign", type=float, default=1.0)
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument("--use_lang", action="store_true")
    p.add_argument("--start_from_demos", action="store_true",
                   help="Use init_states from libero-r demos when possible (instead of sim default)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, sargs = load_flow_policy(Path(args.student_ckpt), device)
    student_img_size = sargs["img_size"]
    use_lang = bool(args.use_lang) and sargs.get("n_tasks", 0) > 0

    # Same task language list used in training (libero-r ordering)
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

    # First pass: run rollouts, accumulate per-trajectory buffers in memory
    print(f"Phase 1: rolling out student on {n_tasks} tasks × {args.rollouts_per_task} init_states each")
    trajectories = []   # list of dicts: imgs_list, wrists_list, states_list, task_idx, success
    total_n = 0

    overrides = json.loads(args.rollouts_per_task_overrides) if args.rollouts_per_task_overrides else {}
    overrides = {int(k): int(v) for k, v in overrides.items()}
    if overrides:
        print(f"Per-task rollout overrides: {overrides}")

    for sim_id in range(n_tasks):
        task = suite.get_task(sim_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = suite.get_task_init_states(sim_id)
        n_for_task = overrides.get(sim_id, args.rollouts_per_task)
        n_rollouts = min(n_for_task, len(init_states))
        train_task_id = sim_to_r.get(sim_id, sim_id) if use_lang else None

        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        for r in range(n_rollouts):
            env.reset()
            env.set_init_state(init_states[r])
            obs = None
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

            traj_imgs, traj_wrists, traj_states = [], [], []
            chunk = None
            chunk_idx = 0
            success = False
            n_steps = 0
            for step in range(args.max_steps):
                n_steps = step + 1
                img_raw, wrist_raw = get_raw_imgs(obs)
                state8 = build_state8(obs)

                # Save EVERY visited state
                traj_imgs.append(img_raw)
                traj_wrists.append(wrist_raw)
                traj_states.append(state8)

                if chunk is None or chunk_idx >= args.exec_horizon or chunk_idx >= len(chunk):
                    chunk = sample_chunk(model, img_raw, wrist_raw, state8, device,
                                         n_steps=args.infer_steps,
                                         task_id=sim_to_r.get(sim_id, sim_id) if use_lang else None,
                                         target_size=student_img_size)
                    chunk_idx = 0

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
                "task_idx": sim_to_r.get(sim_id, sim_id),
                "sim_id": sim_id,
                "n_steps": len(traj_imgs),
                "success": success,
                "imgs": traj_imgs,
                "wrists": traj_wrists,
                "states": traj_states,
            })
            total_n += len(traj_imgs)
            print(f"  sim{sim_id} (train{train_task_id}) init{r}: "
                  f"{'SUCCESS' if success else 'fail'} {n_steps} steps  (total collected: {total_n})")
        env.close()

    # Phase 2: write to memmap
    print(f"\nPhase 2: writing {len(trajectories)} trajectories ({total_n} states) to {out_dir}")
    img_mm = np.memmap(out_dir / "imgs.dat", dtype=np.uint8, mode="w+",
                       shape=(total_n, 256, 256, 3))
    wri_mm = np.memmap(out_dir / "wrists.dat", dtype=np.uint8, mode="w+",
                       shape=(total_n, 256, 256, 3))
    st_mm = np.memmap(out_dir / "states.dat", dtype=np.float32, mode="w+",
                      shape=(total_n, 8))
    starts = np.zeros(len(trajectories), dtype=np.int64)
    lengths = np.zeros(len(trajectories), dtype=np.int64)
    task_indices = np.zeros(len(trajectories), dtype=np.int64)
    success_arr = np.zeros(len(trajectories), dtype=np.int64)

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
        cur += n

    img_mm.flush(); wri_mm.flush(); st_mm.flush()
    np.savez(out_dir / "index.npz",
             episode_starts=starts, episode_lengths=lengths,
             task_indices=task_indices, n_total=np.int64(total_n),
             img_size=np.int64(256), success_per_episode=success_arr)

    success_rate = int(success_arr.sum()) / max(len(trajectories), 1)
    summary = {
        "n_trajectories": len(trajectories),
        "n_states_collected": int(total_n),
        "success_rate": success_rate,
        "successes": int(success_arr.sum()),
        "size_gb": (total_n * 256 * 256 * 3 * 2) / 1e9,
    }
    Path(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. {total_n:,} states from {len(trajectories)} trajs in {out_dir}")
    print(f"Success rate during collection: {summary['successes']}/{summary['n_trajectories']} = {success_rate:.0%}")
    print(f"Total size: {summary['size_gb']:.2f} GB")


if __name__ == "__main__":
    main()
