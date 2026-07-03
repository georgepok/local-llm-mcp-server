"""Closed-loop LIBERO sim rollout: Liquid student vs. demonstrator-only baseline.

For each LIBERO_10 task, runs N rollouts using fixed init_states from the
benchmark. At each step:
  1. Read obs (256x256 video.image, video.wrist_image, state.* dicts)
  2. Resize images to student's training resolution (96x96)
  3. Build state[8] = [x, y, z, roll, pitch, yaw, gripper(2)]
  4. Predict action_chunk[K, 7] via student
  5. Execute the first `--exec_horizon` actions, then re-predict (receding horizon)
  6. Record success at end (env.check_success())

Reports per-task success rate + average rollout length + average inference Hz.

Run on Spark *inside* the LIBERO sim venv:
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python rollout_libero.py \\
    --student_ckpt /tmp/distill_groot_v1/step_030000.pt \\
    --task_suite libero_10 --rollouts_per_task 5 \\
    --max_steps 300 --exec_horizon 8 --img_size 96 \\
    --gripper_convention plus_minus_one \\
    --out_json /tmp/distill_groot_v1/rollout_libero.json
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

# Force EGL for headless rendering
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot import LiquidStudent

print = functools.partial(print, flush=True)
torch.set_float32_matmul_precision("high")


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


def build_state8(obs_raw: dict) -> np.ndarray:
    """Build [8] state array from raw robosuite obs (matches libero-r/demo modality)."""
    xyz = obs_raw["robot0_eef_pos"]
    rpy = quat2axisangle(obs_raw["robot0_eef_quat"])
    grip = obs_raw["robot0_gripper_qpos"]  # [2]
    return np.concatenate([xyz, rpy, grip], axis=0).astype(np.float32)


def preprocess_imgs(obs_raw: dict, target_size: int):
    """Return (img, wrist_img) both shape (target_size, target_size, 3) uint8.

    Sim raw images need vertical+horizontal flip to match training distribution
    (per libero_env.py the wrapper applies obs['agentview_image'][::-1, ::-1]).
    """
    img = obs_raw["agentview_image"][::-1, ::-1]
    wrist = obs_raw["robot0_eye_in_hand_image"][::-1, ::-1]
    img_r = np.array(Image.fromarray(img).resize((target_size, target_size)), dtype=np.uint8)
    wrist_r = np.array(Image.fromarray(wrist).resize((target_size, target_size)), dtype=np.uint8)
    return img_r, wrist_r


def load_student(ckpt_path: Path, device: torch.device) -> tuple[LiquidStudent, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa = ckpt["args"]
    halt_mode = "learned" if sa["policy"] == "liquid_halt" else "none"
    model = LiquidStudent(
        state_dim=8, action_dim=7, action_horizon=sa["action_horizon"],
        d=sa["d"], d_vis=sa["d"], img_size=sa["img_size"], k_max=sa["k"],
        halt_mode=halt_mode, min_steps=sa["halting_min_steps"],
    ).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v); loaded += 1
    print(f"[student] loaded {loaded}/{len(own)} tensors (step={ckpt.get('step')})")
    model.eval()
    return model, sa


@torch.no_grad()
def predict_chunk(student: LiquidStudent, img: np.ndarray, wrist: np.ndarray,
                  state8: np.ndarray, device: torch.device) -> np.ndarray:
    img_t = torch.from_numpy(img).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    wri_t = torch.from_numpy(wrist).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    st_t = torch.from_numpy(state8).to(device).float().unsqueeze(0)
    pred, _ = student(img_t, wri_t, st_t)
    return pred[0].cpu().numpy()  # [K, 7]


def run_rollout(env_underlying, student, init_state, task_description: str,
                args, device: torch.device, gripper_sign: float = 1.0):
    """Run one rollout. Returns (success_bool, n_steps, mean_inference_ms)."""
    del task_description  # available via env wrapper if needed; unused here
    env_underlying.reset()
    env_underlying.set_init_state(init_state)
    # Let physics settle a few steps; obs is updated by every step
    obs = None
    for _ in range(5):
        obs, _, _, _ = env_underlying.step(np.zeros(7, dtype=np.float32))

    inference_times = []
    success = False
    chunk = None
    chunk_idx = 0
    n_steps = 0
    for step in range(args.max_steps):
        n_steps = step + 1
        if chunk is None or chunk_idx >= args.exec_horizon or chunk_idx >= len(chunk):
            img, wrist = preprocess_imgs(obs, args.img_size)
            state8 = build_state8(obs)
            t0 = time.perf_counter()
            chunk = predict_chunk(student, img, wrist, state8, device)
            inference_times.append(time.perf_counter() - t0)
            chunk_idx = 0

        action7 = chunk[chunk_idx].copy()
        # Robosuite OSC_POSE expects gripper in [-1, +1]; +1 = close, -1 = open.
        # Student output convention depends on training data; pass `gripper_sign`
        # to flip if needed. Binarize when above a small threshold.
        g = action7[-1]
        action7[-1] = gripper_sign * np.sign(g) if abs(g) > 0.1 else 0.0

        obs, _, done, _ = env_underlying.step(action7.astype(np.float32))
        chunk_idx += 1

        if env_underlying.check_success():
            success = True
            break
        if done:
            break

    mean_ms = float(np.mean(inference_times) * 1000) if inference_times else 0.0
    return success, n_steps, mean_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", required=True, type=str)
    p.add_argument("--task_suite", default="libero_10", type=str,
                   choices=["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"])
    p.add_argument("--rollouts_per_task", type=int, default=5,
                   help="Number of init_states to use per task")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--exec_horizon", type=int, default=8,
                   help="N actions to execute from each predicted chunk before re-predicting")
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--gripper_sign", type=float, default=1.0,
                   help="Multiplier for student's gripper output (use -1.0 to invert)")
    p.add_argument("--task_indices", type=str, default="",
                   help="Optional comma-separated subset of task IDs (e.g. '0,3,7')")
    p.add_argument("--out_json", default="", type=str)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # Load student (uses our distill_groot.LiquidStudent — needs torch + CUDA)
    student, sargs = load_student(Path(args.student_ckpt), device)
    args.img_size = sargs["img_size"]  # match training resolution

    # Load LIBERO task suite (needs the LIBERO sim venv)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    n_tasks = task_suite.get_num_tasks()
    if args.task_indices:
        task_ids = [int(t) for t in args.task_indices.split(",")]
    else:
        task_ids = list(range(n_tasks))
    print(f"Suite: {args.task_suite}, total tasks: {n_tasks}, evaluating: {task_ids}")

    summary = {"task_suite": args.task_suite, "tasks": []}
    overall_successes = 0
    overall_total = 0

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_desc = task.language
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = task_suite.get_task_init_states(task_id)
        n_rollouts = min(args.rollouts_per_task, len(init_states))

        print(f"\n=== Task {task_id}: {task_name} ===")
        print(f"  desc: {task_desc[:80]}")
        print(f"  rollouts: {n_rollouts}/{len(init_states)} init_states")

        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        successes = 0
        rollout_records = []
        for r in range(n_rollouts):
            t0 = time.time()
            success, n_steps, infer_ms = run_rollout(
                env, student, init_states[r], task_desc, args, device,
                gripper_sign=args.gripper_sign,
            )
            wall = time.time() - t0
            successes += int(success)
            rollout_records.append({
                "init_state": r, "success": bool(success),
                "n_steps": n_steps, "infer_ms": infer_ms, "wall_s": wall,
            })
            print(f"  rollout {r}: {'SUCCESS' if success else 'fail'}  "
                  f"n_steps={n_steps:3d}  infer={infer_ms:.1f}ms  wall={wall:.1f}s")
        env.close()

        rate = successes / n_rollouts
        overall_successes += successes
        overall_total += n_rollouts
        print(f"  TASK {task_id} success rate: {successes}/{n_rollouts} = {rate:.0%}")
        summary["tasks"].append({
            "task_id": task_id, "task_name": task_name, "task_desc": task_desc,
            "n_rollouts": n_rollouts, "n_successes": successes, "success_rate": rate,
            "rollouts": rollout_records,
        })

    print("\n" + "=" * 80)
    print("OVERALL SUCCESS RATE: {}/{} = {:.0%}".format(
        overall_successes, overall_total, overall_successes / max(overall_total, 1)))
    print("=" * 80)
    summary["overall_successes"] = overall_successes
    summary["overall_total"] = overall_total
    summary["overall_success_rate"] = overall_successes / max(overall_total, 1)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"Saved summary to {args.out_json}")


if __name__ == "__main__":
    main()
