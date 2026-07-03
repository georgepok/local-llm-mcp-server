"""Closed-loop LIBERO rollout for flow-matching policy (LiquidFlowPolicy).

Differs from rollout_libero.py only in: how the policy is loaded and how an
action chunk is predicted (ODE sampling instead of direct forward).

Run inside the LIBERO sim venv (which has CPU torch — flow sampling will be
slow but works):
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python rollout_libero_flow.py \\
    --student_ckpt /tmp/distill_groot_flow_v1/step_030000.pt \\
    --task_suite libero_10 --rollouts_per_task 5 \\
    --max_steps 300 --exec_horizon 8 --infer_steps 10
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
# Same SDPA backend tweak as training (libero venv may also need it)
try:
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
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


def preprocess_imgs(obs_raw, target_size: int):
    img = obs_raw["agentview_image"][::-1, ::-1]
    wrist = obs_raw["robot0_eye_in_hand_image"][::-1, ::-1]
    img_r = np.array(Image.fromarray(img).resize((target_size, target_size)), dtype=np.uint8)
    wrist_r = np.array(Image.fromarray(wrist).resize((target_size, target_size)), dtype=np.uint8)
    return img_r, wrist_r


def load_flow_policy(ckpt_path: Path, device: torch.device) -> tuple[LiquidFlowPolicy, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa = ckpt["args"]
    halt_mode = "learned" if sa["policy"] == "liquid_halt" else "none"
    model = LiquidFlowPolicy(
        state_dim=8, action_dim=7, action_horizon=sa["action_horizon"],
        d=sa["d"], d_vis=sa["d"], img_size=sa["img_size"], k_max=sa["k"],
        halt_mode=halt_mode, min_steps=sa["halting_min_steps"],
        n_tasks=sa["n_tasks"], d_task=sa["d_task"],
        head_d=sa["head_d"], head_layers=sa["head_layers"], head_heads=sa["head_heads"],
        n_task_heads=sa.get("n_task_heads", 0),
    ).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v); loaded += 1
    print(f"[flow] loaded {loaded}/{len(own)} tensors (step={ckpt.get('step')})")
    model.eval()
    return model, sa


@torch.no_grad()
def predict_chunk(model, img, wrist, state, device, n_steps: int,
                  task_id: int | None):
    img_t = torch.from_numpy(img).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    wri_t = torch.from_numpy(wrist).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    st_t = torch.from_numpy(state).to(device).float().unsqueeze(0)
    tid_t = None if task_id is None else torch.tensor([task_id], dtype=torch.long, device=device)
    chunk = model.sample(img_t, wri_t, st_t, task_id=tid_t, n_steps=n_steps)
    return chunk[0].cpu().numpy()


def run_rollout(env_underlying, model, init_state, task_idx_for_lang,
                args, device):
    env_underlying.reset()
    env_underlying.set_init_state(init_state)
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
            chunk = predict_chunk(model, img, wrist, state8, device,
                                  n_steps=args.infer_steps,
                                  task_id=task_idx_for_lang)
            inference_times.append(time.perf_counter() - t0)
            chunk_idx = 0

        action7 = chunk[chunk_idx].copy()
        # Robosuite OSC_POSE: gripper +1=close, -1=open. Student trained on
        # demo convention where gripper is in {-1,+1} (binary). Binarize.
        g = action7[-1]
        action7[-1] = args.gripper_sign * np.sign(g) if abs(g) > 0.1 else 0.0

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
    p.add_argument("--task_suite", default="libero_10", type=str)
    p.add_argument("--rollouts_per_task", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--infer_steps", type=int, default=10,
                   help="ODE integration steps for flow sampling")
    p.add_argument("--gripper_sign", type=float, default=1.0)
    p.add_argument("--task_indices", type=str, default="")
    p.add_argument("--out_json", default="", type=str)
    p.add_argument("--use_lang", action="store_true",
                   help="Pass task_id to the policy (only valid if model trained with n_tasks>0)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model, sargs = load_flow_policy(Path(args.student_ckpt), device)
    args.img_size = sargs["img_size"]
    use_lang = bool(args.use_lang) and sargs.get("n_tasks", 0) > 0
    if not args.use_lang and sargs.get("n_tasks", 0) > 0:
        print("[warn] checkpoint trained with task conditioning but --use_lang not set; passing None task_id (zeros)")

    # libero-r training task index -> libero_10 sim task id mapping (must match sim ordering)
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

    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[args.task_suite]()
    n_tasks = suite.get_num_tasks()
    sim_task_lang = {i: suite.get_task(i).language.strip() for i in range(n_tasks)}
    # Map sim_id -> libero_r training_id (the task id the model trained with for embedding lookup)
    sim_to_r = {}
    for sim_id, lang in sim_task_lang.items():
        for r_id, r_lang in enumerate(libero_r_tasks):
            if lang == r_lang.strip():
                sim_to_r[sim_id] = r_id
                break

    if args.task_indices:
        task_ids = [int(t) for t in args.task_indices.split(",")]
    else:
        task_ids = list(range(n_tasks))
    print(f"Suite: {args.task_suite}  evaluating tasks: {task_ids}")

    summary = {"task_suite": args.task_suite, "tasks": []}
    overall_succ, overall_total = 0, 0

    for sim_id in task_ids:
        task = suite.get_task(sim_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = suite.get_task_init_states(sim_id)
        n_rollouts = min(args.rollouts_per_task, len(init_states))
        train_task_id = sim_to_r.get(sim_id, sim_id) if use_lang else None

        print(f"\n=== Task sim{sim_id} (train_id={train_task_id}): {sim_task_lang[sim_id][:70]} ===")

        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        successes = 0
        rollout_records = []
        for r in range(n_rollouts):
            t0 = time.time()
            success, n_steps, infer_ms = run_rollout(
                env, model, init_states[r], train_task_id, args, device,
            )
            wall = time.time() - t0
            successes += int(success)
            rollout_records.append({
                "init_state": r, "success": bool(success),
                "n_steps": n_steps, "infer_ms": infer_ms, "wall_s": wall,
            })
            print(f"  rollout {r}: {'SUCCESS' if success else 'fail'}  n_steps={n_steps:3d}  "
                  f"infer={infer_ms:.1f}ms  wall={wall:.1f}s")
        env.close()

        rate = successes / max(n_rollouts, 1)
        overall_succ += successes
        overall_total += n_rollouts
        print(f"  TASK sim{sim_id} success: {successes}/{n_rollouts} = {rate:.0%}")
        summary["tasks"].append({
            "sim_id": sim_id, "train_task_id": train_task_id,
            "task_name": task.name, "task_desc": sim_task_lang[sim_id],
            "n_rollouts": n_rollouts, "n_successes": successes, "success_rate": rate,
            "rollouts": rollout_records,
        })

    print("\n" + "=" * 80)
    print(f"OVERALL: {overall_succ}/{overall_total} = {overall_succ/max(overall_total,1):.0%}")
    print("=" * 80)
    summary["overall_successes"] = overall_succ
    summary["overall_total"] = overall_total
    summary["overall_success_rate"] = overall_succ / max(overall_total, 1)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"Saved summary to {args.out_json}")


if __name__ == "__main__":
    main()
