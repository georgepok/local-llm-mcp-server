"""Animal-level motor competence eval (OOD-friendly).

Doesn't require the student to understand the task. Measures whether the body
does *coherent things* in any LIBERO scene:

  - reach: did the gripper get within R cm of any object centroid (any object)?
  - grasp_close: did the gripper close while near an object?
  - object_moved: did any object's position change > T cm during the episode?
  - action_smoothness: 1 / (1 + std of consecutive action diffs)  (anti-thrash)
  - exploration: total path length of EE through workspace / max possible

Tests "is there an animal in there" rather than "did it solve the task".
Run inside the LIBERO sim venv:
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python motor_competence_eval.py \\
    --student_ckpt /tmp/distill_groot_flow_v3_dagger2/step_015000.pt \\
    --task_suite libero_spatial --rollouts_per_task 3 --max_steps 400
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


def get_object_positions(env_underlying, obs=None):
    """Return dict of object_name -> [x, y, z] from obs `*_pos` keys.

    Falls back to scanning sim.model.body_names for any non-robot/world body.
    """
    pos = {}
    # Path 1: parse obs for any key ending in _pos (excluding robot)
    if obs is not None:
        for k, v in obs.items():
            if k.endswith("_pos") and not k.startswith("robot0"):
                try:
                    arr = np.asarray(v, dtype=np.float32)
                    if arr.shape == (3,):
                        pos[k[:-4]] = arr
                except Exception:
                    pass
    # Path 2: fall back to sim model bodies
    if not pos:
        try:
            sim = env_underlying.sim
            model = sim.model
            for body_id in range(model.nbody):
                try:
                    name = model.body_id2name(body_id)
                except Exception:
                    name = sim.model.body_names[body_id] if hasattr(sim.model, "body_names") else f"body_{body_id}"
                if name in (None, "world", "robot0_base") or name.startswith("robot0"):
                    continue
                try:
                    pos[name] = np.array(sim.data.body_xpos[body_id])
                except Exception:
                    pass
        except Exception:
            pass
    return pos


def get_raw_imgs(obs_raw):
    return (
        obs_raw["agentview_image"][::-1, ::-1].copy(),
        obs_raw["robot0_eye_in_hand_image"][::-1, ::-1].copy(),
    )


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
        n_task_heads=sa.get("n_task_heads", 0),
    ).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own = model.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
    model.eval()
    return model, sa


@torch.no_grad()
def predict_chunk(model, img_raw, wrist_raw, state8, device, n_steps, target_size):
    img = np.array(Image.fromarray(img_raw).resize((target_size, target_size)), dtype=np.uint8)
    wri = np.array(Image.fromarray(wrist_raw).resize((target_size, target_size)), dtype=np.uint8)
    img_t = torch.from_numpy(img).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    wri_t = torch.from_numpy(wri).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    st_t = torch.from_numpy(state8).to(device).float().unsqueeze(0)
    chunk = model.sample(img_t, wri_t, st_t, task_id=None, n_steps=n_steps)
    return chunk[0].cpu().numpy()


def compute_metrics(traj):
    """traj is a list of dicts with keys: ee_xyz, gripper, action, obj_positions(t0)."""
    if len(traj) < 5:
        return {}
    ee_xyz = np.stack([t["ee_xyz"] for t in traj])              # [T, 3]
    grip = np.stack([t["gripper"] for t in traj])                # [T]
    actions = np.stack([t["action"] for t in traj])              # [T, 7]
    obj_t0 = traj[0]["obj_positions"]                            # dict at start
    obj_tf = traj[-1]["obj_positions"]                           # dict at end

    # 1. Reach: minimum gripper-to-any-object distance over episode
    if obj_t0:
        all_dists = []
        for name in obj_t0:
            pos = obj_t0[name]
            dists = np.linalg.norm(ee_xyz - pos[None, :], axis=1)
            all_dists.append(dists.min())
        reach_min = float(min(all_dists))
        reach_threshold = 0.10  # 10 cm
        reached = reach_min < reach_threshold
    else:
        reach_min = float("nan"); reached = False

    # 2. Grasp event: gripper closed (binary < 0) while EE near an object
    grasp_event = False
    for i, t in enumerate(traj):
        if grip[i] < 0:  # closed
            for name, pos in obj_t0.items():
                if np.linalg.norm(ee_xyz[i] - pos) < 0.10:
                    grasp_event = True; break
            if grasp_event: break

    # 3. Object moved: max delta of any object's position from start to end
    moved_dists = {}
    for name in obj_t0:
        if name in obj_tf:
            d = float(np.linalg.norm(obj_tf[name] - obj_t0[name]))
            moved_dists[name] = d
    max_obj_moved = float(max(moved_dists.values())) if moved_dists else 0.0
    object_moved = max_obj_moved > 0.05  # 5 cm

    # 4. Action smoothness: 1 / (1 + std of consecutive action diffs)
    action_diffs = np.diff(actions, axis=0)
    action_jitter = float(np.linalg.norm(action_diffs, axis=1).std())
    action_smoothness = 1.0 / (1.0 + action_jitter)

    # 5. Exploration: total EE path length over a "max possible" of ~3m budget
    path_len = float(np.linalg.norm(np.diff(ee_xyz, axis=0), axis=1).sum())
    exploration = min(path_len / 3.0, 1.0)

    return {
        "reach_min_cm": reach_min * 100,
        "reached": bool(reached),
        "grasp_event": bool(grasp_event),
        "max_obj_moved_cm": max_obj_moved * 100,
        "object_moved": bool(object_moved),
        "action_smoothness": action_smoothness,
        "exploration": exploration,
        "n_steps": len(traj),
    }


def run_rollout(env_underlying, model, init_state, args, device, target_size):
    env_underlying.reset()
    env_underlying.set_init_state(init_state)
    obs = None
    for _ in range(5):
        obs, _, _, _ = env_underlying.step(np.zeros(7, dtype=np.float32))
    obj_t0 = get_object_positions(env_underlying, obs=obs)
    traj = []
    chunk = None
    chunk_idx = 0
    for step in range(args.max_steps):
        img_raw, wrist_raw = get_raw_imgs(obs)
        state8 = build_state8(obs)
        if chunk is None or chunk_idx >= args.exec_horizon or chunk_idx >= len(chunk):
            chunk = predict_chunk(model, img_raw, wrist_raw, state8, device,
                                  n_steps=args.infer_steps, target_size=target_size)
            chunk_idx = 0
        action7 = chunk[chunk_idx].copy()
        g = action7[-1]
        action7[-1] = args.gripper_sign * np.sign(g) if abs(g) > 0.1 else 0.0
        traj.append({
            "ee_xyz": obs["robot0_eef_pos"].copy(),
            "gripper": float(action7[-1]),
            "action": action7.copy(),
            "obj_positions": obj_t0,
        })
        obs, _, done, _ = env_underlying.step(action7.astype(np.float32))
        chunk_idx += 1
        if done: break
    obj_tf = get_object_positions(env_underlying, obs=obs)
    if traj:
        traj[0]["obj_positions"] = obj_t0
        traj[-1]["obj_positions"] = obj_tf
    success = env_underlying.check_success()
    metrics = compute_metrics(traj)
    metrics["task_success"] = bool(success)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", required=True, type=str)
    p.add_argument("--task_suite", default="libero_spatial", type=str)
    p.add_argument("--rollouts_per_task", type=int, default=3)
    p.add_argument("--task_indices", type=str, default="")
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--infer_steps", type=int, default=10)
    p.add_argument("--gripper_sign", type=float, default=1.0)
    p.add_argument("--out_json", default="", type=str)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sargs = load_flow_policy(Path(args.student_ckpt), device)
    target_size = sargs["img_size"]

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.get_num_tasks()
    if args.task_indices:
        task_ids = [int(t) for t in args.task_indices.split(",")]
    else:
        task_ids = list(range(n_tasks))
    print(f"Suite={args.task_suite}, evaluating {len(task_ids)} tasks × {args.rollouts_per_task} rollouts")

    summary = {"task_suite": args.task_suite, "tasks": []}
    agg = {"reached": 0, "grasp_event": 0, "object_moved": 0, "task_success": 0,
           "smoothness": [], "exploration": [], "reach_min_cm": []}
    n_total = 0

    for sim_id in task_ids:
        task = suite.get_task(sim_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = suite.get_task_init_states(sim_id)
        n_rollouts = min(args.rollouts_per_task, len(init_states))
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        task_metrics = []
        print(f"\n=== sim{sim_id}: {task.language[:80]} ===")
        for r in range(n_rollouts):
            t0 = time.time()
            m = run_rollout(env, model, init_states[r], args, device, target_size)
            wall = time.time() - t0
            task_metrics.append(m)
            print(f"  r{r}: success={m['task_success']} reach={m['reached']} "
                  f"grasp={m['grasp_event']} obj_moved={m['object_moved']}  "
                  f"reach_cm={m['reach_min_cm']:.1f} obj_max={m['max_obj_moved_cm']:.1f}cm  "
                  f"smooth={m['action_smoothness']:.2f}  wall={wall:.1f}s")
            n_total += 1
            for k in ["reached", "grasp_event", "object_moved", "task_success"]:
                agg[k] += int(m[k])
            agg["smoothness"].append(m["action_smoothness"])
            agg["exploration"].append(m["exploration"])
            agg["reach_min_cm"].append(m["reach_min_cm"])
        env.close()
        summary["tasks"].append({"sim_id": sim_id, "task": task.language, "metrics": task_metrics})

    print("\n" + "=" * 80)
    print(f"AGGREGATE over {n_total} rollouts:")
    print(f"  reached (gripper got <10cm to an object):     {agg['reached']}/{n_total} = {100*agg['reached']/n_total:.0f}%")
    print(f"  grasp_event (gripper closed near an object):  {agg['grasp_event']}/{n_total} = {100*agg['grasp_event']/n_total:.0f}%")
    print(f"  object_moved (any object moved >5cm):         {agg['object_moved']}/{n_total} = {100*agg['object_moved']/n_total:.0f}%")
    print(f"  task_success (full task complete):            {agg['task_success']}/{n_total} = {100*agg['task_success']/n_total:.0f}%")
    print(f"  mean reach distance:                          {np.mean(agg['reach_min_cm']):.1f} cm")
    print(f"  mean action smoothness (1=smooth, 0=jittery): {np.mean(agg['smoothness']):.2f}")
    print(f"  mean exploration (path/3m budget):            {np.mean(agg['exploration']):.2f}")

    summary["aggregate"] = {
        "n_total": n_total,
        "reached_rate": agg["reached"] / max(n_total, 1),
        "grasp_event_rate": agg["grasp_event"] / max(n_total, 1),
        "object_moved_rate": agg["object_moved"] / max(n_total, 1),
        "task_success_rate": agg["task_success"] / max(n_total, 1),
        "mean_reach_cm": float(np.mean(agg["reach_min_cm"])),
        "mean_smoothness": float(np.mean(agg["smoothness"])),
        "mean_exploration": float(np.mean(agg["exploration"])),
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
