"""Motor competence eval for GR00T (via ZMQ server) — apples-to-apples vs student.

Same metrics as motor_competence_eval.py (reach/grasp/object_moved/smoothness/etc),
but runs GR00T's policy via the ZMQ server instead of a local student checkpoint.

Run inside the LIBERO sim venv. Requires groot_server.py running in main venv:
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python motor_competence_groot.py \\
    --task_suite libero_spatial --rollouts_per_task 3 --max_steps 400 --port 5555
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import zmq

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motor_competence_eval import (
    build_state8, get_object_positions, get_raw_imgs, compute_metrics, quat2axisangle,
)

print = functools.partial(print, flush=True)


def query_groot(sock, obs_dict):
    sock.send(pickle.dumps(obs_dict))
    r = pickle.loads(sock.recv())
    if "error" in r:
        raise RuntimeError(f"server error: {r['error']}")
    return r["chunk"]


def run_groot_rollout(env_underlying, sock, init_state, args, task_lang,
                      state_keys, video_keys, language_keys, state_slots):
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
            new_obs = {"video": {}, "state": {}, "language": {}}
            for k in video_keys:
                arr = img_raw if k == "image" else wrist_raw
                new_obs["video"][k] = arr[None, None, ...]
            for k in state_keys:
                lo, hi = state_slots[k]
                new_obs["state"][k] = state8[lo:hi].astype(np.float32)[None, None, :]
            for lk in language_keys:
                new_obs["language"][lk] = [[task_lang]]
            chunk = query_groot(sock, new_obs)
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
    p.add_argument("--task_suite", default="libero_spatial", type=str)
    p.add_argument("--rollouts_per_task", type=int, default=3)
    p.add_argument("--task_indices", type=str, default="")
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--gripper_sign", type=float, default=1.0)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--out_json", default="", type=str)
    args = p.parse_args()

    # GR00T modality (mirror server)
    state_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    video_keys = ["image", "wrist_image"]
    language_keys = ["annotation.human.action.task_description"]
    state_slots = {"x": (0, 1), "y": (1, 2), "z": (2, 3), "roll": (3, 4),
                   "pitch": (4, 5), "yaw": (5, 6), "gripper": (6, 8)}

    ctx = zmq.Context(); sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    sock.setsockopt(zmq.RCVTIMEO, 60000)
    print(f"Connected to GR00T server on tcp://127.0.0.1:{args.port}")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.get_num_tasks()
    if args.task_indices:
        task_ids = [int(t) for t in args.task_indices.split(",")]
    else:
        task_ids = list(range(n_tasks))

    summary = {"task_suite": args.task_suite, "policy": "groot", "tasks": []}
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
        print(f"\n=== sim{sim_id} GR00T: {task.language[:80]} ===")
        for r in range(n_rollouts):
            t0 = time.time()
            m = run_groot_rollout(env, sock, init_states[r], args, task.language,
                                   state_keys, video_keys, language_keys, state_slots)
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
    print(f"GR00T AGGREGATE over {n_total} rollouts:")
    print(f"  reached:      {agg['reached']}/{n_total} = {100*agg['reached']/n_total:.0f}%")
    print(f"  grasp_event:  {agg['grasp_event']}/{n_total} = {100*agg['grasp_event']/n_total:.0f}%")
    print(f"  object_moved: {agg['object_moved']}/{n_total} = {100*agg['object_moved']/n_total:.0f}%")
    print(f"  task_success: {agg['task_success']}/{n_total} = {100*agg['task_success']/n_total:.0f}%")
    print(f"  mean reach distance:  {np.mean(agg['reach_min_cm']):.1f} cm")
    print(f"  mean smoothness:      {np.mean(agg['smoothness']):.2f}")
    print(f"  mean exploration:     {np.mean(agg['exploration']):.2f}")

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


if __name__ == "__main__":
    main()
