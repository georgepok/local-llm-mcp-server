"""Chained-LIBERO eval with PROPRIO substrate as a CONJUNCTIVE stuck-detector.

Why conjunctive: A2 diagnostic showed substrate's single-signal output was uniform
across success/failure. Scalar-gate fired -26pp on L=5 hard with distance alone.
Proprio substrate has three signals — combining them sharpens the decision.

Per-call:
  state8 from env, z_t from GR00T → substrate predicts:
    1. goal_distance (R²=0.64 held-out)
    2. P(gripper_moving) [binary, acc 0.85 held-out]
    3. state_delta magnitude (R²=0.55 held-out)

Stuck-trigger fires ONLY when (conjunction over `window` calls):
  - P(gripper_moving) < gripper_thresh  (substrate predicts gripper stuck)
  AND
  - state_delta_pred < state_delta_thresh  (substrate predicts no motion)
  AND optionally
  - distance NOT decreasing

Caps: max_triggers_per_subtask, trigger_cooldown_calls.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pickle
import torch
import zmq

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SELF_DIR))

from liquid_goal_tracker_proprio import JEPA_LGT_Proprio  # type: ignore
from rollout_libero_v11_client import (  # type: ignore
    build_state8, get_raw_imgs, build_groot_obs,
)


DATASET_ROOT = Path("/home/pokazge/datasets")


def extract_z_goal_via_groot(sock, suite_name, task_id):
    suite_short = suite_name.replace("libero_", "")
    sd = DATASET_ROOT / f"libero-{suite_short}-expert-v1"
    if not sd.exists():
        return None
    idx = np.load(sd / "index.npz")
    starts = idx["episode_starts"]; lengths = idx["episode_lengths"]
    task_indices = idx["task_indices"]
    success = idx.get("success_per_episode", np.ones(len(lengths), dtype=bool))
    n_total = int(idx["n_total"]); img_size = int(idx["img_size"])
    imgs = np.memmap(sd / "imgs.dat", dtype=np.uint8, mode="r",
                     shape=(n_total, img_size, img_size, 3))
    wrists = np.memmap(sd / "wrists.dat", dtype=np.uint8, mode="r",
                       shape=(n_total, img_size, img_size, 3))
    states = np.memmap(sd / "states.dat", dtype=np.float32, mode="r",
                       shape=(n_total, 8))
    task_lang = {}
    lp = sd / "task_languages.json"
    if lp.exists():
        task_lang = json.loads(lp.read_text())
    for ep in range(len(lengths)):
        if int(task_indices[ep]) != task_id or not bool(success[ep]):
            continue
        ep_start = int(starts[ep]); ep_len = int(lengths[ep])
        last_gi = ep_start + ep_len - 1
        img = np.array(imgs[last_gi])[::-1, ::-1].copy()
        wri = np.array(wrists[last_gi])[::-1, ::-1].copy()
        st = np.array(states[last_gi])
        lang = (task_lang.get(str(task_id)) or task_lang.get(task_id)
                or "do the task")
        obs = build_groot_obs(img, wri, st, lang)
        sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs}))
        resp = pickle.loads(sock.recv())
        return resp["z_vl"].astype(np.float32)
    return None


class ProprioGateClient:
    def __init__(self, port, substrate_ckpt, suite_name, device=None,
                 window=4, gripper_thresh=0.5, state_delta_thresh=0.6,
                 require_distance_stall=True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.suite_name = suite_name
        self.window = window
        self.gripper_thresh = gripper_thresh
        self.state_delta_thresh = state_delta_thresh
        self.require_distance_stall = require_distance_stall
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, 60000)
        self.sock.connect(f"tcp://localhost:{port}")

        ck = torch.load(substrate_ckpt, map_location=device, weights_only=False)
        sa = ck["args"]
        z_dim = ck["z_vl_dim"]
        action_dim = ck["action_dim"]
        horizon = ck["horizon"]
        state_dim = ck["state_dim"]
        self.substrate = JEPA_LGT_Proprio(
            z_vl_dim=z_dim, action_dim=action_dim, horizon=horizon,
            state_dim=state_dim, d=sa["d_substrate"], K=sa["K_belief"],
        ).to(device)
        self.substrate.load_state_dict(ck["substrate_state_dict"])
        self.substrate.eval()
        self.action_dim = action_dim
        self.horizon = horizon
        self.z_dim = z_dim
        self.state_dim = state_dim
        self.dist_mean = ck["dist_mean"]; self.dist_std = ck["dist_std"]
        self.sd_mean = ck["sd_mean"]; self.sd_std = ck["sd_std"]
        self.h_goal = None
        self.z_goal_cache = {}
        self.current_z_goal = None
        # Histories
        self.gripper_history = []
        self.state_delta_history = []
        self.distance_history = []
        self.n_chunks = 0
        print(f"[proprio-gate] loaded {substrate_ckpt}, d={sa['d_substrate']}, "
              f"K={sa['K_belief']}, window={window}, "
              f"gripper_thresh={gripper_thresh}, "
              f"state_delta_thresh={state_delta_thresh}, "
              f"require_dist_stall={require_distance_stall}", flush=True)

    def set_subtask(self, task_id):
        self.h_goal = None
        self.gripper_history = []
        self.state_delta_history = []
        self.distance_history = []
        if task_id not in self.z_goal_cache:
            z_g = extract_z_goal_via_groot(self.sock, self.suite_name, task_id)
            if z_g is None:
                z_g = np.zeros(self.z_dim, dtype=np.float32)
            self.z_goal_cache[task_id] = z_g
        self.current_z_goal = self.z_goal_cache[task_id]

    def query(self, obs_dict, state8_now):
        """Returns (chunk, is_stuck, diag)."""
        # 1. Baseline GR00T action + z_t
        self.sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs_dict}))
        resp = pickle.loads(self.sock.recv())
        chunk = np.asarray(resp["chunk"], dtype=np.float32)
        z_t = resp["z_vl"]

        # 2. Substrate predictions
        with torch.no_grad():
            z_t_t = torch.from_numpy(z_t).to(self.device).unsqueeze(0)
            z_g_t = torch.from_numpy(self.current_z_goal).to(self.device).unsqueeze(0)
            if self.h_goal is None:
                self.h_goal = self.substrate.init_state(1, self.device)
            chunk_t = torch.from_numpy(chunk).to(self.device).unsqueeze(0).float()
            state8_t = torch.from_numpy(state8_now).to(self.device).unsqueeze(0).float()
            h_new, p_d, aux, _ = self.substrate.step(
                self.h_goal, z_t_t, z_g_t, chunk_t, state8_t)
            self.h_goal = h_new
            p_dist = float(p_d) * self.dist_std + self.dist_mean
            p_gripper = float(torch.sigmoid(aux["pred_gripper_moving_logit"]))
            p_state_delta = float(aux["pred_state_delta"]) * self.sd_std + self.sd_mean

        # 3. Track + decide
        self.gripper_history.append(p_gripper)
        self.state_delta_history.append(p_state_delta)
        self.distance_history.append(p_dist)
        is_stuck = False
        if len(self.gripper_history) >= 2 * self.window:
            recent_g = np.mean(self.gripper_history[-self.window:])
            recent_s = np.mean(self.state_delta_history[-self.window:])
            cond_gripper = (recent_g < self.gripper_thresh)
            cond_state = (recent_s < self.state_delta_thresh)
            cond_distance = True
            if self.require_distance_stall:
                recent_d = np.mean(self.distance_history[-self.window:])
                older_d = np.mean(self.distance_history[
                    -2 * self.window:-self.window])
                cond_distance = (recent_d >= older_d - 0.1)
            is_stuck = cond_gripper and cond_state and cond_distance
        self.n_chunks += 1
        return chunk, is_stuck, {
            "p_dist": p_dist, "p_gripper": p_gripper, "p_state_delta": p_state_delta
        }


def step_subtask(env, sub_task, obs, args, client, max_steps=None):
    sub_lang = sub_task.language
    chunk = None
    chunk_idx = 0
    sub_steps = 0
    n_calls = 0
    n_stuck = 0
    calls_since_last_trigger = 0
    cap = max_steps if max_steps is not None else args.max_steps_per_task
    while sub_steps < cap:
        if chunk is None or chunk_idx >= args.exec_horizon or chunk_idx >= len(chunk):
            img_raw, wrist_raw = get_raw_imgs(obs)
            state8 = build_state8(obs)
            groot_img = img_raw[::-1, ::-1].copy()
            groot_wrist = wrist_raw[::-1, ::-1].copy()
            groot_obs = build_groot_obs(groot_img, groot_wrist, state8, sub_lang)
            chunk, is_stuck, diag = client.query(groot_obs, state8)
            chunk_idx = 0
            n_calls += 1
            calls_since_last_trigger += 1
            should_trigger = (
                args.use_gate and is_stuck
                and n_calls > args.stuck_min_calls
                and n_stuck < args.max_triggers_per_subtask
                and calls_since_last_trigger >= args.trigger_cooldown_calls
            )
            if should_trigger:
                n_stuck += 1
                calls_since_last_trigger = 0
                open_lift = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, -1.0],
                                      dtype=np.float32)
                for _ in range(args.stuck_retract_steps):
                    if sub_steps >= cap: break
                    obs, _, done, _ = env.step(open_lift)
                    sub_steps += 1
                    if env.check_success():
                        return True, sub_steps, obs, n_calls, n_stuck
                    if done:
                        return False, sub_steps, obs, n_calls, n_stuck
                chunk = None
                continue
        action7 = chunk[chunk_idx].copy()
        g = action7[-1]
        action7[-1] = np.sign(g) if abs(g) > 0.1 else 0.0
        obs, _, done, _ = env.step(action7.astype(np.float32))
        chunk_idx += 1
        sub_steps += 1
        if env.check_success():
            return True, sub_steps, obs, n_calls, n_stuck
        if done:
            return False, sub_steps, obs, n_calls, n_stuck
    return False, sub_steps, obs, n_calls, n_stuck


def transfer_state(src_env, dst_env, src_obs):
    try:
        dst_env.reset()
        src_joints = None
        if isinstance(src_obs, dict) and "robot0_joint_pos" in src_obs:
            src_joints = np.asarray(src_obs["robot0_joint_pos"], dtype=np.float64)
        elif hasattr(src_env, "robots") and src_env.robots:
            src_joints = np.asarray(src_env.robots[0]._joint_positions, dtype=np.float64)
        if src_joints is None or len(src_joints) == 0:
            return False
        if hasattr(dst_env, "robots") and dst_env.robots:
            dst_env.robots[0].set_robot_joint_positions(src_joints)
            if hasattr(dst_env, "sim"):
                dst_env.sim.forward()
            return True
    except Exception:
        pass
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate_ckpt", required=True, type=str)
    p.add_argument("--groot_port", type=int, default=5555)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--n_chains", type=int, default=6)
    p.add_argument("--rollouts_per_chain", type=int, default=4)
    p.add_argument("--chain_length", type=int, default=5)
    p.add_argument("--task_suite", default="libero_10")
    p.add_argument("--max_steps_per_task", type=int, default=720)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--retract_steps", type=int, default=30)
    p.add_argument("--retract_grip_thr", type=float, default=0.030)
    p.add_argument("--shared_budget", type=int, default=1100)
    # Gate params
    p.add_argument("--use_gate", action="store_true")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--gripper_thresh", type=float, default=0.5)
    p.add_argument("--state_delta_thresh", type=float, default=0.6)
    p.add_argument("--require_distance_stall", action="store_true",
                   help="Also require distance not decreasing.")
    p.add_argument("--stuck_min_calls", type=int, default=8)
    p.add_argument("--stuck_retract_steps", type=int, default=20)
    p.add_argument("--max_triggers_per_subtask", type=int, default=2)
    p.add_argument("--trigger_cooldown_calls", type=int, default=10)
    p.add_argument("--out_json", default="/tmp/chained_libero_proprio_gate.json")
    args = p.parse_args()

    client = ProprioGateClient(
        args.groot_port, args.substrate_ckpt, args.task_suite,
        window=args.window, gripper_thresh=args.gripper_thresh,
        state_delta_thresh=args.state_delta_thresh,
        require_distance_stall=args.require_distance_stall)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.get_num_tasks()
    rng = np.random.default_rng(args.seed)
    chains = [sorted(rng.choice(n_tasks, size=args.chain_length, replace=False).tolist())
               for _ in range(args.n_chains)]
    print(f"[proprio-gate] suite={args.task_suite} chains={len(chains)} "
          f"L={args.chain_length} use_gate={args.use_gate} "
          f"budget={args.shared_budget}", flush=True)

    envs = {}
    def get_env(task_id):
        if task_id in envs: return envs[task_id]
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        e = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        envs[task_id] = e
        return e

    total_succ = 0; total_attempts = 0
    chain_completions = []
    total_stuck = 0
    summary = {"chains": []}

    for chain_idx, chain in enumerate(chains):
        print(f"\n=== chain{chain_idx} {chain} ===", flush=True)
        first_env = get_env(chain[0])
        init_states = suite.get_task_init_states(chain[0])
        n_rollouts = min(args.rollouts_per_chain, len(init_states))
        for r in range(n_rollouts):
            t0 = time.time()
            n_complete = 0
            first_env.reset()
            first_env.set_init_state(init_states[r])
            obs = None
            for _ in range(5):
                obs, _, _, _ = first_env.step(np.zeros(7, dtype=np.float32))
            current_env = first_env
            budget_remaining = args.shared_budget if args.shared_budget > 0 else None
            sub_details = []
            rollout_stuck = 0
            for sub_idx, sub_id in enumerate(chain):
                sub_task = suite.get_task(sub_id)
                client.set_subtask(sub_id)
                if sub_idx > 0:
                    next_env = get_env(sub_id)
                    next_env.reset()
                    next_init_states = suite.get_task_init_states(sub_id)
                    next_env.set_init_state(next_init_states[r % len(next_init_states)])
                    for _ in range(3):
                        obs, _, _, _ = next_env.step(np.zeros(7, dtype=np.float32))
                    transfer_state(current_env, next_env, obs)
                    current_env = next_env
                    for _ in range(5):
                        obs, _, _, _ = current_env.step(np.zeros(7, dtype=np.float32))
                if budget_remaining is not None and budget_remaining <= 0:
                    sub_details.append({"sub_id": sub_id, "succ": False, "steps": 0,
                                         "skipped_no_budget": True})
                    break
                step_cap = (min(args.max_steps_per_task, budget_remaining)
                            if budget_remaining is not None
                            else args.max_steps_per_task)
                succ, steps, obs, n_calls, n_stuck = step_subtask(
                    current_env, sub_task, obs, args, client, max_steps=step_cap)
                rollout_stuck += n_stuck
                if budget_remaining is not None:
                    budget_remaining -= steps
                sub_details.append({"sub_id": sub_id, "succ": succ, "steps": steps,
                                     "n_calls": n_calls, "n_stuck": n_stuck,
                                     "budget_left": budget_remaining})
                if succ and args.retract_steps > 0 and sub_idx < len(chain) - 1:
                    open_lift = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, -1.0],
                                          dtype=np.float32)
                    for _ in range(args.retract_steps):
                        if budget_remaining is not None and budget_remaining <= 0: break
                        if isinstance(obs, dict):
                            gq = obs.get("robot0_gripper_qpos")
                            if gq is not None and float(np.asarray(gq).mean()) > args.retract_grip_thr:
                                break
                        obs, _, _, _ = current_env.step(open_lift)
                        if budget_remaining is not None:
                            budget_remaining -= 1
                if succ: n_complete += 1
                else: break
            wall = time.time() - t0
            chain_completions.append(n_complete)
            total_succ += n_complete
            total_attempts += len(chain)
            total_stuck += rollout_stuck
            steps_str = ",".join(str(s["steps"]) for s in sub_details if "steps" in s)
            print(f"  r{r}: {n_complete}/{len(chain)} sub-tasks  wall={wall:.0f}s  "
                  f"steps=[{steps_str}]  stuck={rollout_stuck}", flush=True)
            summary["chains"].append({
                "chain_idx": chain_idx, "rollout": r, "chain": chain,
                "n_complete": n_complete, "wall_s": wall, "details": sub_details,
                "n_stuck": rollout_stuck,
            })

    for e in envs.values():
        try: e.close()
        except: pass

    summary["total_succ"] = total_succ
    summary["total_attempts"] = total_attempts
    summary["completion_rate"] = total_succ / max(total_attempts, 1)
    summary["mean_chain_completion"] = (
        float(np.mean(chain_completions)) if chain_completions else 0)
    summary["total_stuck_triggers"] = total_stuck
    print(f"\n{'='*60}", flush=True)
    print(f"OVERALL: {total_succ}/{total_attempts} sub-tasks = "
          f"{100*summary['completion_rate']:.0f}%  stuck_triggers={total_stuck}",
          flush=True)
    print(f"Mean chain completion: {summary['mean_chain_completion']:.2f} "
          f"/ {args.chain_length}", flush=True)
    print(f"{'='*60}", flush=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
