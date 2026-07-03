"""A2: tangent direction analysis on success vs failure.

Run big v2 hybrid substrate eval with per-call detailed logging.
Capture per turn:
  - tangent norm (substrate's output magnitude)
  - cos(tangent, z_goal - z_t) — alignment with goal direction
  - tangent vs noise component (in hybrid mode, decompose)
  - sub-task outcome (success/failure)
  - sub-task progress (env steps used, gripper state)

Offline: aggregate tangent stats by (success, failure) and per sub-task class.
Question: does substrate produce qualitatively different output in successful
vs failed trajectories?
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from collections import defaultdict
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

from liquid_goal_tracker_jepa_big import JEPA_LGT_Big  # type: ignore
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


class LoggingClient:
    """Client that logs per-call tangent + diagnostic data."""
    def __init__(self, port, substrate_ckpt, suite_name, device=None,
                 noise_mix=0.5, noise_norm=7.0, noise_seed=0,
                 lookahead_lambda=1.0):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.lookahead_lambda = lookahead_lambda
        self.noise_mix = noise_mix
        self.noise_norm = noise_norm
        self.noise_rng = np.random.default_rng(noise_seed)
        self.suite_name = suite_name
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, 60000)
        self.sock.connect(f"tcp://localhost:{port}")
        ck = torch.load(substrate_ckpt, map_location=device, weights_only=False)
        sa = ck["args"]
        z_dim = ck["z_vl_dim"]
        action_dim = ck["action_dim"]
        horizon = ck["horizon"]
        self.substrate = JEPA_LGT_Big(
            z_vl_dim=z_dim, action_dim=action_dim, horizon=horizon,
            d=sa["d_substrate"], K=sa["K_belief"],
            tangent_scale=sa["tangent_scale"],
        ).to(device)
        self.substrate.load_state_dict(ck["substrate_state_dict"])
        self.substrate.eval()
        self.action_dim = action_dim
        self.horizon = horizon
        self.z_dim = z_dim
        self.h_goal = None
        self.last_chunk = None
        self.z_goal_cache = {}
        self.current_z_goal = None
        # PER-CALL LOG (cleared at sub-task reset)
        self.call_log = []
        print(f"[a2] loaded {substrate_ckpt}, noise_mix={noise_mix}", flush=True)

    def set_subtask(self, task_id):
        self.h_goal = None
        self.last_chunk = None
        self.call_log = []
        if task_id not in self.z_goal_cache:
            z_g = extract_z_goal_via_groot(self.sock, self.suite_name, task_id)
            if z_g is None:
                z_g = np.zeros(self.z_dim, dtype=np.float32)
            self.z_goal_cache[task_id] = z_g
        self.current_z_goal = self.z_goal_cache[task_id]

    def query(self, obs_dict):
        self.sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs_dict}))
        resp1 = pickle.loads(self.sock.recv())
        z_t = resp1["z_vl"]
        with torch.no_grad():
            z_t_torch = torch.from_numpy(z_t).to(self.device).unsqueeze(0)
            z_goal_torch = torch.from_numpy(self.current_z_goal).to(
                self.device).unsqueeze(0)
            if self.h_goal is None:
                self.h_goal = self.substrate.init_state(1, self.device)
            if self.last_chunk is None:
                lc = torch.zeros(1, self.horizon, self.action_dim, device=self.device)
            else:
                lc = torch.from_numpy(self.last_chunk).to(self.device).unsqueeze(0).float()
            h_new, _, tangent, info = self.substrate.step(
                self.h_goal, z_t_torch, z_goal_torch, lc)
            self.h_goal = h_new
            sub_tangent = (tangent[0].cpu().numpy() * self.lookahead_lambda
                            ).astype(np.float32)

        # Optional noise mix
        if self.noise_mix > 0:
            n = self.noise_rng.standard_normal(self.z_dim).astype(np.float32)
            n = n / (np.linalg.norm(n) + 1e-8) * self.noise_norm
            tangent_np = ((1.0 - self.noise_mix) * sub_tangent
                          + self.noise_mix * n).astype(np.float32)
        else:
            tangent_np = sub_tangent

        # ==== LOGGING ====
        gd = self.current_z_goal - z_t
        gd_norm = np.linalg.norm(gd) + 1e-8
        st_norm = np.linalg.norm(sub_tangent) + 1e-8
        cos_sub_goal = float(np.dot(sub_tangent, gd) / (st_norm * gd_norm))
        tn_norm = np.linalg.norm(tangent_np) + 1e-8
        cos_final_goal = float(np.dot(tangent_np, gd) / (tn_norm * gd_norm))
        self.call_log.append({
            "sub_tangent_norm": float(st_norm),
            "final_tangent_norm": float(tn_norm),
            "cos_sub_to_goaldir": cos_sub_goal,
            "cos_final_to_goaldir": cos_final_goal,
            "z_t_to_goal_dist": float(gd_norm),
            "cv": float(info["metric_cv"]),
        })

        # 3. GR00T action with tangent override
        self.sock.send(pickle.dumps({"op": "get_action_with_zvl_override",
                                       "obs": obs_dict, "zvl_residual": tangent_np}))
        resp2 = pickle.loads(self.sock.recv())
        chunk = np.asarray(resp2["chunk"], dtype=np.float32)
        self.last_chunk = chunk.copy()
        return chunk


def step_subtask(env, sub_task, obs, args, client, max_steps=None):
    sub_lang = sub_task.language
    chunk = None
    chunk_idx = 0
    sub_steps = 0
    n_calls = 0
    cap = max_steps if max_steps is not None else args.max_steps_per_task
    gripper_qpos_log = []
    while sub_steps < cap:
        if chunk is None or chunk_idx >= args.exec_horizon or chunk_idx >= len(chunk):
            img_raw, wrist_raw = get_raw_imgs(obs)
            state8 = build_state8(obs)
            groot_img = img_raw[::-1, ::-1].copy()
            groot_wrist = wrist_raw[::-1, ::-1].copy()
            groot_obs = build_groot_obs(groot_img, groot_wrist, state8, sub_lang)
            chunk = client.query(groot_obs)
            chunk_idx = 0
            n_calls += 1
        action7 = chunk[chunk_idx].copy()
        g = action7[-1]
        action7[-1] = np.sign(g) if abs(g) > 0.1 else 0.0
        obs, _, done, _ = env.step(action7.astype(np.float32))
        chunk_idx += 1
        sub_steps += 1
        if isinstance(obs, dict):
            gq = obs.get("robot0_gripper_qpos")
            if gq is not None:
                gripper_qpos_log.append(float(np.asarray(gq).mean()))
        if env.check_success():
            return True, sub_steps, obs, n_calls, gripper_qpos_log
        if done:
            return False, sub_steps, obs, n_calls, gripper_qpos_log
    return False, sub_steps, obs, n_calls, gripper_qpos_log


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
    p.add_argument("--substrate_ckpt", default="/tmp/lgt_jepa_big_K4_v2.pt")
    p.add_argument("--groot_port", type=int, default=5555)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--n_chains", type=int, default=3)
    p.add_argument("--rollouts_per_chain", type=int, default=4)
    p.add_argument("--chain_length", type=int, default=5)
    p.add_argument("--task_suite", default="libero_10")
    p.add_argument("--max_steps_per_task", type=int, default=720)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--retract_steps", type=int, default=30)
    p.add_argument("--retract_grip_thr", type=float, default=0.030)
    p.add_argument("--noise_mix", type=float, default=0.5)
    p.add_argument("--shared_budget", type=int, default=1100)
    p.add_argument("--out_json", default="/tmp/a2_tangent_success.json")
    args = p.parse_args()

    client = LoggingClient(args.groot_port, args.substrate_ckpt, args.task_suite,
                             noise_mix=args.noise_mix, noise_seed=args.seed)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.get_num_tasks()
    rng = np.random.default_rng(args.seed)
    chains = [sorted(rng.choice(n_tasks, size=args.chain_length, replace=False).tolist())
               for _ in range(args.n_chains)]
    print(f"[a2] suite={args.task_suite} chains={len(chains)} L={args.chain_length}",
          flush=True)

    envs = {}
    def get_env(task_id):
        if task_id in envs: return envs[task_id]
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        e = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        envs[task_id] = e
        return e

    all_subtask_records = []
    for chain_idx, chain in enumerate(chains):
        print(f"\n=== chain{chain_idx} {chain} ===", flush=True)
        first_env = get_env(chain[0])
        init_states = suite.get_task_init_states(chain[0])
        n_rollouts = min(args.rollouts_per_chain, len(init_states))
        for r in range(n_rollouts):
            t0 = time.time()
            first_env.reset()
            first_env.set_init_state(init_states[r])
            obs = None
            for _ in range(5):
                obs, _, _, _ = first_env.step(np.zeros(7, dtype=np.float32))
            current_env = first_env
            budget_remaining = args.shared_budget
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
                if budget_remaining <= 0:
                    break
                step_cap = min(args.max_steps_per_task, budget_remaining)
                succ, steps, obs, n_calls, gq_log = step_subtask(
                    current_env, sub_task, obs, args, client, max_steps=step_cap)
                budget_remaining -= steps
                # Save per-sub-task record with all per-call data
                all_subtask_records.append({
                    "chain_idx": chain_idx, "rollout": r, "sub_idx": sub_idx,
                    "sub_id": sub_id, "succ": succ, "steps": steps,
                    "n_calls": n_calls,
                    "calls": list(client.call_log),  # per-call data
                    "gripper_qpos": gq_log,
                })
                if succ and args.retract_steps > 0 and sub_idx < len(chain) - 1:
                    open_lift = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, -1.0],
                                          dtype=np.float32)
                    for _ in range(args.retract_steps):
                        if budget_remaining <= 0: break
                        if isinstance(obs, dict):
                            gq = obs.get("robot0_gripper_qpos")
                            if gq is not None and float(np.asarray(gq).mean()) > args.retract_grip_thr:
                                break
                        obs, _, _, _ = current_env.step(open_lift)
                        budget_remaining -= 1
                if not succ: break
            wall = time.time() - t0
            print(f"  r{r}: {sum(1 for rec in all_subtask_records[-len(chain):] if rec.get('succ', False))}/"
                  f"{len(chain)} sub-tasks  wall={wall:.0f}s", flush=True)

    for e in envs.values():
        try: e.close()
        except: pass

    # ============== Aggregation ==============
    succ_recs = [r for r in all_subtask_records if r["succ"]]
    fail_recs = [r for r in all_subtask_records if not r["succ"]]
    print(f"\n=== A2 AGGREGATION ===")
    print(f"  succ_subtasks={len(succ_recs)}  fail_subtasks={len(fail_recs)}", flush=True)

    def call_stats(records, label):
        if not records:
            return {"label": label, "n_subtasks": 0}
        all_calls = []
        for rec in records:
            all_calls.extend(rec["calls"])
        if not all_calls:
            return {"label": label, "n_subtasks": len(records), "n_calls": 0}
        sub_norms = [c["sub_tangent_norm"] for c in all_calls]
        final_norms = [c["final_tangent_norm"] for c in all_calls]
        cos_sub = [c["cos_sub_to_goaldir"] for c in all_calls]
        cos_final = [c["cos_final_to_goaldir"] for c in all_calls]
        cvs = [c["cv"] for c in all_calls]
        z_dists = [c["z_t_to_goal_dist"] for c in all_calls]
        return {
            "label": label, "n_subtasks": len(records), "n_calls": len(all_calls),
            "sub_tangent_norm_mean": float(np.mean(sub_norms)),
            "sub_tangent_norm_std": float(np.std(sub_norms)),
            "final_tangent_norm_mean": float(np.mean(final_norms)),
            "cos_sub_to_goal_mean": float(np.mean(cos_sub)),
            "cos_sub_to_goal_std": float(np.std(cos_sub)),
            "cos_final_to_goal_mean": float(np.mean(cos_final)),
            "cv_mean": float(np.mean(cvs)),
            "z_dist_mean": float(np.mean(z_dists)),
        }

    succ_stats = call_stats(succ_recs, "SUCCESS")
    fail_stats = call_stats(fail_recs, "FAILURE")
    print(f"\n=== SUCCESS subtasks ===")
    for k, v in succ_stats.items(): print(f"  {k}: {v}")
    print(f"\n=== FAILURE subtasks ===")
    for k, v in fail_stats.items(): print(f"  {k}: {v}")

    out = {
        "succ_count": len(succ_recs), "fail_count": len(fail_recs),
        "succ_stats": succ_stats, "fail_stats": fail_stats,
        "per_subtask_records": all_subtask_records,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[a2] saved → {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
