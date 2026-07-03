"""Chained-LIBERO eval with JEPA-LGT (action-conditioned world model).

Per turn:
1. Get z_t from GR00T (op=get_action_with_state) — actual current state
2. Substrate.step(h_goal, z_t, last_chunk) → ẑ_{t+1}, tangent = ẑ_{t+1} - z_t
3. Send λ·tangent to GR00T via get_action_with_zvl_override
   → action head decodes from (bb_features + λ·tangent) = ẑ_{t+1} preview
4. Execute returned chunk; this BECOMES last_chunk for the next call

At t=0: last_chunk is the substrate's `init_chunk` (zeros).
At sub-task boundary: reset h_goal AND last_chunk to zero.
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


SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SELF_DIR))

from liquid_goal_tracker_jepa import JEPA_LGT  # type: ignore
from rollout_libero_v11_client import (  # type: ignore
    build_state8, get_raw_imgs, build_groot_obs,
)


class GrootJepaClient:
    def __init__(self, port, substrate_ckpt, device=None, lookahead_lambda=1.0,
                 random_residual_norm=0.0, random_seed=0):
        """
        random_residual_norm > 0: ABLATION mode — bypass substrate, send a fresh
        Gaussian random vector of the given norm to GR00T's zvl_override each turn.
        This tests whether the substrate's *learned* tangent matters or any
        same-magnitude perturbation suffices.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.lookahead_lambda = lookahead_lambda
        self.random_residual_norm = random_residual_norm
        self.rng = np.random.default_rng(random_seed)
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, 60000)
        self.sock.connect(f"tcp://localhost:{port}")

        if self.random_residual_norm > 0:
            self.substrate = None
            ck = torch.load(substrate_ckpt, map_location=device, weights_only=False)
            self.action_dim = ck["action_dim"]
            self.horizon = ck["horizon"]
            self.z_dim = ck["z_vl_dim"]
            print(f"[ABLATION] random residual mode: norm={random_residual_norm}, "
                  f"seed={random_seed}; substrate ignored", flush=True)
        else:
            ck = torch.load(substrate_ckpt, map_location=device, weights_only=False)
            sa = ck["args"]
            z_dim = ck["z_vl_dim"]
            action_dim = ck["action_dim"]
            horizon = ck["horizon"]
            self.substrate = JEPA_LGT(
                z_vl_dim=z_dim, action_dim=action_dim, horizon=horizon,
                d=sa["d_substrate"], K=sa["K_belief"],
                tangent_scale=sa["tangent_scale"],
            ).to(device)
            self.substrate.load_state_dict(ck["substrate_state_dict"])
            self.substrate.eval()
            self.action_dim = action_dim
            self.horizon = horizon
            self.z_dim = z_dim
            print(f"[jepa-eval] loaded {substrate_ckpt}, "
                  f"d={sa['d_substrate']}, K={sa['K_belief']}, "
                  f"tangent_scale={sa['tangent_scale']}, λ={lookahead_lambda}",
                  flush=True)
        self.h_goal = None
        self.last_chunk = None
        self.total_tang_norm = 0.0
        self.n_chunks = 0

    def reset(self):
        self.h_goal = None
        self.last_chunk = None

    def query(self, obs_dict):
        # 1. z_t from GR00T
        self.sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs_dict}))
        resp1 = pickle.loads(self.sock.recv())
        z_t = resp1["z_vl"]  # [2048] np.float32

        # 2. substrate predicts ẑ_{t+1}, OR ablation generates random tangent
        if self.random_residual_norm > 0:
            r = self.rng.standard_normal(self.z_dim).astype(np.float32)
            r = r / (np.linalg.norm(r) + 1e-8) * self.random_residual_norm
            tangent_np = r * self.lookahead_lambda
        else:
            with torch.no_grad():
                z_t_torch = torch.from_numpy(z_t).to(self.device).unsqueeze(0)
                if self.h_goal is None:
                    self.h_goal = self.substrate.init_state(1, self.device)
                if self.last_chunk is None:
                    lc = torch.zeros(1, self.horizon, self.action_dim,
                                      device=self.device)
                else:
                    lc = torch.from_numpy(self.last_chunk).to(self.device).unsqueeze(0).float()
                h_new, z_pred, tangent, _ = self.substrate.step(
                    self.h_goal, z_t_torch, lc)
                self.h_goal = h_new
                tangent_np = (tangent[0].cpu().numpy() * self.lookahead_lambda
                              ).astype(np.float32)

        # 3. GR00T action with tangent as zvl override (forward-step preview)
        self.sock.send(pickle.dumps({"op": "get_action_with_zvl_override",
                                       "obs": obs_dict, "zvl_residual": tangent_np}))
        resp2 = pickle.loads(self.sock.recv())
        chunk = np.asarray(resp2["chunk"], dtype=np.float32)  # [horizon, action_dim]

        # 4. remember last executed chunk for next call
        self.last_chunk = chunk.copy()
        self.total_tang_norm += float(np.linalg.norm(tangent_np))
        self.n_chunks += 1
        return chunk


def step_subtask(env, sub_task, obs, args, client):
    sub_lang = sub_task.language
    chunk = None
    chunk_idx = 0
    sub_steps = 0
    n_calls = 0
    while sub_steps < args.max_steps_per_task:
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
        if env.check_success():
            return True, sub_steps, obs, n_calls
        if done:
            return False, sub_steps, obs, n_calls
    return False, sub_steps, obs, n_calls


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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_chains", type=int, default=3)
    p.add_argument("--rollouts_per_chain", type=int, default=2)
    p.add_argument("--chain_length", type=int, default=3)
    p.add_argument("--task_suite", default="libero_10", type=str)
    p.add_argument("--max_steps_per_task", type=int, default=720)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--retract_steps", type=int, default=30)
    p.add_argument("--retract_grip_thr", type=float, default=0.030)
    p.add_argument("--lookahead_lambda", type=float, default=1.0,
                   help="Strength of tangent (ẑ_{t+1}-z_t) sent to GR00T action head.")
    p.add_argument("--random_residual_norm", type=float, default=0.0,
                   help="ABLATION: if >0, bypass substrate and send a random "
                        "Gaussian vector of this norm each turn. Tests whether "
                        "substrate's learned tangent matters or any perturbation.")
    p.add_argument("--out_json", default="/tmp/chained_libero_jepa.json")
    args = p.parse_args()

    client = GrootJepaClient(args.groot_port, args.substrate_ckpt,
                              lookahead_lambda=args.lookahead_lambda,
                              random_residual_norm=args.random_residual_norm,
                              random_seed=args.seed)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.get_num_tasks()
    rng = np.random.default_rng(args.seed)
    chains = [sorted(rng.choice(n_tasks, size=args.chain_length, replace=False).tolist())
               for _ in range(args.n_chains)]
    print(f"[jepa-eval] suite={args.task_suite} chains={len(chains)} "
          f"chain_len={args.chain_length}", flush=True)

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
    summary = {"chains": []}

    for chain_idx, chain in enumerate(chains):
        print(f"\n=== chain{chain_idx} {chain} ===", flush=True)
        first_env = get_env(chain[0])
        init_states = suite.get_task_init_states(chain[0])
        n_rollouts = min(args.rollouts_per_chain, len(init_states))
        for r in range(n_rollouts):
            t0 = time.time()
            n_complete = 0
            client.reset()
            first_env.reset()
            first_env.set_init_state(init_states[r])
            obs = None
            for _ in range(5):
                obs, _, _, _ = first_env.step(np.zeros(7, dtype=np.float32))
            current_env = first_env
            sub_details = []
            for sub_idx, sub_id in enumerate(chain):
                sub_task = suite.get_task(sub_id)
                client.reset()
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
                succ, steps, obs, n_calls = step_subtask(
                    current_env, sub_task, obs, args, client)
                sub_details.append({"sub_id": sub_id, "succ": succ,
                                     "steps": steps, "n_calls": n_calls})
                if succ and args.retract_steps > 0 and sub_idx < len(chain) - 1:
                    open_lift = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, -1.0],
                                          dtype=np.float32)
                    for _ in range(args.retract_steps):
                        if isinstance(obs, dict):
                            gq = obs.get("robot0_gripper_qpos")
                            if gq is not None and float(np.asarray(gq).mean()) > args.retract_grip_thr:
                                break
                        obs, _, _, _ = current_env.step(open_lift)
                if succ: n_complete += 1
                else: break
            wall = time.time() - t0
            chain_completions.append(n_complete)
            total_succ += n_complete
            total_attempts += len(chain)
            tang_avg = client.total_tang_norm / max(client.n_chunks, 1)
            steps_str = ",".join(str(s["steps"]) for s in sub_details if "steps" in s)
            print(f"  r{r}: {n_complete}/{len(chain)} sub-tasks  wall={wall:.0f}s  "
                  f"steps=[{steps_str}]  tang_norm_mean={tang_avg:.2f}", flush=True)
            summary["chains"].append({
                "chain_idx": chain_idx, "rollout": r, "chain": chain,
                "n_complete": n_complete, "wall_s": wall, "details": sub_details,
                "tang_norm_mean": tang_avg,
            })

    for e in envs.values():
        try: e.close()
        except: pass

    summary["total_succ"] = total_succ
    summary["total_attempts"] = total_attempts
    summary["completion_rate"] = total_succ / max(total_attempts, 1)
    summary["mean_chain_completion"] = (
        float(np.mean(chain_completions)) if chain_completions else 0)
    print(f"\n{'='*60}", flush=True)
    print(f"OVERALL: {total_succ}/{total_attempts} sub-tasks = "
          f"{100*summary['completion_rate']:.0f}%", flush=True)
    print(f"Mean chain completion: {summary['mean_chain_completion']:.2f} "
          f"/ {args.chain_length}", flush=True)
    print(f"{'='*60}", flush=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
