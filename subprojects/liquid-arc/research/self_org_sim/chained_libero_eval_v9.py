"""Chained-LIBERO eval with variant #9 substrate (z_vl in/out via GR00T override).

Per turn:
1. Query GR00T (port 5555 libero_10) for z_vl
2. Substrate.step(h_goal, z_vl) → h_goal_new, residual
3. Query GR00T with zvl_override(residual) → action_chunk_modulated
4. Execute action_chunk_modulated

Plus: conditional retract heuristic between sub-tasks (the proven +9.2pp baseline).

Tests whether substrate + retract beats retract-alone on chained-LIBERO L=3.
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

from liquid_goal_tracker import LiquidGoalTracker  # type: ignore
from rollout_libero_v11_client import (  # type: ignore
    build_state8, get_raw_imgs, build_groot_obs,
)


class GrootV9Client:
    """GR00T client + substrate inside. Per chunk request:
       1) get z_vl via get_action_with_state
       2) substrate.step → residual
       3) get_action_with_zvl_override → modulated chunk
    Substrate h_goal persists across chunks within a sub-task; reset between subs.
    """
    def __init__(self, port, substrate_ckpt, device=None, residual_scale=1.0):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.residual_scale = residual_scale
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, 60000)
        self.sock.connect(f"tcp://localhost:{port}")

        ck = torch.load(substrate_ckpt, map_location=device, weights_only=False)
        sa = ck["args"]
        self.substrate = LiquidGoalTracker(
            z_vl_dim=ck["z_vl_dim"], d=sa["d_substrate"], K=sa["K_belief"],
            out_scale=sa["out_scale"],
        ).to(device)
        self.substrate.load_state_dict(ck["substrate_state_dict"])
        self.substrate.eval()
        self.h_goal = None
        self.total_res_norm = 0.0
        self.n_chunks = 0
        print(f"[v9] substrate loaded from {substrate_ckpt}, d={sa['d_substrate']}, "
              f"K={sa['K_belief']}, out_scale={sa['out_scale']}")

    def reset(self):
        """Reset substrate h_goal — called at sub-task boundary."""
        self.h_goal = None

    def query(self, obs_dict):
        """Get modulated action chunk via 2-step query + substrate."""
        # Step 1: z_vl from GR00T (baseline)
        self.sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs_dict}))
        resp1 = pickle.loads(self.sock.recv())
        z_vl = resp1["z_vl"]  # [2048] np.float32
        # Step 2: substrate produces residual
        with torch.no_grad():
            z_vl_t = torch.from_numpy(z_vl).to(self.device).unsqueeze(0)
            if self.h_goal is None:
                self.h_goal = self.substrate.init_state(1, self.device)
            h_new, residual, _ = self.substrate.step(self.h_goal, z_vl_t)
            self.h_goal = h_new
            residual_np = (residual[0].cpu().numpy() * self.residual_scale).astype(np.float32)
        # Step 3: GR00T with zvl override
        self.sock.send(pickle.dumps({"op": "get_action_with_zvl_override",
                                       "obs": obs_dict, "zvl_residual": residual_np}))
        resp2 = pickle.loads(self.sock.recv())
        chunk = np.asarray(resp2["chunk"], dtype=np.float32)
        self.total_res_norm += float(np.linalg.norm(residual_np))
        self.n_chunks += 1
        return chunk


def step_subtask(env, sub_task, obs, args, client):
    """Step one sub-task: GR00T+substrate every exec_horizon, env stepping."""
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
    """Same as rollout_chained_libero — only transfer robot joints (different scenes)."""
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
    p.add_argument("--n_chains", type=int, default=6)
    p.add_argument("--rollouts_per_chain", type=int, default=3)
    p.add_argument("--chain_length", type=int, default=3)
    p.add_argument("--task_suite", default="libero_10", type=str)
    p.add_argument("--max_steps_per_task", type=int, default=720)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--retract_steps", type=int, default=30)
    p.add_argument("--retract_grip_thr", type=float, default=0.030)
    p.add_argument("--residual_scale", type=float, default=1.0,
                   help="Multiply substrate residual at inference. <1.0 = gentler nudge")
    p.add_argument("--out_json", default="/tmp/chained_libero_v9.json")
    args = p.parse_args()

    client = GrootV9Client(args.groot_port, args.substrate_ckpt,
                              residual_scale=args.residual_scale)
    print(f"[v9] residual_scale={args.residual_scale}")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.get_num_tasks()
    rng = np.random.default_rng(args.seed)
    chains = [sorted(rng.choice(n_tasks, size=args.chain_length, replace=False).tolist())
               for _ in range(args.n_chains)]
    print(f"[v9] suite={args.task_suite} chains={len(chains)} chain_len={args.chain_length}")

    envs = {}
    def get_env(task_id):
        if task_id in envs:
            return envs[task_id]
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        e = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        envs[task_id] = e
        return e

    total_succ = 0
    total_attempts = 0
    chain_completions = []
    summary = {"chains": []}

    for chain_idx, chain in enumerate(chains):
        print(f"\n=== chain{chain_idx} {chain} ===")
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
                client.reset()  # h_goal reset per sub-task
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
                succ, steps, obs, n_calls = step_subtask(current_env, sub_task, obs, args, client)
                sub_details.append({"sub_id": sub_id, "succ": succ, "steps": steps,
                                     "n_calls": n_calls})
                # Conditional retract between sub-tasks (proven +9.2pp baseline)
                if succ and args.retract_steps > 0 and sub_idx < len(chain) - 1:
                    open_lift = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, -1.0],
                                          dtype=np.float32)
                    for _ in range(args.retract_steps):
                        if isinstance(obs, dict):
                            gq = obs.get("robot0_gripper_qpos")
                            if gq is not None and float(np.asarray(gq).mean()) > args.retract_grip_thr:
                                break
                        obs, _, _, _ = current_env.step(open_lift)
                if succ:
                    n_complete += 1
                else:
                    break
            wall = time.time() - t0
            chain_completions.append(n_complete)
            total_succ += n_complete
            total_attempts += len(chain)
            res_norm_avg = client.total_res_norm / max(client.n_chunks, 1)
            steps_str = ",".join(str(s["steps"]) for s in sub_details if "steps" in s)
            print(f"  r{r}: {n_complete}/{len(chain)} sub-tasks  wall={wall:.0f}s  "
                  f"steps=[{steps_str}]  res_norm_mean={res_norm_avg:.2f}")
            summary["chains"].append({
                "chain_idx": chain_idx, "rollout": r, "chain": chain,
                "n_complete": n_complete, "wall_s": wall, "details": sub_details,
                "res_norm_mean": res_norm_avg,
            })

    for e in envs.values():
        try: e.close()
        except: pass

    summary["total_succ"] = total_succ
    summary["total_attempts"] = total_attempts
    summary["completion_rate"] = total_succ / max(total_attempts, 1)
    summary["mean_chain_completion"] = float(np.mean(chain_completions)) if chain_completions else 0
    print(f"\n{'='*60}")
    print(f"OVERALL: {total_succ}/{total_attempts} sub-tasks = {100*summary['completion_rate']:.0f}%")
    print(f"Mean chain completion: {summary['mean_chain_completion']:.2f} / {args.chain_length}")
    print(f"{'='*60}")
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
