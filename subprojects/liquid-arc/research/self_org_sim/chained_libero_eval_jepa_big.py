"""Chained-LIBERO eval with goal-conditioned JEPA-LGT.

Per sub-task: at entry, extract z_goal once (from LIBERO expert demo's final frame
for that task_id, run through GR00T backbone). Hold z_goal constant across all
turns of that sub-task. Reset z_goal + h_goal at sub-task boundary.

Per turn:
1. Get z_t from GR00T (op=get_action_with_state)
2. substrate.step(h_goal, z_t, z_goal, last_chunk) → ẑ_{t+1}, tangent
3. Send λ·tangent as zvl_override → action head decodes from bb + tangent
4. Execute chunk; chunk becomes last_chunk for next call
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

from liquid_goal_tracker_jepa_big import JEPA_LGT_Big as JEPA_LGT_GoalDelta as JEPA_LGT_Goal  # type: ignore
from rollout_libero_v11_client import (  # type: ignore
    build_state8, get_raw_imgs, build_groot_obs,
)


DATASET_ROOT = Path("/home/pokazge/datasets")


def extract_z_goal_via_groot(sock, suite_name, task_id):
    """Get z_goal for a task by sending its final expert frame through GR00T."""
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


class GrootJepaGoalClient:
    def __init__(self, port, substrate_ckpt, suite_name, device=None,
                 lookahead_lambda=1.0, noise_mix=0.0, noise_norm=7.0,
                 noise_seed=0):
        """
        noise_mix > 0: hybrid — final_tangent = (1-noise_mix)·substrate_tangent
                       + noise_mix · gaussian_noise(norm=noise_norm).
        """
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
        self.substrate = JEPA_LGT_Goal(
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
        self.z_goal_cache = {}  # task_id → z_goal (extracted once, cached)
        self.current_z_goal = None
        self.total_tang_norm = 0.0
        self.n_chunks = 0
        self.cos_goal_sum = 0.0
        self.cos_goal_n = 0
        print(f"[jepa-goal-eval] loaded {substrate_ckpt}, d={sa['d_substrate']}, "
              f"K={sa['K_belief']}, tangent_scale={sa['tangent_scale']}, "
              f"λ={lookahead_lambda}", flush=True)

    def set_subtask(self, task_id):
        """Reset substrate state + load z_goal for this sub-task."""
        self.h_goal = None
        self.last_chunk = None
        if task_id not in self.z_goal_cache:
            z_g = extract_z_goal_via_groot(self.sock, self.suite_name, task_id)
            if z_g is None:
                print(f"[jepa-goal-eval] WARN: no z_goal for task_id={task_id}; "
                      f"using zeros", flush=True)
                z_g = np.zeros(self.z_dim, dtype=np.float32)
            self.z_goal_cache[task_id] = z_g
        self.current_z_goal = self.z_goal_cache[task_id]

    def query(self, obs_dict):
        # 1. z_t from GR00T
        self.sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs_dict}))
        resp1 = pickle.loads(self.sock.recv())
        z_t = resp1["z_vl"]

        # 2. substrate predicts ẑ_{t+1}
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
            h_new, z_pred, tangent, _ = self.substrate.step(
                self.h_goal, z_t_torch, z_goal_torch, lc)
            self.h_goal = h_new
            substrate_tangent = (tangent[0].cpu().numpy() * self.lookahead_lambda
                                  ).astype(np.float32)
            # DIAGNOSTIC: is substrate's tangent pointing TOWARD z_goal?
            goal_dir = self.current_z_goal - z_t
            gd_norm = np.linalg.norm(goal_dir) + 1e-8
            tg_norm = np.linalg.norm(substrate_tangent) + 1e-8
            cos_to_goal = float(np.dot(substrate_tangent, goal_dir) / (gd_norm * tg_norm))
            self.cos_goal_sum += cos_to_goal
            self.cos_goal_n += 1
        # Optional hybrid with random noise (escape-stuck-states regularization)
        if self.noise_mix > 0:
            n = self.noise_rng.standard_normal(self.z_dim).astype(np.float32)
            n = n / (np.linalg.norm(n) + 1e-8) * self.noise_norm
            tangent_np = ((1.0 - self.noise_mix) * substrate_tangent
                          + self.noise_mix * n).astype(np.float32)
        else:
            tangent_np = substrate_tangent

        # 3. GR00T action with tangent as zvl override
        self.sock.send(pickle.dumps({"op": "get_action_with_zvl_override",
                                       "obs": obs_dict, "zvl_residual": tangent_np}))
        resp2 = pickle.loads(self.sock.recv())
        chunk = np.asarray(resp2["chunk"], dtype=np.float32)

        self.last_chunk = chunk.copy()
        self.total_tang_norm += float(np.linalg.norm(tangent_np))
        self.n_chunks += 1
        return chunk


def step_subtask(env, sub_task, obs, args, client, max_steps=None):
    sub_lang = sub_task.language
    chunk = None
    chunk_idx = 0
    sub_steps = 0
    n_calls = 0
    cap = max_steps if max_steps is not None else args.max_steps_per_task
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
    p.add_argument("--lookahead_lambda", type=float, default=1.0)
    p.add_argument("--noise_mix", type=float, default=0.0,
                   help="Hybrid mix factor [0,1]: 0 = pure substrate, 1 = pure noise.")
    p.add_argument("--noise_norm", type=float, default=7.0,
                   help="Norm of the random component when noise_mix>0.")
    p.add_argument("--shared_budget", type=int, default=0,
                   help="If >0: chain has total step budget; each sub-task uses "
                        "min(max_steps_per_task, remaining). Failures consume their budget.")
    p.add_argument("--out_json", default="/tmp/chained_libero_jepa_goal.json")
    args = p.parse_args()

    client = GrootJepaGoalClient(
        args.groot_port, args.substrate_ckpt, args.task_suite,
        lookahead_lambda=args.lookahead_lambda,
        noise_mix=args.noise_mix, noise_norm=args.noise_norm,
        noise_seed=args.seed)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.get_num_tasks()
    rng = np.random.default_rng(args.seed)
    chains = [sorted(rng.choice(n_tasks, size=args.chain_length, replace=False).tolist())
               for _ in range(args.n_chains)]
    print(f"[jepa-goal-eval] suite={args.task_suite} chains={len(chains)} "
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
            first_env.reset()
            first_env.set_init_state(init_states[r])
            obs = None
            for _ in range(5):
                obs, _, _, _ = first_env.step(np.zeros(7, dtype=np.float32))
            current_env = first_env
            sub_details = []
            budget_remaining = args.shared_budget if args.shared_budget > 0 else None
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
                    # Shared budget exhausted → remaining sub-tasks auto-fail
                    sub_details.append({"sub_id": sub_id, "succ": False,
                                         "steps": 0, "n_calls": 0,
                                         "skipped_no_budget": True})
                    break
                step_cap = (min(args.max_steps_per_task, budget_remaining)
                            if budget_remaining is not None
                            else args.max_steps_per_task)
                succ, steps, obs, n_calls = step_subtask(
                    current_env, sub_task, obs, args, client, max_steps=step_cap)
                if budget_remaining is not None:
                    budget_remaining -= steps
                sub_details.append({"sub_id": sub_id, "succ": succ,
                                     "steps": steps, "n_calls": n_calls,
                                     "budget_left": budget_remaining})
                if succ and args.retract_steps > 0 and sub_idx < len(chain) - 1:
                    open_lift = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, -1.0],
                                          dtype=np.float32)
                    retract_used = 0
                    for _ in range(args.retract_steps):
                        if budget_remaining is not None and budget_remaining <= 0:
                            break
                        if isinstance(obs, dict):
                            gq = obs.get("robot0_gripper_qpos")
                            if gq is not None and float(np.asarray(gq).mean()) > args.retract_grip_thr:
                                break
                        obs, _, _, _ = current_env.step(open_lift)
                        retract_used += 1
                        if budget_remaining is not None:
                            budget_remaining -= 1
                if succ: n_complete += 1
                else: break
            wall = time.time() - t0
            chain_completions.append(n_complete)
            total_succ += n_complete
            total_attempts += len(chain)
            tang_avg = client.total_tang_norm / max(client.n_chunks, 1)
            cos_goal_avg = client.cos_goal_sum / max(client.cos_goal_n, 1)
            steps_str = ",".join(str(s["steps"]) for s in sub_details if "steps" in s)
            print(f"  r{r}: {n_complete}/{len(chain)} sub-tasks  wall={wall:.0f}s  "
                  f"steps=[{steps_str}]  tang_norm={tang_avg:.2f}  "
                  f"cos(tang,goal)={cos_goal_avg:+.3f}", flush=True)
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
