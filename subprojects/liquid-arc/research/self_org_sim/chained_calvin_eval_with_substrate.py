"""Chained-CALVIN eval with LiquidGoalTracker substrate in the loop.

Per turn:
1. Query GR00T (port 5559) via get_action_with_state → z_vl, chunk_baseline
2. Substrate.step(h_goal, z_vl) → h_goal_new, residual
3. Query GR00T via get_action_with_zvl_override(zvl_residual=residual) → chunk_modified
4. Execute chunk_modified in env

Same harness as chained_calvin_eval.py but wraps GrootCalvinModel with substrate.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pickle
import torch
import zmq

CALVIN_ROOT = Path("/home/pokazge/calvin")
SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CALVIN_ROOT / "calvin_env"))
sys.path.insert(0, str(CALVIN_ROOT / "calvin_models"))
sys.path.insert(0, str(SELF_DIR))

from rollout_calvin_zeroshot import (  # type: ignore
    calvin_obs_to_groot, make_env, make_task_oracle, load_episode_state,
)
from chained_calvin_eval import load_eval_sequences  # type: ignore
from liquid_goal_tracker import LiquidGoalTracker  # type: ignore


class GrootWithSubstrate:
    def __init__(self, groot_port=5559, substrate_ckpt="", exec_horizon=8, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, 60000)
        self.sock.connect(f"tcp://localhost:{groot_port}")
        self.exec_horizon = exec_horizon
        self.device = device

        if substrate_ckpt and Path(substrate_ckpt).exists():
            ck = torch.load(substrate_ckpt, map_location=device, weights_only=False)
            sa = ck["args"]
            self.substrate = LiquidGoalTracker(
                z_vl_dim=ck["z_vl_dim"], d=sa["d_substrate"], K=sa["K_belief"],
                out_scale=sa["out_scale"],
            ).to(device)
            self.substrate.load_state_dict(ck["substrate_state_dict"])
            self.substrate.eval()
            print(f"[gwsub] loaded substrate from {substrate_ckpt}")
        else:
            # Untrained — residual will be ~0 due to init
            self.substrate = LiquidGoalTracker(z_vl_dim=2048).to(device)
            self.substrate.eval()
            print(f"[gwsub] using UNTRAINED substrate (residual ≈ 0 by init)")

        self.reset()

    def reset(self):
        self.h_goal = None
        self.cached_chunk = None
        self.chunk_idx = 0
        self.last_gripper = 1
        self.residual_norm_total = 0.0
        self.n_chunks = 0

    def set_subtask(self, name):
        pass  # substrate doesn't take subtask name

    def step(self, obs, lang):
        if self.cached_chunk is None or self.chunk_idx >= self.exec_horizon:
            groot_obs, _ = calvin_obs_to_groot(obs, lang)
            # Step 1: get z_vl from GR00T (baseline call)
            self.sock.send(pickle.dumps({"op": "get_action_with_state", "obs": groot_obs}))
            resp = pickle.loads(self.sock.recv())
            z_vl_np = resp["z_vl"]  # [2048]
            # Step 2: substrate produces residual
            with torch.no_grad():
                z_vl_t = torch.from_numpy(z_vl_np).to(self.device).unsqueeze(0)
                if self.h_goal is None:
                    self.h_goal = self.substrate.init_state(1, self.device)
                h_new, residual, _ = self.substrate.step(self.h_goal, z_vl_t)
                self.h_goal = h_new
                residual_np = residual[0].cpu().numpy().astype(np.float32)
            # Step 3: query GR00T with residual override
            self.sock.send(pickle.dumps({
                "op": "get_action_with_zvl_override", "obs": groot_obs,
                "zvl_residual": residual_np,
            }))
            resp2 = pickle.loads(self.sock.recv())
            self.cached_chunk = np.asarray(resp2["chunk"], dtype=np.float32)
            self.chunk_idx = 0
            self.residual_norm_total += float(np.linalg.norm(residual_np))
            self.n_chunks += 1

        action7 = self.cached_chunk[self.chunk_idx].copy()
        self.chunk_idx += 1
        g = action7[-1]
        if abs(g) > 0.1:
            self.last_gripper = int(np.sign(g))
        action7[-1] = self.last_gripper
        return action7.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groot_port", type=int, default=5559)
    p.add_argument("--substrate_ckpt", default="", type=str)
    p.add_argument("--n_sequences", type=int, default=20)
    p.add_argument("--max_steps_per_subtask", type=int, default=120)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--out_json", default="/tmp/chained_calvin_substrate.json", type=str)
    args = p.parse_args()

    print(f"[calvin] env init")
    env = make_env()
    task_oracle = make_task_oracle()

    model = GrootWithSubstrate(
        groot_port=args.groot_port, substrate_ckpt=args.substrate_ckpt,
        exec_horizon=args.exec_horizon,
    )
    sequences = load_eval_sequences(n_sequences=args.n_sequences)
    print(f"[calvin] {len(sequences)} eval sequences")

    success_counts = [0, 0, 0, 0, 0]
    total_completed = 0
    summary = {"sequences": []}
    for i, seq in enumerate(sequences):
        # Reset env to seq initial
        initial = load_episode_state(seq["initial_state_frame"])
        try:
            env.reset(robot_obs=initial["robot_obs"], scene_obs=initial["scene_obs"])
        except TypeError:
            env.reset()
        settle = np.zeros(7, dtype=np.float32); settle[-1] = 1.0
        obs, _, _, _ = env.step(settle)

        t0 = time.time()
        counter = 0
        sub_details = []
        model.reset()
        for sub in seq["subtasks"]:
            model.set_subtask(sub["task_name"])
            start_info = env.get_info()
            succeeded = False
            for step_i in range(args.max_steps_per_subtask):
                action = model.step(obs, sub["lang"])
                obs, _, _, current_info = env.step(action)
                done_tasks = task_oracle.get_task_info_for_set(
                    start_info, current_info, {sub["task_name"]},
                )
                if len(done_tasks) > 0:
                    succeeded = True
                    break
            sub_details.append({"task": sub["task_name"], "succ": succeeded,
                                 "steps": step_i + 1})
            if succeeded:
                counter += 1
            else:
                break
        for j in range(counter):
            success_counts[j] += 1
        total_completed += counter
        wall = time.time() - t0
        avg_resnorm = model.residual_norm_total / max(model.n_chunks, 1)
        print(f"  seq{i}: {counter}/5  wall={wall:.0f}s  "
              f"running: {' '.join(f'{c}' for c in success_counts)}  "
              f"res_norm_mean={avg_resnorm:.3f}")
        summary["sequences"].append({"seq_idx": i, "chain_length_succ": counter,
                                       "wall_s": wall, "details": sub_details,
                                       "residual_norm_mean": avg_resnorm})

    print(f"\n=== CALVIN substrate eval (N={len(sequences)}) ===")
    for k, c in enumerate(success_counts):
        print(f"  task {k+1}: {c}/{len(sequences)} = {100*c/len(sequences):.1f}%")
    print(f"  avg chain length: {total_completed/len(sequences):.2f} / 5")

    summary["success_counts"] = success_counts
    summary["avg_chain_length"] = total_completed / max(len(sequences), 1)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[calvin] saved → {args.out_json}")


if __name__ == "__main__":
    main()
