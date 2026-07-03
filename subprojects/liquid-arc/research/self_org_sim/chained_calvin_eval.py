"""Chained-CALVIN eval: GR00T-CALVIN (finetuned) on CALVIN's long-horizon protocol.

CALVIN's eval: sample N initial-state + 5-task sequences. For each sequence:
  for subtask in sequence:
    if rollout(env, model, subtask) succeeds → counter+=1, continue
    else: break (chain ends)
Score = consecutive successful subtasks.

This harness wraps GR00T (via groot_server) as CALVIN's expected `model.step(obs, lang)`
interface. Substrate overlays (variants 1-6 from prior session) can wrap the model
to test "does substrate guide GR00T thru CALVIN long-horizon".
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pickle
import zmq

CALVIN_ROOT = Path("/home/pokazge/calvin")
SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CALVIN_ROOT / "calvin_env"))
sys.path.insert(0, str(CALVIN_ROOT / "calvin_models"))
sys.path.insert(0, str(SELF_DIR))

import hydra
from omegaconf import OmegaConf
try:
    import cv2
except ImportError:
    cv2 = None

# Reuse CALVIN→GR00T adapter from zero-shot probe
from rollout_calvin_zeroshot import (  # type: ignore
    calvin_obs_to_groot, make_env, make_task_oracle, load_episode_state,
)


class GrootCalvinModel:
    """CALVIN-protocol-compatible wrapper around groot_server (CALVIN-finetuned)."""
    def __init__(self, groot_port=5555, chunk_horizon=16, exec_horizon=8,
                 substrate_overlay=None):
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, 60000)
        self.sock.connect(f"tcp://localhost:{groot_port}")
        self.chunk_horizon = chunk_horizon
        self.exec_horizon = exec_horizon
        self.substrate_overlay = substrate_overlay
        self.reset()

    def reset(self):
        self.cached_chunk = None
        self.chunk_idx = 0
        self.last_gripper = 1  # CALVIN requires ±1
        if self.substrate_overlay is not None:
            self.substrate_overlay.reset()

    def set_subtask(self, subtask_name):
        """Hook for substrate overlay to know what sub-task is active."""
        if self.substrate_overlay is not None and hasattr(self.substrate_overlay, "set_subtask"):
            self.substrate_overlay.set_subtask(subtask_name)

    def step(self, obs, lang_annotation):
        """CALVIN model interface: (obs_dict, language_str) -> action_7."""
        # Re-query GR00T every exec_horizon steps
        if self.cached_chunk is None or self.chunk_idx >= self.exec_horizon:
            groot_obs, _ = calvin_obs_to_groot(obs, lang_annotation)
            self.sock.send(pickle.dumps({"op": "get_action_with_state", "obs": groot_obs}))
            resp = pickle.loads(self.sock.recv())
            chunk = np.asarray(resp["chunk"], dtype=np.float32)
            if self.substrate_overlay is not None:
                chunk = self.substrate_overlay.modulate(chunk, obs)
            self.cached_chunk = chunk
            self.chunk_idx = 0
        action7 = self.cached_chunk[self.chunk_idx].copy()
        self.chunk_idx += 1
        g = action7[-1]
        if abs(g) > 0.1:
            self.last_gripper = int(np.sign(g))
        action7[-1] = self.last_gripper
        return action7.astype(np.float32)


def load_eval_sequences(n_sequences=20, val_only=True):
    """Build N sequences of 5 CALVIN subtasks (CALVIN's standard long-horizon eval).
    Uses validation lang annotations + initial states.
    """
    split = "validation" if val_only else "training"
    ann_path = (CALVIN_ROOT / "dataset" / "calvin_debug_dataset" / split
                 / "lang_annotations" / "auto_lang_ann.npy")
    if not ann_path.exists():
        # Fall back to full dataset
        ann_path = (CALVIN_ROOT / "dataset" / "task_D_D" / split
                     / "lang_annotations" / "auto_lang_ann.npy")
    d = np.load(ann_path, allow_pickle=True).item()
    n = len(d["language"]["ann"])

    rng = np.random.default_rng(0)
    # Sample N sequences of 5 sub-tasks each from distinct annotations
    sequences = []
    for s in range(n_sequences):
        # Pick 5 distinct annotations
        idxs = rng.choice(n, size=min(5, n), replace=False)
        seq = []
        for i in idxs:
            seq.append({
                "task_name": d["language"]["task"][int(i)],
                "lang": d["language"]["ann"][int(i)],
                "start_frame": int(d["info"]["indx"][int(i)][0]),
            })
        # Use first subtask's start as initial state
        sequences.append({
            "initial_state_frame": seq[0]["start_frame"],
            "subtasks": seq,
        })
    return sequences


def evaluate_sequence(env, model, task_oracle, sequence, max_steps_per_subtask=120):
    """One chained-CALVIN sequence: run subtasks consecutively, count consecutive successes."""
    initial = load_episode_state(sequence["initial_state_frame"])
    try:
        env.reset(robot_obs=initial["robot_obs"], scene_obs=initial["scene_obs"])
    except TypeError:
        env.reset()
    settle = np.zeros(7, dtype=np.float32); settle[-1] = 1.0
    obs, _, _, _ = env.step(settle)

    success_counter = 0
    sub_details = []
    for sub in sequence["subtasks"]:
        model.reset()
        model.set_subtask(sub["task_name"])
        start_info = env.get_info()
        succeeded = False
        for step in range(max_steps_per_subtask):
            action = model.step(obs, sub["lang"])
            obs, _, _, current_info = env.step(action)
            done_tasks = task_oracle.get_task_info_for_set(
                start_info, current_info, {sub["task_name"]},
            )
            if len(done_tasks) > 0:
                succeeded = True
                break
        sub_details.append({"task": sub["task_name"], "succ": succeeded,
                             "steps": step + 1})
        if succeeded:
            success_counter += 1
        else:
            break  # chain ends on first failure
    return success_counter, sub_details


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groot_port", type=int, default=5555,
                   help="GR00T server port (use CALVIN-finetuned checkpoint server)")
    p.add_argument("--n_sequences", type=int, default=20,
                   help="Number of 5-task sequences (CALVIN standard = 1000)")
    p.add_argument("--max_steps_per_subtask", type=int, default=120)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--out_json", default="/tmp/chained_calvin.json", type=str)
    args = p.parse_args()

    print(f"[calvin] env init")
    env = make_env()
    task_oracle = make_task_oracle()
    print(f"[calvin] task oracle: {type(task_oracle).__name__}")

    model = GrootCalvinModel(groot_port=args.groot_port, exec_horizon=args.exec_horizon)
    print(f"[calvin] GR00T connected port {args.groot_port}")

    sequences = load_eval_sequences(n_sequences=args.n_sequences)
    print(f"[calvin] {len(sequences)} eval sequences, max {args.max_steps_per_subtask} steps/subtask")

    # CALVIN's standard metric: success per position 1-5 (cumulative)
    success_counts = [0, 0, 0, 0, 0]
    total_completed = 0
    summary = {"sequences": []}
    for i, seq in enumerate(sequences):
        t0 = time.time()
        counter, details = evaluate_sequence(env, model, task_oracle, seq,
                                                args.max_steps_per_subtask)
        for j in range(counter):
            success_counts[j] += 1
        total_completed += counter
        wall = time.time() - t0
        seq_summary = {"seq_idx": i, "chain_length_succ": counter,
                        "wall_s": wall, "details": details}
        summary["sequences"].append(seq_summary)
        print(f"  seq{i}: {counter}/5  wall={wall:.0f}s  "
              f"running: {' '.join(f'{c}' for c in success_counts)}")

    # Success rate at each chain position
    print(f"\n=== CALVIN long-horizon results (N={len(sequences)}) ===")
    for k, c in enumerate(success_counts):
        rate = c / max(len(sequences), 1)
        print(f"  task {k+1}: {c}/{len(sequences)} = {100*rate:.1f}%")
    avg_len = total_completed / max(len(sequences), 1)
    print(f"  avg chain length: {avg_len:.2f} / 5")

    summary["success_counts"] = success_counts
    summary["avg_chain_length"] = avg_len
    summary["n_sequences"] = len(sequences)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[calvin] saved → {args.out_json}")


if __name__ == "__main__":
    main()
