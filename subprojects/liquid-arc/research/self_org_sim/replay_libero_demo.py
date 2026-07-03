"""Replay a libero-r demonstration through the LIBERO sim env to test format alignment.

For each pairing of (libero-r task index, sim task index), pick one episode of
libero-r data, set sim init_state=0, and execute the demonstrated actions
verbatim. If success_rate is high, sim+actions are compatible (and the student
is just imprecise). If success_rate is low, there's a fundamental format
mismatch (fps, coord frame, gripper convention).

Run inside the LIBERO sim venv:
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python replay_libero_demo.py \\
    --raw_data_root /home/pokazge/datasets/libero-10-r-raw/libero-10-r \\
    --task_suite libero_10 --max_steps 400
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

print = functools.partial(print, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_data_root", required=True, type=str)
    p.add_argument("--task_suite", default="libero_10", type=str)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--n_episodes_per_task", type=int, default=1)
    p.add_argument("--n_init_states", type=int, default=3,
                   help="Number of init_states to try per (demo, sim_task) pairing")
    args = p.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    raw_root = Path(args.raw_data_root)

    # Load libero-r tasks list (training-time naming)
    libero_r_tasks = {}
    with open(raw_root / "meta" / "tasks.jsonl") as f:
        for line in f:
            d = json.loads(line)
            libero_r_tasks[d["task_index"]] = d["task"].strip()

    # Build sim task name → sim task index map
    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[args.task_suite]()
    sim_tasks = {i: suite.get_task(i).language.strip() for i in range(suite.get_num_tasks())}

    # Match by language string equality
    r_to_sim = {}
    for r_id, r_lang in libero_r_tasks.items():
        for sim_id, sim_lang in sim_tasks.items():
            if r_lang == sim_lang:
                r_to_sim[r_id] = sim_id
                break
    print("libero-r → sim_task mapping:")
    for k, v in sorted(r_to_sim.items()):
        print(f"  r{k} -> sim{v}: {libero_r_tasks[k][:60]}")

    # Episode index by task in libero-r data
    eps_by_task = {}
    eps_root = raw_root / "data" / "chunk-000"
    for ep_path in sorted(eps_root.glob("episode_*.parquet")):
        df_one = pd.read_parquet(ep_path, columns=["task_index"])
        ti = int(df_one["task_index"].iloc[0])
        eps_by_task.setdefault(ti, []).append(ep_path)
    print(f"\nEpisodes per task (libero-r): {{ti: count for ti, ls in eps_by_task.items()}}")

    # Replay: for each task in r_to_sim, take 1 episode, replay actions in sim
    print("\n" + "=" * 80)
    print("REPLAY RESULTS")
    print("=" * 80)
    overall_succ = 0
    overall_total = 0
    for r_id in sorted(r_to_sim):
        sim_id = r_to_sim[r_id]
        if r_id not in eps_by_task:
            print(f"r{r_id} -> sim{sim_id}: NO EPISODES IN libero-r")
            continue
        ep_path = eps_by_task[r_id][0]
        ep_df = pd.read_parquet(ep_path)
        actions = np.stack(ep_df["actions"].values).astype(np.float32)  # [T, 7]
        n_demo_steps = min(len(actions), args.max_steps)
        print(f"\n=== r{r_id} -> sim{sim_id}: {libero_r_tasks[r_id][:70]} ===")
        print(f"  demo episode={ep_path.name}, length={len(actions)}, replaying {n_demo_steps} steps")

        task = suite.get_task(sim_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = suite.get_task_init_states(sim_id)

        for s_id in range(min(args.n_init_states, len(init_states))):
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
            env.reset()
            env.set_init_state(init_states[s_id])
            success = False
            n_steps = 0
            for t in range(n_demo_steps):
                a = actions[t].astype(np.float32)
                # Try the actions as-is from libero-r
                _, _, done, _ = env.step(a)
                n_steps = t + 1
                if env.check_success():
                    success = True
                    break
                if done:
                    break
            print(f"  init_state {s_id}: {'SUCCESS' if success else 'fail'} after {n_steps} steps")
            overall_succ += int(success)
            overall_total += 1
            env.close()

    print("\n" + "=" * 80)
    print(f"OVERALL: {overall_succ}/{overall_total} = {overall_succ/max(overall_total,1):.0%}")
    print("=" * 80)
    if overall_succ == 0:
        print("⚠ All demos failed — fundamental action-format mismatch (fps, coord frame, etc.)")
    elif overall_succ < overall_total / 2:
        print("⚠ Demos partially work — possible format issue + init_state sensitivity")
    else:
        print("✓ Demos work in sim — format is compatible; student must be the bottleneck")


if __name__ == "__main__":
    main()
