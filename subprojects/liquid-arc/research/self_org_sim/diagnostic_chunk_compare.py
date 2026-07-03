"""Diagnostic: compare Liquid's first-N action chunks to expert chunks at the
same starting state. Tells us if Liquid's failure mode is:

  (A) Liquid produces ~the same chunk regardless of task → no task discrimination
  (B) Liquid produces wildly different from expert → broken representation
  (C) Liquid produces close-to-expert → action is fine, deployment regression elsewhere

Run on Spark in libero venv with groot_server on port 5555:
  python diagnostic_chunk_compare.py \\
    --ckpt /tmp/distill_multisuite_v6/step_016000.pt \\
    --suite libero_spatial --n_chunks 5
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import zmq

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot_flow import LiquidFlowPolicy
from rollout_libero_s1s2 import (
    build_state8, get_raw_imgs,
    preprocess_for_liquid, build_groot_obs, query_groot_full,
)

print = functools.partial(print, flush=True)
torch.set_float32_matmul_precision("high")


def load_policy(ckpt_path: Path, device):
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
        z_groot_dim=sa.get("z_groot_dim", 0),
        gated_mixture=sa.get("gated_mixture", False),
        z_channel_dims=sa.get("z_channel_dims", None),
        query_bank=sa.get("use_query_bank", False),
        query_dim=sa.get("query_dim", 8),
        cadence_head=sa.get("cadence_head", False),
    ).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own = model.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
    print(f"[load] {ckpt_path}")
    return model, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--suite", default="libero_spatial", type=str)
    p.add_argument("--task_idx", type=int, default=0)
    p.add_argument("--n_chunks", type=int, default=10)
    p.add_argument("--port", type=int, default=5555)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model, ckpt = load_policy(Path(args.ckpt), device)
    sa = ckpt["args"]
    target_size = sa["img_size"]
    model.eval()

    # ZMQ to GR00T (we need z_vl from GR00T)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    sock.setsockopt(zmq.RCVTIMEO, 60000)
    print(f"[zmq] connected port {args.port}")

    # Load libero env + first init state
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_idx)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = suite.get_task_init_states(args.task_idx)
    task_lang = task.language
    print(f"[task] {args.suite} sim{args.task_idx}: {task_lang!r}")

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.reset()
    env.set_init_state(init_states[0])
    obs = None
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

    # Query GR00T once (uninitialized fire)
    img_raw, wrist_raw = get_raw_imgs(obs)
    state8 = build_state8(obs)
    groot_obs = build_groot_obs(img_raw, wrist_raw, state8, task_lang)
    resp = query_groot_full(sock, groot_obs)
    bank = np.stack([resp["traj_model_output"][i] for i in [0, 1, 2, 3]]).astype(np.float32)
    delta = np.eye(4, dtype=np.float32)
    groot_chunk = np.array(resp["chunk"], dtype=np.float32)

    # Sample Liquid's chunk
    img_r, wrist_r = preprocess_for_liquid(img_raw, wrist_raw, target_size)
    img_t = torch.from_numpy(img_r).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    wri_t = torch.from_numpy(wrist_r).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    st_t = torch.from_numpy(state8).to(device).float().unsqueeze(0)
    bank_t = torch.from_numpy(bank).to(device).float().unsqueeze(0)
    delta_t = torch.from_numpy(delta).to(device).float().unsqueeze(0)

    with torch.no_grad():
        # Sample 3 times to see noise variance
        liquid_chunks = []
        for _ in range(3):
            chunk_torch = model.sample(img_t, wri_t, st_t, task_id=None, n_steps=10,
                                        z_bank=bank_t, delta_bank=delta_t)
            liquid_chunks.append(chunk_torch[0].cpu().numpy())
        liquid_chunks = np.stack(liquid_chunks)  # [3, 16, 7]

    # Load expert chunks from libero hdf5 for the SAME init state if possible.
    # Just take first demo's first chunks.
    suite_dir = f"/home/pokazge/Isaac-GR00T/external_dependencies/LIBERO/libero/datasets/{args.suite}"
    hdf5_files = sorted(Path(suite_dir).glob("*.hdf5"))
    h_path = hdf5_files[args.task_idx]
    with h5py.File(h_path, "r") as f:
        demo0 = f["data/demo_0"]
        expert_actions = demo0["actions"][:].astype(np.float32)
        # Take first 16 actions (one chunk)
        expert_chunk = expert_actions[:16] if len(expert_actions) >= 16 else expert_actions

    # Print all chunks for visual comparison
    np.set_printoptions(precision=3, suppress=True, linewidth=140)
    print(f"\n=== Expert first chunk (libero hdf5 demo_0, first 16 actions) ===")
    print(expert_chunk)
    print(f"\n=== GR00T's first chunk (queried at env step 0) ===")
    print(groot_chunk)
    print(f"\n=== Liquid's first chunk (sample 1 of 3) ===")
    print(liquid_chunks[0])
    print(f"\n=== Liquid's first chunk (sample 2 of 3) ===")
    print(liquid_chunks[1])

    # Statistics
    print("\n=== Statistics ===")
    expert_mse_groot = float(np.mean((groot_chunk - expert_chunk) ** 2))
    expert_mse_liquid = float(np.mean((liquid_chunks.mean(0) - expert_chunk) ** 2))
    expert_mse_liquid_per_sample = [float(np.mean((c - expert_chunk) ** 2)) for c in liquid_chunks]
    groot_mse_liquid = float(np.mean((liquid_chunks.mean(0) - groot_chunk) ** 2))

    liquid_var = float(liquid_chunks.std(axis=0).mean())  # variance across 3 samples
    print(f"expert chunk std (action dims): {expert_chunk.std(axis=0).round(3)}")
    print(f"groot chunk std (action dims):  {groot_chunk.std(axis=0).round(3)}")
    print(f"liquid chunk std across samples: {liquid_var:.4f}")
    print(f"MSE(expert, groot): {expert_mse_groot:.4f}")
    print(f"MSE(expert, liquid mean): {expert_mse_liquid:.4f}")
    print(f"MSE(expert, liquid per sample): {[f'{v:.4f}' for v in expert_mse_liquid_per_sample]}")
    print(f"MSE(groot, liquid mean): {groot_mse_liquid:.4f}")

    # Now do the same thing for libero_10 sim3 for comparison (a task we DO solve well)
    print(f"\n\n=== COMPARISON: libero_10 sim3 (we solve this at 80%) ===")
    suite10 = benchmark.get_benchmark_dict()['libero_10']()
    task10 = suite10.get_task(3)
    bddl10 = os.path.join(get_libero_path("bddl_files"), task10.problem_folder, task10.bddl_file)
    init10 = suite10.get_task_init_states(3)
    task_lang10 = task10.language
    print(f"[task10] {task_lang10!r}")
    env.close()
    env = OffScreenRenderEnv(bddl_file_name=bddl10, camera_heights=256, camera_widths=256)
    env.reset()
    env.set_init_state(init10[0])
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
    img_raw, wrist_raw = get_raw_imgs(obs)
    state8 = build_state8(obs)
    groot_obs10 = build_groot_obs(img_raw, wrist_raw, state8, task_lang10)
    resp10 = query_groot_full(sock, groot_obs10)
    bank10 = np.stack([resp10["traj_model_output"][i] for i in [0, 1, 2, 3]]).astype(np.float32)
    img_r, wrist_r = preprocess_for_liquid(img_raw, wrist_raw, target_size)
    img_t = torch.from_numpy(img_r).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    wri_t = torch.from_numpy(wrist_r).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    st_t = torch.from_numpy(state8).to(device).float().unsqueeze(0)
    bank_t10 = torch.from_numpy(bank10).to(device).float().unsqueeze(0)
    delta_t = torch.from_numpy(delta).to(device).float().unsqueeze(0)

    with torch.no_grad():
        liquid10 = []
        for _ in range(3):
            chunk_torch = model.sample(img_t, wri_t, st_t, task_id=None, n_steps=10,
                                        z_bank=bank_t10, delta_bank=delta_t)
            liquid10.append(chunk_torch[0].cpu().numpy())
        liquid10 = np.stack(liquid10)

    print(f"\nLiquid's first chunk on libero_10 sim3 (sample 1):")
    print(liquid10[0])
    print(f"GR00T's first chunk on libero_10 sim3:")
    print(np.array(resp10["chunk"], dtype=np.float32))

    # Key comparison: does Liquid produce DIFFERENT chunks for the two different tasks?
    print(f"\n=== KEY CHECK: Is Liquid producing different chunks for different tasks? ===")
    diff = float(np.mean((liquid_chunks[0] - liquid10[0]) ** 2))
    print(f"MSE(libero_spatial sim0 chunk, libero_10 sim3 chunk): {diff:.4f}")
    print(f"  (if ~0: Liquid outputs same chunk regardless of task → BAD)")
    print(f"  (if large: Liquid IS task-aware in its outputs)")

    env.close()


if __name__ == "__main__":
    main()
