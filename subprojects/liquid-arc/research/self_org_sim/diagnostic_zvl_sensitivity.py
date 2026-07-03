"""Diagnostic: how sensitive is Liquid's chunk to z_vl, holding image+state fixed?
If we swap z_vl (e.g., from libero_spatial sim0 → libero_10 sim3), does the chunk
change meaningfully? If not, z_vl conditioning is broken — model ignores the signal.

Run on Spark in libero venv with groot_server on port 5555:
  python diagnostic_zvl_sensitivity.py --ckpt /tmp/distill_multisuite_v6/step_016000.pt
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from pathlib import Path

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
    return model, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--n_samples", type=int, default=5)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_policy(Path(args.ckpt), device)
    sa = ckpt["args"]; target_size = sa["img_size"]
    model.eval()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    sock.setsockopt(zmq.RCVTIMEO, 60000)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    # Get z_vl/bank from libero_spatial sim0
    suite_s = benchmark.get_benchmark_dict()['libero_spatial']()
    task_s = suite_s.get_task(0)
    bddl_s = os.path.join(get_libero_path("bddl_files"), task_s.problem_folder, task_s.bddl_file)
    init_s = suite_s.get_task_init_states(0)
    env_s = OffScreenRenderEnv(bddl_file_name=bddl_s, camera_heights=256, camera_widths=256)
    env_s.reset()
    env_s.set_init_state(init_s[0])
    obs_s = None
    for _ in range(5):
        obs_s, _, _, _ = env_s.step(np.zeros(7, dtype=np.float32))
    img_raw_s, wrist_raw_s = get_raw_imgs(obs_s)
    state8_s = build_state8(obs_s)
    groot_obs_s = build_groot_obs(img_raw_s, wrist_raw_s, state8_s, task_s.language)
    resp_s = query_groot_full(sock, groot_obs_s)
    bank_s = np.stack([resp_s["traj_model_output"][i] for i in [0, 1, 2, 3]]).astype(np.float32)
    env_s.close()
    print(f"[spatial] task: {task_s.language!r}")
    print(f"[spatial] state8: {state8_s.round(3)}")
    print(f"[spatial] bank shape: {bank_s.shape}, norm: {np.linalg.norm(bank_s):.3f}")

    # Get z_vl/bank from libero_10 sim3 (a different task with different scene)
    suite_10 = benchmark.get_benchmark_dict()['libero_10']()
    task_10 = suite_10.get_task(3)
    bddl_10 = os.path.join(get_libero_path("bddl_files"), task_10.problem_folder, task_10.bddl_file)
    init_10 = suite_10.get_task_init_states(3)
    env_10 = OffScreenRenderEnv(bddl_file_name=bddl_10, camera_heights=256, camera_widths=256)
    env_10.reset()
    env_10.set_init_state(init_10[0])
    obs_10 = None
    for _ in range(5):
        obs_10, _, _, _ = env_10.step(np.zeros(7, dtype=np.float32))
    img_raw_10, wrist_raw_10 = get_raw_imgs(obs_10)
    state8_10 = build_state8(obs_10)
    groot_obs_10 = build_groot_obs(img_raw_10, wrist_raw_10, state8_10, task_10.language)
    resp_10 = query_groot_full(sock, groot_obs_10)
    bank_10 = np.stack([resp_10["traj_model_output"][i] for i in [0, 1, 2, 3]]).astype(np.float32)
    env_10.close()
    print(f"[libero10] task: {task_10.language!r}")
    print(f"[libero10] state8: {state8_10.round(3)}")
    print(f"[libero10] bank shape: {bank_10.shape}, norm: {np.linalg.norm(bank_10):.3f}")
    print(f"\n[delta] ||bank_spatial - bank_10|| = {np.linalg.norm(bank_s - bank_10):.3f}")

    # Now: with libero_spatial IMAGE+STATE (everything held fixed),
    # try Liquid with three different z_vl/bank conditions:
    #   A) bank_s (its own correct z_vl)
    #   B) bank_10 (z_vl from a different task)
    #   C) zero bank (no z_vl signal)
    img_r, wrist_r = preprocess_for_liquid(img_raw_s, wrist_raw_s, target_size)
    img_t = torch.from_numpy(img_r).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    wri_t = torch.from_numpy(wrist_r).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    st_t = torch.from_numpy(state8_s).to(device).float().unsqueeze(0)
    delta = np.eye(4, dtype=np.float32)
    delta_t = torch.from_numpy(delta).to(device).float().unsqueeze(0)

    bank_s_t = torch.from_numpy(bank_s).to(device).float().unsqueeze(0)
    bank_10_t = torch.from_numpy(bank_10).to(device).float().unsqueeze(0)
    bank_zero_t = torch.zeros_like(bank_s_t)

    chunks_A = []  # correct z_vl
    chunks_B = []  # mismatched z_vl
    chunks_C = []  # zero z_vl
    torch.manual_seed(42)
    with torch.no_grad():
        for i in range(args.n_samples):
            torch.manual_seed(42 + i)  # consistent noise across A/B/C
            cA = model.sample(img_t, wri_t, st_t, task_id=None, n_steps=10,
                               z_bank=bank_s_t, delta_bank=delta_t)[0].cpu().numpy()
            torch.manual_seed(42 + i)
            cB = model.sample(img_t, wri_t, st_t, task_id=None, n_steps=10,
                               z_bank=bank_10_t, delta_bank=delta_t)[0].cpu().numpy()
            torch.manual_seed(42 + i)
            cC = model.sample(img_t, wri_t, st_t, task_id=None, n_steps=10,
                               z_bank=bank_zero_t, delta_bank=delta_t)[0].cpu().numpy()
            chunks_A.append(cA); chunks_B.append(cB); chunks_C.append(cC)

    chunks_A = np.stack(chunks_A); chunks_B = np.stack(chunks_B); chunks_C = np.stack(chunks_C)

    np.set_printoptions(precision=3, suppress=True, linewidth=140)
    print(f"\n=== chunk first action under different z_vl (image+state held fixed) ===")
    print(f"A. correct z_vl:   {chunks_A.mean(0)[0]}")
    print(f"B. mismatched z_vl:{chunks_B.mean(0)[0]}")
    print(f"C. zero z_vl:      {chunks_C.mean(0)[0]}")

    print(f"\n=== Stats across {args.n_samples} samples ===")
    A_var = chunks_A.std(axis=0).mean()
    B_var = chunks_B.std(axis=0).mean()
    A_B_diff = float(np.mean((chunks_A.mean(0) - chunks_B.mean(0)) ** 2))
    A_C_diff = float(np.mean((chunks_A.mean(0) - chunks_C.mean(0)) ** 2))
    print(f"intra-condition variance (A): {A_var:.5f}")
    print(f"intra-condition variance (B): {B_var:.5f}")
    print(f"MSE(A.mean, B.mean) — correct z_vl vs mismatched z_vl: {A_B_diff:.5f}")
    print(f"MSE(A.mean, C.mean) — correct z_vl vs zero z_vl:       {A_C_diff:.5f}")
    print()
    print(f"=== INTERPRETATION ===")
    if A_B_diff < A_var * 2:
        print(f"  z_vl swap effect ({A_B_diff:.5f}) is < 2x intra-noise ({A_var:.5f})")
        print(f"  → Liquid is INSENSITIVE to z_vl. Conditioning is broken / weak.")
    else:
        print(f"  z_vl swap effect ({A_B_diff:.5f}) is significantly > intra-noise ({A_var:.5f})")
        print(f"  → Liquid IS sensitive to z_vl, just maybe not enough at output level.")


if __name__ == "__main__":
    main()
