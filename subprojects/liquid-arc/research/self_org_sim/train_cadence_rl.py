"""Stage 2 Phase B: REINFORCE training of cadence_head.

External signal only: episode reward = task_success − λ × n_fires.
Architecture:
  - Stage 1 checkpoint loaded with cadence_head=True (fresh head, bias=-1)
  - All weights frozen EXCEPT cadence_head.{weight, bias}
  - Per-chunk: encoder forward (with grad), sample fire from sigmoid(cadence_logit),
    log_prob accumulated for backward
  - Per-episode: R = success − λ × n_fires, advantage = R − baseline (running mean)
  - Loss = -Σ log_prob × advantage, averaged over batch of B episodes

Curriculum on λ: phase 1 (λ=0) learns fire-when-helpful; phase 2 (λ=0.005) adds
sparsity preference; phase 3 (λ=0.02) tightens.

Run inside LIBERO sim venv (CPU torch) with groot_server active on port 5555:
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python train_cadence_rl.py \\
    --base_ckpt /tmp/distill_v_pressure_v1/step_008000.pt \\
    --out_ckpt /tmp/distill_stage2_phaseB/step_final.pt \\
    --total_episodes 500 --batch_size 8 --port 5555
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zmq
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot_flow import LiquidFlowPolicy
from rollout_libero_s1s2 import (
    quat2axisangle, build_state8, get_raw_imgs,
    preprocess_for_liquid, build_groot_obs, query_groot_full,
)

print = functools.partial(print, flush=True)
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
except Exception:
    pass


def load_with_cadence_head(ckpt_path: Path, device):
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
        cadence_head=True,
    ).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v); loaded += 1
    print(f"[load] {loaded}/{len(own)} tensors from {ckpt_path}")
    return model, ckpt


def run_episode_with_grad(env, init_state, model, task_lang, sock, args, device,
                          target_size, depth_indices_list):
    """Run one libero episode. Cadence decided by cadence_head (stochastic).
    Returns: (success, log_probs[], n_fires, n_chunks, n_steps).
    """
    env.reset()
    env.set_init_state(init_state)
    obs = None
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

    chunk = None
    chunk_idx = 0
    cached_bank = None
    cached_delta_bank = None
    log_probs = []  # tensors, with grad
    n_fires = 0
    n_chunks = 0
    success = False
    n_steps = 0

    for step in range(args.max_steps):
        n_steps = step + 1
        if chunk is None or chunk_idx >= args.exec_horizon or chunk_idx >= len(chunk):
            img_raw, wrist_raw = get_raw_imgs(obs)
            state8 = build_state8(obs)
            img_r, wrist_r = preprocess_for_liquid(img_raw, wrist_raw, target_size)
            img_t = torch.from_numpy(img_r).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            wri_t = torch.from_numpy(wrist_r).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            st_t = torch.from_numpy(state8).to(device).float().unsqueeze(0)
            bank_t = (torch.from_numpy(cached_bank).to(device).float().unsqueeze(0)
                      if cached_bank is not None else None)
            delta_t = (torch.from_numpy(cached_delta_bank).to(device).float().unsqueeze(0)
                       if cached_delta_bank is not None else None)

            uninitialized = (cached_bank is None)
            if uninitialized:
                # Mandatory init fire (no decision needed; no log_prob)
                fired = True
            else:
                # WITH grad: encoder forward + cadence_logit
                cond, info = model.forward_encoder(
                    img_t, wri_t, st_t, task_id=None,
                    z_bank=bank_t, delta_bank=delta_t,
                )
                fire_logit = info["cadence_logit"].squeeze()  # [1]
                fire_prob = torch.sigmoid(fire_logit)
                # Sample stochastically
                u = torch.rand((), device=device)
                fired = (u < fire_prob).item()
                # Log prob of the action taken
                log_p = torch.log(fire_prob + 1e-8) if fired else torch.log(1.0 - fire_prob + 1e-8)
                log_probs.append(log_p)

            if fired:
                groot_obs = build_groot_obs(img_raw, wrist_raw, state8, task_lang)
                resp = query_groot_full(sock, groot_obs)
                full_traj = resp["traj_model_output"]
                cached_bank = np.stack([full_traj[d] for d in depth_indices_list]).astype(np.float32)
                K = cached_bank.shape[0]
                cached_delta_bank = np.eye(K, dtype=np.float32)
                n_fires += 1

            # Sample action chunk (no grad — flow head is frozen)
            with torch.no_grad():
                bank_t_use = (torch.from_numpy(cached_bank).to(device).float().unsqueeze(0)
                              if cached_bank is not None else None)
                delta_t_use = (torch.from_numpy(cached_delta_bank).to(device).float().unsqueeze(0)
                               if cached_delta_bank is not None else None)
                chunk_torch = model.sample(
                    img_t, wri_t, st_t, task_id=None,
                    n_steps=args.infer_steps,
                    z_bank=bank_t_use, delta_bank=delta_t_use,
                )
                chunk = chunk_torch[0].cpu().numpy()
            chunk_idx = 0
            n_chunks += 1

        action7 = chunk[chunk_idx].copy()
        g = action7[-1]
        action7[-1] = args.gripper_sign * np.sign(g) if abs(g) > 0.1 else 0.0
        obs, _, done, _ = env.step(action7.astype(np.float32))
        chunk_idx += 1
        if env.check_success():
            success = True; break
        if done: break

    return success, log_probs, n_fires, n_chunks, n_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", required=True, type=str)
    p.add_argument("--out_ckpt", required=True, type=str)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--task_suite", default="libero_10", type=str)
    p.add_argument("--total_episodes", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Episodes per gradient step (smaller=faster updates, "
                        "larger=lower variance).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=720)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--infer_steps", type=int, default=10)
    p.add_argument("--gripper_sign", type=float, default=1.0)
    p.add_argument("--depth_indices", default="0,1,2,3", type=str)
    p.add_argument("--lambda_init", type=float, default=0.0,
                   help="Initial λ (call-cost weight). Curriculum increases it.")
    p.add_argument("--lambda_max", type=float, default=0.02)
    p.add_argument("--curriculum_warmup_eps", type=int, default=100,
                   help="Episodes at λ=lambda_init before linear ramp to lambda_max.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_path", default="", type=str)
    args = p.parse_args()
    args.depth_indices_list = [int(x) for x in args.depth_indices.split(",")]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model, ckpt = load_with_cadence_head(Path(args.base_ckpt), device)
    sa = ckpt["args"]
    target_size = sa["img_size"]

    # Freeze everything except cadence_head
    for p_ in model.parameters():
        p_.requires_grad = False
    cadence_params = []
    for p_ in model.encoder.cadence_head.parameters():
        p_.requires_grad = True
        cadence_params.append(p_)
    n_train = sum(p_.numel() for p_ in cadence_params)
    n_total = sum(p_.numel() for p_ in model.parameters())
    print(f"[freeze] training {n_train:,} / {n_total:,} params ({100*n_train/n_total:.2f}%)")

    optimizer = torch.optim.Adam(cadence_params, lr=args.lr)

    # ZMQ to GR00T server
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    sock.setsockopt(zmq.RCVTIMEO, 60000)
    print(f"[zmq] connected to groot_server on port {args.port}")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.get_num_tasks()

    # Pre-collect bddl + init_states for cycling
    tasks_info = []
    for sim_id in range(n_tasks):
        task = suite.get_task(sim_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = suite.get_task_init_states(sim_id)
        tasks_info.append({
            "sim_id": sim_id, "bddl": bddl, "init_states": init_states,
            "lang": task.language,
        })

    # Running baseline for variance reduction (running mean of returns)
    baseline = 0.0
    baseline_alpha = 0.05  # EMA factor

    # Training log
    history = []
    success_window = []  # last 50 episode successes
    fires_window = []    # last 50 fires
    losses_window = []   # losses per batch
    batch_log_probs = []
    batch_returns = []
    batch_count = 0

    print(f"[train] {args.total_episodes} episodes, batch={args.batch_size}, "
          f"λ={args.lambda_init}→{args.lambda_max} ramp at ep {args.curriculum_warmup_eps}")
    t_start = time.time()
    env = None
    cur_sim_id = -1
    for ep in range(args.total_episodes):
        # Curriculum: λ ramps from init to max after warmup
        if ep < args.curriculum_warmup_eps:
            lam = args.lambda_init
        else:
            t = (ep - args.curriculum_warmup_eps) / max(args.total_episodes - args.curriculum_warmup_eps, 1)
            lam = args.lambda_init + min(t, 1.0) * (args.lambda_max - args.lambda_init)

        # Cycle through tasks; one init_state per episode (rotate within task)
        ti = ep % n_tasks
        if ti != cur_sim_id:
            if env is not None:
                env.close()
            env = OffScreenRenderEnv(bddl_file_name=tasks_info[ti]["bddl"],
                                      camera_heights=256, camera_widths=256)
            cur_sim_id = ti
        init_idx = (ep // n_tasks) % len(tasks_info[ti]["init_states"])
        init_state = tasks_info[ti]["init_states"][init_idx]
        task_lang = tasks_info[ti]["lang"]

        success, log_probs, n_fires, n_chunks, n_steps = run_episode_with_grad(
            env, init_state, model, task_lang, sock, args, device,
            target_size, args.depth_indices_list,
        )

        # Compute return: R = success - λ * n_fires (per-episode)
        R = float(success) - lam * n_fires
        baseline = (1 - baseline_alpha) * baseline + baseline_alpha * R
        advantage = R - baseline

        # Stack log_probs for this episode (could be empty if all chunks were init)
        if len(log_probs) > 0:
            ep_log_p_sum = torch.stack(log_probs).sum()
            batch_log_probs.append(ep_log_p_sum)
            batch_returns.append(advantage)

        # Tracking windows
        success_window.append(int(success))
        fires_window.append(n_fires)
        if len(success_window) > 50:
            success_window.pop(0)
            fires_window.pop(0)

        # Gradient step every batch_size episodes
        if (ep + 1) % args.batch_size == 0 and len(batch_log_probs) > 0:
            optimizer.zero_grad()
            loss = -torch.stack([
                lp * adv for lp, adv in zip(batch_log_probs, batch_returns)
            ]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cadence_params, 1.0)
            optimizer.step()
            losses_window.append(loss.item())
            if len(losses_window) > 20:
                losses_window.pop(0)
            batch_log_probs = []
            batch_returns = []
            batch_count += 1

        # Periodic logging
        if (ep + 1) % 5 == 0:
            sw = sum(success_window) / len(success_window)
            fw = sum(fires_window) / len(fires_window)
            lw = sum(losses_window) / max(len(losses_window), 1)
            elapsed = time.time() - t_start
            print(f"  ep {ep+1:4d}/{args.total_episodes}  "
                  f"sim{cur_sim_id} init{init_idx}  "
                  f"{'SUCC' if success else 'fail'}  fires={n_fires:2d}  "
                  f"R={R:+.3f}  λ={lam:.4f}  baseline={baseline:+.3f}  "
                  f"win50_succ={sw*100:.0f}%  win50_fires={fw:.1f}  "
                  f"loss={lw:.3f}  ({elapsed/60:.1f}min)")
            history.append({
                "ep": ep + 1, "sim_id": cur_sim_id, "init_idx": init_idx,
                "success": bool(success), "n_fires": n_fires, "n_chunks": n_chunks,
                "n_steps": n_steps, "R": R, "lambda": lam, "baseline": baseline,
                "win50_succ": sw, "win50_fires": fw, "loss": lw,
                "elapsed_min": elapsed / 60,
            })

    if env is not None:
        env.close()

    # Save
    out = Path(args.out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": ckpt.get("step"),
        "policy": model.state_dict(),
        "args": {**sa, "cadence_head": True, "stage2_phase_b": True},
        "phaseB_history": history,
    }, out)
    print(f"[save] {out}")
    if args.log_path:
        Path(args.log_path).write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
