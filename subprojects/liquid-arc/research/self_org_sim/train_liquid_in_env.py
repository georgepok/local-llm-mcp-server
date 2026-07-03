"""Environment-validated training: Liquid acts in libero env, GR00T is consulted
on uncertainty (physics_cadence), the ENVIRONMENT decides what's correct.

Paradigm shift from offline distillation:
  - Liquid is the stateful driver, not a passive distillation target.
  - GR00T is a hypothesis source on uncertainty, not a teacher with correct answers.
  - The environment's success signal is the only ground truth.
  - SUCCESSFUL episodes pull Liquid toward what it actually did (which may include
    GR00T's suggestions if they were executed). FAILED episodes produce no gradient.

Per-episode flow:
  1. Reset env, reset adaptive optimizer state? NO — adaptive params accumulate
     across episodes (no per-episode reset; this is real online learning).
  2. Run libero rollout. At each chunk decision:
     - Liquid samples chunk from cond.
     - Physics-cadence may fire GR00T → get GR00T's chunk.
     - Mixing rule (--mix_strategy): "liquid_only" always uses Liquid's chunk;
       "groot_when_fired" uses GR00T's chunk on fire chunks (so GR00T's hypotheses
       actually get tested in the env). Default "groot_when_fired" — that's how
       suggestions get validated.
     - Record (img, wrist, state, cached_bank, cached_delta_bank, EXECUTED chunk).
  3. Check env.check_success() at end.
  4. If success: backprop flow-matching MSE on stored (cond, executed_chunk) pairs,
     step optimizer.
  5. If failure: no update. (Optional: use as negative-reward signal in v2.)

Ckpts every --save_every episodes. Eval every --eval_every episodes (no env
contamination — separate eval task suite).

Run inside libero venv with groot_server on port 5555:
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python train_liquid_in_env.py \\
    --base_ckpt /tmp/distill_multisuite_v6/step_016000.pt \\
    --train_suite libero_spatial \\
    --eval_suite libero_spatial \\
    --total_episodes 200 --rollouts_per_eval_task 2 \\
    --mix_strategy groot_when_fired \\
    --out_dir /tmp/distill_v6_envtrain
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
try:
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
except Exception:
    pass


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
    loaded = 0
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v); loaded += 1
    print(f"[load] {loaded}/{len(own)} tensors from {ckpt_path}")
    return model, ckpt


def setup_adaptive_params(model):
    """Freeze everything except drift + tau_raw + z_groot_proj — V8's adaptive set."""
    for p in model.parameters():
        p.requires_grad = False
    adaptive_params: list = []
    for p in model.encoder.drift.parameters():
        p.requires_grad = True
        adaptive_params.append(p)
    if hasattr(model.encoder, "tau_raw") and isinstance(model.encoder.tau_raw, torch.nn.Parameter):
        model.encoder.tau_raw.requires_grad = True
        adaptive_params.append(model.encoder.tau_raw)
    if model.encoder.z_groot_proj is not None:
        for p in model.encoder.z_groot_proj.parameters():
            p.requires_grad = True
            adaptive_params.append(p)
    n = sum(p.numel() for p in adaptive_params)
    print(f"[adapt] {n:,} trainable params (drift + tau_raw + z_groot_proj)")
    return adaptive_params


def run_episode_collect(env, init_state, model, task_lang, sock, args, device,
                         target_size, depth_indices_list, mix_strategy):
    """Run one libero episode. Records per-chunk (img, wrist, state, bank, delta,
    executed_chunk) for later success-filtered training. Returns (success, records).

    physics_cadence drives the cadence. When fired, GR00T's chunk is obtained and
    (per mix_strategy) optionally executed instead of Liquid's chunk.
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
    # Physics-cadence state
    physics_prev_cond = None
    physics_cum_drift = 0.0
    physics_drifts_at_fires: list[float] = []
    physics_chunks_since_fire = 0

    records: list[dict] = []
    n_fires = 0
    n_groot_executed = 0
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

            uninitialized = (cached_bank is None)
            # ---- Physics cadence decision ----
            if uninitialized:
                fired = True
            else:
                bank_t_dec = torch.from_numpy(cached_bank).to(device).float().unsqueeze(0)
                delta_t_dec = torch.from_numpy(cached_delta_bank).to(device).float().unsqueeze(0)
                with torch.no_grad():
                    cond_dec, _ = model.forward_encoder(
                        img_t, wri_t, st_t, task_id=None,
                        z_bank=bank_t_dec, delta_bank=delta_t_dec,
                    )
                    cond_dec_np = cond_dec.squeeze(0).cpu().numpy()
                physics_chunks_since_fire += 1
                if physics_prev_cond is not None:
                    delta = float(np.linalg.norm(cond_dec_np - physics_prev_cond))
                    physics_cum_drift += delta
                physics_prev_cond = cond_dec_np
                if len(physics_drifts_at_fires) > 0:
                    threshold = float(np.median(physics_drifts_at_fires))
                    fired = (physics_cum_drift > threshold
                              and physics_chunks_since_fire >= 2)
                else:
                    fired = (physics_chunks_since_fire >= 4)
                if physics_chunks_since_fire >= 32:
                    fired = True
                if fired:
                    physics_drifts_at_fires.append(physics_cum_drift)
                    physics_cum_drift = 0.0
                    physics_chunks_since_fire = 0
                    physics_prev_cond = None

            # ---- GR00T query if fired ----
            groot_chunk = None
            if fired:
                groot_obs = build_groot_obs(img_raw, wrist_raw, state8, task_lang)
                resp = query_groot_full(sock, groot_obs)
                full_traj = resp["traj_model_output"]
                cached_bank = np.stack([full_traj[di] for di in depth_indices_list]).astype(np.float32)
                K = cached_bank.shape[0]
                cached_delta_bank = np.eye(K, dtype=np.float32)
                groot_chunk = np.array(resp["chunk"], dtype=np.float32)
                n_fires += 1

            # ---- Sample Liquid's chunk ----
            with torch.no_grad():
                bank_t_use = torch.from_numpy(cached_bank).to(device).float().unsqueeze(0)
                delta_t_use = torch.from_numpy(cached_delta_bank).to(device).float().unsqueeze(0)
                liquid_chunk_t = model.sample(
                    img_t, wri_t, st_t, task_id=None,
                    n_steps=args.infer_steps,
                    z_bank=bank_t_use, delta_bank=delta_t_use,
                )
                liquid_chunk = liquid_chunk_t[0].cpu().numpy()

            # ---- Mixing rule: which chunk to actually execute ----
            if fired and groot_chunk is not None and mix_strategy == "groot_when_fired":
                executed_chunk = groot_chunk
                n_groot_executed += 1
            elif fired and groot_chunk is not None and mix_strategy == "alternating":
                # Alternate: execute GR00T half the time on fires, Liquid the other half.
                if n_fires % 2 == 0:
                    executed_chunk = groot_chunk
                    n_groot_executed += 1
                else:
                    executed_chunk = liquid_chunk
            else:
                executed_chunk = liquid_chunk

            # Record everything needed to reconstruct cond + train flow MSE
            records.append({
                "img": img_t.squeeze(0).detach().cpu().clone(),
                "wrist": wri_t.squeeze(0).detach().cpu().clone(),
                "state": st_t.squeeze(0).detach().cpu().clone(),
                "bank": torch.from_numpy(cached_bank.copy()),
                "delta_bank": torch.from_numpy(cached_delta_bank.copy()),
                "executed_chunk": torch.from_numpy(executed_chunk.copy()),
            })

            chunk = executed_chunk
            chunk_idx = 0

        action7 = chunk[chunk_idx].copy()
        g = action7[-1]
        action7[-1] = args.gripper_sign * np.sign(g) if abs(g) > 0.1 else 0.0
        obs, _, done, _ = env.step(action7.astype(np.float32))
        chunk_idx += 1
        if env.check_success():
            success = True
            break
        if done:
            break

    return success, records, n_fires, n_groot_executed, n_steps


def train_on_records(model, records, optimizer, action_params, args, device):
    """Run flow-matching MSE backprop on stored episode records.

    For each record, re-encode (img, wrist, state, bank, delta) with grad enabled
    through trainable action layers, compute v_target = chunk − noise, predict
    velocity via flow head, MSE loss. Backprop, step.
    """
    if not records:
        return 0.0
    # Stack records into batch tensors
    imgs = torch.stack([r["img"] for r in records]).to(device)
    wrists = torch.stack([r["wrist"] for r in records]).to(device)
    states = torch.stack([r["state"] for r in records]).to(device)
    banks = torch.stack([r["bank"] for r in records]).to(device)
    deltas = torch.stack([r["delta_bank"] for r in records]).to(device)
    chunks = torch.stack([r["executed_chunk"] for r in records]).to(device)

    n = imgs.shape[0]
    mb_size = min(args.minibatch_size, n)
    losses_local: list[float] = []
    for _ in range(args.update_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, mb_size):
            idx = perm[start:start + mb_size]
            img_mb = imgs[idx]
            wri_mb = wrists[idx]
            st_mb = states[idx]
            bank_mb = banks[idx]
            delta_mb = deltas[idx]
            chunk_mb = chunks[idx]

            cond_grad, _ = model.forward_encoder(
                img_mb, wri_mb, st_mb, task_id=None,
                z_bank=bank_mb, delta_bank=delta_mb,
            )
            t_rand = torch.rand(cond_grad.shape[0], device=device)
            noise = torch.randn_like(chunk_mb)
            noisy = ((1.0 - t_rand.view(-1, 1, 1)) * noise
                     + t_rand.view(-1, 1, 1) * chunk_mb)
            v_target = chunk_mb - noise
            v_pred = model.velocity(noisy, t_rand, cond_grad, task_id=None)
            loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(action_params, args.max_grad_norm)
            optimizer.step()
            losses_local.append(loss.item())
    return float(np.mean(losses_local))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", required=True, type=str)
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--train_suite", default="libero_spatial", type=str,
                   help="LIBERO suite to train on. Liquid will collect successes here.")
    p.add_argument("--eval_suite", default="", type=str,
                   help="(Optional) Suite to eval periodically. Defaults to train_suite.")
    p.add_argument("--total_episodes", type=int, default=200)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--rollouts_per_eval_task", type=int, default=2)
    p.add_argument("--mix_strategy", choices=["liquid_only", "groot_when_fired", "alternating"],
                   default="groot_when_fired",
                   help="On a fire, which chunk to execute. groot_when_fired: test "
                        "GR00T's hypothesis in the env; if it works, Liquid learns from it.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--update_epochs", type=int, default=2,
                   help="Per-success episode: how many gradient passes over the records.")
    p.add_argument("--minibatch_size", type=int, default=16)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--max_steps", type=int, default=720)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--infer_steps", type=int, default=10)
    p.add_argument("--gripper_sign", type=float, default=1.0)
    p.add_argument("--depth_indices", default="0,1,2,3", type=str)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_path", default="", type=str)
    args = p.parse_args()
    args.depth_indices_list = [int(x) for x in args.depth_indices.split(",")]
    if not args.eval_suite:
        args.eval_suite = args.train_suite

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model, ckpt = load_policy(Path(args.base_ckpt), device)
    sa = ckpt["args"]
    target_size = sa["img_size"]

    adaptive_params = setup_adaptive_params(model)
    optimizer = torch.optim.SGD(adaptive_params, lr=args.lr)

    # ZMQ to GR00T
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    sock.setsockopt(zmq.RCVTIMEO, 60000)
    print(f"[zmq] connected to groot_server on port {args.port}")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.train_suite]()
    n_tasks = suite.get_num_tasks()
    tasks_info = []
    for sim_id in range(n_tasks):
        task = suite.get_task(sim_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = suite.get_task_init_states(sim_id)
        tasks_info.append({
            "sim_id": sim_id, "bddl": bddl, "init_states": init_states,
            "lang": task.language,
        })

    history: list[dict] = []
    success_window: list[int] = []
    fires_window: list[int] = []

    print(f"[train] {args.total_episodes} episodes  suite={args.train_suite}  "
          f"mix_strategy={args.mix_strategy}  lr={args.lr}")
    t_start = time.time()
    env = None
    cur_sim_id = -1
    n_success_total = 0

    for ep in range(args.total_episodes):
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

        success, records, n_fires, n_groot_exec, n_steps = run_episode_collect(
            env, init_state, model, task_lang, sock, args, device,
            target_size, args.depth_indices_list, args.mix_strategy,
        )

        # Success-filtered update
        train_loss = 0.0
        if success:
            n_success_total += 1
            train_loss = train_on_records(model, records, optimizer,
                                           adaptive_params, args, device)

        success_window.append(int(success))
        fires_window.append(n_fires)
        if len(success_window) > 50:
            success_window.pop(0); fires_window.pop(0)

        if (ep + 1) % 5 == 0:
            sw = sum(success_window) / max(len(success_window), 1)
            fw = sum(fires_window) / max(len(fires_window), 1)
            elapsed = time.time() - t_start
            print(f"  ep {ep+1:4d}/{args.total_episodes}  "
                  f"sim{cur_sim_id} init{init_idx}  "
                  f"{'SUCC' if success else 'fail'}  fires={n_fires:2d}  "
                  f"groot_exec={n_groot_exec:2d}  "
                  f"win50_succ={sw*100:.0f}%  win50_fires={fw:.1f}  "
                  f"loss={train_loss:.4f}  total_succ={n_success_total}  "
                  f"({elapsed/60:.1f}min)")
            history.append({
                "ep": ep + 1, "sim_id": cur_sim_id, "init_idx": init_idx,
                "success": bool(success), "n_fires": n_fires,
                "n_groot_executed": n_groot_exec, "n_steps": n_steps,
                "train_loss": train_loss,
                "win50_succ": sw, "win50_fires": fw,
                "total_successes": n_success_total,
                "elapsed_min": elapsed / 60,
            })

        # Periodic checkpoint
        if (ep + 1) % args.save_every == 0:
            out = out_dir / f"step_{ep+1:06d}.pt"
            torch.save({
                "step": ckpt.get("step"),
                "policy": model.state_dict(),
                "args": {**sa, "envtrain_episode": ep + 1,
                         "envtrain_total_successes": n_success_total},
                "envtrain_history": history,
            }, out)
            print(f"  [save] {out}")

    if env is not None:
        env.close()

    # Final save
    out = out_dir / "step_final.pt"
    torch.save({
        "step": ckpt.get("step"),
        "policy": model.state_dict(),
        "args": {**sa, "envtrain_episode": args.total_episodes,
                 "envtrain_total_successes": n_success_total},
        "envtrain_history": history,
    }, out)
    print(f"[save] {out}")
    print(f"[done] total successes: {n_success_total}/{args.total_episodes}")
    if args.log_path:
        Path(args.log_path).write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
