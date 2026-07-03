"""Stage 2 Phase C: PPO + GAE training of cadence_head (+ optional action fast layers).

Phase B (vanilla REINFORCE) failed to converge: episode-level advantage with EMA
baseline can't separate "fired too much" from "task hard". Phase C addresses this:

  - Value head V(s): small MLP from cond → scalar, co-trained with cadence_head.
  - Per-chunk reward: r_t = -λ if cadence fired at chunk t (bookkeeping cost),
    plus terminal reward 1.0 at the last chunk if task succeeded.
  - GAE: A_t = δ_t + γλ_gae A_{t+1}, where δ_t = r_t + γ V_{t+1} − V_t.
  - PPO clipped surrogate: L = min(ratio*A, clip(ratio,1-ε,1+ε)*A).
  - Entropy bonus on Bernoulli(p_fire) for exploration.
  - K PPO epochs per rollout batch (re-evaluate stored cond, compute new logp/V).

Optional `--adapt_action_layers` enables the option-c HYBRID:
  - drift + tau_raw + z_groot_proj become trainable.
  - For every chunk where cadence fired (and we received GR00T's chunk + fresh bank),
    we store (img, wrist, state, bank, delta_bank, groot_chunk) and run a parallel
    flow-matching auxiliary loss in each update epoch:
      cond_grad = forward_encoder(img, wrist, state, bank, delta) ← grad through
                                                                    trainable action
                                                                    layers only
      v_pred = model.velocity(noisy, t, cond_grad)  ← flow head frozen
      flow_loss = MSE(v_pred, target_chunk - noise)
    Backprop accumulates gradient on drift/tau/z_groot_proj only (other model
    params are frozen and skip the optimizer.step). This is V8-style adaptation
    integrated into PPO training — addresses the Phase C plateau where cadence
    alone can't compensate when the action policy itself can't solve a task.

Run inside LIBERO sim venv (CPU torch) with groot_server active on port 5555:
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python train_cadence_ppo.py \\
    --base_ckpt /tmp/distill_v_pressure_v1/step_008000.pt \\
    --out_ckpt /tmp/distill_stage2_phaseC/step_final.pt \\
    --total_episodes 800 --rollout_batch 8 --ppo_epochs 4 \\
    --lambda_init 0.005 --lambda_max 0.03 --port 5555
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
import torch.nn as nn
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


CADENCE_LOGIT_BOUND = 3.0  # bounds p_fire ∈ [0.047, 0.953] via soft tanh clip


def soft_clip_logit(raw: torch.Tensor) -> torch.Tensor:
    """Bound cadence logit through tanh to prevent sigmoid saturation.

    raw → bound · tanh(raw / bound).  Differentiable everywhere; gradient
    monotonically shrinks as |raw| grows but never vanishes. Prevents the
    Phase C-hybrid v1 collapse mode where cadence_logit → −∞ and the head
    gets stuck at p≈0 with zero gradient.
    """
    return CADENCE_LOGIT_BOUND * torch.tanh(raw / CADENCE_LOGIT_BOUND)


class ValueHead(nn.Module):
    """Small MLP critic over post-encoder cond [d] → scalar V(s).

    Last-layer zero-init so V(s)=0 at start; learns from returns over training.
    """
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, d] or [d] → [B] or scalar
        return self.fc2(F.silu(self.fc1(h))).squeeze(-1)


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


def collect_episode(env, init_state, model, value_head, task_lang, sock, args, device,
                    target_size, depth_indices_list, lam_fire,
                    action_optimizer=None, action_params=None):
    """Run one libero episode. Two-timescale (option B):
      - cadence sample: with no_grad on encoder (PPO will re-evaluate with grad later)
      - within-episode action adaptation: when fired, take ONE flow-matching SGD step
        through trainable action layers (drift, tau_raw, z_groot_proj). Action layers
        evolve within the episode; main() resets them at the next episode start.
        This is V8's per-episode adaptation reorganized to live inside PPO data
        collection — slow plastic (cadence PPO) + fast plastic (action SGD), separated.

    Returns dict with:
      cond_t       [T_dec, d]   detached encoder outputs at each decision chunk
      fired_t      [T_dec]      1/0 actions
      logp_old     [T_dec]      detached log-prob of action taken at collection time
      value_old    [T_dec]      detached critic V(cond_t)
      reward_t     [T_dec]      per-chunk reward (-λ if fired) + terminal +1 at last
      success      bool
      n_fires      int
      n_steps      int
      flow_losses  list[float]  per-fire within-episode flow MSE values
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
    conds, fires, logps, values, rewards = [], [], [], [], []
    flow_losses: list = []
    n_fires = 0
    success = False
    n_steps = 0
    do_within_episode_flow = (action_optimizer is not None and action_params)

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
            if uninitialized:
                # Mandatory init fire — not a learned decision; not stored.
                fired = True
            else:
                bank_t = torch.from_numpy(cached_bank).to(device).float().unsqueeze(0)
                delta_t = torch.from_numpy(cached_delta_bank).to(device).float().unsqueeze(0)
                with torch.no_grad():
                    cond, info = model.forward_encoder(
                        img_t, wri_t, st_t, task_id=None,
                        z_bank=bank_t, delta_bank=delta_t,
                    )
                    fire_logit = soft_clip_logit(info["cadence_logit"].squeeze())
                    fire_prob = torch.sigmoid(fire_logit)
                    u = torch.rand((), device=device)
                    fired = bool((u < fire_prob).item())
                    logp = (torch.log(fire_prob + 1e-8) if fired
                            else torch.log(1.0 - fire_prob + 1e-8))
                    v = value_head(cond.squeeze(0))
                conds.append(cond.squeeze(0).detach().cpu())
                fires.append(int(fired))
                logps.append(logp.detach().cpu())
                values.append(v.detach().cpu())
                rewards.append(-lam_fire if fired else 0.0)

            if fired:
                groot_obs = build_groot_obs(img_raw, wrist_raw, state8, task_lang)
                resp = query_groot_full(sock, groot_obs)
                full_traj = resp["traj_model_output"]
                cached_bank = np.stack([full_traj[di] for di in depth_indices_list]).astype(np.float32)
                K = cached_bank.shape[0]
                cached_delta_bank = np.eye(K, dtype=np.float32)
                n_fires += 1

                if do_within_episode_flow:
                    assert action_optimizer is not None and action_params is not None
                    # Two-timescale: ONE flow-matching SGD step on action fast layers
                    # using GR00T's chunk as target. Re-encode with grad (encoder forward
                    # is mostly frozen; only drift/tau_raw/z_groot_proj receive gradient).
                    target_chunk = torch.from_numpy(np.array(resp["chunk"],
                                                              dtype=np.float32)
                                                     ).to(device).unsqueeze(0)
                    bank_t_g = torch.from_numpy(cached_bank).to(device).float().unsqueeze(0)
                    delta_t_g = torch.from_numpy(cached_delta_bank).to(device).float().unsqueeze(0)
                    cond_g, _ = model.forward_encoder(
                        img_t, wri_t, st_t, task_id=None,
                        z_bank=bank_t_g, delta_bank=delta_t_g,
                    )
                    t_rand = torch.rand(1, device=device)
                    noise = torch.randn_like(target_chunk)
                    noisy = ((1.0 - t_rand.view(-1, 1, 1)) * noise
                             + t_rand.view(-1, 1, 1) * target_chunk)
                    v_target = target_chunk - noise
                    v_pred = model.velocity(noisy, t_rand, cond_g, task_id=None)
                    flow_loss = F.mse_loss(v_pred, v_target) * args.flow_coef
                    action_optimizer.zero_grad()
                    flow_loss.backward()
                    torch.nn.utils.clip_grad_norm_(action_params, args.max_grad_norm)
                    action_optimizer.step()
                    flow_losses.append(flow_loss.item())

            with torch.no_grad():
                bank_t_use = torch.from_numpy(cached_bank).to(device).float().unsqueeze(0)
                delta_t_use = torch.from_numpy(cached_delta_bank).to(device).float().unsqueeze(0)
                chunk_torch = model.sample(
                    img_t, wri_t, st_t, task_id=None,
                    n_steps=args.infer_steps,
                    z_bank=bank_t_use, delta_bank=delta_t_use,
                )
                chunk = chunk_torch[0].cpu().numpy()
            chunk_idx = 0

        action7 = chunk[chunk_idx].copy()
        g = action7[-1]
        action7[-1] = args.gripper_sign * np.sign(g) if abs(g) > 0.1 else 0.0
        obs, _, done, _ = env.step(action7.astype(np.float32))
        chunk_idx += 1
        if env.check_success():
            success = True; break
        if done: break

    # Terminal reward attached to the last decision chunk (if any decisions were made).
    if len(rewards) > 0:
        rewards[-1] += 1.0 if success else 0.0

    return {
        "cond": torch.stack(conds) if conds else torch.zeros(0),
        "fired": torch.tensor(fires, dtype=torch.float32),
        "logp_old": torch.stack(logps) if logps else torch.zeros(0),
        "value_old": torch.stack(values) if values else torch.zeros(0),
        "reward": torch.tensor(rewards, dtype=torch.float32),
        "success": success,
        "n_fires": n_fires,
        "n_steps": n_steps,
        "flow_losses": flow_losses,
    }


def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                gamma: float, lam_gae: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-chunk GAE. rewards, values are [T]. Bootstrap V(T)=0 (episodic)."""
    T = rewards.shape[0]
    if T == 0:
        return torch.zeros(0), torch.zeros(0)
    advantages = torch.zeros(T)
    last_gae = 0.0
    next_v = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_v - values[t]
        last_gae = delta + gamma * lam_gae * last_gae
        advantages[t] = last_gae
        next_v = values[t]
    returns = advantages + values
    return advantages, returns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", required=True, type=str)
    p.add_argument("--out_ckpt", required=True, type=str)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--task_suite", default="libero_10", type=str)
    p.add_argument("--total_episodes", type=int, default=800)
    p.add_argument("--rollout_batch", type=int, default=8,
                   help="Episodes per PPO update.")
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--minibatch_size", type=int, default=256,
                   help="Per-chunk minibatch within a PPO epoch.")
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_coef", type=float, default=0.015,
                   help="Bernoulli entropy bonus weight. Soft-clip is the primary anti-"
                        "collapse mechanism; this is just a soft regularizer. Higher (~0.05)"
                        " freezes the policy at p≈0.5 — wastes the gradient signal.")
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="(Legacy) Default LR if --lr_cadence and --lr_value are unset.")
    p.add_argument("--lr_cadence", type=float, default=6e-5,
                   help="LR for cadence_head only. Lower than action LR so cadence head"
                        " can't out-pace action policy adaptation. Prevents v2 'stuck at"
                        " p=0.5 random firing' mode where high ent_coef kept exploration"
                        " on but cadence couldn't track the changing action policy.")
    p.add_argument("--lr_value", type=float, default=3e-4,
                   help="LR for value head. Critic is fast; this is the only place we keep"
                        " the legacy 3e-4.")
    p.add_argument("--lr_action", type=float, default=3e-5,
                   help="LR for action fast layers if --adapt_action_layers.")
    p.add_argument("--max_steps", type=int, default=720)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--infer_steps", type=int, default=10)
    p.add_argument("--gripper_sign", type=float, default=1.0)
    p.add_argument("--depth_indices", default="0,1,2,3", type=str)
    p.add_argument("--lambda_init", type=float, default=0.0,
                   help="Initial λ_fire (per-fire reward cost). Default 0 to let "
                        "action layers mature on flow loss before cadence is penalized "
                        "for firing.")
    p.add_argument("--lambda_max", type=float, default=0.03)
    p.add_argument("--curriculum_warmup_eps", type=int, default=200,
                   help="Episodes at λ=lambda_init before linear ramp to lambda_max.")
    p.add_argument("--adapt_action_layers", action="store_true",
                   help="Co-train drift + tau_raw + z_groot_proj alongside cadence_head + value head."
                        " Addresses Phase B failure mode where frozen action policy bottlenecks."
                        " Adds parallel flow-matching aux loss using GR00T's chunk as target,"
                        " backprop'd through trainable action layers only.")
    p.add_argument("--flow_minibatch_size", type=int, default=16,
                   help="Per-fired-chunk minibatch for flow-matching aux loss. Smaller "
                        "because re-encoding requires vision forward.")
    p.add_argument("--flow_coef", type=float, default=1.0,
                   help="Weight on flow-matching aux loss in option-c hybrid mode.")
    p.add_argument("--kl_coef", type=float, default=0.0,
                   help="KL(π_current ‖ π_ref) regularization weight. π_ref is the "
                        "cadence_head's state at the start of training (uniform p≈0.27 "
                        "from bias=-1 init). >0 keeps PPO close to baseline cadence — "
                        "task-awareness emerges as small per-state perturbations rather "
                        "than full policy reinvention. Variant (5) of S1/S2 search.")
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
    d = sa["d"]

    value_head = ValueHead(d).to(device)

    # Freeze everything except the trainable set.
    for p_ in model.parameters():
        p_.requires_grad = False
    cadence_params = list(model.encoder.cadence_head.parameters())
    for p_ in cadence_params:
        p_.requires_grad = True
    action_params: list[torch.nn.Parameter] = []
    if args.adapt_action_layers:
        for p_ in model.encoder.drift.parameters():
            p_.requires_grad = True; action_params.append(p_)
        if hasattr(model.encoder, "tau_raw") and isinstance(model.encoder.tau_raw, torch.nn.Parameter):
            model.encoder.tau_raw.requires_grad = True
            action_params.append(model.encoder.tau_raw)
        if model.encoder.z_groot_proj is not None:
            for p_ in model.encoder.z_groot_proj.parameters():
                p_.requires_grad = True; action_params.append(p_)
    value_params = list(value_head.parameters())
    for p_ in value_params:
        p_.requires_grad = True

    n_cadence = sum(p_.numel() for p_ in cadence_params)
    n_value = sum(p_.numel() for p_ in value_params)
    n_action = sum(p_.numel() for p_ in action_params)
    n_total = sum(p_.numel() for p_ in model.parameters()) + n_value
    print(f"[freeze] cadence={n_cadence:,}  value={n_value:,}  "
          f"action_fast={n_action:,}  total_model={n_total:,}")

    # PPO optimizer: cadence_head + value_head only (slow plastic, across-episode learning).
    optimizer = torch.optim.Adam([
        {"params": cadence_params, "lr": args.lr_cadence},
        {"params": value_params, "lr": args.lr_value},
    ])

    # Variant (5): save reference cadence_head state for KL regularization.
    # The reference is the cadence_head at the start of training — uniform p≈0.27 from
    # bias=-1 init, weights=0 → state-independent baseline cadence (≈ K=4 in expectation).
    # KL regularization in the PPO loss anchors the current policy near this reference,
    # so task-awareness emerges as small per-state perturbations rather than full
    # policy specialization (which produces erratic timing — the v6/v10 failure mode).
    reference_cadence_head = None
    if args.kl_coef > 0:
        reference_cadence_head = nn.Linear(d, 1).to(device)
        reference_cadence_head.load_state_dict(model.encoder.cadence_head.state_dict())
        for p_ in reference_cadence_head.parameters():
            p_.requires_grad = False
        print(f"[kl] reference cadence_head saved; kl_coef={args.kl_coef}")

    # Within-episode action-layer optimizer (fast plastic). SGD per V8's proven recipe.
    # Reset between episodes via action_layer_initial_state (saved below).
    action_optimizer = None
    action_layer_initial_state = None
    if args.adapt_action_layers and action_params:
        action_optimizer = torch.optim.SGD(action_params, lr=args.lr_action)
        action_layer_initial_state = {
            id(p): p.detach().clone() for p in action_params
        }
        print(f"[two-timescale] action layers will reset per episode and adapt "
              f"within-episode via SGD lr={args.lr_action}")

    print(f"[lr] cadence={args.lr_cadence:.0e}  value={args.lr_value:.0e}  "
          f"action={args.lr_action:.0e}  ent_coef={args.ent_coef}")

    def reset_action_layers():
        if action_layer_initial_state is None:
            return
        with torch.no_grad():
            for p in action_params:
                p.copy_(action_layer_initial_state[id(p)])
        # Reset SGD optimizer state too (no momentum carryover between episodes).
        if action_optimizer is not None:
            action_optimizer.state = {}

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

    tasks_info = []
    for sim_id in range(n_tasks):
        task = suite.get_task(sim_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = suite.get_task_init_states(sim_id)
        tasks_info.append({
            "sim_id": sim_id, "bddl": bddl, "init_states": init_states,
            "lang": task.language,
        })

    history = []
    success_window: list[int] = []
    fires_window: list[int] = []
    losses_window: list[dict] = []

    print(f"[train] {args.total_episodes} episodes  rollout_batch={args.rollout_batch}  "
          f"ppo_epochs={args.ppo_epochs}  γ={args.gamma}  λ_GAE={args.gae_lambda}  "
          f"clip={args.clip_eps}  λ_fire={args.lambda_init}→{args.lambda_max} "
          f"ramp at ep {args.curriculum_warmup_eps}  "
          f"adapt_action={args.adapt_action_layers}")

    t_start = time.time()
    env = None
    cur_sim_id = -1
    rollout_buf: list[dict] = []
    update_count = 0

    # Variant (v14): track best-so-far snapshot. Save the model when win50_succ
    # hits a new maximum. PPO oscillates around an attractor; the peak ckpt is
    # what we want for deployment, not the endpoint.
    best_win50_succ = -1.0
    best_at_ep = 0
    best_ckpt_path = Path(args.out_ckpt).parent / "step_best.pt"

    for ep in range(args.total_episodes):
        if ep < args.curriculum_warmup_eps:
            lam_fire = args.lambda_init
        else:
            t = (ep - args.curriculum_warmup_eps) / max(args.total_episodes - args.curriculum_warmup_eps, 1)
            lam_fire = args.lambda_init + min(t, 1.0) * (args.lambda_max - args.lambda_init)

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

        # Two-timescale: reset action layers to base ckpt at episode start,
        # then within-episode SGD adapts them as cadence fires.
        reset_action_layers()
        rollout = collect_episode(
            env, init_state, model, value_head, task_lang, sock, args, device,
            target_size, args.depth_indices_list, lam_fire,
            action_optimizer=action_optimizer, action_params=action_params,
        )
        rollout_buf.append(rollout)
        success_window.append(int(rollout["success"]))
        fires_window.append(rollout["n_fires"])
        if len(success_window) > 50:
            success_window.pop(0); fires_window.pop(0)

        # Variant (v14): if win50_succ hits a new max, save best ckpt.
        # Reset action layers before save so the snapshot is the clean base
        # (deployment will re-adapt within episode anyway).
        if len(success_window) >= 50:
            cur_succ = sum(success_window) / 50.0
            if cur_succ > best_win50_succ:
                best_win50_succ = cur_succ
                best_at_ep = ep + 1
                reset_action_layers()
                torch.save({
                    "step": ckpt.get("step"),
                    "policy": model.state_dict(),
                    "value_head": value_head.state_dict(),
                    "args": {**sa, "cadence_head": True, "stage2_phase_c": True,
                             "adapt_action_layers": args.adapt_action_layers,
                             "best_win50_succ": cur_succ,
                             "best_at_ep": ep + 1},
                }, best_ckpt_path)
                print(f"  [best] saved at ep {ep+1}  win50_succ={cur_succ*100:.0f}%")

        # PPO update every rollout_batch episodes.
        if (ep + 1) % args.rollout_batch == 0:
            # Aggregate per-chunk experience from buffer.
            cond_all, fired_all, logp_old_all, val_old_all = [], [], [], []
            adv_all, ret_all = [], []
            within_ep_flow_losses: list = []
            for r in rollout_buf:
                within_ep_flow_losses.extend(r.get("flow_losses", []))
                if r["cond"].shape[0] == 0:
                    continue
                adv, ret = compute_gae(r["reward"], r["value_old"],
                                        args.gamma, args.gae_lambda)
                cond_all.append(r["cond"])
                fired_all.append(r["fired"])
                logp_old_all.append(r["logp_old"])
                val_old_all.append(r["value_old"])
                adv_all.append(adv); ret_all.append(ret)
            rollout_buf = []
            if not cond_all:
                continue
            cond_all = torch.cat(cond_all).to(device)
            fired_all = torch.cat(fired_all).to(device)
            logp_old_all = torch.cat(logp_old_all).to(device)
            val_old_all = torch.cat(val_old_all).to(device)
            adv_all = torch.cat(adv_all).to(device)
            ret_all = torch.cat(ret_all).to(device)
            # Standardize advantage (PPO standard practice).
            if adv_all.shape[0] > 1:
                adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

            N = cond_all.shape[0]
            mb_size = min(args.minibatch_size, N)
            losses_local = {"policy": [], "value": [], "entropy": [],
                            "approx_kl": [], "clip_frac": [], "flow": [],
                            "kl_to_ref": []}
            for _ in range(args.ppo_epochs):
                perm = torch.randperm(N, device=device)
                for start in range(0, N, mb_size):
                    idx = perm[start:start + mb_size]
                    cond_mb = cond_all[idx]
                    fired_mb = fired_all[idx]
                    logp_old_mb = logp_old_all[idx]
                    adv_mb = adv_all[idx]
                    ret_mb = ret_all[idx]

                    # Re-evaluate policy and value on stored cond. Apply the
                    # SAME soft clip used at collection time so the importance
                    # ratio is exact for ratios near the policy boundary.
                    new_logit = soft_clip_logit(
                        model.encoder.cadence_head(cond_mb).squeeze(-1))
                    new_prob = torch.sigmoid(new_logit)
                    new_logp = (fired_mb * torch.log(new_prob + 1e-8) +
                                (1.0 - fired_mb) * torch.log(1.0 - new_prob + 1e-8))
                    new_value = value_head(cond_mb)

                    ratio = torch.exp(new_logp - logp_old_mb)
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_eps,
                                        1.0 + args.clip_eps) * adv_mb
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(new_value, ret_mb)
                    # Bernoulli entropy: H(p) = -[p log p + (1-p) log(1-p)]
                    entropy = -(new_prob * torch.log(new_prob + 1e-8) +
                                (1.0 - new_prob) * torch.log(1.0 - new_prob + 1e-8))
                    entropy = entropy.mean()

                    # Variant (5): KL(π_current ‖ π_ref) on Bernoulli cadence policy.
                    # Anchors the policy near the uniform-baseline reference, preventing
                    # PPO from drifting into the high-variance/erratic regime that hurt
                    # v6/v10. KL = p·log(p/p_ref) + (1-p)·log((1-p)/(1-p_ref)) per state,
                    # averaged across the minibatch.
                    if reference_cadence_head is not None:
                        with torch.no_grad():
                            ref_logit = soft_clip_logit(
                                reference_cadence_head(cond_mb).squeeze(-1))
                            ref_prob = torch.sigmoid(ref_logit)
                        kl_to_ref = (
                            new_prob * (torch.log(new_prob + 1e-8)
                                         - torch.log(ref_prob + 1e-8))
                            + (1.0 - new_prob) * (torch.log(1.0 - new_prob + 1e-8)
                                                   - torch.log(1.0 - ref_prob + 1e-8))
                        ).mean()
                    else:
                        kl_to_ref = torch.tensor(0.0, device=device)

                    loss = (policy_loss
                            + args.vf_coef * value_loss
                            - args.ent_coef * entropy
                            + args.kl_coef * kl_to_ref)

                    optimizer.zero_grad()
                    loss.backward()
                    trainable = cadence_params + value_params + action_params
                    torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                    optimizer.step()

                    with torch.no_grad():
                        approx_kl = (logp_old_mb - new_logp).mean().item()
                        clip_frac = ((ratio - 1.0).abs() > args.clip_eps).float().mean().item()
                    losses_local["policy"].append(policy_loss.item())
                    losses_local["value"].append(value_loss.item())
                    losses_local["entropy"].append(entropy.item())
                    losses_local["approx_kl"].append(approx_kl)
                    losses_local["clip_frac"].append(clip_frac)
                    losses_local["kl_to_ref"].append(kl_to_ref.item())

            # Two-timescale: flow-matching is now done WITHIN each episode in
            # collect_episode (action_optimizer.step() per fire). Aggregate the
            # within-episode flow losses for logging only — no further optimizer step.
            losses_local["flow"].extend(within_ep_flow_losses)
            agg = {k: float(np.mean(v)) if v else 0.0 for k, v in losses_local.items()}
            losses_window.append(agg)
            if len(losses_window) > 10:
                losses_window.pop(0)
            update_count += 1

        # Periodic logging
        if (ep + 1) % 5 == 0:
            sw = sum(success_window) / max(len(success_window), 1)
            fw = sum(fires_window) / max(len(fires_window), 1)
            elapsed = time.time() - t_start
            if losses_window:
                lw = {k: float(np.mean([w[k] for w in losses_window]))
                      for k in losses_window[0]}
            else:
                lw = {"policy": 0.0, "value": 0.0, "entropy": 0.0,
                      "approx_kl": 0.0, "clip_frac": 0.0, "flow": 0.0,
                      "kl_to_ref": 0.0}
            print(f"  ep {ep+1:4d}/{args.total_episodes}  "
                  f"sim{cur_sim_id} init{init_idx}  "
                  f"{'SUCC' if rollout['success'] else 'fail'}  fires={rollout['n_fires']:2d}  "
                  f"λ={lam_fire:.4f}  "
                  f"win50_succ={sw*100:.0f}%  win50_fires={fw:.1f}  "
                  f"pol={lw['policy']:+.3f} v={lw['value']:.3f} "
                  f"ent={lw['entropy']:.3f} kl={lw['approx_kl']:+.3f} "
                  f"klref={lw.get('kl_to_ref', 0.0):.4f} "
                  f"clip={lw['clip_frac']:.2f} flow={lw.get('flow', 0.0):.3f}  "
                  f"updates={update_count}  ({elapsed/60:.1f}min)")
            history.append({
                "ep": ep + 1, "sim_id": cur_sim_id, "init_idx": init_idx,
                "success": bool(rollout["success"]), "n_fires": rollout["n_fires"],
                "n_steps": rollout["n_steps"], "lambda": lam_fire,
                "win50_succ": sw, "win50_fires": fw,
                "loss_policy": lw["policy"], "loss_value": lw["value"],
                "entropy": lw["entropy"], "approx_kl": lw["approx_kl"],
                "clip_frac": lw["clip_frac"], "loss_flow": lw.get("flow", 0.0),
                "kl_to_ref": lw.get("kl_to_ref", 0.0),
                "updates": update_count,
                "elapsed_min": elapsed / 60,
            })

    if env is not None:
        env.close()

    out = Path(args.out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": ckpt.get("step"),
        "policy": model.state_dict(),
        "value_head": value_head.state_dict(),
        "args": {**sa, "cadence_head": True, "stage2_phase_c": True,
                 "adapt_action_layers": args.adapt_action_layers},
        "phaseC_history": history,
    }, out)
    print(f"[save] {out}")
    if best_win50_succ >= 0:
        print(f"[best] step_best.pt at ep {best_at_ep} with win50_succ="
              f"{best_win50_succ*100:.0f}%  (final ep {args.total_episodes} "
              f"ended at {sum(success_window)/max(len(success_window),1)*100:.0f}%)")
    if args.log_path:
        Path(args.log_path).write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
