"""Cartpole on Isaac Sim using our simplified architectures.

Bypasses LiquidARCRoboticsModel (5M params, full LiquidARC). Instead, takes
the small Flat and Liquid+halt policies that worked in `predator_hunt_ppo.py`
and adapts them to Isaac Cartpole's 4-dim observation + 1-dim continuous
action space. Same PPO loop as the existing `train_isaac.py`.

Goal: directly test whether adaptive ODE depth (halting) helps over a flat
MLP on a real physics task (pole balance), with comparable param counts and
validated PPO.

Run on Spark host:
  cd /home/pokazge/IsaacLab
  ./isaaclab.sh -p /home/pokazge/liquid-arc/research/self_org_sim/cartpole_isaac.py \\
      --policy flat --headless --num_envs 1024 --total_steps 500000

  ./isaaclab.sh -p /home/pokazge/liquid-arc/research/self_org_sim/cartpole_isaac.py \\
      --policy liquid_halt --k 16 --halting_min_steps 4 --ponder_lambda 0.05 \\
      --headless --num_envs 1024 --total_steps 500000
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)

os.environ.setdefault("WARP_CACHE_PATH", os.path.expanduser("~/.cache/warp"))
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Policies — same shape as predator_hunt.py, adapted for Gaussian action
# ---------------------------------------------------------------------------

class FlatGaussianPolicy(nn.Module):
    """MLP encoder → mean head + learned log_std + value head."""

    def __init__(self, obs_dim: int, action_dim: int, d: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
        )
        self.mean_head = nn.Linear(d, action_dim)
        self.value_head = nn.Linear(d, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, obs: torch.Tensor):
        h = self.encoder(obs)
        mean = self.mean_head(h)
        value = self.value_head(h).squeeze(-1)
        return mean, value, {
            "steps_mean": torch.tensor(0.0, device=h.device),
            "ponder_cost": torch.tensor(0.0, device=h.device),
        }


class LiquidGaussianPolicy(nn.Module):
    """Continuous-time recurrent policy + optional adaptive halting + Gaussian head.

    Same dynamics module as predator_hunt.py (LTC contraction with per-dim tau).
    h_0 = encoder(obs); h_{k+1} = h_k + dt/tau * (drift(h_k) - h_k).
    With halt_enabled, halt head outputs p_halt(h_k); after min_steps,
    still_active *= (1 - p_halt). Hard inputs use more iterations.
    """

    def __init__(self, obs_dim: int, action_dim: int, d: int = 64,
                 k_max: int = 16, halt_mode: str = "none",
                 min_steps: int = 4, dt: float = 0.5,
                 conv_eps: float = 0.01, conv_eps_scale: float = 0.005):
        super().__init__()
        assert halt_mode in {"none", "learned", "convergence"}
        self.k_max = k_max
        self.halt_mode = halt_mode
        self.min_steps = min_steps
        self.dt = dt
        self.conv_eps = conv_eps
        self.conv_eps_scale = conv_eps_scale
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
        )
        # Drift outputs a learnable VELOCITY for h: h_new = h + dt/tau · drift(h).
        # Zero-init last linear → drift(h) ≈ 0 at init → h doesn't change across
        # iterations → liquid policy ≈ flat policy at init.
        self.drift = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(),
            nn.Linear(d * 2, d),
        )
        nn.init.zeros_(self.drift[-1].weight)
        nn.init.zeros_(self.drift[-1].bias)
        self.tau_raw = nn.Parameter(torch.zeros(d))
        if halt_mode == "learned":
            self.halt_head = nn.Linear(d, 1)
            with torch.no_grad():
                self.halt_head.bias.fill_(-3.0)
        self.mean_head = nn.Linear(d, action_dim)
        self.value_head = nn.Linear(d, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, obs: torch.Tensor):
        h = self.encoder(obs)
        B = h.shape[0]
        tau = F.softplus(self.tau_raw) + 0.1
        steps_used = torch.zeros(B, 1, device=h.device)
        still_active = torch.ones(B, 1, device=h.device)
        ponder_cost = torch.zeros((), device=h.device)
        h_init_norms = h.norm(dim=-1).detach()  # for CV diagnostics
        for k in range(self.k_max):
            # Residual velocity-field formulation. drift(h)=0 at init ⇒ h_new=h.
            dh = self.drift(h) / tau
            if self.halt_mode == "convergence":
                # Mandatory min_steps then dynamics-driven SOFT stop.
                # active=sigmoid((||dh||-eps)/scale): close to 1 when dh is
                # large, close to 0 when small. h stops evolving where dh~0.
                # No learned halt parameters → no PPO ratio mismatch from
                # halt-distribution shift between rollout and update.
                if k < self.min_steps:
                    active = torch.ones(B, 1, device=h.device)
                else:
                    norm = dh.norm(dim=-1, keepdim=True)
                    active = torch.sigmoid(
                        (norm - self.conv_eps) / self.conv_eps_scale)
                h = h + self.dt * dh * active
                steps_used = steps_used + active
                # Track residual for logging only (NOT added to loss —
                # convergence is parameter-free, has no compute pressure)
                ponder_cost = ponder_cost + dh.norm(dim=-1).mean()
            elif self.halt_mode == "learned":
                h_new = h + self.dt * dh
                h = still_active * h_new + (1.0 - still_active) * h
                p_halt = torch.sigmoid(self.halt_head(h))
                steps_used = steps_used + still_active
                if k >= self.min_steps:
                    still_active = still_active * (1.0 - p_halt)
                ponder_cost = ponder_cost + still_active.mean()
            else:  # "none" — fixed K
                h = h + self.dt * dh
                steps_used = steps_used + 1.0
        if self.halt_mode in ("learned", "convergence"):
            ponder_cost = ponder_cost / self.k_max
        else:
            ponder_cost = torch.tensor(1.0, device=h.device)
        mean = self.mean_head(h)
        value = self.value_head(h).squeeze(-1)
        # SoC: coefficient of variation of hidden state magnitudes across batch.
        # Low CV ⇒ representations have collapsed to a small manifold (degenerate
        # policy, advantages become noisy under PPO). High CV ⇒ diverse states.
        # The crit_loss term (added in PPO update) penalizes collapse below target,
        # creating a critical-point attractor on the substrate. This is
        # SELF-stabilization, not external constraint — the dynamics learn to
        # maintain geometric variation, surviving big PPO updates without clipping.
        h_norms_final = h.norm(dim=-1)  # [B]
        h_cv = h_norms_final.std() / (h_norms_final.mean() + 1e-8)
        return mean, value, {
            "steps_mean": steps_used.mean().detach(),
            "steps_min": steps_used.min().detach(),
            "steps_max": steps_used.max().detach(),
            "ponder_cost": ponder_cost.detach() if not torch.is_grad_enabled()
                else ponder_cost,
            "h_cv": h_cv if torch.is_grad_enabled() else h_cv.detach(),
            "h_norm_mean": h_norms_final.mean().detach(),
        }


def build_policy(name, obs_dim, action_dim, d, k, min_steps, device,
                  conv_eps=0.01, conv_eps_scale=0.005):
    if name == "flat":
        p = FlatGaussianPolicy(obs_dim, action_dim, d=d)
    elif name == "liquid_fixed":
        p = LiquidGaussianPolicy(obs_dim, action_dim, d=d, k_max=k,
                                  halt_mode="none", min_steps=min_steps)
    elif name == "liquid_halt":
        p = LiquidGaussianPolicy(obs_dim, action_dim, d=d, k_max=k,
                                  halt_mode="learned", min_steps=min_steps)
    elif name == "liquid_conv":
        p = LiquidGaussianPolicy(obs_dim, action_dim, d=d, k_max=k,
                                  halt_mode="convergence", min_steps=min_steps,
                                  conv_eps=conv_eps,
                                  conv_eps_scale=conv_eps_scale)
    else:
        raise ValueError(name)
    return p.to(device)


def get_action_dist(policy, obs):
    mean, value, info = policy(obs)
    std = policy.log_std.exp().expand_as(mean)
    dist = torch.distributions.Normal(mean, std)
    return dist, value, info


# ---------------------------------------------------------------------------
# PPO with GAE — same as predator_hunt_ppo.py + Isaac env loop
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = values[t]  # bootstrap = current (no terminal value here)
            next_nonterminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-Direct-v0")
    parser.add_argument("--policy",
                        choices=["flat", "liquid_fixed", "liquid_halt", "liquid_conv"],
                        default="flat")
    parser.add_argument("--conv_eps", type=float, default=0.01,
                        help="Convergence threshold ||dh|| for liquid_conv halting")
    parser.add_argument("--conv_eps_scale", type=float, default=0.005,
                        help="Smoothness of soft convergence transition")
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--k", type=int, default=16,
                        help="ODE iterations (fixed K) or K_max (with halting)")
    parser.add_argument("--halting_min_steps", type=int, default=4)
    parser.add_argument("--ponder_lambda", type=float, default=0.05)
    # Self-organized criticality: maintain geometric diversity via CV penalty.
    # Penalizes collapse of hidden-state norms below target — keeps the substrate
    # at a critical-point attractor, self-stabilizing against PPO overcorrection.
    parser.add_argument("--crit_lambda", type=float, default=0.0,
                        help="SoC aux loss weight (0 = disabled)")
    parser.add_argument("--cv_target", type=float, default=0.5,
                        help="Target CV of hidden state norms across batch")
    # Curriculum (quadcopter only — controls QC_GUST_STRENGTH and goal range
    # by writing to os.environ each update; env reads via os.environ.get).
    parser.add_argument("--curriculum_warmup", type=float, default=0.25,
                        help="Fraction of training to keep difficulty=0 (no gust, default goal range)")
    parser.add_argument("--curriculum_full", type=float, default=0.75,
                        help="Fraction of training at which to reach max difficulty")
    parser.add_argument("--gust_max", type=float, default=0.5,
                        help="Max gust strength reached at curriculum_full")
    parser.add_argument("--goal_xy_max", type=float, default=4.0,
                        help="Max target xy range")
    parser.add_argument("--goal_z_max_max", type=float, default=2.5,
                        help="Max target z upper bound")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true",
                        help="Required for vision-based Isaac tasks (e.g. Repose-Cube-Shadow-Vision)")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--total_steps", type=int, default=500000)
    parser.add_argument("--rollout_length", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=4096)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_compile", action="store_true", default=False)
    # PPO stabilization: reward normalization + value-loss clipping. Standard
    # PPO practice; mitigates value-net overshoot when rewards have high
    # magnitude / sparse bonuses (e.g. shadow_hand reach_goal_bonus=250).
    parser.add_argument("--reward_norm", action="store_true",
                        help="Normalize rewards by running std of returns (PPO standard)")
    parser.add_argument("--vf_clip", type=float, default=0.0,
                        help="Value-loss clipping range (0 disables; 0.5-1.0 typical)")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Save checkpoint to this path every save_every updates")
    parser.add_argument("--save_every", type=int, default=20,
                        help="Save checkpoint every N PPO updates")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a saved checkpoint (loads policy + optimizer state)")
    # Elastic Weight Consolidation: penalize movement of important weights from
    # their prior-phase values, weighted by per-weight Fisher information.
    # Prevents catastrophic forgetting in curriculum / multi-phase training.
    parser.add_argument("--ewc_lambda", type=float, default=0.0,
                        help="EWC penalty weight (0 disables; ~100-1000 typical)")
    parser.add_argument("--fisher_samples", type=int, default=2048,
                        help="Number of rollout samples used to estimate Fisher diag at end of phase")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip PPO updates; collect rollouts and report reward stats only (frozen-policy evaluation)")
    args = parser.parse_args()

    # Isaac Lab simulation app must launch before any Isaac/sim imports
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless,
                                enable_cameras=args.enable_cameras)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # registers Isaac tasks
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device="cuda", num_envs=args.num_envs)
    # Shadow_hand reward overrides via env vars — fix the parking-spot reward
    # hack: original env has fall_penalty=0 and rot_reward_scale=1.0 which lets
    # policy "park" cube somewhere stable and ignore rotation goal entirely.
    if "Shadow" in args.task:
        if "SH_ROT_SCALE" in os.environ and hasattr(env_cfg, "rot_reward_scale"):
            env_cfg.rot_reward_scale = float(os.environ["SH_ROT_SCALE"])
            print(f"  Override: rot_reward_scale = {env_cfg.rot_reward_scale}")
        if "SH_FALL_PENALTY" in os.environ and hasattr(env_cfg, "fall_penalty"):
            env_cfg.fall_penalty = float(os.environ["SH_FALL_PENALTY"])
            print(f"  Override: fall_penalty = {env_cfg.fall_penalty}")
        if "SH_DIST_SCALE" in os.environ and hasattr(env_cfg, "dist_reward_scale"):
            env_cfg.dist_reward_scale = float(os.environ["SH_DIST_SCALE"])
            print(f"  Override: dist_reward_scale = {env_cfg.dist_reward_scale}")
    env = gym.make(args.task, cfg=env_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_space = env.unwrapped.single_observation_space["policy"]
    action_space = env.unwrapped.single_action_space
    obs_dim = obs_space.shape[0]
    action_dim = action_space.shape[0]
    print(f"obs_dim={obs_dim} action_dim={action_dim} num_envs={args.num_envs}")

    torch.manual_seed(args.seed)
    policy = build_policy(args.policy, obs_dim, action_dim, args.d, args.k,
                            args.halting_min_steps, device,
                            conv_eps=args.conv_eps,
                            conv_eps_scale=args.conv_eps_scale)
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy: {args.policy} ({n_params:,} params) "
          f"d={args.d} K={args.k}")

    # Resume from checkpoint if requested. Loads policy + optimizer state so
    # we can continue training from prior phase. Action/value heads expected
    # to match (same task / action_dim); for cross-task transfer use --resume
    # then re-init heads via separate flag (not implemented yet — single-task
    # continuous learning first).
    if args.resume is not None and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # Strip torch.compile wrapper prefix if present
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
        # Filter to only matching shapes (heads may differ across tasks)
        own_sd = policy.state_dict()
        loaded, skipped = 0, 0
        for k, v in sd.items():
            if k in own_sd and own_sd[k].shape == v.shape:
                own_sd[k].copy_(v)
                loaded += 1
            else:
                skipped += 1
        if "optim" in ckpt and skipped == 0:
            try:
                optim.load_state_dict(ckpt["optim"])
            except Exception as e:
                print(f"  optim state skipped: {e}")
        print(f"  RESUMED from {args.resume}: loaded {loaded} tensors, "
              f"skipped {skipped} (head/shape mismatch)")
        # Load EWC anchors if present in checkpoint and EWC is enabled
        ewc_fisher = ckpt.get("fisher", None)
        ewc_prior = ckpt.get("prior_params", None)
        if args.ewc_lambda > 0 and ewc_fisher is not None and ewc_prior is not None:
            ewc_fisher = {n: t.to(device) for n, t in ewc_fisher.items()}
            ewc_prior = {n: t.to(device) for n, t in ewc_prior.items()}
            print(f"  EWC anchors loaded: {len(ewc_fisher)} tensors, "
                  f"lambda={args.ewc_lambda}")
        else:
            ewc_fisher = None
            ewc_prior = None
    else:
        ewc_fisher = None
        ewc_prior = None

    if not args.no_compile and device.type == "cuda":
        # Compile only for liquid policies (the iteration is the bottleneck);
        # flat is too small to benefit. Compile dynamically to handle varying
        # batch sizes between rollout and PPO minibatches.
        if args.policy != "flat":
            policy = torch.compile(policy, mode="default", dynamic=True)
        print(f"torch.compile: {'on' if args.policy != 'flat' else 'flat skipped'}")

    # Set initial difficulty BEFORE first env.reset() so the FIRST episodes
    # spawn at the warmup difficulty (typically gust=0), not the env's default.
    # Without this, the env's first reset uses default QC_DR_GUST_MAX=0.5,
    # giving early episodes random heavy gusts before the policy can learn.
    is_quadcopter_init = "Quadcopter" in args.task
    is_shadow_init = "Shadow" in args.task
    qc_dr_init = bool(int(os.environ.get("QC_DR", "0")))
    sh_dr_init = bool(int(os.environ.get("SH_DR", "0")))
    if is_quadcopter_init and qc_dr_init and not (
            "QC_DR_GUST_MAX" in os.environ or "QC_GUST_STRENGTH" in os.environ):
        os.environ["QC_DR_GUST_MAX"] = "0.0"
        os.environ["QC_GOAL_XY"] = "2.0"
    if is_shadow_init and sh_dr_init and "SH_DR_FORCE_MAX" not in os.environ:
        os.environ["SH_DR_FORCE_MAX"] = "0.0"

    obs, _ = env.reset()
    obs_t = obs["policy"]  # [num_envs, obs_dim]
    num_envs = obs_t.shape[0]
    # Per-step n_used distribution tracking (records min/max within each batch)
    steps_mins, steps_maxs = [], []
    total_updates = args.total_steps // (num_envs * args.rollout_length)
    print(f"Training: {total_updates} updates (~{args.total_steps:,} env steps, "
          f"batch={num_envs * args.rollout_length})")

    ep_rewards = torch.zeros(num_envs, device=device)
    ep_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)
    completed_rewards: list[float] = []
    completed_lengths: list[int] = []

    t0 = time.time()
    global_step = 0
    # Reward normalization state: track running discounted return per env,
    # maintain EMA of variance. Normalize rewards by sqrt(var) before GAE.
    reward_running_return = None
    reward_var_ema = 1.0
    reward_var_decay = 0.99
    is_quadcopter = "Quadcopter" in args.task
    is_shadow = "Shadow" in args.task
    # If a difficulty bound is set in env at script entry, user is controlling
    # difficulty externally (e.g., explicit phase). Disable internal curriculum.
    external_difficulty = (
        "QC_GUST_STRENGTH" in os.environ
        or "QC_GOAL_XY" in os.environ
        or "QC_DR_GUST_MAX" in os.environ
        or "QC_DR_XY_MAX" in os.environ
        or "SH_DR_FORCE_MAX" in os.environ
    )
    dr_mode = bool(int(os.environ.get("QC_DR", "0")) or int(os.environ.get("SH_DR", "0")))
    if external_difficulty:
        print(f"  External difficulty detected (QC_GUST_STRENGTH={os.environ.get('QC_GUST_STRENGTH','-')} "
              f"QC_GOAL_XY={os.environ.get('QC_GOAL_XY','-')}); internal curriculum disabled")
    for update in range(total_updates):
        # Curriculum ramp (quadcopter only): linear from 0 (warmup) → 1 (full)
        if (is_quadcopter or is_shadow) and not external_difficulty:
            progress = update / max(total_updates - 1, 1)
            warmup = args.curriculum_warmup
            full = args.curriculum_full
            if progress <= warmup:
                diff = 0.0
            elif progress >= full:
                diff = 1.0
            else:
                diff = (progress - warmup) / max(full - warmup, 1e-6)
            if is_shadow and dr_mode:
                # Shadow_hand DR (multi-axis): ramp force perturbation, cube
                # mass variability, and actuator stiffness variability — all
                # together. Industry sim-to-real DR for in-hand manipulation.
                os.environ["SH_DR_FORCE_MAX"] = f"{diff * args.gust_max:.4f}"
                os.environ["SH_DR_MASS_VAR"] = f"{diff * 0.5:.4f}"   # ±50% mass at full
                os.environ["SH_DR_STIFF_VAR"] = f"{diff * 0.3:.4f}"  # ±30% stiffness at full
            elif is_quadcopter and dr_mode:
                # DR mode: ramp the upper bound of GUST randomization range.
                # Each episode samples random gust within [0, gust_max·diff].
                # Target xy is also expanded as difficulty rises so the policy
                # learns to track varied goals.
                os.environ["QC_DR_GUST_MAX"] = f"{diff * args.gust_max:.4f}"
                os.environ["QC_GOAL_XY"] = f"{2.0 + diff * (args.goal_xy_max - 2.0):.4f}"
            elif is_quadcopter:
                # Non-DR: ramp the fixed gust strength
                os.environ["QC_GUST_STRENGTH"] = f"{diff * args.gust_max:.4f}"
                os.environ["QC_GOAL_XY"] = f"{2.0 + diff * (args.goal_xy_max - 2.0):.4f}"
                os.environ["QC_GOAL_Z_MIN"] = f"{0.5 - diff * 0.2:.4f}"
                os.environ["QC_GOAL_Z_MAX"] = f"{1.5 + diff * (args.goal_z_max_max - 1.5):.4f}"
        # === Rollout (no_grad) ===
        roll_obs, roll_actions, roll_log_probs, roll_values = [], [], [], []
        roll_rewards, roll_dones = [], []
        steps_means = []
        steps_mins_buf, steps_maxs_buf = [], []
        with torch.no_grad():
            for t in range(args.rollout_length):
                dist, value, info = get_action_dist(policy, obs_t)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                roll_obs.append(obs_t.clone())
                roll_actions.append(action)
                roll_log_probs.append(log_prob)
                roll_values.append(value)
                roll_rewards.append(reward)
                roll_dones.append(done.float())
                steps_means.append(float(info["steps_mean"].item()))
                if "steps_min" in info:
                    steps_mins_buf.append(float(info["steps_min"].item()))
                    steps_maxs_buf.append(float(info["steps_max"].item()))
                ep_rewards = ep_rewards + reward
                ep_lengths = ep_lengths + 1
                # Episode bookkeeping
                done_idx = done.nonzero(as_tuple=True)[0]
                if len(done_idx) > 0:
                    completed_rewards.extend(ep_rewards[done_idx].tolist())
                    completed_lengths.extend(ep_lengths[done_idx].tolist())
                    ep_rewards[done_idx] = 0
                    ep_lengths[done_idx] = 0
                obs_t = next_obs["policy"]
                global_step += num_envs

        rewards = torch.stack(roll_rewards)
        values = torch.stack(roll_values)
        dones = torch.stack(roll_dones)
        actions = torch.stack(roll_actions)
        log_probs = torch.stack(roll_log_probs)
        observations = torch.stack(roll_obs)

        # Reward normalization (PPO standard): scale rewards by running std of
        # the discounted return so value net targets stay in [-3, 3] range
        # regardless of absolute reward magnitude. Critical when rewards have
        # high-magnitude bonuses (shadow_hand reach_goal_bonus=250 × 5x scale).
        if args.reward_norm:
            T_r, B_r = rewards.shape
            if reward_running_return is None or reward_running_return.shape != (B_r,):
                reward_running_return = torch.zeros(B_r, device=rewards.device)
            # Update running return per env, accumulate variance over rollout
            new_var_samples = []
            for t in range(T_r):
                reward_running_return = reward_running_return * args.gamma * (1.0 - dones[t]) + rewards[t]
                new_var_samples.append(reward_running_return)
            batch_var = torch.stack(new_var_samples).var().item()
            reward_var_ema = reward_var_decay * reward_var_ema + (1 - reward_var_decay) * batch_var
            scale = (reward_var_ema + 1e-8) ** 0.5
            rewards_norm = rewards / scale
        else:
            rewards_norm = rewards
        advantages, returns = compute_gae(rewards_norm, values, dones,
                                            gamma=args.gamma, lam=args.gae_lambda)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Skip PPO updates in eval-only mode (frozen-policy evaluation)
        if args.eval_only:
            T, B = observations.shape[:2]
            # Set dummy losses for logging
            class _Dummy:
                def __init__(self, v): self.v = v
                def item(self): return self.v
            pol_loss, val_loss = _Dummy(0.0), _Dummy(0.0)
            if args.log_every > 0 and update % args.log_every == 0:
                elapsed = time.time() - t0
                fps = global_step / max(elapsed, 1e-6)
                recent = completed_rewards[-100:] if completed_rewards else [0]
                recent_l = completed_lengths[-100:] if completed_lengths else [0]
                mean_r = sum(recent) / max(len(recent), 1)
                mean_l = sum(recent_l) / max(len(recent_l), 1)
                n_used_avg = sum(steps_means) / max(len(steps_means), 1)
                halt_str = ""
                if args.policy != "flat":
                    if steps_mins_buf:
                        n_min = sum(steps_mins_buf) / len(steps_mins_buf)
                        n_max = sum(steps_maxs_buf) / len(steps_maxs_buf)
                        halt_str = (f" n_used={n_used_avg:5.2f}/{args.k}"
                                    f" [min={n_min:.1f} max={n_max:.1f}]")
                    else:
                        halt_str = f" n_used={n_used_avg:5.2f}/{args.k}"
                print(f"  [EVAL u={update:3d}] step={global_step:>7d} "
                      f"reward={mean_r:7.1f} ep_len={mean_l:6.1f} "
                      f"fps={fps:5.0f}{halt_str}", flush=True)
            continue

        # === PPO update ===
        T, B = observations.shape[:2]
        flat_obs = observations.reshape(T * B, obs_dim)
        flat_act = actions.reshape(T * B, action_dim)
        flat_logp = log_probs.reshape(T * B)
        flat_adv = advantages.reshape(T * B)
        flat_ret = returns.reshape(T * B)
        flat_val = values.reshape(T * B)
        N = T * B
        for epoch in range(args.n_epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, args.minibatch_size):
                idx = perm[start:start + args.minibatch_size]
                mb_obs = flat_obs[idx]
                mb_act = flat_act[idx]
                mb_old_logp = flat_logp[idx]
                mb_adv = flat_adv[idx]
                mb_ret = flat_ret[idx]
                mb_val_old = flat_val[idx]

                dist, new_value, info = get_action_dist(policy, mb_obs)
                new_logp = dist.log_prob(mb_act).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * mb_adv
                pol_loss = -torch.min(surr1, surr2).mean()
                if args.vf_clip > 0:
                    # Value-loss clipping (PPO standard): caps how much the
                    # value net can update per iteration. Pessimistic max of
                    # clipped + unclipped MSE prevents value-net overshoot
                    # when rewards have high magnitude/sparse bonuses.
                    val_clipped = mb_val_old + torch.clamp(
                        new_value - mb_val_old, -args.vf_clip, args.vf_clip)
                    v_unclipped = (new_value - mb_ret) ** 2
                    v_clipped = (val_clipped - mb_ret) ** 2
                    val_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    val_loss = F.mse_loss(new_value, mb_ret)
                loss = pol_loss + args.vf_coef * val_loss - args.entropy_coef * entropy
                if args.policy == "liquid_halt" and "ponder_cost" in info:
                    pc = info["ponder_cost"]
                    if pc.requires_grad:
                        loss = loss + args.ponder_lambda * pc
                # SoC: hinged penalty for CV collapsing below target. Only
                # pushes UP when degenerate; allows CV > target naturally.
                if args.crit_lambda > 0 and "h_cv" in info:
                    cv = info["h_cv"]
                    if cv.requires_grad:
                        crit_loss = (args.cv_target - cv).clamp(min=0) ** 2
                        loss = loss + args.crit_lambda * crit_loss
                # EWC: penalize movement of weights important for prior phase
                # weighted by Fisher diagonal. Preserves prior skill while
                # allowing adaptation to current phase.
                if (args.ewc_lambda > 0 and ewc_fisher is not None
                        and ewc_prior is not None):
                    ewc_loss = torch.zeros((), device=device)
                    for n, p in policy.named_parameters():
                        # Strip torch.compile prefix if present
                        key = n.replace("_orig_mod.", "")
                        if key in ewc_fisher:
                            ewc_loss = ewc_loss + (
                                ewc_fisher[key] * (p - ewc_prior[key]) ** 2
                            ).sum()
                    loss = loss + args.ewc_lambda * ewc_loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  WARN: NaN/Inf loss at update={update}, skipping")
                    continue
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optim.step()

        if update % args.log_every == 0:
            elapsed = time.time() - t0
            fps = global_step / max(elapsed, 1e-6)
            recent = completed_rewards[-100:] if completed_rewards else [0]
            recent_l = completed_lengths[-100:] if completed_lengths else [0]
            mean_r = sum(recent) / max(len(recent), 1)
            mean_l = sum(recent_l) / max(len(recent_l), 1)
            n_used_avg = sum(steps_means) / max(len(steps_means), 1)
            halt_str = ""
            if args.policy != "flat":
                if steps_mins_buf:
                    n_min = sum(steps_mins_buf) / len(steps_mins_buf)
                    n_max = sum(steps_maxs_buf) / len(steps_maxs_buf)
                    halt_str = (f" n_used={n_used_avg:5.2f}/{args.k}"
                                f" [min={n_min:.1f} max={n_max:.1f}]")
                else:
                    halt_str = f" n_used={n_used_avg:5.2f}/{args.k}"
            curr_str = ""
            if is_quadcopter:
                curr_str = (f" gust={os.environ.get('QC_GUST_STRENGTH', '0'):>5}"
                            f" xy={os.environ.get('QC_GOAL_XY', '2'):>4}")
            cv_str = ""
            if args.policy != "flat" and "h_cv" in info:
                cv_str = f" cv={float(info['h_cv'].item()):.2f}"
            print(f"  [u={update:3d}] step={global_step:>7d} "
                  f"reward={mean_r:7.1f} ep_len={mean_l:6.1f} fps={fps:5.0f} "
                  f"pol_loss={pol_loss.item():+.4f} val_loss={val_loss.item():.3f} "
                  f"log_std={policy.log_std.data.mean().item():+.3f}{halt_str}{cv_str}{curr_str}")

        # Periodic checkpoint save
        if (args.save_path is not None
                and args.save_every > 0
                and (update + 1) % args.save_every == 0):
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save({
                "policy": policy.state_dict(),
                "optim": optim.state_dict(),
                "global_step": global_step,
                "update": update,
                "args": vars(args),
            }, args.save_path)

    # Final save at end of training (always if save_path set)
    if args.save_path is not None:
        # Compute Fisher information diagonal for EWC. Estimates per-weight
        # importance for the CURRENT phase's task. Saved with checkpoint so
        # the next phase can use it as an anchor against forgetting.
        # Fisher_i ≈ E[(∂ log π(a|s) / ∂ θ_i)^2]
        print(f"  Computing Fisher diagonal ({args.fisher_samples} samples)...")
        fisher = {n: torch.zeros_like(p) for n, p in policy.named_parameters()}
        prior_params = {n: p.detach().clone() for n, p in policy.named_parameters()}
        n_samples_done = 0
        # Reuse current obs from last rollout
        with torch.enable_grad():
            while n_samples_done < args.fisher_samples:
                obs_in = obs_t.detach()
                dist, _, _ = get_action_dist(policy, obs_in)
                action = dist.sample().detach()
                log_prob = dist.log_prob(action).sum(dim=-1)
                # Sum over batch (Fisher is diagonal so squaring per-element gradients
                # is what matters; equiv to per-sample fisher averaged over batch)
                policy.zero_grad()
                log_prob.sum().backward()
                for n, p in policy.named_parameters():
                    if p.grad is not None:
                        fisher[n] = fisher[n] + p.grad.detach() ** 2
                n_samples_done += obs_in.shape[0]
                # Step env to advance to a new state for next batch
                with torch.no_grad():
                    next_obs, _, _, _, _ = env.step(action)
                    obs_t = next_obs["policy"]
        # Normalize fisher by total samples
        for n in fisher:
            fisher[n] = fisher[n] / max(n_samples_done, 1)
        # Strip compile prefix from keys for compatibility with resume
        fisher_clean = {n.replace("_orig_mod.", ""): t.cpu()
                        for n, t in fisher.items()}
        prior_clean = {n.replace("_orig_mod.", ""): t.cpu()
                       for n, t in prior_params.items()}
        print(f"  Fisher computed (mean magnitude: "
              f"{sum(t.mean().item() for t in fisher_clean.values()) / len(fisher_clean):.6f})")

        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save({
            "policy": policy.state_dict(),
            "optim": optim.state_dict(),
            "fisher": fisher_clean,
            "prior_params": prior_clean,
            "global_step": global_step,
            "update": total_updates,
            "args": vars(args),
        }, args.save_path)
        print(f"  Saved checkpoint: {args.save_path}")

    print(f"\nFINAL  policy={args.policy} params={n_params:,} "
          f"reward(last100)={sum(completed_rewards[-100:]) / max(len(completed_rewards[-100:]), 1):.1f} "
          f"ep_len(last100)={sum(completed_lengths[-100:]) / max(len(completed_lengths[-100:]), 1):.1f} "
          f"({time.time() - t0:.0f}s)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
