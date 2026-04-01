"""Train ContinuousLifecycleRunner as a robotics controller in Isaac Lab.

The key distinction from train_isaac.py: the ODE state _h persists across
rollout steps. Observations inject as sensory forcing. Episode resets zero
_h for terminated environments via model.handle_resets(done).

Curiosity here is prediction error — how surprised was the model by the
new observation? This is computed by the lifecycle runner itself as the
L2 distance between embedded obs and the current h state before forcing.

Usage:
    cd /home/pokazge/IsaacLab
    ./isaaclab.sh -p /home/pokazge/liquid-arc/scripts/train_lifecycle.py \\
        --checkpoint /home/pokazge/liquid-arc/output_30m/checkpoints/step_10000.pt \\
        --headless --num_envs 1024
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Set Warp cache BEFORE any imports that trigger Warp init
os.environ["WARP_CACHE_PATH"] = os.path.expanduser("~/.cache/warp")
# Triton needs system ptxas for SM 12.1 (Blackwell) — same as fgn-train container
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

# Force unbuffered output so logs appear in real-time
import functools
print = functools.partial(print, flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add liquid-arc to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.lifecycle import ContinuousLifecycleRunner


class CuriosityNormalizer:
    """Running normalization for intrinsic reward stability."""

    def __init__(self, decay: float = 0.99):
        self.mean = 0.0
        self.var = 1.0
        self.decay = decay
        self.count = 0

    def normalize(self, curiosity: torch.Tensor) -> torch.Tensor:
        batch_mean = curiosity.mean().item()
        batch_var = curiosity.var().item() + 1e-8
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean
            self.var = self.decay * self.var + (1 - self.decay) * batch_var
        self.count += 1
        return (curiosity - self.mean) / (self.var ** 0.5 + 1e-8)


def get_lambda_eff(update_idx, total_updates, lam_init, lam_final, ramp_start):
    """Efficiency regularizer schedule: low early (explore), high late (optimize)."""
    ramp_begin = int(total_updates * ramp_start)
    if update_idx < ramp_begin:
        return lam_init
    progress = (update_idx - ramp_begin) / max(total_updates - ramp_begin, 1)
    return lam_init + (lam_final - lam_init) * min(progress, 1.0)


def get_beta(
    update_idx: int, total_updates: int,
    beta_init: float, beta_final: float, decay_fraction: float,
) -> float:
    """Decay intrinsic motivation coefficient over training."""
    progress = min(update_idx / max(total_updates * decay_fraction, 1), 1.0)
    return beta_init * (1 - progress) + beta_final * progress


class GaussianPolicy(nn.Module):
    """Wraps ContinuousLifecycleRunner with a learned log_std for PPO.

    The lifecycle runner maintains persistent ODE state internally.
    This wrapper only adds the exploration noise parameter and exposes
    the standard PPO interface (sample, evaluate_actions).
    """

    def __init__(self, model: ContinuousLifecycleRunner, action_dim: int):
        super().__init__()
        self.model = model
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def get_action_dist(
        self, tokens: dict, actuated_indices: torch.Tensor,
    ):
        """Forward pass through the lifecycle model, return distribution and result."""
        result = self.model.step(tokens, actuated_indices)
        mean = result['actions']
        # NaN/Inf protection on mean and std before creating distribution
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.zeros_like(mean)
        mean = mean.clamp(-20.0, 20.0)
        std = self.log_std.clamp(min=-0.5, max=0.5).exp().expand_as(mean)  # std in [0.6, 1.65]
        dist = torch.distributions.Normal(mean, std)
        return dist, result

    def get_action_dist_eval(
        self, tokens: dict, actuated_indices: torch.Tensor,
    ):
        """Forward pass for PPO eval — skip autonomous steps (h starts fresh)."""
        result = self.model.step(tokens, actuated_indices, skip_autonomous=True)
        mean = result['actions']
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.zeros_like(mean)
        mean = mean.clamp(-20.0, 20.0)
        std = self.log_std.clamp(min=-0.5, max=0.5).exp().expand_as(mean)  # std in [0.6, 1.65]
        dist = torch.distributions.Normal(mean, std)
        return dist, result

    def evaluate_actions(
        self, tokens: dict, actuated_indices: torch.Tensor, actions: torch.Tensor,
    ):
        """Re-evaluate collected actions for the PPO update.

        Note: During the PPO update epochs we run fresh forward passes on
        stored observations. Because _h is detached at each step during
        rollout collection, calling step() here re-initialises _h from
        zeros for the minibatch obs, which is intentional — we only need
        the log_prob and entropy for the gradient, not the full trajectory.
        """
        dist, result = self.get_action_dist_eval(tokens, actuated_indices)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, result


class ValueNet(nn.Module):
    """Simple MLP value network operating on the flat observation vector."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
):
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: [T, B] reward tensor
        values:  [T, B] value estimates
        dones:   [T, B] float done flags (1.0 = episode ended)
        gamma:   discount factor
        lam:     GAE lambda

    Returns:
        advantages: [T, B]
        returns:    [T, B] (advantages + values, for value loss target)
    """
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = 0.0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

    returns = advantages + values
    return advantages, returns


def main():
    parser = argparse.ArgumentParser(
        description="Train ContinuousLifecycleRunner on Isaac-Velocity-Flat-Anymal-C-Direct-v0"
    )
    # Core
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to ARC pre-trained checkpoint (dynamics initialisation)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--total_steps", type=int, default=100_000)
    # Rollout
    parser.add_argument("--rollout_length", type=int, default=32)
    # Lifecycle-specific
    parser.add_argument("--internal_steps", type=int, default=16,
                        help="ODE steps per physics timestep (sensory forcing)")
    parser.add_argument("--autonomous_steps", type=int, default=0,
                        help="Additional autonomous ODE steps after each observation")
    # PPO hypers
    parser.add_argument("--lr", type=float, default=1e-3)  # match rl_games baseline
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.02)  # higher than MLP's 0.005 to force exploration
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=1024)
    # Dynamics freeze/unfreeze
    parser.add_argument("--freeze_dynamics", action="store_true", default=True)
    parser.add_argument("--unfreeze_dynamics", action="store_true", default=False)
    # Curiosity (prediction error, NOT ||dh/dt||)
    parser.add_argument("--intrinsic_beta", type=float, default=0.0,
                        help="Intrinsic motivation coefficient (0 = standard PPO)")
    parser.add_argument("--beta_final", type=float, default=0.01,
                        help="Final intrinsic coefficient after decay")
    parser.add_argument("--beta_decay_fraction", type=float, default=0.5,
                        help="Fraction of training over which beta decays from init to final")
    parser.add_argument("--extrinsic_weight", type=float, default=1.0,
                        help="Weight on extrinsic reward (0 = curiosity-only mode)")
    parser.add_argument("--lambda_eff", type=float, default=0.001,
                        help="Initial efficiency regularizer weight")
    parser.add_argument("--lambda_eff_final", type=float, default=0.005,
                        help="Final efficiency regularizer weight")
    parser.add_argument("--lambda_eff_ramp_start", type=float, default=0.5,
                        help="Fraction of training when λ_eff starts ramping up")
    parser.add_argument("--reward_scale", type=float, default=0.6,
                        help="Reward scaling factor (rl_games uses 0.6)")
    parser.add_argument("--curiosity_min_steps", type=int, default=200,
                        help="Episode steps the agent must survive before curiosity activates")
    # Misc
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--no_compile", action="store_true", default=False,
                        help="Disable torch.compile (slower per step, instant startup)")
    parser.add_argument("--resume_isaac", type=str, default=None,
                        help="Resume from a previously saved lifecycle Isaac checkpoint")
    # Hierarchical tau: layered timescales (fast reflexes → slow consolidation)
    parser.add_argument("--hierarchical_tau", action="store_true", default=False,
                        help="Initialize tau_step_embed with hierarchical schedule")
    parser.add_argument("--tau_fast", type=float, default=0.2,
                        help="Target tau for early ODE steps (fast reactive)")
    parser.add_argument("--tau_slow", type=float, default=0.9,
                        help="Target tau for late ODE steps (slow consolidation)")
    parser.add_argument("--unfreeze_tau_step", type=int, default=0,
                        help="Unfreeze tau from step 0 (default: immediately)")
    # Standing penalty (deprecated — use push_interval instead)
    parser.add_argument("--standing_penalty", type=float, default=0.0,
                        help="DEPRECATED. Use --push_interval for intrinsic environmental richness")
    # Environmental perturbations: make standing boring, not penalized
    parser.add_argument("--push_interval", type=int, default=0,
                        help="Apply random push every N steps (0=off, 50=moderate, 20=frequent)")
    parser.add_argument("--push_force", type=float, default=1.0,
                        help="Max push velocity impulse in m/s (0.5=gentle, 2.0=strong)")
    # Rough terrain: varied terrain makes standing inherently boring
    parser.add_argument("--rough_terrain", action="store_true", default=False,
                        help="Use rough terrain (stairs, slopes, random) instead of flat")
    # Exploratory action init: large initial actions for locomotion discovery
    parser.add_argument("--exploratory_init", action="store_true", default=False,
                        help="Init action head with Xavier weights instead of zeros")
    parser.add_argument("--exploratory_scale", type=float, default=0.5,
                        help="Xavier gain for exploratory init (0.5=moderate, 1.0=full)")
    args = parser.parse_args()

    freeze = not args.unfreeze_dynamics

    # Isaac Lab requires AppLauncher to initialise before any simulation imports
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app  # noqa: F841 — must be kept alive

    # NOW we can import Isaac Lab / gym modules (after AppLauncher init)
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401 — registers all tasks

    # Environment selection: flat or rough terrain
    if args.rough_terrain:
        TASK = "Isaac-Velocity-Rough-Anymal-C-Direct-v0"
        N_ENTITIES = 14   # 13 + terrain token
        OBS_DIM = 235
        from isaaclab_tasks.direct.anymal_c.anymal_c_env_cfg import AnymalCRoughEnvCfg
        env_cfg = AnymalCRoughEnvCfg()
    else:
        TASK = "Isaac-Velocity-Flat-Anymal-C-Direct-v0"
        N_ENTITIES = 13
        OBS_DIM = 48
        from isaaclab_tasks.direct.anymal_c.anymal_c_env_cfg import AnymalCFlatEnvCfg
        env_cfg = AnymalCFlatEnvCfg()

    N_ACTUATED = 12
    ACTION_DIM = 12
    STATE_DIM = 16

    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(TASK, cfg=env_cfg)

    print(f"Environment: {TASK}")
    print(f"Obs space:   {env.observation_space}")
    print(f"Action space:{env.action_space}")
    print(f"Entities: {N_ENTITIES}, Actuated: {N_ACTUATED}")
    print(f"Terrain: {'rough' if args.rough_terrain else 'flat'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LiquidARC config — 5M model (d=768)
    config = LiquidARCConfig(
        d_model=768, d_metric=192, d_ffn=1536,
        n_ode_steps=16, tau_min=0.5, tau_max=1.0,
        t_diffusion_init=1.0, dropout=0.0,
        n_colors=10, n_roles=8, n_sep_types=4,
        max_grid_size=30, max_grids=16,
        max_seq_len=64,
    )
    config.integration_time = 2.0  # T=2.0 — stability via adaptive damping in dynamics
    config.persist_alpha = 1.0

    # Build ContinuousLifecycleRunner from ARC checkpoint
    model = ContinuousLifecycleRunner.from_pretrained(
        checkpoint_path=args.checkpoint,
        config=config,
        n_entities=N_ENTITIES,
        n_actuated=N_ACTUATED,
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        internal_steps=args.internal_steps,
        autonomous_steps=args.autonomous_steps,
        freeze_dynamics=freeze,
        device=str(device),
    )

    # Resume from a previous lifecycle Isaac checkpoint (full model)
    if args.resume_isaac:
        isaac_ckpt = torch.load(args.resume_isaac, map_location=device, weights_only=False)
        state = isaac_ckpt['model']
        # Strip _orig_mod. prefix from torch.compile'd checkpoints
        cleaned = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
        print(f"Resumed model from {args.resume_isaac}")

    # Hierarchical tau: layered timescales for locomotion discovery
    # Early ODE steps = fast reflexes (low tau), late steps = slow consolidation (high tau)
    if args.hierarchical_tau:
        model.dynamics.init_hierarchical_tau(
            tau_fast=args.tau_fast, tau_slow=args.tau_slow,
            n_steps=args.internal_steps,
        )
        # Unfreeze tau immediately so the hierarchy is active from step 0
        model.dynamics.freeze_tau = False
        print(f"Tau unfrozen with hierarchical init: {args.tau_fast}→{args.tau_slow}")
    elif args.unfreeze_tau_step == 0:
        # Unfreeze tau from start even without hierarchical init
        model.dynamics.freeze_tau = False
        print("Tau unfrozen (flat init)")

    # torch.compile the dynamics (same as discrete)
    if not args.no_compile and device.type == "cuda":
        model.dynamics = torch.compile(model.dynamics, mode="default", dynamic=True)
        print("torch.compile: dynamics compiled")

    # Exploratory action head init: large diverse initial actions for locomotion
    if args.exploratory_init:
        model.action_head.init_exploratory(scale=args.exploratory_scale)
        print(f"Action head: exploratory init (scale={args.exploratory_scale})")

    print(f"Frozen params:    {model.frozen_parameter_count():,}")
    print(f"Trainable params: {model.trainable_parameter_count():,}")

    # Wrap in policy (adds log_std parameter for Gaussian exploration noise)
    policy = GaussianPolicy(model, action_dim=ACTION_DIM).to(device)
    value_net = ValueNet(obs_dim=OBS_DIM, hidden=256).to(device)

    # Tokenizer: splits flat obs into entity tokens
    if args.rough_terrain:
        from liquid_arc.isaac_wrapper import AnymalRoughTokenizer
        tokenizer = AnymalRoughTokenizer(obs_dim=OBS_DIM)
    else:
        from liquid_arc.isaac_wrapper import AnymalTokenizer
        tokenizer = AnymalTokenizer(obs_dim=OBS_DIM)

    # Restore policy/value state from Isaac checkpoint if resuming
    if args.resume_isaac:
        if 'policy_log_std' in isaac_ckpt:
            policy.log_std.data = isaac_ckpt['policy_log_std']
        if 'value_net' in isaac_ckpt:
            value_net.load_state_dict(isaac_ckpt['value_net'])

    # Optimizer — 10x lower LR for dynamics when unfrozen to prevent NaN
    if not freeze:
        dynamics_params = list(model.dynamics.parameters())
        dynamics_ids = {id(p) for p in dynamics_params}
        new_params = [p for p in policy.parameters() if id(p) not in dynamics_ids]
        optimizer = torch.optim.Adam([
            {'params': new_params, 'lr': args.lr},
            {'params': dynamics_params, 'lr': args.lr * 0.1},
            {'params': list(value_net.parameters()), 'lr': args.lr},
        ])
        print(f"  Dynamics LR: {args.lr * 0.1:.1e} (0.1x)")
    else:
        optimizer = torch.optim.Adam([
            {'params': list(policy.parameters()), 'lr': args.lr},
            {'params': list(value_net.parameters()), 'lr': args.lr},
        ])

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
    num_envs = obs.shape[0]
    total_updates = args.total_steps // (num_envs * args.rollout_length)

    # Curiosity setup
    curiosity_normalizer = CuriosityNormalizer()
    use_curiosity = args.intrinsic_beta > 0

    print(f"\nTraining: {total_updates} updates, {num_envs} envs, "
          f"rollout={args.rollout_length}, freeze={'yes' if freeze else 'no'}")
    print(f"Effective batch: {num_envs * args.rollout_length}")
    print(f"internal_steps={args.internal_steps}, autonomous_steps={args.autonomous_steps}")
    if use_curiosity:
        print(f"Curiosity (prediction error): beta={args.intrinsic_beta}→{args.beta_final}, "
              f"decay over {args.beta_decay_fraction*100:.0f}% of training, "
              f"extrinsic_weight={args.extrinsic_weight}, "
              f"min_steps={args.curiosity_min_steps}")

    t0 = time.time()
    global_step = 0
    ep_rewards = torch.zeros(num_envs, device=device)
    ep_lengths = torch.zeros(num_envs, device=device)
    completed_rewards = []
    completed_lengths = []

    # Actuated indices are fixed for Anymal: joints 1-12 (token indices 1..12)
    actuated_indices = tokenizer.tokenize(obs[:1])['actuated_indices']

    for update in range(total_updates):
        # ------------------------------------------------------------------
        # Rollout collection — persistent _h across steps, reset on done
        # ------------------------------------------------------------------
        rollout_obs = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []
        rollout_pred_errors = []  # per-step mean prediction errors [B]

        for step in range(args.rollout_length):
            with torch.no_grad():
                tokens = tokenizer.tokenize(obs)

                if update == 0 and step == 0:
                    print(f"  DEBUG: obs={obs.shape}, "
                          f"state_features={tokens['state_features'].shape}, "
                          f"actuated_indices={actuated_indices.shape}")

                # model.step() uses persistent _h — this is the lifecycle difference
                dist, result = policy.get_action_dist(tokens, actuated_indices)

                if update == 0 and step == 0:
                    print(f"  DEBUG: actions={result['actions'].shape}, "
                          f"log_std={policy.log_std.shape}")

                actions = dist.sample()

                # NaN protection
                if torch.isnan(actions).any() or torch.isinf(actions).any():
                    actions = torch.zeros_like(actions)
                    print(f"  WARNING: NaN/Inf in actions at update={update} step={step}, zeroed")
                actions = actions.clamp(-10.0, 10.0)

                log_probs = dist.log_prob(actions).sum(dim=-1)
                if torch.isnan(log_probs).any():
                    log_probs = torch.zeros_like(log_probs)

                values = value_net(obs)

                # Prediction error per entity [B, N] → mean per env [B]
                pred_error = result['prediction_error'].mean(dim=-1)

            rollout_obs.append(obs)
            rollout_actions.append(actions)
            rollout_log_probs.append(log_probs)
            rollout_values.append(values)
            rollout_pred_errors.append(pred_error.detach())

            # Step environment
            obs_dict, rewards, terminated, truncated, infos = env.step(actions)
            obs_next = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            dones = terminated | truncated

            # Handle episode resets — zero _h for environments that terminated
            # This is the core lifecycle mechanism: the ODE state is cleared on death
            if dones.any():
                model.handle_resets(dones)

            # Environmental perturbations: make standing inherently unstable
            # Random velocity impulses create novel sensory patterns — standing is boring,
            # recovery is informationally rich. No penalty needed.
            if args.push_interval > 0 and global_step % args.push_interval == 0:
                unwrapped = env.unwrapped
                if hasattr(unwrapped, '_robot'):
                    robot = unwrapped._robot
                    n = robot.num_instances
                    # Random velocity impulse on root — like a physical push
                    vel = robot.data.root_vel_w.clone()
                    push_mag = args.push_force  # reused as velocity magnitude (m/s)
                    vel[:, 0] += (torch.rand(n, device=device) - 0.5) * 2 * push_mag
                    vel[:, 1] += (torch.rand(n, device=device) - 0.5) * 2 * push_mag
                    robot.write_root_velocity_to_sim(vel)

            # Curiosity-augmented reward uses prediction error (not ||dh/dt||)
            # Gated on episode survival: agent must walk first, then gets curiosity
            if use_curiosity:
                beta = get_beta(
                    update, total_updates,
                    args.intrinsic_beta, args.beta_final, args.beta_decay_fraction,
                )
                norm_curiosity = curiosity_normalizer.normalize(pred_error).clamp(-5, 5)
                alive_mask = (ep_lengths >= args.curiosity_min_steps).float()
                gated_curiosity = norm_curiosity * alive_mask
                total_reward = args.extrinsic_weight * rewards + beta * gated_curiosity
            else:
                total_reward = rewards

            # Reward scaling (rl_games baseline uses 0.6)
            total_reward = total_reward * args.reward_scale

            rollout_rewards.append(total_reward)
            rollout_dones.append(dones.float())

            # Track episode statistics
            ep_rewards += rewards
            ep_lengths += 1
            done_mask = dones.bool()
            if done_mask.any():
                completed_rewards.extend(ep_rewards[done_mask].tolist())
                completed_lengths.extend(ep_lengths[done_mask].tolist())
                ep_rewards[done_mask] = 0.0
                ep_lengths[done_mask] = 0.0

            global_step += num_envs
            obs = obs_next

        # Stack rollout tensors
        rollout_obs = torch.stack(rollout_obs)              # [T, B, obs_dim]
        rollout_actions = torch.stack(rollout_actions)      # [T, B, action_dim]
        rollout_log_probs = torch.stack(rollout_log_probs)  # [T, B]
        rollout_rewards = torch.stack(rollout_rewards)      # [T, B]
        rollout_dones = torch.stack(rollout_dones)          # [T, B]
        rollout_values = torch.stack(rollout_values)        # [T, B]
        rollout_pred_errors = torch.stack(rollout_pred_errors)  # [T, B]

        # Compute GAE advantages and returns
        advantages, returns = compute_gae(
            rollout_rewards, rollout_values, rollout_dones,
            args.gamma, args.gae_lambda,
        )

        # Flatten for minibatch sampling
        T, B = rollout_obs.shape[:2]
        flat_obs = rollout_obs.reshape(T * B, -1)
        flat_actions = rollout_actions.reshape(T * B, -1)
        flat_log_probs = rollout_log_probs.reshape(T * B)
        flat_advantages = advantages.reshape(T * B)
        flat_returns = returns.reshape(T * B)

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-8
        )

        # ------------------------------------------------------------------
        # PPO update epochs
        # ------------------------------------------------------------------
        total_samples = T * B
        for epoch in range(args.n_epochs):
            indices = torch.randperm(total_samples, device=device)
            for start in range(0, total_samples, args.minibatch_size):
                end = min(start + args.minibatch_size, total_samples)
                mb_idx = indices[start:end]

                mb_obs = flat_obs[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_old_log_probs = flat_log_probs[mb_idx]
                mb_advantages = flat_advantages[mb_idx]
                mb_returns = flat_returns[mb_idx]

                # Re-tokenize the minibatch observations
                mb_tokens = tokenizer.tokenize(mb_obs)

                # PPO re-evaluation (skip autonomous steps to avoid divergence)
                model._h = None
                new_log_probs, entropy, eval_result = policy.evaluate_actions(
                    mb_tokens, actuated_indices, mb_actions,
                )

                # Clipped surrogate objective
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                mb_values = value_net(mb_obs)
                value_loss = F.mse_loss(mb_values, mb_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Efficiency regularizer: penalize ||dh/dt||² to encourage
                # tau self-regulation. STABILIZING (opposite of curiosity).
                eff_cost = eval_result.get('efficiency_cost', torch.tensor(0.0, device=device))
                # Fixed λ — the prediction-error gate in the lifecycle runner
                # handles the explore/exploit tradeoff automatically
                efficiency_loss = args.lambda_eff * eff_cost

                loss = policy_loss + 2.0 * value_loss + args.entropy_coef * entropy_loss + efficiency_loss

                # NaN protection: skip minibatch update if loss is invalid
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  WARNING: NaN/Inf loss at update={update} epoch={epoch}, skipping")
                    continue

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
                optimizer.step()

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if update % args.log_every == 0:
            elapsed = time.time() - t0
            fps = global_step / max(elapsed, 1e-6)

            if completed_rewards:
                mean_reward = sum(completed_rewards[-100:]) / len(completed_rewards[-100:])
                mean_length = sum(completed_lengths[-100:]) / len(completed_lengths[-100:])
            else:
                mean_reward = 0.0
                mean_length = 0.0

            # Fresh diagnostics pass on a single env (lightweight)
            with torch.no_grad():
                diag_tokens = tokenizer.tokenize(obs[:1])
                diag_result = model.step(diag_tokens, actuated_indices)
                cv = diag_result['metric_cv']
                tau = diag_result['tau_mean']

            # Prediction error stats over last rollout
            pred_error_mean = rollout_pred_errors.mean().item()

            # Per-entity beta from SensoryForcing (body=0, feet=1..12 → summarised)
            beta_vals = diag_result['beta'].cpu()  # [N]
            beta_body = beta_vals[0].item()
            beta_feet = beta_vals[1:].mean().item()

            # Curiosity logging
            curiosity_str = ""
            if use_curiosity:
                beta_now = get_beta(
                    update, total_updates,
                    args.intrinsic_beta, args.beta_final, args.beta_decay_fraction,
                )
                curiosity_str = (
                    f" pred_err={pred_error_mean:.4f}"
                    f" beta_body={beta_body:.3f}"
                    f" beta_feet={beta_feet:.3f}"
                    f" curiosity_beta={beta_now:.4f}"
                )

            print(
                f"  [update={update:5d}] step={global_step:8d}"
                f" reward={mean_reward:7.2f}"
                f" ep_len={mean_length:6.0f}"
                f" fps={fps:6.0f}"
                f" cv={cv:.3f}"
                f" tau={tau:.3f}"
                f" pred_err={pred_error_mean:.4f}"
                f" beta_body={beta_body:.3f}"
                f" beta_feet={beta_feet:.3f}"
                f" policy_loss={policy_loss.item():.4f}"
                f" value_loss={value_loss.item():.4f}"
                f" log_std={policy.log_std.data.mean().item():.3f}"
                + (f" eff={eff_cost.item():.4f}" if args.lambda_eff > 0 else "")
                + (f" curiosity_beta={get_beta(update, total_updates, args.intrinsic_beta, args.beta_final, args.beta_decay_fraction):.4f}" if use_curiosity else "")
            )

        # Periodic checkpoint save (every 50 updates)
        if update > 0 and update % 50 == 0:
            save_dir = "/home/pokazge/liquid-arc/output_isaac"
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = f"{save_dir}/lifecycle_step_{global_step}.pt"
            torch.save({
                'model': model.state_dict(),
                'policy_log_std': policy.log_std.data,
                'value_net': value_net.state_dict(),
                'update': update,
                'step': global_step,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # -------------------------------------------------------------------------
    # Save final checkpoint
    # -------------------------------------------------------------------------
    save_path = "/home/pokazge/liquid-arc/output_isaac/lifecycle_anymal_final.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'policy_log_std': policy.log_std.data,
        'value_net': value_net.state_dict(),
    }, save_path)
    print(f"Checkpoint saved to {save_path}")

    print(f"\nTraining complete. {global_step:,} total steps.")
    if completed_rewards:
        last100 = completed_rewards[-100:]
        print(f"Final avg reward (last 100 episodes): {sum(last100)/len(last100):.2f}")

    env.close()


if __name__ == "__main__":
    main()
