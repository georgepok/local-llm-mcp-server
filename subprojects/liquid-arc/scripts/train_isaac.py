"""Train LiquidARC as a robotics controller in Isaac Lab.

Simple PPO implementation — no external RL framework dependency.
Connects LiquidARC's pre-trained geometric ODE to Isaac Sim physics.

Usage:
    cd /home/pokazge/IsaacLab
    ./isaaclab.sh -p /home/pokazge/liquid-arc/scripts/train_isaac.py \
        --task Isaac-Cartpole-Direct-v0 \
        --checkpoint /home/pokazge/liquid-arc/output_30m/checkpoints/step_10000.pt \
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

# Note: Isaac Lab source patched for optional pxr/omni imports (Newton standalone)
# Force unbuffered output so logs appear in real-time
import functools
print = functools.partial(print, flush=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add liquid-arc to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.robotics_model import LiquidARCRoboticsModel


class CuriosityNormalizer:
    """Running normalization for intrinsic reward stability."""
    def __init__(self, decay=0.99):
        self.mean = 0.0
        self.var = 1.0
        self.decay = decay
        self.count = 0

    def normalize(self, curiosity):
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


def get_beta(update_idx, total_updates, beta_init, beta_final, decay_fraction):
    """Decay intrinsic motivation over training."""
    progress = min(update_idx / max(total_updates * decay_fraction, 1), 1.0)
    return beta_init * (1 - progress) + beta_final * progress


class GaussianPolicy(nn.Module):
    """Wraps LiquidARC model with a learned log_std for PPO."""

    def __init__(self, model: LiquidARCRoboticsModel, action_dim: int):
        super().__init__()
        self.model = model
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, tokens):
        result = self.model(**tokens)
        return result

    def get_action_dist(self, tokens):
        result = self.forward(tokens)
        mean = result['actions']
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist, result

    def evaluate_actions(self, tokens, actions):
        dist, result = self.get_action_dist(tokens)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, result


class ValueNet(nn.Module):
    """Simple value network operating on the same tokenized input."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute GAE advantages and returns."""
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

    returns = advantages + values
    return advantages, returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-Direct-v0")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--rollout_length", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=1024)
    parser.add_argument("--freeze_dynamics", action="store_true", default=True)
    parser.add_argument("--unfreeze_dynamics", action="store_true", default=False)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--resume_isaac", type=str, default=None,
                        help="Resume from a previously saved Isaac checkpoint (full model)")
    # Curiosity-augmented PPO
    parser.add_argument("--intrinsic_beta", type=float, default=0.0,
                        help="Intrinsic motivation coefficient (0=standard PPO)")
    parser.add_argument("--beta_final", type=float, default=0.01)
    parser.add_argument("--beta_decay_fraction", type=float, default=0.5,
                        help="Fraction of training over which beta decays")
    parser.add_argument("--extrinsic_weight", type=float, default=1.0,
                        help="Weight for extrinsic reward (0=curiosity-only)")
    parser.add_argument("--curiosity_min_steps", type=int, default=200,
                        help="Min episode steps before curiosity reward activates")
    parser.add_argument("--no_compile", action="store_true", default=False,
                        help="Disable torch.compile (slower per step, instant startup)")
    args = parser.parse_args()

    freeze = not args.unfreeze_dynamics

    # Isaac Lab requires its AppLauncher to initialize before any sim imports
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    # NOW we can import Isaac Lab modules (after AppLauncher init)
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401 — registers tasks

    # Task-specific configuration
    TASK_CONFIGS = {
        "Isaac-Cartpole-Direct-v0": {
            "cfg_module": "isaaclab_tasks.direct.cartpole.cartpole_env",
            "cfg_class": "CartpoleEnvCfg",
            "n_entities": 2, "n_actuated": 1, "action_dim": 1,
            "state_dim": 4, "obs_dim": 4,
            "tokenizer": "cartpole",
        },
        "Isaac-Velocity-Flat-Anymal-C-Direct-v0": {
            "cfg_module": "isaaclab_tasks.direct.anymal_c.anymal_c_env_cfg",
            "cfg_class": "AnymalCFlatEnvCfg",
            "n_entities": 13, "n_actuated": 12, "action_dim": 12,
            "state_dim": 16, "obs_dim": 48,
            "tokenizer": "anymal",
        },
        "Isaac-Humanoid-Direct-v0": {
            "cfg_module": "isaaclab_tasks.direct.humanoid.humanoid_env",
            "cfg_class": "HumanoidEnvCfg",
            "n_entities": 22, "n_actuated": 21, "action_dim": 21,
            "state_dim": 16, "obs_dim": 75,
            "tokenizer": "humanoid",
        },
    }

    task_cfg = TASK_CONFIGS.get(args.task)
    if task_cfg is None:
        raise ValueError(f"Unknown task: {args.task}. Supported: {list(TASK_CONFIGS.keys())}")

    # Create environment
    import importlib
    cfg_mod = importlib.import_module(task_cfg["cfg_module"])
    EnvCfgClass = getattr(cfg_mod, task_cfg["cfg_class"])
    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=env_cfg)

    obs_dim = task_cfg["obs_dim"]
    action_dim = task_cfg["action_dim"]
    print(f"Environment: {args.task}")
    print(f"Obs space: {env.observation_space}, Action space: {env.action_space}")
    print(f"Entities: {task_cfg['n_entities']}, Actuated: {task_cfg['n_actuated']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load LiquidARC config (5M)
    config = LiquidARCConfig(
        d_model=768, d_metric=192, d_ffn=1536,
        n_ode_steps=16, tau_min=0.5, tau_max=1.0,
        t_diffusion_init=1.0, dropout=0.0,
        n_colors=10, n_roles=8, n_sep_types=4,
        max_grid_size=30, max_grids=16,
        max_seq_len=64,
    )
    config.integration_time = 2.0
    config.persist_alpha = 1.0

    # Create model
    model = LiquidARCRoboticsModel.from_pretrained(
        checkpoint_path=args.checkpoint,
        config=config,
        action_dim=action_dim,
        n_entities=task_cfg["n_entities"],
        n_actuated=task_cfg["n_actuated"],
        state_dim_per_entity=task_cfg["state_dim"],
        freeze_dynamics=freeze,
        device=str(device),
    )

    # Resume from Isaac checkpoint if provided (overrides ARC checkpoint load)
    if args.resume_isaac:
        isaac_ckpt = torch.load(args.resume_isaac, map_location=device, weights_only=False)
        # Strip _orig_mod. prefix from torch.compile'd checkpoints
        state = isaac_ckpt['model']
        cleaned = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)
        print(f"Resumed full model from {args.resume_isaac}")

    # torch.compile the dynamics (TRITON_PTXAS_PATH must be set for SM 12.1)
    if not args.no_compile and device.type == "cuda":
        model.dynamics = torch.compile(model.dynamics, mode="default", dynamic=True)
        print("torch.compile: dynamics compiled")

    print(f"Frozen params: {model.frozen_parameter_count():,}")
    print(f"Trainable params: {model.trainable_parameter_count():,}")

    # Wrap in policy
    policy = GaussianPolicy(model, action_dim=action_dim).to(device)
    value_net = ValueNet(obs_dim=obs_dim, hidden=256).to(device)

    # Select tokenizer
    from liquid_arc.isaac_wrapper import CartpoleTokenizer, AnymalTokenizer, HumanoidTokenizer
    if task_cfg["tokenizer"] == "cartpole":
        tokenizer = CartpoleTokenizer()
    elif task_cfg["tokenizer"] == "anymal":
        tokenizer = AnymalTokenizer(obs_dim=obs_dim)
    elif task_cfg["tokenizer"] == "humanoid":
        tokenizer = HumanoidTokenizer(obs_dim=obs_dim)

    # Load policy/value state if resuming
    if args.resume_isaac:
        if 'policy_log_std' in isaac_ckpt:
            policy.log_std.data = isaac_ckpt['policy_log_std']
        if 'value_net' in isaac_ckpt:
            value_net.load_state_dict(isaac_ckpt['value_net'])

    # Optimizer — lower LR for dynamics when unfrozen to prevent NaN
    if not freeze:
        dynamics_params = list(model.dynamics.parameters())
        dynamics_ids = {id(p) for p in dynamics_params}
        new_params = [p for p in policy.parameters() if id(p) not in dynamics_ids]
        optimizer = torch.optim.Adam([
            {'params': new_params, 'lr': args.lr},
            {'params': dynamics_params, 'lr': args.lr * 0.1},  # 10x lower for dynamics
            {'params': list(value_net.parameters()), 'lr': args.lr},
        ])
        print(f"  Dynamics LR: {args.lr * 0.1:.1e} (0.1x)")
    else:
        optimizer = torch.optim.Adam([
            {'params': list(policy.parameters()), 'lr': args.lr},
            {'params': list(value_net.parameters()), 'lr': args.lr},
        ])

    # Training loop
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
    if use_curiosity:
        print(f"Curiosity: beta={args.intrinsic_beta}→{args.beta_final}, "
              f"decay over {args.beta_decay_fraction*100:.0f}% of training, "
              f"extrinsic_weight={args.extrinsic_weight}")

    t0 = time.time()
    global_step = 0
    ep_rewards = torch.zeros(num_envs, device=device)
    ep_lengths = torch.zeros(num_envs, device=device)
    completed_rewards = []
    completed_lengths = []

    for update in range(total_updates):
        # Collect rollout
        rollout_obs = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []

        for step in range(args.rollout_length):
            with torch.no_grad():
                tokens = tokenizer.tokenize(obs)
                if update == 0 and step == 0:
                    print(f"  DEBUG: obs={obs.shape}, state_features={tokens['state_features'].shape}, "
                          f"actuated_indices={tokens['actuated_indices'].shape}")
                dist, result = policy.get_action_dist(tokens)
                if update == 0 and step == 0:
                    print(f"  DEBUG: actions={result['actions'].shape}, log_std={policy.log_std.shape}")
                actions = dist.sample()
                # NaN protection: clamp actions and skip if model produces invalid values
                if torch.isnan(actions).any() or torch.isinf(actions).any():
                    actions = torch.zeros_like(actions)
                    print(f"  WARNING: NaN/Inf in actions at update={update} step={step}, zeroed")
                actions = actions.clamp(-10.0, 10.0)
                log_probs = dist.log_prob(actions).sum(dim=-1)
                if torch.isnan(log_probs).any():
                    log_probs = torch.zeros_like(log_probs)
                values = value_net(obs)

            rollout_obs.append(obs)
            rollout_actions.append(actions)
            rollout_log_probs.append(log_probs)
            rollout_values.append(values)

            # Step environment
            obs_dict, rewards, terminated, truncated, infos = env.step(actions)
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            dones = terminated | truncated

            # Curiosity-augmented reward (gated on episode survival)
            if use_curiosity:
                beta = get_beta(update, total_updates, args.intrinsic_beta,
                                args.beta_final, args.beta_decay_fraction)
                raw_curiosity = result['curiosity'].detach()
                norm_curiosity = curiosity_normalizer.normalize(raw_curiosity).clamp(-5, 5)
                # Only grant curiosity reward after surviving min_steps
                # Robot must learn to move FIRST, then gets rewarded for exploring
                alive_mask = (ep_lengths >= args.curiosity_min_steps).float()
                gated_curiosity = norm_curiosity * alive_mask
                total_reward = args.extrinsic_weight * rewards + beta * gated_curiosity
            else:
                total_reward = rewards
                raw_curiosity = None

            rollout_rewards.append(total_reward)
            rollout_dones.append(dones.float())

            # Track episode stats
            ep_rewards += rewards
            ep_lengths += 1
            done_mask = dones.bool()
            if done_mask.any():
                completed_rewards.extend(ep_rewards[done_mask].tolist())
                completed_lengths.extend(ep_lengths[done_mask].tolist())
                ep_rewards[done_mask] = 0
                ep_lengths[done_mask] = 0

            global_step += num_envs

        # Stack rollout
        rollout_obs = torch.stack(rollout_obs)        # [T, B, obs_dim]
        rollout_actions = torch.stack(rollout_actions)  # [T, B, action_dim]
        rollout_log_probs = torch.stack(rollout_log_probs)  # [T, B]
        rollout_rewards = torch.stack(rollout_rewards)  # [T, B]
        rollout_dones = torch.stack(rollout_dones)      # [T, B]
        rollout_values = torch.stack(rollout_values)    # [T, B]

        # Compute GAE
        advantages, returns = compute_gae(
            rollout_rewards, rollout_values, rollout_dones,
            args.gamma, args.gae_lambda)

        # Flatten
        T, B = rollout_obs.shape[:2]
        flat_obs = rollout_obs.reshape(T * B, -1)
        flat_actions = rollout_actions.reshape(T * B, -1)
        flat_log_probs = rollout_log_probs.reshape(T * B)
        flat_advantages = advantages.reshape(T * B)
        flat_returns = returns.reshape(T * B)

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # PPO update
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

                # Policy loss
                mb_tokens = tokenizer.tokenize(mb_obs)
                new_log_probs, entropy, _ = policy.evaluate_actions(mb_tokens, mb_actions)

                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                mb_values = value_net(mb_obs)
                value_loss = F.mse_loss(mb_values, mb_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = policy_loss + 0.5 * value_loss + args.entropy_coef * entropy_loss

                # NaN protection: skip update if loss is invalid
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  WARNING: NaN/Inf loss at update={update}, skipping")
                    continue

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
                optimizer.step()

        # Logging
        if update % args.log_every == 0:
            elapsed = time.time() - t0
            fps = global_step / max(elapsed, 1e-6)

            if completed_rewards:
                mean_reward = sum(completed_rewards[-100:]) / len(completed_rewards[-100:])
                mean_length = sum(completed_lengths[-100:]) / len(completed_lengths[-100:])
            else:
                mean_reward = 0.0
                mean_length = 0.0

            # Get model diagnostics from last forward pass
            with torch.no_grad():
                tokens = tokenizer.tokenize(obs[:1])
                result = policy.model(**tokens)
                cv = result['metric_cv']
                tau = result['tau_mean']

            curiosity_str = ""
            if use_curiosity:
                cur_mean = result['curiosity'].mean().item() if 'curiosity' in result else 0
                beta_now = get_beta(update, total_updates, args.intrinsic_beta,
                                    args.beta_final, args.beta_decay_fraction)
                curiosity_str = f" cur={cur_mean:.3f} beta={beta_now:.3f}"

            print(f"  [update={update}] step={global_step} reward={mean_reward:.1f} "
                  f"ep_len={mean_length:.0f} fps={fps:.0f} "
                  f"cv={cv:.3f} tau={tau:.3f} "
                  f"policy_loss={policy_loss.item():.4f} "
                  f"value_loss={value_loss.item():.4f} "
                  f"log_std={policy.log_std.data.mean().item():.3f}{curiosity_str}")

    # Save final checkpoint (full model — dynamics + embedding + action head)
    task_short = args.task.split("-")[1].lower()  # e.g., "cartpole" or "velocity"
    save_path = f"/home/pokazge/liquid-arc/output_isaac/{task_short}_final.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'policy_log_std': policy.log_std.data,
        'value_net': value_net.state_dict(),
    }, save_path)
    print(f"Checkpoint saved to {save_path}")

    print(f"\nTraining complete. {global_step} total steps.")
    if completed_rewards:
        print(f"Final avg reward (last 100): {sum(completed_rewards[-100:])/len(completed_rewards[-100:]):.1f}")

    env.close()


if __name__ == "__main__":
    main()
