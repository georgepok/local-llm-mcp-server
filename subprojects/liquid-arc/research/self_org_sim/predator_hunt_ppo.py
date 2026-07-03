"""Predator-Hunt with PPO + reward shaping (replaces REINFORCE).

Two changes vs `predator_hunt.py`:

1. **Hunter reward shaping**: per step, hunter gets +shaping_coef × (old_dist
   - new_dist) where dist = L_inf between agents. Positive when hunter closes
   distance. This converts sparse reward (catch only) → dense (closing too).

2. **PPO update** (multiple epochs per rollout, clipped objective, GAE):
   replaces REINFORCE's high-variance MC returns. Standard for sparse-reward
   continuous tasks; what handles the "9% catch rate" sparsity.

Run:
  python predator_hunt_ppo.py --policy_a flat --policy_b liquid_halt --train_steps 2000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Env (with hunter shaping reward)
# ---------------------------------------------------------------------------

class HuntEnv:
    """Hunter-Prey asymmetric roles + distance-closing shaping reward.

    Reward (per agent):
      Hunter: +remaining_steps on catch (sparse) +shaping_coef·(old_dist
              - new_dist) per step (dense, reward for closing)
      Prey:   +1/max_steps per step alive
    """

    def __init__(self, grid_size: int = 8, sensor_size: int = 5,
                 max_steps: int = 64, device: str = "cpu",
                 min_start_distance: int = 3,
                 shaping_coef: float = 0.05):
        assert sensor_size % 2 == 1
        self.G = grid_size
        self.S = sensor_size
        self.R = sensor_size // 2
        self.max_steps = max_steps
        self.min_start_distance = min_start_distance
        self.shaping_coef = shaping_coef
        self.device = device
        self.A = 5
        self.act_delta = torch.tensor([
            [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0],
        ], device=device, dtype=torch.long)

    def reset(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.B = batch_size
        self.pos_a = torch.randint(0, self.G, (batch_size, 2), device=self.device)
        self.pos_b = torch.zeros_like(self.pos_a)
        for b in range(batch_size):
            while True:
                p = torch.randint(0, self.G, (2,), device=self.device)
                if (p - self.pos_a[b]).abs().max().item() >= self.min_start_distance:
                    self.pos_b[b] = p
                    break
        self.role_a = (torch.rand(batch_size, device=self.device) < 0.5)
        self.role_b = ~self.role_a
        self.steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.alive_a = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        self.alive_b = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        self.caught = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        return self.get_obs()

    def _patch(self, self_pos, other_pos, self_alive, other_alive,
                self_role_hunter):
        B, S, R, G = self.B, self.S, self.R, self.G
        obs = torch.zeros(B, 4, S, S, device=self.device)
        obs[:, 0, R, R] = self_alive.float()
        di = torch.arange(-R, R + 1, device=self.device)
        dj = torch.arange(-R, R + 1, device=self.device)
        gi, gj = torch.meshgrid(di, dj, indexing="ij")
        agent_xy = self_pos.unsqueeze(1).unsqueeze(1)
        world = agent_xy + torch.stack([gi, gj], dim=-1)
        oob = ((world[..., 0] < 0) | (world[..., 0] >= G)
               | (world[..., 1] < 0) | (world[..., 1] >= G))
        obs[:, 2] = oob.float()
        rel = other_pos - self_pos
        visible = (rel.abs().max(dim=-1).values <= R) & other_alive
        si = (rel[:, 0] + R).clamp(0, S - 1)
        sj = (rel[:, 1] + R).clamp(0, S - 1)
        for b in range(B):
            if visible[b]:
                obs[b, 1, si[b], sj[b]] = 1.0
        obs[:, 3] = self_role_hunter.float().view(B, 1, 1).expand(B, S, S)
        return obs

    def get_obs(self):
        oa = self._patch(self.pos_a, self.pos_b, self.alive_a, self.alive_b,
                          self.role_a)
        ob = self._patch(self.pos_b, self.pos_a, self.alive_b, self.alive_a,
                          self.role_b)
        return oa, ob

    def step(self, action_a, action_b):
        B, G = self.B, self.G
        mask_a = self.alive_a.unsqueeze(-1)
        mask_b = self.alive_b.unsqueeze(-1)
        old_dist = (self.pos_a - self.pos_b).abs().max(dim=-1).values.float()

        new_a = (self.pos_a + self.act_delta[action_a]).clamp(0, G - 1)
        new_b = (self.pos_b + self.act_delta[action_b]).clamp(0, G - 1)
        new_a = torch.where(mask_a, new_a, self.pos_a)
        new_b = torch.where(mask_b, new_b, self.pos_b)

        new_dist = (new_a - new_b).abs().max(dim=-1).values.float()
        # Closing distance: positive when hunter got closer
        closing = old_dist - new_dist

        both_alive = self.alive_a & self.alive_b
        same_cell = (new_a == new_b).all(dim=-1)
        caught_now = both_alive & same_cell

        self.pos_a = new_a
        self.pos_b = new_b
        any_alive = self.alive_a | self.alive_b
        self.steps = self.steps + any_alive.long()

        remaining = (self.max_steps - self.steps).clamp(min=0).float()
        a_hunter = self.role_a
        b_hunter = self.role_b
        a_caught_b = caught_now & a_hunter
        b_caught_a = caught_now & b_hunter

        # Catch bonus
        hunter_reward_a = a_caught_b.float() * (remaining + 1.0)
        hunter_reward_b = b_caught_a.float() * (remaining + 1.0)
        # Shaping: hunter rewarded for closing distance (only while alive
        # and prey alive, no catch yet)
        shaping_a = a_hunter.float() * closing * self.shaping_coef * both_alive.float()
        shaping_b = b_hunter.float() * closing * self.shaping_coef * both_alive.float()
        # Prey survival reward
        prey_step = (1.0 / float(self.max_steps))
        prey_reward_a = (~a_hunter).float() * self.alive_a.float() * prey_step
        prey_reward_b = (~b_hunter).float() * self.alive_b.float() * prey_step

        reward_a = hunter_reward_a + shaping_a + prey_reward_a
        reward_b = hunter_reward_b + shaping_b + prey_reward_b

        self.caught = self.caught | caught_now
        ep_terminal = self.caught | (self.steps >= self.max_steps)
        prey_dies_a = caught_now & ~a_hunter
        prey_dies_b = caught_now & ~b_hunter
        self.alive_a = self.alive_a & ~prey_dies_a
        self.alive_b = self.alive_b & ~prey_dies_b

        oa, ob = self.get_obs()
        info = {"caught_now": caught_now, "steps": self.steps.clone()}
        return (oa, ob), (reward_a, reward_b), (ep_terminal, ep_terminal), info

    def outcomes(self):
        a_won = (self.role_a & self.caught) | (~self.role_a & ~self.caught)
        b_won = (self.role_b & self.caught) | (~self.role_b & ~self.caught)
        return a_won, b_won


# ---------------------------------------------------------------------------
# Policies (same as predator_hunt.py)
# ---------------------------------------------------------------------------

class FlatPolicy(nn.Module):
    def __init__(self, obs_dim: int, d: int = 64, n_actions: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
        )
        self.action_head = nn.Linear(d, n_actions)
        self.value_head = nn.Linear(d, 1)

    def forward(self, obs_flat):
        h = self.encoder(obs_flat)
        return self.action_head(h), self.value_head(h).squeeze(-1), {
            "steps_mean": torch.tensor(0.0, device=h.device),
            "ponder_cost": torch.tensor(0.0, device=h.device),
        }


class LiquidPolicy(nn.Module):
    def __init__(self, obs_dim: int, d: int = 64, n_actions: int = 5,
                 k_max: int = 8, halt_enabled: bool = False,
                 min_steps: int = 2, dt: float = 0.5):
        super().__init__()
        self.k_max = k_max
        self.halt_enabled = halt_enabled
        self.min_steps = min_steps
        self.dt = dt
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
        )
        self.drift = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(),
            nn.Linear(d * 2, d),
        )
        self.tau_raw = nn.Parameter(torch.zeros(d))
        if halt_enabled:
            self.halt_head = nn.Linear(d, 1)
            with torch.no_grad():
                self.halt_head.bias.fill_(-3.0)
        self.action_head = nn.Linear(d, n_actions)
        self.value_head = nn.Linear(d, 1)

    def forward(self, obs_flat):
        h = self.encoder(obs_flat)
        B = h.shape[0]
        tau = F.softplus(self.tau_raw) + 0.1
        steps_used = torch.zeros(B, 1, device=h.device)
        still_active = torch.ones(B, 1, device=h.device)
        ponder_cost = torch.zeros((), device=h.device)
        for k in range(self.k_max):
            target = self.drift(h)
            dh = (target - h) / tau
            h_new = h + self.dt * dh
            if self.halt_enabled:
                h = still_active * h_new + (1.0 - still_active) * h
                p_halt = torch.sigmoid(self.halt_head(h))
                steps_used = steps_used + still_active
                if k >= self.min_steps:
                    still_active = still_active * (1.0 - p_halt)
                ponder_cost = ponder_cost + still_active.mean()
            else:
                h = h_new
                steps_used = steps_used + 1.0
        if self.halt_enabled:
            ponder_cost = ponder_cost / self.k_max
        else:
            ponder_cost = torch.tensor(1.0, device=h.device)
        return self.action_head(h), self.value_head(h).squeeze(-1), {
            "steps_mean": steps_used.mean().detach(),
            "ponder_cost": ponder_cost.detach() if not torch.is_grad_enabled()
                else ponder_cost,
        }


def build_policy(name, obs_dim, d, k, device):
    if name == "flat":
        p = FlatPolicy(obs_dim, d=d)
    elif name == "liquid_fixed":
        p = LiquidPolicy(obs_dim, d=d, k_max=k, halt_enabled=False)
    elif name == "liquid_halt":
        p = LiquidPolicy(obs_dim, d=d, k_max=k, halt_enabled=True)
    else:
        raise ValueError(name)
    return p.to(device)


# ---------------------------------------------------------------------------
# PPO with GAE
# ---------------------------------------------------------------------------

def collect_rollouts(env, policy_a, policy_b, batch_size, max_steps):
    """Collect rollout WITHOUT gradient (saves old_logp for PPO ratio)."""
    obs_a, obs_b = env.reset(batch_size)
    bufs_a = {"obs": [], "act": [], "logp": [], "val": [], "rew": [], "done": []}
    bufs_b = {"obs": [], "act": [], "logp": [], "val": [], "rew": [], "done": []}
    info_steps_a, info_steps_b = 0.0, 0.0
    info_n = 0
    with torch.no_grad():
        for t in range(max_steps):
            oa = obs_a.reshape(batch_size, -1)
            ob = obs_b.reshape(batch_size, -1)
            la, va, ia = policy_a(oa)
            lb, vb, ib = policy_b(ob)
            da = torch.distributions.Categorical(logits=la)
            db = torch.distributions.Categorical(logits=lb)
            act_a = da.sample()
            act_b = db.sample()
            logp_a = da.log_prob(act_a)
            logp_b = db.log_prob(act_b)
            (next_a, next_b), (rew_a, rew_b), (done_a, done_b), _ = env.step(
                act_a, act_b)
            bufs_a["obs"].append(oa); bufs_a["act"].append(act_a)
            bufs_a["logp"].append(logp_a); bufs_a["val"].append(va)
            bufs_a["rew"].append(rew_a); bufs_a["done"].append(done_a.float())
            bufs_b["obs"].append(ob); bufs_b["act"].append(act_b)
            bufs_b["logp"].append(logp_b); bufs_b["val"].append(vb)
            bufs_b["rew"].append(rew_b); bufs_b["done"].append(done_b.float())
            info_steps_a += float(ia["steps_mean"].item())
            info_steps_b += float(ib["steps_mean"].item())
            info_n += 1
            obs_a, obs_b = next_a, next_b
            if done_a.all() and done_b.all():
                break
        # Final value bootstrap (V(s_T)) for GAE
        oa = obs_a.reshape(batch_size, -1)
        ob = obs_b.reshape(batch_size, -1)
        _, val_a_T, _ = policy_a(oa)
        _, val_b_T, _ = policy_b(ob)

    a_won, b_won = env.outcomes()
    rollout = {}
    for k, v in bufs_a.items():
        rollout[f"a_{k}"] = torch.stack(v)
    for k, v in bufs_b.items():
        rollout[f"b_{k}"] = torch.stack(v)
    rollout["a_val_T"] = val_a_T
    rollout["b_val_T"] = val_b_T
    rollout["a_won"] = a_won.float().mean().item()
    rollout["b_won"] = b_won.float().mean().item()
    rollout["caught_rate"] = env.caught.float().mean().item()
    rollout["mean_episode_length"] = env.steps.float().mean().item()
    rollout["a_inference_steps"] = info_steps_a / max(info_n, 1)
    rollout["b_inference_steps"] = info_steps_b / max(info_n, 1)
    return rollout


def gae(rew, done, val, val_T, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation. Inputs:
        rew, done, val: [T, B]
        val_T: [B] (bootstrap value at episode end)
    Returns: returns [T,B], advantages [T,B]"""
    T, B = rew.shape
    advantages = torch.zeros_like(rew)
    last_gae = torch.zeros(B, device=rew.device)
    for t in range(T - 1, -1, -1):
        next_val = val_T if t == T - 1 else val[t + 1]
        next_nonterminal = 1.0 - done[t]
        delta = rew[t] + gamma * next_val * next_nonterminal - val[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + val
    return returns, advantages


def ppo_update(policy, optim, obs, act, old_logp, returns, advantages, val_old,
                ponder_lambda=0.0, n_epochs=4, batch_size_mb=4096,
                clip_eps=0.2, vf_clip=0.2, ent_coef=0.01, vf_coef=0.5,
                obs_dim=None):
    """PPO update. Flattens [T,B,...] → [T*B,...] for shuffling.

    Returns dict of mean losses.
    """
    T, B = obs.shape[:2]
    obs_flat = obs.reshape(T * B, obs_dim)
    act_flat = act.reshape(T * B)
    old_logp_flat = old_logp.reshape(T * B).detach()
    returns_flat = returns.reshape(T * B).detach()
    adv_flat = advantages.reshape(T * B).detach()
    val_old_flat = val_old.reshape(T * B).detach()
    # Normalize advantages
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    N = T * B
    losses = []
    for epoch in range(n_epochs):
        perm = torch.randperm(N, device=obs.device)
        for start in range(0, N, batch_size_mb):
            idx = perm[start:start + batch_size_mb]
            mb_obs = obs_flat[idx]
            mb_act = act_flat[idx]
            mb_old_logp = old_logp_flat[idx]
            mb_returns = returns_flat[idx]
            mb_adv = adv_flat[idx]
            mb_val_old = val_old_flat[idx]
            new_logits, new_val, info = policy(mb_obs)
            new_dist = torch.distributions.Categorical(logits=new_logits)
            new_logp = new_dist.log_prob(mb_act)
            entropy = new_dist.entropy().mean()
            ratio = torch.exp(new_logp - mb_old_logp)
            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            pol_loss = -torch.min(unclipped, clipped).mean()
            # Value clipping
            val_clipped = mb_val_old + torch.clamp(new_val - mb_val_old,
                                                     -vf_clip, vf_clip)
            v1 = (new_val - mb_returns) ** 2
            v2 = (val_clipped - mb_returns) ** 2
            val_loss = torch.max(v1, v2).mean()
            loss = pol_loss + vf_coef * val_loss - ent_coef * entropy
            if ponder_lambda > 0 and "ponder_cost" in info:
                loss = loss + ponder_lambda * info["ponder_cost"]
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            losses.append(float(loss.item()))
    return {"loss": sum(losses) / max(len(losses), 1)}


def evaluate(env, policy_a, policy_b, n_episodes, max_steps):
    policy_a.eval(); policy_b.eval()
    with torch.no_grad():
        obs_a, obs_b = env.reset(n_episodes)
        steps_means_a, steps_means_b = [], []
        for t in range(max_steps):
            oa = obs_a.reshape(n_episodes, -1)
            ob = obs_b.reshape(n_episodes, -1)
            la, _, ia = policy_a(oa)
            lb, _, ib = policy_b(ob)
            act_a = la.argmax(dim=-1)
            act_b = lb.argmax(dim=-1)
            (obs_a, obs_b), _, (done_a, done_b), _ = env.step(act_a, act_b)
            steps_means_a.append(float(ia["steps_mean"].item()))
            steps_means_b.append(float(ib["steps_mean"].item()))
            if done_a.all() and done_b.all():
                break
        a_won, b_won = env.outcomes()
        a_was_hunter = env.role_a
        a_was_prey = ~env.role_a
        a_hunter_succ = (a_was_hunter & env.caught).float().sum().item()
        a_prey_succ = (a_was_prey & ~env.caught).float().sum().item()
        n_a_hunter = a_was_hunter.sum().item()
        n_a_prey = a_was_prey.sum().item()
        b_was_hunter = env.role_b
        b_was_prey = ~env.role_b
        b_hunter_succ = (b_was_hunter & env.caught).float().sum().item()
        b_prey_succ = (b_was_prey & ~env.caught).float().sum().item()
        n_b_hunter = b_was_hunter.sum().item()
        n_b_prey = b_was_prey.sum().item()
    policy_a.train(); policy_b.train()
    return {
        "a_winrate": a_won.float().mean().item(),
        "b_winrate": b_won.float().mean().item(),
        "caught_rate": env.caught.float().mean().item(),
        "mean_episode_length": env.steps.float().mean().item(),
        "a_hunter_winrate": a_hunter_succ / max(n_a_hunter, 1),
        "a_prey_winrate": a_prey_succ / max(n_a_prey, 1),
        "b_hunter_winrate": b_hunter_succ / max(n_b_hunter, 1),
        "b_prey_winrate": b_prey_succ / max(n_b_prey, 1),
        "a_inference_steps": sum(steps_means_a) / max(len(steps_means_a), 1),
        "b_inference_steps": sum(steps_means_b) / max(len(steps_means_b), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_a", choices=["flat", "liquid_fixed", "liquid_halt"],
                    default="flat")
    ap.add_argument("--policy_b", choices=["flat", "liquid_fixed", "liquid_halt"],
                    default="liquid_halt")
    ap.add_argument("--grid_size", type=int, default=8)
    ap.add_argument("--sensor_size", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=64)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--train_steps", type=int, default=2000)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--minibatch_size", type=int, default=4096)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--shaping_coef", type=float, default=0.05)
    ap.add_argument("--ponder_lambda", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_episodes", type=int, default=400)
    ap.add_argument("--final_eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="output_predator_ppo")
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    torch.manual_seed(args.seed)
    env = HuntEnv(grid_size=args.grid_size, sensor_size=args.sensor_size,
                    max_steps=args.max_steps, device=device,
                    shaping_coef=args.shaping_coef)
    obs_dim = 4 * args.sensor_size * args.sensor_size
    policy_a = build_policy(args.policy_a, obs_dim=obs_dim, d=args.d, k=args.k,
                              device=device)
    policy_b = build_policy(args.policy_b, obs_dim=obs_dim, d=args.d, k=args.k,
                              device=device)
    optim_a = torch.optim.AdamW(policy_a.parameters(), lr=args.lr)
    optim_b = torch.optim.AdamW(policy_b.parameters(), lr=args.lr)
    np_a = sum(p.numel() for p in policy_a.parameters() if p.requires_grad)
    np_b = sum(p.numel() for p in policy_b.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"A={args.policy_a} ({np_a:,})  vs  B={args.policy_b} ({np_b:,})  "
          f"obs_dim={obs_dim} d={args.d} K={args.k}  PPO ({args.ppo_epochs} ep) "
          f"shape={args.shaping_coef}")

    history = []
    t0 = time.time()
    for step in range(1, args.train_steps + 1):
        rollout = collect_rollouts(env, policy_a, policy_b, args.batch_size,
                                    args.max_steps)

        # Compute returns + advantages (per agent) using GAE
        ret_a, adv_a = gae(rollout["a_rew"], rollout["a_done"], rollout["a_val"],
                             rollout["a_val_T"], gamma=args.gamma, lam=args.lam)
        ret_b, adv_b = gae(rollout["b_rew"], rollout["b_done"], rollout["b_val"],
                             rollout["b_val_T"], gamma=args.gamma, lam=args.lam)
        ppo_update(policy_a, optim_a, rollout["a_obs"], rollout["a_act"],
                    rollout["a_logp"], ret_a, adv_a, rollout["a_val"],
                    ponder_lambda=args.ponder_lambda
                        if args.policy_a == "liquid_halt" else 0.0,
                    n_epochs=args.ppo_epochs,
                    batch_size_mb=args.minibatch_size,
                    clip_eps=args.clip_eps, ent_coef=args.ent_coef,
                    obs_dim=obs_dim)
        ppo_update(policy_b, optim_b, rollout["b_obs"], rollout["b_act"],
                    rollout["b_logp"], ret_b, adv_b, rollout["b_val"],
                    ponder_lambda=args.ponder_lambda
                        if args.policy_b == "liquid_halt" else 0.0,
                    n_epochs=args.ppo_epochs,
                    batch_size_mb=args.minibatch_size,
                    clip_eps=args.clip_eps, ent_coef=args.ent_coef,
                    obs_dim=obs_dim)

        if args.log_every > 0 and step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"  step {step:5d}  "
                  f"a_win {rollout['a_won']:.3f} b_win {rollout['b_won']:.3f} "
                  f"caught {rollout['caught_rate']:.3f} "
                  f"mean_ep {rollout['mean_episode_length']:5.1f}  "
                  f"n_a={rollout['a_inference_steps']:5.2f} "
                  f"n_b={rollout['b_inference_steps']:5.2f}  "
                  f"({elapsed:.0f}s)", flush=True)
        if step == 1 or step % args.eval_every == 0 or step == args.train_steps:
            ev = evaluate(env, policy_a, policy_b, args.eval_episodes,
                           args.max_steps)
            log = {"step": step, **ev}
            history.append(log)
            print(f"  step {step:5d}  EVAL  "
                  f"a_win {ev['a_winrate']:.3f} b_win {ev['b_winrate']:.3f} "
                  f"caught {ev['caught_rate']:.3f}  "
                  f"a_H {ev['a_hunter_winrate']:.3f} a_P {ev['a_prey_winrate']:.3f}  "
                  f"b_H {ev['b_hunter_winrate']:.3f} b_P {ev['b_prey_winrate']:.3f}  "
                  f"n_a={ev['a_inference_steps']:5.2f} "
                  f"n_b={ev['b_inference_steps']:5.2f}", flush=True)
    final = evaluate(env, policy_a, policy_b, args.final_eval, args.max_steps)
    print(f"\nFINAL  A={args.policy_a} vs B={args.policy_b}  "
          f"a_win={final['a_winrate']:.3f} b_win={final['b_winrate']:.3f}  "
          f"a_H={final['a_hunter_winrate']:.3f} a_P={final['a_prey_winrate']:.3f}  "
          f"b_H={final['b_hunter_winrate']:.3f} b_P={final['b_prey_winrate']:.3f}  "
          f"caught={final['caught_rate']:.3f}  ({time.time() - t0:.0f}s)")
    os.makedirs(args.out_dir, exist_ok=True)
    out = {"a": args.policy_a, "b": args.policy_b,
            "a_params": np_a, "b_params": np_b,
            "config": vars(args), "history": history, "final": final}
    out_path = os.path.join(args.out_dir,
                             f"ppo_{args.policy_a}_vs_{args.policy_b}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
