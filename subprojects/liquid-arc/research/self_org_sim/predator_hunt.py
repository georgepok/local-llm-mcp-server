"""Predator-Hunt: asymmetric roles for cross-architecture comparison.

Design (resolves the pacifist-Nash collapse seen in predator_mixed.py):
  - Each episode randomly assigns one agent the role of HUNTER, the other PREY
  - Role is encoded in the observation (4th channel of the sensor patch)
  - Hunter wins by catching prey (same cell after both move)
  - Prey wins by surviving max_steps
  - Asymmetric rewards force engagement: hunter must chase, prey must dodge

Each policy network plays BOTH roles (role rotates per episode). Two
architectures matched head-to-head: A=Flat, B=Liquid_halt (or other pairs).
Aggregate win rate measures architectural advantage.

Variable inherent depth:
  - Hunter at distance: simple chase, low depth
  - Hunter near prey: predict prey's dodge, high depth
  - Prey at distance: random walk, low depth
  - Prey near hunter: anticipate hunter's attack, plan escape, high depth

Run:
  python predator_hunt.py --policy_a flat --policy_b liquid_halt --train_steps 5000
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
# Env
# ---------------------------------------------------------------------------

class HuntEnv:
    """Hunter-Prey asymmetric roles, random assignment per episode.

    Channels in sensor obs (4):
        0: self (1 at center if alive)
        1: opponent (1 at relative position if visible)
        2: wall / OOB
        3: role flag (1.0 broadcast across patch if you are HUNTER, else 0)

    Role assignment: per env in batch, role_a ∈ {0=prey, 1=hunter}; role_b
    is the complement.

    Catch: after both move, same cell ⇒ hunter caught prey (regardless of
    which moved into which). Position swap ⇒ pass-through, no catch.

    Rewards (per agent):
        Hunter: +1 × remaining_steps if catches prey on this step. 0 otherwise.
        Prey: +1 / max_steps per step alive (cumulative max = 1 if survives
              entire episode). Effectively a per-step normalised survival reward.
    """

    def __init__(self, grid_size: int = 8, sensor_size: int = 5,
                 max_steps: int = 64, device: str = "cpu",
                 min_start_distance: int = 3):
        assert sensor_size % 2 == 1
        self.G = grid_size
        self.S = sensor_size
        self.R = sensor_size // 2
        self.max_steps = max_steps
        self.min_start_distance = min_start_distance
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
        # Role assignment: role_a[b] = 1 if A is hunter, 0 if prey
        self.role_a = (torch.rand(batch_size, device=self.device) < 0.5)
        self.role_b = ~self.role_a  # exact complement
        self.steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.alive_a = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        self.alive_b = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        self.caught = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        return self.get_obs()

    def _patch(self, self_pos: torch.Tensor, other_pos: torch.Tensor,
                self_alive: torch.Tensor, other_alive: torch.Tensor,
                self_role_hunter: torch.Tensor) -> torch.Tensor:
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
        # Role flag: broadcast across all positions
        obs[:, 3] = self_role_hunter.float().view(B, 1, 1).expand(B, S, S)
        return obs

    def get_obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_a = self._patch(self.pos_a, self.pos_b, self.alive_a, self.alive_b,
                              self.role_a)
        obs_b = self._patch(self.pos_b, self.pos_a, self.alive_b, self.alive_a,
                              self.role_b)
        return obs_a, obs_b

    def step(self, action_a: torch.Tensor, action_b: torch.Tensor):
        B, G = self.B, self.G
        mask_a = self.alive_a.unsqueeze(-1)
        mask_b = self.alive_b.unsqueeze(-1)
        new_a = (self.pos_a + self.act_delta[action_a]).clamp(0, G - 1)
        new_b = (self.pos_b + self.act_delta[action_b]).clamp(0, G - 1)
        new_a = torch.where(mask_a, new_a, self.pos_a)
        new_b = torch.where(mask_b, new_b, self.pos_b)

        # Catch: after both move, they share a cell. Position swap is a
        # pass-through (no catch).
        both_alive = self.alive_a & self.alive_b
        same_cell = (new_a == new_b).all(dim=-1)
        caught_now = both_alive & same_cell
        # Position swap: prey "passed through" hunter — no catch.

        # Update positions
        self.pos_a = new_a
        self.pos_b = new_b
        any_alive = self.alive_a | self.alive_b
        self.steps = self.steps + any_alive.long()

        # Rewards
        # remaining: steps left after this one
        remaining = (self.max_steps - self.steps).clamp(min=0).float()
        # If A is hunter: a_reward includes catch bonus
        a_hunter = self.role_a
        b_hunter = self.role_b
        # Catch bonus: hunter gets remaining_steps reward. Prey gets 0.
        # If caught_now: catch happened
        a_caught_b = caught_now & a_hunter   # A was hunter, caught prey B
        b_caught_a = caught_now & b_hunter   # B was hunter, caught prey A
        # Hunter reward on catch: remaining
        hunter_reward_a = a_caught_b.float() * (remaining + 1.0)  # +1 for the step
        hunter_reward_b = b_caught_a.float() * (remaining + 1.0)
        # Prey reward per step: 1 / max_steps (so total ≤ 1.0 for full survival)
        prey_step = (1.0 / float(self.max_steps))
        prey_reward_a = (~a_hunter).float() * self.alive_a.float() * prey_step
        prey_reward_b = (~b_hunter).float() * self.alive_b.float() * prey_step

        reward_a = hunter_reward_a + prey_reward_a
        reward_b = hunter_reward_b + prey_reward_b

        # Episode terminates: caught (any) OR max_steps reached
        self.caught = self.caught | caught_now
        ep_terminal = self.caught | (self.steps >= self.max_steps)
        done_a = ep_terminal
        done_b = ep_terminal
        # Mark dead post-catch (so they don't keep accumulating)
        # In hunter-prey: prey is "killed" on catch; hunter "succeeds"
        prey_dies = caught_now & ~a_hunter   # A is prey
        prey_dies_b = caught_now & ~b_hunter # B is prey
        self.alive_a = self.alive_a & ~prey_dies
        self.alive_b = self.alive_b & ~prey_dies_b

        obs_a, obs_b = self.get_obs()
        info = {"caught_now": caught_now, "steps": self.steps.clone()}
        return (obs_a, obs_b), (reward_a, reward_b), (done_a, done_b), info

    def outcomes(self):
        """At episode end:
            a_hunter_caught: A was hunter and caught prey
            a_prey_survived: A was prey and survived (not caught)
            b_hunter_caught, b_prey_survived: same for B
        Each agent "wins" iff they succeeded in their role.
        """
        a_was_hunter = self.role_a
        a_was_prey = ~self.role_a
        a_won = (a_was_hunter & self.caught) | (a_was_prey & ~self.caught)
        b_was_hunter = self.role_b
        b_was_prey = ~self.role_b
        b_won = (b_was_hunter & self.caught) | (b_was_prey & ~self.caught)
        # In this asymmetric framing: exactly one wins per episode.
        # If caught: hunter wins, prey loses. If timeout: prey wins, hunter loses.
        return a_won, b_won


# ---------------------------------------------------------------------------
# Policies
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

    def forward(self, obs_flat: torch.Tensor):
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

    def forward(self, obs_flat: torch.Tensor):
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


def build_policy(name: str, obs_dim: int, d: int, k: int, device: str) -> nn.Module:
    if name == "flat":
        p = FlatPolicy(obs_dim, d=d)
    elif name == "liquid_fixed":
        p = LiquidPolicy(obs_dim, d=d, k_max=k, halt_enabled=False)
    elif name == "liquid_halt":
        p = LiquidPolicy(obs_dim, d=d, k_max=k, halt_enabled=True)
    else:
        raise ValueError(f"Unknown policy: {name}")
    return p.to(device)


# ---------------------------------------------------------------------------
# REINFORCE
# ---------------------------------------------------------------------------

def collect_rollouts(env: HuntEnv, policy_a: nn.Module, policy_b: nn.Module,
                     batch_size: int, max_steps: int):
    obs_a, obs_b = env.reset(batch_size)
    bufs_a = {"obs": [], "act": [], "logp": [], "val": [], "rew": [], "done": []}
    bufs_b = {"obs": [], "act": [], "logp": [], "val": [], "rew": [], "done": []}
    info_steps_a, info_steps_b = 0.0, 0.0
    info_n = 0
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
        (next_obs_a, next_obs_b), (rew_a, rew_b), (done_a, done_b), _ = env.step(
            act_a, act_b)
        bufs_a["obs"].append(oa); bufs_a["act"].append(act_a)
        bufs_a["logp"].append(logp_a); bufs_a["val"].append(va)
        bufs_a["rew"].append(rew_a); bufs_a["done"].append(done_a.float())
        bufs_b["obs"].append(ob); bufs_b["act"].append(act_b)
        bufs_b["logp"].append(logp_b); bufs_b["val"].append(vb)
        bufs_b["rew"].append(rew_b); bufs_b["done"].append(done_b.float())
        info_steps_a += float(ia.get("steps_mean").item())
        info_steps_b += float(ib.get("steps_mean").item())
        info_n += 1
        obs_a, obs_b = next_obs_a, next_obs_b
        if done_a.all() and done_b.all():
            break
    a_won, b_won = env.outcomes()
    rollout = {}
    for k, v in bufs_a.items():
        rollout[f"a_{k}"] = torch.stack(v)
    for k, v in bufs_b.items():
        rollout[f"b_{k}"] = torch.stack(v)
    rollout["a_won"] = a_won.float().mean().item()
    rollout["b_won"] = b_won.float().mean().item()
    rollout["caught_rate"] = env.caught.float().mean().item()
    rollout["mean_episode_length"] = env.steps.float().mean().item()
    rollout["a_inference_steps"] = info_steps_a / max(info_n, 1)
    rollout["b_inference_steps"] = info_steps_b / max(info_n, 1)
    # Per-role outcomes — useful for debugging
    a_was_hunter = env.role_a
    a_hunter_succ = (a_was_hunter & env.caught).float().mean().item()
    a_was_prey = ~env.role_a
    a_prey_succ = (a_was_prey & ~env.caught).float().mean().item()
    rollout["a_hunter_succ_rate"] = a_hunter_succ * 2.0   # normalize: half episodes had A as hunter
    rollout["a_prey_succ_rate"] = a_prey_succ * 2.0
    return rollout


def compute_returns(rewards, dones, gamma=0.99):
    T, B = rewards.shape
    returns = torch.zeros_like(rewards)
    G = torch.zeros(B, device=rewards.device)
    for t in range(T - 1, -1, -1):
        G = rewards[t] + gamma * G * (1.0 - dones[t])
        returns[t] = G
    return returns


def reinforce_loss(rew, done, val, logp, gamma=0.99,
                   ponder_lambda=0.0, ponder_cost=None):
    returns = compute_returns(rew, done, gamma)
    adv = (returns - val).detach()
    adv_norm = (adv - adv.mean()) / (adv.std() + 1e-6)
    pre_done = torch.zeros_like(done)
    pre_done[1:] = done[:-1]
    active = (1.0 - pre_done.cumsum(dim=0).clamp(0, 1)).detach()
    pol_loss = -(logp * adv_norm * active).sum() / active.sum().clamp(min=1)
    val_loss = (((returns - val) ** 2) * active).sum() / active.sum().clamp(min=1)
    entropy = -((logp.exp() * logp) * active).sum() / active.sum().clamp(min=1)
    total = pol_loss + 0.5 * val_loss - 0.01 * entropy
    if ponder_lambda > 0 and ponder_cost is not None:
        total = total + ponder_lambda * ponder_cost
    return total


def evaluate(env: HuntEnv, policy_a, policy_b, n_episodes: int, max_steps: int):
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
        a_hunter_succ = (a_was_hunter & env.caught).float().sum().item()
        a_was_prey = ~env.role_a
        a_prey_succ = (a_was_prey & ~env.caught).float().sum().item()
        n_a_hunter = a_was_hunter.sum().item()
        n_a_prey = a_was_prey.sum().item()
    policy_a.train(); policy_b.train()
    return {
        "a_winrate": a_won.float().mean().item(),
        "b_winrate": b_won.float().mean().item(),
        "caught_rate": env.caught.float().mean().item(),
        "mean_episode_length": env.steps.float().mean().item(),
        "a_hunter_winrate": a_hunter_succ / max(n_a_hunter, 1),
        "a_prey_winrate": a_prey_succ / max(n_a_prey, 1),
        "b_hunter_winrate": (~env.role_a & env.caught).float().sum().item() / max((~env.role_a).sum().item(), 1),
        "b_prey_winrate": (env.role_a & ~env.caught).float().sum().item() / max(env.role_a.sum().item(), 1),
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
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--train_steps", type=int, default=5000)
    ap.add_argument("--ponder_lambda", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_episodes", type=int, default=400)
    ap.add_argument("--final_eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="output_predator_hunt")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    torch.manual_seed(args.seed)
    env = HuntEnv(grid_size=args.grid_size, sensor_size=args.sensor_size,
                    max_steps=args.max_steps, device=device)
    obs_dim = 4 * args.sensor_size * args.sensor_size  # 4 channels now
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
          f"obs_dim={obs_dim} d={args.d} K={args.k}")

    history = []
    t0 = time.time()
    for step in range(1, args.train_steps + 1):
        rollout = collect_rollouts(env, policy_a, policy_b,
                                    args.batch_size, args.max_steps)
        pc_a, pc_b = None, None
        if args.policy_a == "liquid_halt":
            _, _, ia = policy_a(rollout["a_obs"].reshape(-1, obs_dim))
            pc_a = ia["ponder_cost"]
        if args.policy_b == "liquid_halt":
            _, _, ib = policy_b(rollout["b_obs"].reshape(-1, obs_dim))
            pc_b = ib["ponder_cost"]
        loss_a = reinforce_loss(rollout["a_rew"], rollout["a_done"],
                                  rollout["a_val"], rollout["a_logp"],
                                  gamma=args.gamma,
                                  ponder_lambda=args.ponder_lambda
                                      if args.policy_a == "liquid_halt" else 0.0,
                                  ponder_cost=pc_a)
        loss_b = reinforce_loss(rollout["b_rew"], rollout["b_done"],
                                  rollout["b_val"], rollout["b_logp"],
                                  gamma=args.gamma,
                                  ponder_lambda=args.ponder_lambda
                                      if args.policy_b == "liquid_halt" else 0.0,
                                  ponder_cost=pc_b)
        optim_a.zero_grad(); loss_a.backward()
        torch.nn.utils.clip_grad_norm_(policy_a.parameters(), 1.0)
        optim_a.step()
        optim_b.zero_grad(); loss_b.backward()
        torch.nn.utils.clip_grad_norm_(policy_b.parameters(), 1.0)
        optim_b.step()

        if args.log_every > 0 and step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"  step {step:5d}  "
                  f"a_win {rollout['a_won']:.3f} b_win {rollout['b_won']:.3f} "
                  f"caught {rollout['caught_rate']:.3f} "
                  f"mean_ep {rollout['mean_episode_length']:5.1f}  "
                  f"n_a={rollout['a_inference_steps']:5.2f} "
                  f"n_b={rollout['b_inference_steps']:5.2f} "
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
                             f"hunt_{args.policy_a}_vs_{args.policy_b}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
