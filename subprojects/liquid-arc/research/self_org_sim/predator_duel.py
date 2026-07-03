"""Predator-Duel: two agents with shared policy, mutual avoidance for survival.

Pivots from `predator_survival.py` (1 agent vs greedy NPCs) to symmetric
self-play: 2 agents, same policy network, each tries to outlive the other.
Collision = both die (same cell, or position swap). Reward: +1 per step
alive. Confined 16x16 grid forces eventual encounter — pure avoidance is
not viable forever.

Why this is a better test of adaptive depth:
  - When opponent is far → simple movement, low depth needed
  - When opponent is medium-range → must predict their next move, medium depth
  - Near walls / corners → multi-step planning to avoid being herded
  - Co-evolution: as the shared policy gets better, opponent gets better,
    forcing deeper play

The symmetric setup means each rollout yields trajectories from BOTH
agents' perspectives, doubling the training signal per environment step.

Run:
  python predator_duel.py --policy flat --train_steps 5000
  python predator_duel.py --policy liquid_fixed --k 8 --train_steps 5000
  python predator_duel.py --policy liquid_halt --k 16 --train_steps 5000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DuelEnv:
    """Batched 2-agent duel on a grid. Both agents share a policy.

    Observation per agent: sensor_size x sensor_size x 3 patch around self.
        ch0: self (always 1 at center if alive)
        ch1: opponent (1 at the relative position if visible & alive)
        ch2: wall / out-of-bounds
    """

    def __init__(self, grid_size: int = 16, sensor_size: int = 5,
                 max_steps: int = 200, device: str = "cpu",
                 min_start_distance: int = 4):
        assert sensor_size % 2 == 1
        self.G = grid_size
        self.S = sensor_size
        self.R = sensor_size // 2
        self.max_steps = max_steps
        self.min_start_distance = min_start_distance
        self.device = device
        self.A = 5  # actions: U,D,L,R,STAY
        self.act_delta = torch.tensor([
            [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0],
        ], device=device, dtype=torch.long)

    def reset(self, batch_size: int) -> torch.Tensor:
        """Return [2B, 3, S, S] — agent A's obs concatenated with B's."""
        self.B = batch_size
        # Place 2 agents per env, with min L_inf distance
        self.pos_a = torch.randint(
            0, self.G, (batch_size, 2), device=self.device)
        self.pos_b = torch.zeros_like(self.pos_a)
        for b in range(batch_size):
            while True:
                p = torch.randint(0, self.G, (2,), device=self.device)
                if (p - self.pos_a[b]).abs().max().item() >= self.min_start_distance:
                    self.pos_b[b] = p
                    break
        self.steps = torch.zeros(
            batch_size, dtype=torch.long, device=self.device)
        self.alive = torch.ones(
            batch_size, dtype=torch.bool, device=self.device)
        return self.get_obs()

    def _patch(self, self_pos: torch.Tensor, other_pos: torch.Tensor,
                alive: torch.Tensor) -> torch.Tensor:
        """Build [B, 3, S, S] sensor patches from `self_pos`, with `other_pos`
        as the visible opponent."""
        B, S, R, G = self.B, self.S, self.R, self.G
        obs = torch.zeros(B, 3, S, S, device=self.device)
        obs[:, 0, R, R] = alive.float()
        di = torch.arange(-R, R + 1, device=self.device)
        dj = torch.arange(-R, R + 1, device=self.device)
        gi, gj = torch.meshgrid(di, dj, indexing="ij")
        agent_xy = self_pos.unsqueeze(1).unsqueeze(1)
        world = agent_xy + torch.stack([gi, gj], dim=-1)
        oob = ((world[..., 0] < 0) | (world[..., 0] >= G)
               | (world[..., 1] < 0) | (world[..., 1] >= G))
        obs[:, 2] = oob.float()
        rel = other_pos - self_pos
        visible = (rel.abs().max(dim=-1).values <= R) & alive
        si = (rel[:, 0] + R).clamp(0, S - 1)
        sj = (rel[:, 1] + R).clamp(0, S - 1)
        for b in range(B):
            if visible[b]:
                obs[b, 1, si[b], sj[b]] = 1.0
        return obs

    def get_obs(self) -> torch.Tensor:
        """Return [2B, 3, S, S]. First B rows are A's view, next B are B's view."""
        obs_a = self._patch(self.pos_a, self.pos_b, self.alive)
        obs_b = self._patch(self.pos_b, self.pos_a, self.alive)
        return torch.cat([obs_a, obs_b], dim=0)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                    torch.Tensor, dict]:
        """`actions` is [2B] long: first B are A's, next B are B's.

        Returns:
            obs: [2B, 3, S, S]
            reward: [2B] — +1 if alive at start of step
            done: [2B] — True if not alive after step
            info: dict
        """
        B, G = self.B, self.G
        a_act = actions[:B]
        b_act = actions[B:]
        # Compute deltas, gate by alive (dead agents don't move)
        mask = self.alive.unsqueeze(-1)
        new_a = (self.pos_a + self.act_delta[a_act]).clamp(0, G - 1)
        new_b = (self.pos_b + self.act_delta[b_act]).clamp(0, G - 1)
        new_a = torch.where(mask, new_a, self.pos_a)
        new_b = torch.where(mask, new_b, self.pos_b)

        # Reward computed BEFORE death this step (alive at step entry)
        reward = self.alive.float()

        # Collision detection (only matters for alive episodes):
        #  (a) same cell after move
        #  (b) position swap (they passed through each other)
        same_cell = (new_a == new_b).all(dim=-1)
        swapped = (new_a == self.pos_b).all(dim=-1) & (new_b == self.pos_a).all(dim=-1)
        collision = (same_cell | swapped) & self.alive

        # Update positions
        self.pos_a = new_a
        self.pos_b = new_b

        # Tick step counter only for alive episodes
        self.steps = self.steps + self.alive.long()

        # Both die on collision
        self.alive = self.alive & ~collision

        # Done flag per agent: episode ends when not alive or max_steps reached
        ep_done = ~self.alive | (self.steps >= self.max_steps)
        done = torch.cat([ep_done, ep_done], dim=0)
        # Reward [2B]: same value for both agents (symmetric)
        reward2 = torch.cat([reward, reward], dim=0)

        obs = self.get_obs()
        info = {
            "collision": collision,
            "steps": self.steps.clone(),
            "any_alive": self.alive.any().item(),
        }
        return obs, reward2, done, info


# ---------------------------------------------------------------------------
# Policies (same as predator_survival.py)
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
            "steps_mean": torch.tensor(0.0, device=h.device)}


class LiquidPolicy(nn.Module):
    """Continuous-time recurrent policy with optional adaptive halting."""

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
            "steps_min": steps_used.min().detach(),
            "steps_max": steps_used.max().detach(),
            "ponder_cost": ponder_cost.detach() if not torch.is_grad_enabled()
                else ponder_cost,
        }


# ---------------------------------------------------------------------------
# Rollouts + REINFORCE
# ---------------------------------------------------------------------------

def collect_rollouts(env: DuelEnv, policy: nn.Module, batch_size: int,
                     max_steps: int):
    obs = env.reset(batch_size)
    obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = (
        [], [], [], [], [], [])
    info_steps_sum = 0.0
    info_n = 0
    for t in range(max_steps):
        obs_flat = obs.reshape(2 * batch_size, -1)
        logits, value, info = policy(obs_flat)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        next_obs, reward, done, _ = env.step(action)
        obs_buf.append(obs_flat)
        act_buf.append(action)
        logp_buf.append(logp)
        val_buf.append(value)
        rew_buf.append(reward)
        done_buf.append(done.float())
        info_steps_sum += float(info.get("steps_mean", torch.tensor(0.0)).item())
        info_n += 1
        obs = next_obs
        if done.all():
            break
    ep_lengths = env.steps.float().cpu()
    avg_steps = info_steps_sum / max(info_n, 1)
    return {
        "obs": torch.stack(obs_buf),
        "act": torch.stack(act_buf),
        "logp": torch.stack(logp_buf),
        "val": torch.stack(val_buf),
        "rew": torch.stack(rew_buf),
        "done": torch.stack(done_buf),
        "ep_lengths": ep_lengths,
        "avg_inference_steps": avg_steps,
    }


def compute_returns(rewards: torch.Tensor, dones: torch.Tensor,
                    gamma: float = 0.99) -> torch.Tensor:
    T, B = rewards.shape
    returns = torch.zeros_like(rewards)
    G = torch.zeros(B, device=rewards.device)
    for t in range(T - 1, -1, -1):
        G = rewards[t] + gamma * G * (1.0 - dones[t])
        returns[t] = G
    return returns


def reinforce_loss(rollout: dict, gamma: float = 0.99,
                   ponder_lambda: float = 0.0,
                   ponder_cost: Optional[torch.Tensor] = None):
    rew = rollout["rew"]
    done = rollout["done"]
    val = rollout["val"]
    logp = rollout["logp"]
    returns = compute_returns(rew, done, gamma)
    # Advantage normalized for variance reduction
    adv = (returns - val).detach()
    adv_mean = adv.mean()
    adv_std = adv.std() + 1e-6
    adv_norm = (adv - adv_mean) / adv_std
    # Active mask: 1 for steps before/at episode end, 0 after
    pre_done = torch.zeros_like(done)
    pre_done[1:] = done[:-1]
    active = (1.0 - pre_done.cumsum(dim=0).clamp(0, 1)).detach()
    pol_loss = -(logp * adv_norm * active).sum() / active.sum().clamp(min=1)
    val_loss = (((returns - val) ** 2) * active).sum() / active.sum().clamp(min=1)
    entropy = -((logp.exp() * logp) * active).sum() / active.sum().clamp(min=1)
    total = pol_loss + 0.5 * val_loss - 0.01 * entropy
    if ponder_lambda > 0 and ponder_cost is not None:
        total = total + ponder_lambda * ponder_cost
    return total, {
        "pol_loss": pol_loss.detach(),
        "val_loss": val_loss.detach(),
        "entropy": entropy.detach(),
        "mean_return": returns.mean().detach(),
    }


def evaluate(env: DuelEnv, policy: nn.Module, n_episodes: int,
             max_steps: int) -> dict:
    policy.eval()
    with torch.no_grad():
        obs = env.reset(n_episodes)
        steps_means = []
        for t in range(max_steps):
            obs_flat = obs.reshape(2 * n_episodes, -1)
            logits, _, info = policy(obs_flat)
            action = logits.argmax(dim=-1)
            obs, _, done, _ = env.step(action)
            steps_means.append(float(info.get("steps_mean", torch.tensor(0.0)).item()))
            if done.all():
                break
        ep_lengths = env.steps.float().cpu()
    policy.train()
    return {
        "mean_episode_length": ep_lengths.mean().item(),
        "median_episode_length": ep_lengths.median().item(),
        "max_episode_length": ep_lengths.max().item(),
        "min_episode_length": ep_lengths.min().item(),
        "p10": ep_lengths.kthvalue(max(1, int(0.10 * n_episodes)))[0].item(),
        "p90": ep_lengths.kthvalue(max(1, int(0.90 * n_episodes)))[0].item(),
        "avg_inference_steps": (sum(steps_means) / max(len(steps_means), 1)),
    }


def build_policy(name: str, obs_dim: int, d: int, k: int,
                  device: str) -> nn.Module:
    if name == "flat":
        p = FlatPolicy(obs_dim, d=d)
    elif name == "liquid_fixed":
        p = LiquidPolicy(obs_dim, d=d, k_max=k, halt_enabled=False)
    elif name == "liquid_halt":
        p = LiquidPolicy(obs_dim, d=d, k_max=k, halt_enabled=True)
    else:
        raise ValueError(f"Unknown policy: {name}")
    return p.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["flat", "liquid_fixed", "liquid_halt"],
                    default="flat")
    ap.add_argument("--grid_size", type=int, default=16)
    ap.add_argument("--sensor_size", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--train_steps", type=int, default=5000)
    ap.add_argument("--ponder_lambda", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_episodes", type=int, default=200)
    ap.add_argument("--final_eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="output_predator_duel")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    torch.manual_seed(args.seed)
    env = DuelEnv(grid_size=args.grid_size, sensor_size=args.sensor_size,
                   max_steps=args.max_steps, device=device)
    obs_dim = 3 * args.sensor_size * args.sensor_size
    policy = build_policy(args.policy, obs_dim=obs_dim, d=args.d, k=args.k,
                           device=device)
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"Policy: {args.policy} | params: {n_params:,} | "
          f"obs_dim: {obs_dim} | d: {args.d} | "
          f"K: {args.k if args.policy != 'flat' else 0}")

    history = []
    t0 = time.time()
    losses_window = []
    for step in range(1, args.train_steps + 1):
        rollout = collect_rollouts(env, policy, args.batch_size, args.max_steps)
        if args.policy == "liquid_halt":
            obs_flat = rollout["obs"].reshape(-1, obs_dim)
            _, _, info = policy(obs_flat)
            ponder_cost = info["ponder_cost"]
        else:
            ponder_cost = None
        loss, metrics = reinforce_loss(
            rollout, gamma=args.gamma,
            ponder_lambda=args.ponder_lambda if args.policy == "liquid_halt"
                else 0.0,
            ponder_cost=ponder_cost,
        )
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optim.step()
        losses_window.append(float(loss.item()))
        if len(losses_window) > 100:
            losses_window = losses_window[-100:]
        if args.log_every > 0 and step % args.log_every == 0:
            avg_loss = sum(losses_window) / len(losses_window)
            mean_ep = float(rollout["ep_lengths"].mean().item())
            elapsed = time.time() - t0
            n_used = rollout.get("avg_inference_steps", 0.0)
            print(f"  step {step:5d}  loss {avg_loss:+.4f}  "
                  f"mean_ep {mean_ep:6.2f}  "
                  f"n_used {n_used:5.2f}  ({elapsed:.0f}s)", flush=True)
        if step == 1 or step % args.eval_every == 0 or step == args.train_steps:
            ev = evaluate(env, policy, n_episodes=args.eval_episodes,
                           max_steps=args.max_steps)
            log = {"step": step, "loss": sum(losses_window) / len(losses_window),
                    **ev}
            history.append(log)
            print(f"  step {step:5d}  EVAL  "
                  f"mean_ep {ev['mean_episode_length']:6.2f}  "
                  f"median {ev['median_episode_length']:6.1f}  "
                  f"p10 {ev['p10']:5.1f}  "
                  f"p90 {ev['p90']:5.1f}  "
                  f"n_used {ev['avg_inference_steps']:5.2f}", flush=True)
    final = evaluate(env, policy, n_episodes=args.final_eval,
                      max_steps=args.max_steps)
    print(f"\nFINAL  policy={args.policy}  "
          f"mean_ep={final['mean_episode_length']:6.2f}  "
          f"median={final['median_episode_length']:6.1f}  "
          f"p10={final['p10']:5.1f}  p90={final['p90']:5.1f}  "
          f"n_used={final['avg_inference_steps']:5.2f}  "
          f"params={n_params:,}  ({time.time() - t0:.0f}s)")
    os.makedirs(args.out_dir, exist_ok=True)
    out = {"policy": args.policy, "params": n_params, "config": vars(args),
            "history": history, "final": final}
    out_path = os.path.join(args.out_dir,
                             f"duel_{args.policy}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
