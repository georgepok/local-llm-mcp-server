"""Mixed-architecture duel: Flat vs Liquid in the same environment.

Pivots from `predator_duel.py` (symmetric self-play, mutual death on
collision) to **asymmetric outcomes**: on collision, a random tiebreak
picks one survivor. Each agent has its own architecture, own policy
network, own optimizer. Both train simultaneously by playing against
each other — co-evolution.

The metric is **win rate**: in 1000 evaluation episodes, what fraction
does each architecture survive longer? If Liquid (with adaptive depth)
beats Flat, the architecture provides a real advantage in this dynamic.

Run:
  python predator_mixed.py --policy_a flat --policy_b liquid_halt --train_steps 5000
  python predator_mixed.py --policy_a flat --policy_b liquid_fixed --train_steps 5000
  python predator_mixed.py --policy_a liquid_fixed --policy_b liquid_halt --train_steps 5000
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
# Asymmetric duel environment
# ---------------------------------------------------------------------------

class MixedDuelEnv:
    """2 agents, asymmetric outcomes (random tiebreak on collision).

    State: pos_a [B,2], pos_b [B,2], alive_a [B], alive_b [B], steps [B].
    Episode ends when EITHER dies, OR max_steps reached.

    Reward structure (forces conflict — vanilla +1/step gives mutual-avoidance Nash):
      alive_bonus per step: small constant (0.01)
      kill_bonus when opponent dies in collision (you survived): proportional
        to remaining steps. Killing early ≫ killing late ≫ surviving alone.

    Default grid_size=8 (small enough that 5x5 sensor + min_start_distance=3
    forces near-immediate engagement; mutual avoidance impossible).
    """

    def __init__(self, grid_size: int = 8, sensor_size: int = 5,
                 max_steps: int = 200, device: str = "cpu",
                 min_start_distance: int = 3,
                 alive_bonus: float = 0.01,
                 kill_bonus_per_remaining: float = 1.0):
        assert sensor_size % 2 == 1
        self.G = grid_size
        self.S = sensor_size
        self.R = sensor_size // 2
        self.max_steps = max_steps
        self.min_start_distance = min_start_distance
        self.device = device
        self.alive_bonus = alive_bonus
        self.kill_bonus_per_remaining = kill_bonus_per_remaining
        self.A = 5
        self.act_delta = torch.tensor([
            [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0],
        ], device=device, dtype=torch.long)

    def reset(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.B = batch_size
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
        self.alive_a = torch.ones(
            batch_size, dtype=torch.bool, device=self.device)
        self.alive_b = torch.ones(
            batch_size, dtype=torch.bool, device=self.device)
        return self.get_obs()

    def _patch(self, self_pos: torch.Tensor, other_pos: torch.Tensor,
                self_alive: torch.Tensor,
                other_alive: torch.Tensor) -> torch.Tensor:
        B, S, R, G = self.B, self.S, self.R, self.G
        obs = torch.zeros(B, 3, S, S, device=self.device)
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
        return obs

    def get_obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (obs_a [B,3,S,S], obs_b [B,3,S,S])."""
        obs_a = self._patch(self.pos_a, self.pos_b, self.alive_a, self.alive_b)
        obs_b = self._patch(self.pos_b, self.pos_a, self.alive_b, self.alive_a)
        return obs_a, obs_b

    def step(self, action_a: torch.Tensor, action_b: torch.Tensor):
        """Returns (obs_a, obs_b), (reward_a, reward_b), (done_a, done_b), info."""
        B, G = self.B, self.G
        # Both move simultaneously, gated by alive
        mask_a = self.alive_a.unsqueeze(-1)
        mask_b = self.alive_b.unsqueeze(-1)
        new_a = (self.pos_a + self.act_delta[action_a]).clamp(0, G - 1)
        new_b = (self.pos_b + self.act_delta[action_b]).clamp(0, G - 1)
        new_a = torch.where(mask_a, new_a, self.pos_a)
        new_b = torch.where(mask_b, new_b, self.pos_b)

        # Collision detection: both alive AND (same cell OR position swap)
        both_alive = self.alive_a & self.alive_b
        same_cell = (new_a == new_b).all(dim=-1)
        swapped = ((new_a == self.pos_b).all(dim=-1)
                   & (new_b == self.pos_a).all(dim=-1))
        collision = both_alive & (same_cell | swapped)

        # Random tiebreak: 0 → A dies, 1 → B dies
        coin = (torch.rand(B, device=self.device) < 0.5)
        a_dies_this_step = collision & coin
        b_dies_this_step = collision & ~coin

        # Reward = alive_bonus * (alive at step entry) + kill_bonus when
        # opponent dies this step. The kill_bonus is proportional to the
        # remaining episode steps so early kills ≫ late kills.
        remaining = (self.max_steps - self.steps - 1).clamp(min=0).float()
        reward_a = self.alive_a.float() * self.alive_bonus
        reward_b = self.alive_b.float() * self.alive_bonus
        reward_a = reward_a + b_dies_this_step.float() * remaining * self.kill_bonus_per_remaining
        reward_b = reward_b + a_dies_this_step.float() * remaining * self.kill_bonus_per_remaining

        # Update positions
        self.pos_a = new_a
        self.pos_b = new_b

        # Tick step counter while either is alive (paired episode)
        any_alive = self.alive_a | self.alive_b
        self.steps = self.steps + any_alive.long()

        # Update alive flags
        self.alive_a = self.alive_a & ~a_dies_this_step
        self.alive_b = self.alive_b & ~b_dies_this_step

        # Done: episode ends if either died, OR max_steps reached
        # Once one dies, the other could in principle continue alone, but
        # there's no opponent — terminate cleanly.
        ep_terminal = (~self.alive_a | ~self.alive_b
                       | (self.steps >= self.max_steps))
        done_a = ep_terminal
        done_b = ep_terminal

        obs_a, obs_b = self.get_obs()
        info = {
            "collision": collision,
            "a_died": a_dies_this_step,
            "b_died": b_dies_this_step,
            "steps": self.steps.clone(),
        }
        return (obs_a, obs_b), (reward_a, reward_b), (done_a, done_b), info

    def winners(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """At episode end:
          a_won: alive_a & ~alive_b
          b_won: alive_b & ~alive_a
          tie:   alive_a == alive_b (both alive at timeout, or both dead)
        """
        a_won = self.alive_a & ~self.alive_b
        b_won = self.alive_b & ~self.alive_a
        tie = self.alive_a == self.alive_b
        return a_won, b_won, tie


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
# Rollout, REINFORCE per agent
# ---------------------------------------------------------------------------

def collect_rollouts(env: MixedDuelEnv, policy_a: nn.Module,
                     policy_b: nn.Module, batch_size: int, max_steps: int):
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

    a_won, b_won, tie = env.winners()
    rollout = {}
    for k, v in bufs_a.items():
        rollout[f"a_{k}"] = torch.stack(v)
    for k, v in bufs_b.items():
        rollout[f"b_{k}"] = torch.stack(v)
    rollout["a_won"] = a_won.float().mean().item()
    rollout["b_won"] = b_won.float().mean().item()
    rollout["tie"] = tie.float().mean().item()
    rollout["ep_lengths"] = env.steps.float().cpu()
    rollout["a_inference_steps"] = info_steps_a / max(info_n, 1)
    rollout["b_inference_steps"] = info_steps_b / max(info_n, 1)
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


def evaluate(env: MixedDuelEnv, policy_a, policy_b, n_episodes: int,
             max_steps: int):
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
        a_won, b_won, tie = env.winners()
    policy_a.train(); policy_b.train()
    return {
        "a_winrate": a_won.float().mean().item(),
        "b_winrate": b_won.float().mean().item(),
        "tie_rate": tie.float().mean().item(),
        "mean_episode_length": env.steps.float().mean().item(),
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
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--train_steps", type=int, default=5000)
    ap.add_argument("--ponder_lambda", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_episodes", type=int, default=400)
    ap.add_argument("--final_eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="output_predator_mixed")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    torch.manual_seed(args.seed)
    env = MixedDuelEnv(grid_size=args.grid_size, sensor_size=args.sensor_size,
                        max_steps=args.max_steps, device=device)
    obs_dim = 3 * args.sensor_size * args.sensor_size
    policy_a = build_policy(args.policy_a, obs_dim=obs_dim, d=args.d, k=args.k,
                              device=device)
    policy_b = build_policy(args.policy_b, obs_dim=obs_dim, d=args.d, k=args.k,
                              device=device)
    optim_a = torch.optim.AdamW(policy_a.parameters(), lr=args.lr)
    optim_b = torch.optim.AdamW(policy_b.parameters(), lr=args.lr)
    np_a = sum(p.numel() for p in policy_a.parameters() if p.requires_grad)
    np_b = sum(p.numel() for p in policy_b.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"A={args.policy_a} ({np_a:,} params)  vs  "
          f"B={args.policy_b} ({np_b:,} params)  obs_dim={obs_dim} d={args.d}")

    history = []
    t0 = time.time()
    for step in range(1, args.train_steps + 1):
        rollout = collect_rollouts(env, policy_a, policy_b,
                                    args.batch_size, args.max_steps)

        # Per-agent ponder costs (recompute on grad)
        pc_a, pc_b = None, None
        if args.policy_a == "liquid_halt":
            _, _, ia = policy_a(rollout["a_obs"].reshape(-1, obs_dim))
            pc_a = ia["ponder_cost"]
        if args.policy_b == "liquid_halt":
            _, _, ib = policy_b(rollout["b_obs"].reshape(-1, obs_dim))
            pc_b = ib["ponder_cost"]

        loss_a = reinforce_loss(
            rollout["a_rew"], rollout["a_done"], rollout["a_val"],
            rollout["a_logp"], gamma=args.gamma,
            ponder_lambda=args.ponder_lambda if args.policy_a == "liquid_halt"
                else 0.0,
            ponder_cost=pc_a,
        )
        loss_b = reinforce_loss(
            rollout["b_rew"], rollout["b_done"], rollout["b_val"],
            rollout["b_logp"], gamma=args.gamma,
            ponder_lambda=args.ponder_lambda if args.policy_b == "liquid_halt"
                else 0.0,
            ponder_cost=pc_b,
        )
        optim_a.zero_grad(); loss_a.backward()
        torch.nn.utils.clip_grad_norm_(policy_a.parameters(), 1.0)
        optim_a.step()
        optim_b.zero_grad(); loss_b.backward()
        torch.nn.utils.clip_grad_norm_(policy_b.parameters(), 1.0)
        optim_b.step()

        if args.log_every > 0 and step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"  step {step:5d}  "
                  f"a_win {rollout['a_won']:.3f}  b_win {rollout['b_won']:.3f}  "
                  f"tie {rollout['tie']:.3f}  "
                  f"mean_ep {rollout['ep_lengths'].mean():6.2f}  "
                  f"n_a={rollout['a_inference_steps']:5.2f} "
                  f"n_b={rollout['b_inference_steps']:5.2f}  "
                  f"({elapsed:.0f}s)", flush=True)
        if step == 1 or step % args.eval_every == 0 or step == args.train_steps:
            ev = evaluate(env, policy_a, policy_b, args.eval_episodes,
                           args.max_steps)
            log = {"step": step, **ev}
            history.append(log)
            print(f"  step {step:5d}  EVAL  "
                  f"a_win {ev['a_winrate']:.3f} "
                  f"b_win {ev['b_winrate']:.3f} "
                  f"tie {ev['tie_rate']:.3f}  "
                  f"mean_ep {ev['mean_episode_length']:6.2f}  "
                  f"n_a={ev['a_inference_steps']:5.2f} "
                  f"n_b={ev['b_inference_steps']:5.2f}", flush=True)
    final = evaluate(env, policy_a, policy_b, args.final_eval, args.max_steps)
    print(f"\nFINAL  A={args.policy_a} vs B={args.policy_b}  "
          f"a_win={final['a_winrate']:.3f}  b_win={final['b_winrate']:.3f}  "
          f"tie={final['tie_rate']:.3f}  "
          f"mean_ep={final['mean_episode_length']:6.2f}  "
          f"n_a={final['a_inference_steps']:5.2f} "
          f"n_b={final['b_inference_steps']:5.2f}  ({time.time() - t0:.0f}s)")
    os.makedirs(args.out_dir, exist_ok=True)
    out = {"a": args.policy_a, "b": args.policy_b,
            "a_params": np_a, "b_params": np_b,
            "config": vars(args), "history": history, "final": final}
    out_path = os.path.join(args.out_dir,
                             f"mixed_{args.policy_a}_vs_{args.policy_b}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
