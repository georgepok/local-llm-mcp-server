"""Predator-Survival: minimal robotic environment for testing adaptive depth.

Hypothesis: in a survival task, the inherent computational depth required to
choose a good action varies with situational difficulty. When predators are
far, any random move is fine (low depth needed). When predators are 1-2
cells away with walls behind, planning escape requires more iteration.

This file is self-contained:
- PredatorSurvivalEnv: batched 16x16 gridworld with greedy predators
- Three policy networks: Flat MLP, Fixed-K Liquid (continuous-time RNN),
  Adaptive-Halt Liquid (same + halt head + ponder cost)
- REINFORCE with value baseline
- Comparison: survival rate over evaluation episodes

Run:
    python predator_survival.py --policy flat --train_steps 5000
    python predator_survival.py --policy liquid_fixed --k 8 --train_steps 5000
    python predator_survival.py --policy liquid_halt --k_max 16 --train_steps 5000

The metric: mean episode length (capped at max_steps) over 1000 eval episodes.
Random policy baseline: ~5-10 steps before predator catches up.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PredatorSurvivalEnv:
    """Batched predator-survival gridworld.

    State: agent position (b, 2), predator positions (b, n_pred, 2),
    step counter (b,), alive flag (b,).

    Predators move toward agent each step (greedy L_inf, ties broken
    randomly). Agent dies if a predator lands on its cell.

    Observation: sensor_size x sensor_size patch around agent, 3 channels:
        ch0: self (always 1 at center)
        ch1: predator presence (1 if any predator in this cell)
        ch2: wall / out-of-bounds (1 if outside grid)

    Reward: +1 per timestep alive. Episode terminates on death or max_steps.
    """

    def __init__(self, grid_size: int = 16, n_predators: int = 2,
                 sensor_size: int = 5, max_steps: int = 200,
                 device: str = "cpu"):
        assert sensor_size % 2 == 1
        self.G = grid_size
        self.P = n_predators
        self.S = sensor_size
        self.R = sensor_size // 2  # sensor radius
        self.max_steps = max_steps
        self.device = device
        self.A = 5  # actions: up, down, left, right, stay
        self.act_delta = torch.tensor([
            [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0],
        ], device=device, dtype=torch.long)

    def reset(self, batch_size: int) -> torch.Tensor:
        self.B = batch_size
        # Random agent positions
        self.agent_pos = torch.randint(
            0, self.G, (batch_size, 2), device=self.device)
        # Random predator positions, must not overlap with agent and must
        # have L_inf distance >= 3 from agent at start (give agent time)
        self.pred_pos = torch.zeros(
            batch_size, self.P, 2, dtype=torch.long, device=self.device)
        for p in range(self.P):
            for b in range(batch_size):
                while True:
                    pos = torch.randint(0, self.G, (2,), device=self.device)
                    d = (pos - self.agent_pos[b]).abs().max().item()
                    if d >= 3:
                        # Also check no overlap with previously placed preds
                        ok = True
                        for q in range(p):
                            if torch.equal(pos, self.pred_pos[b, q]):
                                ok = False
                                break
                        if ok:
                            self.pred_pos[b, p] = pos
                            break
        self.steps = torch.zeros(
            batch_size, dtype=torch.long, device=self.device)
        self.alive = torch.ones(
            batch_size, dtype=torch.bool, device=self.device)
        return self.get_obs()

    def get_obs(self) -> torch.Tensor:
        """Build sensor patch [B, 3, S, S]. Agent always at (R, R).

        ch0: self (1 at center for alive agents)
        ch1: predator presence in sensor window
        ch2: wall / OOB flag
        """
        B, S, R, G = self.B, self.S, self.R, self.G
        obs = torch.zeros(B, 3, S, S, device=self.device)
        # ch0: self at center
        obs[:, 0, R, R] = self.alive.float()
        # ch1, ch2: scan grid
        # Compute world coords of each sensor cell: (agent + (di, dj))
        # for di, dj in [-R, R].
        di = torch.arange(-R, R + 1, device=self.device)
        dj = torch.arange(-R, R + 1, device=self.device)
        gi, gj = torch.meshgrid(di, dj, indexing="ij")
        # World position for each sensor cell, broadcast over batch
        # [B, S, S, 2]
        agent_xy = self.agent_pos.unsqueeze(1).unsqueeze(1)  # [B,1,1,2]
        world = agent_xy + torch.stack([gi, gj], dim=-1)
        # Wall mask: cells outside grid
        oob = ((world[..., 0] < 0) | (world[..., 0] >= G)
               | (world[..., 1] < 0) | (world[..., 1] >= G))
        obs[:, 2] = oob.float()
        # Predator mask: for each predator, check if any sensor cell matches
        for p in range(self.P):
            pp = self.pred_pos[:, p]  # [B, 2]
            # Relative offset from agent
            rel = pp - self.agent_pos  # [B, 2]
            # Predator visible iff |rel_i| <= R for both axes
            visible = (rel.abs().max(dim=-1).values <= R)
            # Sensor cell index = rel + R  (clipped if not visible)
            si = (rel[:, 0] + R).clamp(0, S - 1)
            sj = (rel[:, 1] + R).clamp(0, S - 1)
            for b in range(B):
                if visible[b] and self.alive[b]:
                    obs[b, 1, si[b], sj[b]] = 1.0
        return obs  # [B, 3, S, S]

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                    torch.Tensor, dict]:
        """actions: [B] long. Returns (obs, reward, done, info)."""
        B, G = self.B, self.G
        # Move agent (clamped to grid)
        delta = self.act_delta[actions]
        new_pos = (self.agent_pos + delta).clamp(0, G - 1)
        # Only move if alive
        mask = self.alive.unsqueeze(-1)
        self.agent_pos = torch.where(mask, new_pos, self.agent_pos)
        # Check collision after agent moves (predator may have stepped on
        # agent's old cell, but we check after both move below)
        self.steps = self.steps + self.alive.long()
        # Move predators greedily toward agent
        # delta = sign(agent_pos - pred_pos), L_inf greedy
        for p in range(self.P):
            target_diff = self.agent_pos - self.pred_pos[:, p]  # [B, 2]
            # Step: each axis moves +1 / -1 / 0 toward agent
            pred_step = target_diff.sign().clamp(-1, 1)
            # When agent on same row/col, predator moves diagonally if needed
            # (sign=0 contribution). This is fine — keep simple.
            new_pred = (self.pred_pos[:, p] + pred_step).clamp(0, G - 1)
            self.pred_pos[:, p] = torch.where(
                mask, new_pred, self.pred_pos[:, p])
        # Check collisions: any predator at agent's position
        died = torch.zeros(B, dtype=torch.bool, device=self.device)
        for p in range(self.P):
            died = died | (self.pred_pos[:, p] == self.agent_pos).all(dim=-1)
        died = died & self.alive
        # Reward: +1 per timestep alive (granted before death this step)
        reward = self.alive.float()
        # Update alive
        self.alive = self.alive & ~died
        # Done if dead OR exceeded max_steps
        done = ~self.alive | (self.steps >= self.max_steps)
        # Build next obs
        obs = self.get_obs()
        info = {"died": died, "steps": self.steps.clone()}
        return obs, reward, done, info


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

class FlatPolicy(nn.Module):
    """Standard MLP policy. No iteration."""

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
        logits = self.action_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value, {"steps_mean": torch.tensor(0.0)}


class LiquidPolicy(nn.Module):
    """Continuous-time recurrent policy with optional halting.

    h_0 = encoder(obs); for k in 0..K-1: h_{k+1} = h + dt/tau * (f(h) - h).
    LTC-style contraction; tau is per-dim adaptive (positive via softplus).

    If halt_enabled: a halt head outputs p_halt(h_k); after min_steps,
    still_active *= (1 - p_halt). Effective output is the per-batch
    weighted h once still_active drops, but for simplicity we keep the
    soft-stop formulation and report mean steps_used.
    """

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
        # ODE drift: f(h) returns target for LTC: dh/dt = (1/tau)(f - h)
        self.drift = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(),
            nn.Linear(d * 2, d),
        )
        # Per-dim time constant tau via softplus on raw param
        self.tau_raw = nn.Parameter(torch.zeros(d))  # softplus(0) ≈ 0.69
        if halt_enabled:
            self.halt_head = nn.Linear(d, 1)
            # Bias halt head to output small p_halt at init (don't terminate
            # immediately on random dynamics)
            with torch.no_grad():
                self.halt_head.bias.fill_(-3.0)  # sigmoid(-3) ≈ 0.047
        self.action_head = nn.Linear(d, n_actions)
        self.value_head = nn.Linear(d, 1)

    def forward(self, obs_flat: torch.Tensor):
        h = self.encoder(obs_flat)  # [B, d]
        B, d = h.shape
        tau = F.softplus(self.tau_raw) + 0.1  # safe positive
        steps_used = torch.zeros(B, 1, device=h.device)
        still_active = torch.ones(B, 1, device=h.device)
        ponder_cost = torch.zeros((), device=h.device)

        for k in range(self.k_max):
            target = self.drift(h)
            dh = (target - h) / tau
            h_new = h + self.dt * dh
            if self.halt_enabled:
                # Freeze halted-out states — only update active ones
                h = still_active * h_new + (1.0 - still_active) * h
                p_halt = torch.sigmoid(self.halt_head(h))
                # Track soft expected step count
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

        logits = self.action_head(h)
        value = self.value_head(h).squeeze(-1)
        info = {
            "steps_mean": steps_used.mean().detach(),
            "steps_min": steps_used.min().detach(),
            "steps_max": steps_used.max().detach(),
            "ponder_cost": ponder_cost.detach() if not torch.is_grad_enabled()
                else ponder_cost,
        }
        return logits, value, info


# ---------------------------------------------------------------------------
# REINFORCE with value baseline
# ---------------------------------------------------------------------------

def collect_rollouts(env: PredatorSurvivalEnv, policy: nn.Module,
                     batch_size: int, max_steps: int, device: str):
    """Collect a batch of trajectories. Returns flat tensors of obs/actions/
    rewards/values/log_probs/dones, plus per-batch episode length."""
    obs = env.reset(batch_size)
    obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = (
        [], [], [], [], [], [])
    info_steps_sum = 0.0
    info_n = 0
    for t in range(max_steps):
        obs_flat = obs.reshape(batch_size, -1)
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
        "obs": torch.stack(obs_buf),       # [T, B, obs_dim]
        "act": torch.stack(act_buf),       # [T, B]
        "logp": torch.stack(logp_buf),     # [T, B]
        "val": torch.stack(val_buf),       # [T, B]
        "rew": torch.stack(rew_buf),       # [T, B]
        "done": torch.stack(done_buf),     # [T, B]
        "ep_lengths": ep_lengths,          # [B]
        "avg_inference_steps": avg_steps,
    }


def compute_returns(rewards: torch.Tensor, dones: torch.Tensor,
                    gamma: float = 0.99) -> torch.Tensor:
    """Discounted Monte-Carlo returns (no bootstrap; episodes are short)."""
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
    adv = (returns - val).detach()
    # Mask out timesteps after episode death (rew=0, but logp still counted)
    alive_mask = (1.0 - done.cumsum(dim=0).clamp(0, 1)).detach()
    # Active mask = 1 for first dead step too, so we still include the
    # final transition reward; afterwards 0.
    # Use alive_mask shifted by 1 to include the death step itself.
    T, B = rew.shape
    pre_done = torch.zeros_like(done)
    pre_done[1:] = done[:-1]
    active = (1.0 - pre_done.cumsum(dim=0).clamp(0, 1)).detach()

    pol_loss = -(logp * adv * active).sum() / active.sum().clamp(min=1)
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


# ---------------------------------------------------------------------------
# Train + eval
# ---------------------------------------------------------------------------

def evaluate(env: PredatorSurvivalEnv, policy: nn.Module, n_episodes: int,
             max_steps: int, device: str) -> dict:
    policy.eval()
    with torch.no_grad():
        obs = env.reset(n_episodes)
        steps_means = []
        for t in range(max_steps):
            obs_flat = obs.reshape(n_episodes, -1)
            logits, _, info = policy(obs_flat)
            action = logits.argmax(dim=-1)  # greedy at eval
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


def build_policy(name: str, obs_dim: int, d: int, k: int, halt: bool,
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
    ap.add_argument("--n_predators", type=int, default=2)
    ap.add_argument("--sensor_size", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--k", type=int, default=8,
                    help="Fixed K for liquid_fixed, K_max for liquid_halt")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--train_steps", type=int, default=5000)
    ap.add_argument("--ponder_lambda", type=float, default=0.05,
                    help="Compute-cost penalty for liquid_halt")
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_episodes", type=int, default=200)
    ap.add_argument("--final_eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="output_predator")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    torch.manual_seed(args.seed)

    env = PredatorSurvivalEnv(
        grid_size=args.grid_size, n_predators=args.n_predators,
        sensor_size=args.sensor_size, max_steps=args.max_steps,
        device=device,
    )
    obs_dim = 3 * args.sensor_size * args.sensor_size
    policy = build_policy(
        args.policy, obs_dim=obs_dim, d=args.d, k=args.k,
        halt=(args.policy == "liquid_halt"), device=device,
    )
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"Policy: {args.policy} | params: {n_params:,} | "
          f"obs_dim: {obs_dim} | d: {args.d} | "
          f"K: {args.k if args.policy != 'flat' else 0}")

    history = []
    t0 = time.time()
    best_mean = -1.0
    losses_window = []

    for step in range(1, args.train_steps + 1):
        # Collect rollout (re-build env to reset internal state)
        rollout = collect_rollouts(
            env, policy, args.batch_size, args.max_steps, device)
        # Track ponder_cost for liquid_halt by re-running policy to get
        # fresh cost on a sample obs (or just use 0 — for simplicity, the
        # ponder_cost from rollout uses no_grad; we recompute on training
        # batch). Simpler: include ponder term in the policy update by
        # computing on stacked obs.
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
                  f"n_used {n_used:5.2f}  "
                  f"({elapsed:.0f}s)", flush=True)

        if step == 1 or step % args.eval_every == 0 or step == args.train_steps:
            eval_metrics = evaluate(
                env, policy, n_episodes=args.eval_episodes,
                max_steps=args.max_steps, device=device,
            )
            log = {"step": step, "loss": sum(losses_window) / len(losses_window),
                    **eval_metrics}
            history.append(log)
            print(f"  step {step:5d}  EVAL  "
                  f"mean_ep {eval_metrics['mean_episode_length']:6.2f}  "
                  f"median {eval_metrics['median_episode_length']:6.1f}  "
                  f"p10 {eval_metrics['p10']:5.1f}  "
                  f"p90 {eval_metrics['p90']:5.1f}  "
                  f"n_used {eval_metrics['avg_inference_steps']:5.2f}",
                  flush=True)
            if eval_metrics["mean_episode_length"] > best_mean:
                best_mean = eval_metrics["mean_episode_length"]

    # Final long eval
    final = evaluate(
        env, policy, n_episodes=args.final_eval,
        max_steps=args.max_steps, device=device,
    )
    print(f"\nFINAL  policy={args.policy}  "
          f"mean_ep={final['mean_episode_length']:6.2f}  "
          f"median={final['median_episode_length']:6.1f}  "
          f"p10={final['p10']:5.1f}  p90={final['p90']:5.1f}  "
          f"n_used={final['avg_inference_steps']:5.2f}  "
          f"params={n_params:,}  ({time.time() - t0:.0f}s)")

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "policy": args.policy, "params": n_params,
        "config": vars(args), "history": history, "final": final,
    }
    out_path = os.path.join(args.out_dir,
                             f"predator_{args.policy}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
