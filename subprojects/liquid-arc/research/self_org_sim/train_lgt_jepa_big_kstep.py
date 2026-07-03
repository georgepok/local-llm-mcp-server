"""K-step rollout JEPA-LGT — V-JEPA-AC style autoregressive prediction.

For each starting turn t in an episode (with t+K available), unroll K steps:
  z_pred[0] = z_t                                          # real anchor
  for k in 0..K-1:
      h, z_pred[k+1], _ = substrate.step(h, z_pred[k], z_goal, chunk[t+k])
      L_k = smooth_L1(z_pred[k+1], sg(z[t+k+1]))           # JEPA loss per step

  Total = mean over k                                       # multi-step supervision

Autoregressive: each prediction step consumes the substrate's OWN previous
output (not the real z[t+k]). This forces the substrate to maintain coherent
multi-step latent dynamics — V-JEPA-AC's defining training detail.

Substrate is unchanged (compact pooled latent JEPA_LGT_GoalDelta). Only the
training procedure changes — that's the JEPA-faithful way to deepen prediction.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from liquid_goal_tracker_jepa_big import JEPA_LGT_Big as JEPA_LGT_GoalDelta  # type: ignore

torch.set_float32_matmul_precision("high")


def load_triples(path):
    d = np.load(path, allow_pickle=True)
    return {
        "z_t": d["z_t"], "chunks": d["chunks"], "z_next": d["z_next"],
        "z_goal": d["z_goal"], "ep_id": d["ep_id"], "suite": d["suite"],
        "task_id": d["task_id"],
    }


def build_episode_index(ep_id, suite):
    keys = [(str(s), int(e)) for s, e in zip(suite, ep_id)]
    eps = {}
    for i, k in enumerate(keys):
        eps.setdefault(k, []).append(i)
    return [sorted(idxs) for idxs in eps.values()]


def sample_episode_window(data, episodes, max_turns, device):
    ep_idxs = episodes[np.random.randint(len(episodes))]
    if len(ep_idxs) <= max_turns:
        window = ep_idxs
    else:
        start = np.random.randint(0, len(ep_idxs) - max_turns + 1)
        window = ep_idxs[start:start + max_turns]
    z_t = torch.from_numpy(data["z_t"][window]).to(device)
    chunks = torch.from_numpy(data["chunks"][window]).to(device)
    z_next = torch.from_numpy(data["z_next"][window]).to(device)
    z_goal = torch.from_numpy(data["z_goal"][window]).to(device)
    return z_t, chunks, z_goal, z_next


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/tmp/libero_jepa_goal_triples.npz")
    p.add_argument("--output", default="/tmp/lgt_jepa_kstep.pt")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--max_turns_per_ep", type=int, default=16)
    p.add_argument("--K", type=int, default=4,
                   help="Rollout horizon (V-JEPA-2 uses 32; LIBERO turn-cost is "
                        "high so smaller K like 4-8 is fine).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=400)
    p.add_argument("--d_substrate", type=int, default=64)
    p.add_argument("--K_belief", type=int, default=4)
    p.add_argument("--tangent_scale", type=float, default=0.2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--smooth_l1_beta", type=float, default=1.0)
    p.add_argument("--teacher_forcing_prob", type=float, default=0.0,
                   help="Probability of using REAL z[t+k] instead of predicted ẑ "
                        "at each step. 0 = full autoregressive (V-JEPA-AC); "
                        "1 = full teacher-forcing (trivial, no multi-step coherence).")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[kstep] device={device}, K={args.K}, output={args.output}", flush=True)

    data = load_triples(args.data)
    z_vl_dim = data["z_t"].shape[1]
    action_horizon = data["chunks"].shape[1]
    action_dim = data["chunks"].shape[2]
    print(f"[kstep] {len(data['z_t'])} triples, z_vl_dim={z_vl_dim}, "
          f"chunk={action_horizon}x{action_dim}", flush=True)

    episodes = build_episode_index(data["ep_id"], data["suite"])
    long_eps = [e for e in episodes if len(e) > args.K]
    print(f"[kstep] {len(episodes)} episodes total, {len(long_eps)} with >K turns",
          flush=True)

    substrate = JEPA_LGT_GoalDelta(
        z_vl_dim=z_vl_dim, action_dim=action_dim, horizon=action_horizon,
        d=args.d_substrate, K=args.K_belief, tangent_scale=args.tangent_scale,
    ).to(device)
    n_params = sum(p.numel() for p in substrate.parameters())
    print(f"[kstep] substrate params: {n_params:,}, tangent_scale={args.tangent_scale}",
          flush=True)
    opt = torch.optim.AdamW(substrate.parameters(),
                             lr=args.lr, weight_decay=args.weight_decay)

    rolling = {"loss": [], "naive": [], "ratio": [],
               "loss_k0": [], "loss_klast": [],
               "tang_norm": [], "cv": [], "a_gate": [], "g_gate": [], "d_gate": []}
    t_start = time.time()
    info = {"action_gate": torch.tensor(0.0),
            "goal_gate": torch.tensor(0.0),
            "delta_gate": torch.tensor(0.0)}
    K = args.K
    for step in range(args.max_steps):
        if not long_eps:
            print("[kstep] no episodes long enough for K-rollout"); break
        z_t_seq, chunks_seq, z_goal_seq, z_next_seq = sample_episode_window(
            data, long_eps, args.max_turns_per_ep, device)
        T = z_t_seq.shape[0]
        if T < K + 1:
            continue

        # For each starting turn t in 0..T-K-1, unroll K steps
        h_goal = substrate.init_state(1, device)
        rollout_losses = []
        rollout_loss_k0 = []
        rollout_loss_klast = []
        naive_per_step = []
        tang_norm_sum = 0.0
        cv_sum = 0.0
        n_rollouts = 0
        # Single rollout starting at turn 0 (substrate's h_goal evolves across
        # all K steps of one rollout, then is REINITIALIZED for next starting turn)
        for t_start_idx in range(0, T - K, max(1, K // 2)):  # stride K/2 for overlap
            h_goal_local = substrate.init_state(1, device)
            z_pred = z_t_seq[t_start_idx].unsqueeze(0)  # [1, 2048]  initial anchor
            z_goal_local = z_goal_seq[t_start_idx].unsqueeze(0)
            step_losses = []
            for k in range(K):
                chunk_k = chunks_seq[t_start_idx + k].unsqueeze(0)
                h_goal_local, z_pred_next, tangent, info = substrate.step(
                    h_goal_local, z_pred, z_goal_local, chunk_k)
                target = z_next_seq[t_start_idx + k].unsqueeze(0).detach()
                loss_k = F.smooth_l1_loss(z_pred_next, target,
                                            beta=args.smooth_l1_beta)
                step_losses.append(loss_k)
                with torch.no_grad():
                    naive_k = F.smooth_l1_loss(z_pred, target,
                                                  beta=args.smooth_l1_beta)
                    naive_per_step.append(float(naive_k))
                    tang_norm_sum += float(info["tangent_norm"])
                    cv_sum += float(info["metric_cv"])
                # AR vs teacher forcing
                if (args.teacher_forcing_prob > 0
                        and np.random.rand() < args.teacher_forcing_prob):
                    z_pred = z_next_seq[t_start_idx + k].unsqueeze(0).detach()
                else:
                    z_pred = z_pred_next   # autoregressive (V-JEPA-AC default)
            rollout_loss = torch.stack(step_losses).mean()
            rollout_losses.append(rollout_loss)
            rollout_loss_k0.append(float(step_losses[0].detach()))
            rollout_loss_klast.append(float(step_losses[-1].detach()))
            n_rollouts += 1

        if not rollout_losses:
            continue
        avg_loss = torch.stack(rollout_losses).mean()
        opt.zero_grad()
        avg_loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(substrate.parameters(),
                                            args.max_grad_norm)
        opt.step()

        denom = max(n_rollouts * K, 1)
        rolling["loss"].append(float(avg_loss))
        rolling["naive"].append(float(np.mean(naive_per_step)))
        rolling["ratio"].append(float(avg_loss) /
                                  max(float(np.mean(naive_per_step)), 1e-8))
        rolling["loss_k0"].append(float(np.mean(rollout_loss_k0)))
        rolling["loss_klast"].append(float(np.mean(rollout_loss_klast)))
        rolling["tang_norm"].append(tang_norm_sum / denom)
        rolling["cv"].append(cv_sum / denom)
        rolling["a_gate"].append(float(info["action_gate"]))
        rolling["g_gate"].append(float(info["goal_gate"]))
        rolling["d_gate"].append(float(info["delta_gate"]))
        for k in rolling:
            rolling[k] = rolling[k][-50:]

        if step % args.log_every == 0:
            wall = time.time() - t_start
            print(
                f"step {step:>5}  loss={np.mean(rolling['loss']):.5f}  "
                f"naive={np.mean(rolling['naive']):.5f}  "
                f"ratio={np.mean(rolling['ratio']):.3f}  "
                f"k0={np.mean(rolling['loss_k0']):.5f}  "
                f"k{K-1}={np.mean(rolling['loss_klast']):.5f}  "
                f"tang={np.mean(rolling['tang_norm']):.3f}  "
                f"cv={np.mean(rolling['cv']):.3f}  "
                f"g={np.mean(rolling['g_gate']):.2f}  "
                f"d={np.mean(rolling['d_gate']):.2f}  "
                f"a={np.mean(rolling['a_gate']):.2f}  "
                f"rollouts={n_rollouts}  T={T}  wall={wall:.0f}s", flush=True)

        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            torch.save({
                "substrate_state_dict": substrate.state_dict(),
                "args": vars(args), "z_vl_dim": z_vl_dim,
                "action_dim": action_dim, "horizon": action_horizon,
                "step": step,
            }, args.output.replace(".pt", f"_step{step}.pt"))

    torch.save({
        "substrate_state_dict": substrate.state_dict(),
        "args": vars(args), "z_vl_dim": z_vl_dim,
        "action_dim": action_dim, "horizon": action_horizon,
        "step": args.max_steps,
    }, args.output)
    print(f"\n[kstep] saved → {args.output}", flush=True)


if __name__ == "__main__":
    main()
