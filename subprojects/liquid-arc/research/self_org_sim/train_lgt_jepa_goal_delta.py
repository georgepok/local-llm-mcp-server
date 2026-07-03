"""Train goal-conditioned JEPA-LGT: predict z_{t+1} given (z_t, z_goal, action_t).

L_pred = smooth_L1(ẑ_{t+1}, sg(z_{t+1}))   [V-JEPA-AC convention]

Episodes are sequential triples with FIXED z_goal per episode (from final expert frame).
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from liquid_goal_tracker_jepa_goal_delta import JEPA_LGT_GoalDelta as JEPA_LGT_Goal  # type: ignore

torch.set_float32_matmul_precision("high")


def load_triples(path):
    d = np.load(path, allow_pickle=True)
    return {
        "z_t":    d["z_t"],
        "chunks": d["chunks"],
        "z_next": d["z_next"],
        "z_goal": d["z_goal"],
        "ep_id":  d["ep_id"],
        "suite":  d["suite"],
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
    z_goal = torch.from_numpy(data["z_goal"][window]).to(device)  # [T,2048] fixed within ep
    return z_t, chunks, z_goal, z_next


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/tmp/libero_jepa_goal_triples.npz")
    p.add_argument("--output", default="/tmp/lgt_jepa_goal.pt")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--max_turns_per_ep", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=400)
    p.add_argument("--d_substrate", type=int, default=64)
    p.add_argument("--K_belief", type=int, default=4)
    p.add_argument("--tangent_scale", type=float, default=0.2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--loss", choices=["mse", "smooth_l1"], default="smooth_l1")
    p.add_argument("--smooth_l1_beta", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[jepa-goal] device={device}, output={args.output}", flush=True)

    data = load_triples(args.data)
    z_vl_dim = data["z_t"].shape[1]
    action_horizon = data["chunks"].shape[1]
    action_dim = data["chunks"].shape[2]
    print(f"[jepa-goal] {len(data['z_t'])} triples, z_vl_dim={z_vl_dim}, "
          f"chunk={action_horizon}x{action_dim}, "
          f"unique_tasks={len(set(int(t) for t in data['task_id']))}",
          flush=True)
    delta_norm = np.linalg.norm(data["z_next"] - data["z_t"], axis=1).mean()
    goal_distance = np.linalg.norm(data["z_goal"] - data["z_t"], axis=1).mean()
    print(f"[jepa-goal] |z_next - z_t| mean = {delta_norm:.3f}, "
          f"|z_goal - z_t| mean = {goal_distance:.3f}", flush=True)

    episodes = build_episode_index(data["ep_id"], data["suite"])
    print(f"[jepa-goal] {len(episodes)} episodes, mean turns/ep = "
          f"{np.mean([len(e) for e in episodes]):.1f}", flush=True)

    substrate = JEPA_LGT_Goal(
        z_vl_dim=z_vl_dim, action_dim=action_dim, horizon=action_horizon,
        d=args.d_substrate, K=args.K_belief, tangent_scale=args.tangent_scale,
    ).to(device)
    n_params = sum(p.numel() for p in substrate.parameters())
    print(f"[jepa-goal] substrate params: {n_params:,}, "
          f"tangent_scale={args.tangent_scale}", flush=True)
    opt = torch.optim.AdamW(substrate.parameters(),
                             lr=args.lr, weight_decay=args.weight_decay)

    rolling = {"pred": [], "naive": [], "tang_norm": [], "cv": [],
               "a_gate": [], "g_gate": []}
    t_start = time.time()
    info = {"action_gate": torch.tensor(0.0), "goal_gate": torch.tensor(0.0)}
    for step in range(args.max_steps):
        z_t_seq, chunks_seq, z_goal_seq, z_next_seq = sample_episode_window(
            data, episodes, args.max_turns_per_ep, device)
        T = z_t_seq.shape[0]
        if T < 1:
            continue
        h_goal = substrate.init_state(1, device)
        pred_loss_sum = torch.tensor(0.0, device=device)
        naive_loss_sum = 0.0
        tang_norm_sum = 0.0
        cv_sum = 0.0
        for t in range(T):
            z_t = z_t_seq[t].unsqueeze(0)
            chunk_t = chunks_seq[t].unsqueeze(0)
            z_goal_t = z_goal_seq[t].unsqueeze(0)
            z_next = z_next_seq[t].unsqueeze(0)
            h_goal, z_pred, tangent, info = substrate.step(
                h_goal, z_t, z_goal_t, chunk_t)
            target = z_next.detach()
            if args.loss == "smooth_l1":
                loss_t = F.smooth_l1_loss(z_pred, target, beta=args.smooth_l1_beta)
                naive_t = F.smooth_l1_loss(z_t, target, beta=args.smooth_l1_beta)
            else:
                loss_t = F.mse_loss(z_pred, target)
                naive_t = F.mse_loss(z_t, target)
            pred_loss_sum = pred_loss_sum + loss_t
            with torch.no_grad():
                naive_loss_sum += float(naive_t)
                tang_norm_sum += float(info["tangent_norm"])
                cv_sum += float(info["metric_cv"])

        avg_pred = pred_loss_sum / T
        opt.zero_grad()
        avg_pred.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(substrate.parameters(), args.max_grad_norm)
        opt.step()

        rolling["pred"].append(float(avg_pred))
        rolling["naive"].append(naive_loss_sum / T)
        rolling["tang_norm"].append(tang_norm_sum / T)
        rolling["cv"].append(cv_sum / T)
        rolling["a_gate"].append(float(info["action_gate"]))
        rolling["g_gate"].append(float(info["goal_gate"]))
        for k in rolling:
            rolling[k] = rolling[k][-50:]

        if step % args.log_every == 0:
            wall = time.time() - t_start
            ratio = np.mean(rolling['pred']) / max(np.mean(rolling['naive']), 1e-8)
            print(
                f"step {step:>5}  pred={np.mean(rolling['pred']):.5f}  "
                f"naive={np.mean(rolling['naive']):.5f}  "
                f"ratio={ratio:.3f}  "
                f"tang_norm={np.mean(rolling['tang_norm']):.3f}  "
                f"cv={np.mean(rolling['cv']):.3f}  "
                f"a_gate={np.mean(rolling['a_gate']):.3f}  "
                f"g_gate={np.mean(rolling['g_gate']):.3f}  "
                f"T={T}  wall={wall:.0f}s", flush=True)

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
    print(f"\n[jepa-goal] saved → {args.output}", flush=True)


if __name__ == "__main__":
    main()
