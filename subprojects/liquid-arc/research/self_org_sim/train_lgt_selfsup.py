"""Train LiquidGoalTracker self-supervised: predict z_vl evolution across turns.

Substrate learns to model how GR00T's z_vl changes turn-to-turn:
  At turn t, given (h_goal_prev, z_vl_t), substrate produces (h_goal_new, residual_t).
  Target: residual_t ≈ z_vl_{t+1} - z_vl_t  (the next-turn delta)
  Loss = MSE(residual_t, target_delta_t)

At INFERENCE, applying residual_t to z_vl_t gives action_head a "preview" of
the anticipated next-turn vl-embedding. Action head produces actions that are
correct for the ANTICIPATED state, providing forward-looking goal-tracking.

This is the goal-tracking spec per user: small substrate, in/out from GR00T's
transformer, persistent state across turns.
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

from liquid_goal_tracker import LiquidGoalTracker  # type: ignore

torch.set_float32_matmul_precision("high")


def load_episodes(path):
    d = np.load(path)
    return {
        "z_vls": d["z_vls"],
        "episode_starts": d["episode_starts"],
        "episode_lengths": d["episode_lengths"],
    }


def sample_batch(data, batch_size, max_turns, device):
    n_eps = len(data["episode_starts"])
    idxs = np.random.choice(n_eps, batch_size, replace=False)
    z_vl_seqs = []
    for i in idxs:
        s = int(data["episode_starts"][i])
        L = int(data["episode_lengths"][i])
        if L > max_turns:
            start = np.random.randint(0, L - max_turns + 1)
            s2 = s + start
            L2 = max_turns
        else:
            s2 = s
            L2 = L
        z_vl_seqs.append(data["z_vls"][s2:s2 + L2])
    T = min(len(z) for z in z_vl_seqs)
    return torch.from_numpy(np.stack([z[:T] for z in z_vl_seqs])).to(device)  # [B, T, 2048]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/tmp/libero_zvl_episodes.npz")
    p.add_argument("--output", default="/tmp/lgt_libero_selfsup.pt")
    p.add_argument("--batch_episodes", type=int, default=8)
    p.add_argument("--max_turns_per_ep", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--d_substrate", type=int, default=64)
    p.add_argument("--K_belief", type=int, default=4)
    p.add_argument("--out_scale", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lgt-ss] device={device}")

    data = load_episodes(args.data)
    z_vl_dim = data["z_vls"].shape[1]
    print(f"[lgt-ss] {len(data['episode_starts'])} episodes, "
          f"{len(data['z_vls'])} total turns, z_vl_dim={z_vl_dim}")

    substrate = LiquidGoalTracker(
        z_vl_dim=z_vl_dim, d=args.d_substrate, K=args.K_belief,
        out_scale=args.out_scale,
    ).to(device)
    n_params = sum(p.numel() for p in substrate.parameters())
    print(f"[lgt-ss] substrate params: {n_params:,}")

    opt = torch.optim.AdamW(substrate.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)

    # Statistics for normalization — z_vl values have moderate range
    z_mean = data["z_vls"].mean()
    z_std = data["z_vls"].std()
    print(f"[lgt-ss] z_vl stats: mean={z_mean:.4f}, std={z_std:.4f}")

    t_start = time.time()
    for step in range(args.max_steps):
        z_vls = sample_batch(data, args.batch_episodes, args.max_turns_per_ep, device)
        B, T, _ = z_vls.shape
        if T < 2:
            continue

        h_goal = substrate.init_state(B, device)
        total_loss = torch.tensor(0.0, device=device)
        residual_norm_sum = 0.0
        target_norm_sum = 0.0
        cv_sum = 0.0
        improvement_sum = 0.0
        for turn in range(T - 1):  # predict t+1 from t, so loop T-1 times
            z_vl_t = z_vls[:, turn]
            z_vl_next = z_vls[:, turn + 1]
            target_delta = z_vl_next - z_vl_t  # [B, 2048]

            h_goal, residual, info = substrate.step(h_goal, z_vl_t)
            loss_t = F.mse_loss(residual, target_delta)
            total_loss = total_loss + loss_t

            with torch.no_grad():
                residual_norm_sum += float(residual.norm(dim=-1).mean())
                target_norm_sum += float(target_delta.norm(dim=-1).mean())
                cv_sum += float(info["metric_cv"])
                # "Improvement": MSE(residual, target) vs MSE(0, target)
                naive_mse = (target_delta ** 2).mean().item()
                pred_mse = loss_t.item()
                improvement_sum += (naive_mse - pred_mse) / max(naive_mse, 1e-8)

        avg_loss = total_loss / max(T - 1, 1)
        opt.zero_grad()
        avg_loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(substrate.parameters(), args.max_grad_norm)
        opt.step()

        if step % args.log_every == 0:
            denom = max(T - 1, 1)
            print(f"step {step:>5}  loss={float(avg_loss):.5f}  "
                  f"residual_norm={residual_norm_sum/denom:.3f}  "
                  f"target_norm={target_norm_sum/denom:.3f}  "
                  f"improvement_vs_zero={improvement_sum/denom:.1%}  "
                  f"cv={cv_sum/denom:.3f}  T={T}  wall={time.time()-t_start:.0f}s")

        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            torch.save({
                "substrate_state_dict": substrate.state_dict(),
                "args": vars(args), "z_vl_dim": z_vl_dim, "step": step,
            }, args.output.replace(".pt", f"_step{step}.pt"))

    torch.save({
        "substrate_state_dict": substrate.state_dict(),
        "args": vars(args), "z_vl_dim": z_vl_dim, "step": args.max_steps,
    }, args.output)
    print(f"\n[lgt-ss] saved → {args.output}")


if __name__ == "__main__":
    main()
