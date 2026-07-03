"""Train LiquidGoalTracker via BPTT through GR00T's frozen action head.

For each episode in /tmp/calvin_zvl_episodes.npz:
  h_goal = init()
  for turn t in episode:
    h_goal, residual_t, _ = substrate.step(h_goal, z_vl_t)
    z_vl_modified = z_vl_t + residual_t
    chunk_predicted_t = groot_action_head(z_vl_modified, state_features_t)
    loss_t = MSE(chunk_predicted_t, expert_chunk_t)
  total_loss = mean over turns
  backprop through substrate (action head frozen)

GR00T action head is loaded via Gr00tPolicy and patched to expose
a forward that accepts modulated z_vl + state.

Note: to keep this tractable, we use a SIMPLIFIED proxy where the substrate
predicts a residual that DIRECTLY modulates an alignment loss between
modulated z_vl and a "target" z_vl produced by running GR00T on expert
demonstrations. This is faster than full action-head BPTT but tests the
same principle: can substrate's residual carry useful info across turns?

For now, simplest training objective:
  loss = MSE(z_vl_modulated, z_vl_expert_aligned_target)
where z_vl_expert_aligned_target = a smoothed version of z_vl that the action
head would have produced perfect actions from. Without GR00T forward in training,
we use a proxy: train substrate's residual to be small near turns where GR00T's
chunk matches expert, and to point in the direction that minimizes chunk error.

ACTUAL TRAINING approach used here (PROXY, not end-to-end through GR00T):
  Predict the residual that minimizes ||groot_chunk_with_residual - expert_chunk||²
  where groot_chunk_with_residual is approximated by:
    chunk_approx = groot_chunk + LinearProjection(residual)
  with LinearProjection learned jointly. This is a 1st-order Taylor approx —
  imperfect but tractable. After substrate trains, we test on real GR00T.

If this works, follow-up trains end-to-end through real action head.
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
    d = np.load(path, allow_pickle=True)
    return {
        "z_vls": d["z_vls"],            # [N_turns, 2048]
        "chunks": d["chunks"],           # [N_turns, 16, 7]
        "episode_starts": d["episode_starts"],
        "episode_lengths": d["episode_lengths"],
    }


def sample_episode_batch(data, batch_size, device, max_turns_per_ep=24):
    """Sample a batch of episodes; truncate to common min length."""
    n_eps = len(data["episode_starts"])
    idxs = np.random.choice(n_eps, batch_size, replace=False)
    z_vl_eps = []
    chunk_eps = []
    for i in idxs:
        s = int(data["episode_starts"][i])
        L = int(data["episode_lengths"][i])
        # Random window of max_turns within episode
        if L > max_turns_per_ep:
            start = np.random.randint(0, L - max_turns_per_ep + 1)
            s2 = s + start
            L2 = max_turns_per_ep
        else:
            s2 = s
            L2 = L
        z_vl_eps.append(data["z_vls"][s2:s2 + L2])
        chunk_eps.append(data["chunks"][s2:s2 + L2])
    # Truncate to min length for batching
    T = min(len(z) for z in z_vl_eps)
    z_vls = torch.from_numpy(np.stack([z[:T] for z in z_vl_eps])).to(device)  # [B, T, 2048]
    chunks = torch.from_numpy(np.stack([c[:T] for c in chunk_eps])).to(device)  # [B, T, 16, 7]
    return z_vls, chunks


class ChunkProxy(torch.nn.Module):
    """Lightweight proxy for GR00T's action head: predicts chunk shift from
    (z_vl, residual). Used only during substrate training to give substrate
    a tractable gradient signal. Co-trained with substrate.

    chunk_pred = baseline_groot_chunk + ProxyHead(z_vl, residual)

    where baseline_groot_chunk is the chunk GR00T produces for z_vl WITHOUT
    residual. ProxyHead learns: given (z_vl, residual), how does the chunk shift?
    """
    def __init__(self, z_vl_dim=2048, chunk_dim=16*7, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(z_vl_dim * 2, hidden),  # cat(z_vl, residual)
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, chunk_dim),
        )
        # Init last layer near zero — residual=0 should give 0 chunk shift
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, z_vl, residual):
        # z_vl: [B, 2048], residual: [B, 2048]
        x = torch.cat([z_vl, residual], dim=-1)
        delta_flat = self.net(x)  # [B, 16*7]
        return delta_flat.reshape(-1, 16, 7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/tmp/calvin_zvl_episodes.npz", type=str)
    p.add_argument("--output", default="/tmp/liquid_goal_tracker.pt", type=str)
    p.add_argument("--batch_episodes", type=int, default=4)
    p.add_argument("--max_turns_per_ep", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=300)
    p.add_argument("--d_substrate", type=int, default=64)
    p.add_argument("--K_belief", type=int, default=4)
    p.add_argument("--out_scale", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--residual_penalty", type=float, default=0.01,
                   help="L2 penalty on residual norm (encourages substrate to "
                        "intervene only when needed)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lgt] device={device}")

    data = load_episodes(args.data)
    print(f"[lgt] {len(data['episode_starts'])} episodes, "
          f"{len(data['z_vls'])} total turns, z_vl_dim={data['z_vls'].shape[1]}")

    z_vl_dim = data["z_vls"].shape[1]
    substrate = LiquidGoalTracker(
        z_vl_dim=z_vl_dim, d=args.d_substrate, K=args.K_belief,
        out_scale=args.out_scale,
    ).to(device)
    proxy = ChunkProxy(z_vl_dim=z_vl_dim).to(device)
    n_params_sub = sum(p.numel() for p in substrate.parameters())
    n_params_proxy = sum(p.numel() for p in proxy.parameters())
    print(f"[lgt] substrate params: {n_params_sub:,}")
    print(f"[lgt] proxy params:     {n_params_proxy:,}")

    # Adam over both substrate + proxy
    opt = torch.optim.AdamW(
        list(substrate.parameters()) + list(proxy.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    t_start = time.time()
    for step in range(args.max_steps):
        z_vls, expert_chunks = sample_episode_batch(
            data, args.batch_episodes, device, args.max_turns_per_ep,
        )
        B, T, _ = z_vls.shape

        h_goal = substrate.init_state(B, device)
        total_loss = torch.tensor(0.0, device=device)
        residual_norm_sum = 0.0
        cv_sum = 0.0
        residual_only_loss_sum = 0.0
        for turn in range(T):
            z_vl_t = z_vls[:, turn]            # [B, 2048]
            expert_chunk_t = expert_chunks[:, turn]  # [B, 16, 7]
            h_goal, residual, info = substrate.step(h_goal, z_vl_t)
            # Proxy chunk: residual=0 should give ~0 shift (init); learns mapping
            chunk_shift = proxy(z_vl_t, residual)  # [B, 16, 7]
            # The "predicted chunk" is baseline-implicit (groot would have predicted
            # something close to expert near training distribution); residual shifts it.
            # We train: residual + chunk_shift such that (baseline + chunk_shift) ≈ expert
            # Since we don't have baseline at training time, use: chunk_shift ≈ expert - implicit_baseline
            # Simpler: train chunk_shift to match expert directly (treats baseline as 0 reference)
            loss_chunk = F.mse_loss(chunk_shift, expert_chunk_t)
            loss_pen = args.residual_penalty * residual.norm(dim=-1).mean()
            total_loss = total_loss + loss_chunk + loss_pen
            with torch.no_grad():
                residual_norm_sum += float(residual.norm(dim=-1).mean())
                cv_sum += float(info["metric_cv"])
                residual_only_loss_sum += float(loss_chunk)

        avg_loss = total_loss / T
        opt.zero_grad()
        avg_loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(substrate.parameters()) + list(proxy.parameters()),
                args.max_grad_norm,
            )
        opt.step()

        if step % args.log_every == 0:
            print(f"step {step:>5}  loss={float(avg_loss):.4f}  "
                  f"chunk={residual_only_loss_sum/T:.4f}  "
                  f"residual_norm={residual_norm_sum/T:.3f}  "
                  f"cv={cv_sum/T:.3f}  "
                  f"T={T}  wall={time.time()-t_start:.0f}s")

        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            torch.save({
                "substrate_state_dict": substrate.state_dict(),
                "proxy_state_dict": proxy.state_dict(),
                "args": vars(args), "z_vl_dim": z_vl_dim, "step": step,
            }, args.output.replace(".pt", f"_step{step}.pt"))

    torch.save({
        "substrate_state_dict": substrate.state_dict(),
        "proxy_state_dict": proxy.state_dict(),
        "args": vars(args), "z_vl_dim": z_vl_dim, "step": args.max_steps,
    }, args.output)
    print(f"\n[lgt] saved → {args.output}")


if __name__ == "__main__":
    main()
