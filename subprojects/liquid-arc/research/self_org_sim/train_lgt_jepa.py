"""Train JEPA-LGT: action-conditioned latent prediction in z_vl space.

Loss (primary):  L_pred = || ẑ_{t+1} - sg(z_{t+1}) ||²
Loss (aux, opt): L_id = || tangent[t=0] ||²   (small prior to prefer ẑ ≈ z when uncertain)

Training data: (z_t, action_chunk_t, z_{t+1}) triples collected via
collect_libero_jepa_triples.py.

Sequence batching: within an episode, triples are consecutive in time. Reset h_goal
at episode boundary, carry across consecutive triples to give the substrate
multi-turn context.

Baseline comparisons reported each log_every:
  L_naive_zero  = MSE(z_t, z_{t+1})       # substrate output zero → ẑ=z_t
  L_pred / L_naive_zero  ≤ 1.0 means prediction is helping
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

from liquid_goal_tracker_jepa import JEPA_LGT  # type: ignore

torch.set_float32_matmul_precision("high")


def load_triples(path):
    d = np.load(path, allow_pickle=True)
    return {
        "z_t":    d["z_t"],     # [N, 2048]
        "chunks": d["chunks"],  # [N, 16, 7]
        "z_next": d["z_next"],  # [N, 2048]
        "ep_id":  d["ep_id"],   # [N]  episode index
        "suite":  d["suite"],   # [N]  suite name
    }


def build_episode_index(ep_id, suite):
    """Group triples by (suite, ep_id) → list of contiguous index ranges."""
    keys = [(str(s), int(e)) for s, e in zip(suite, ep_id)]
    eps = {}
    for i, k in enumerate(keys):
        eps.setdefault(k, []).append(i)
    return [sorted(idxs) for idxs in eps.values()]


def sample_episode_window(data, episodes, max_turns, device, bf16=False):
    """Sample one episode, contiguous window of up to max_turns triples."""
    ep_idxs = episodes[np.random.randint(len(episodes))]
    if len(ep_idxs) <= max_turns:
        window = ep_idxs
    else:
        start = np.random.randint(0, len(ep_idxs) - max_turns + 1)
        window = ep_idxs[start:start + max_turns]
    z_t = torch.from_numpy(data["z_t"][window]).to(device)          # [T, 2048]
    chunks = torch.from_numpy(data["chunks"][window]).to(device)    # [T, 16, 7]
    z_next = torch.from_numpy(data["z_next"][window]).to(device)    # [T, 2048]
    if bf16:
        z_t = z_t.to(torch.bfloat16)
        chunks = chunks.to(torch.bfloat16)
        z_next = z_next.to(torch.bfloat16)
    return z_t, chunks, z_next


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/tmp/libero_jepa_triples.npz")
    p.add_argument("--output", default="/tmp/lgt_jepa.pt")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--max_turns_per_ep", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--ckpt_every", type=int, default=200)
    p.add_argument("--d_substrate", type=int, default=64)
    p.add_argument("--K_belief", type=int, default=4)
    p.add_argument("--tangent_scale", type=float, default=0.2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--variance_reg", type=float, default=0.0,
                   help="Weight on -log(var(z_pred)) term (VICReg-style anti-collapse).")
    p.add_argument("--loss", choices=["mse", "smooth_l1"], default="smooth_l1",
                   help="V-JEPA 2-AC convention is smooth_l1; more robust to outlier z dims.")
    p.add_argument("--smooth_l1_beta", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[jepa] device={device}, output={args.output}", flush=True)

    data = load_triples(args.data)
    z_vl_dim = data["z_t"].shape[1]
    action_horizon = data["chunks"].shape[1]
    action_dim = data["chunks"].shape[2]
    print(f"[jepa] {len(data['z_t'])} triples, z_vl_dim={z_vl_dim}, "
          f"chunk={action_horizon}x{action_dim}", flush=True)
    z_mean = data["z_t"].mean(); z_std = data["z_t"].std()
    print(f"[jepa] z_vl stats: mean={z_mean:.4f}, std={z_std:.4f}", flush=True)
    delta_norm = np.linalg.norm(data["z_next"] - data["z_t"], axis=1).mean()
    z_t_norm = np.linalg.norm(data["z_t"], axis=1).mean()
    print(f"[jepa] |z_next - z_t| mean = {delta_norm:.3f}  "
          f"(vs |z_t| mean = {z_t_norm:.3f})", flush=True)

    episodes = build_episode_index(data["ep_id"], data["suite"])
    print(f"[jepa] {len(episodes)} episodes, mean turns/ep = "
          f"{np.mean([len(e) for e in episodes]):.1f}", flush=True)

    substrate = JEPA_LGT(
        z_vl_dim=z_vl_dim, action_dim=action_dim, horizon=action_horizon,
        d=args.d_substrate, K=args.K_belief, tangent_scale=args.tangent_scale,
    ).to(device)
    n_params = sum(p.numel() for p in substrate.parameters())
    print(f"[jepa] substrate params: {n_params:,}, tangent_scale={args.tangent_scale}",
          flush=True)
    opt = torch.optim.AdamW(substrate.parameters(),
                             lr=args.lr, weight_decay=args.weight_decay)

    rolling = {"pred": [], "naive": [], "ratio": [], "tang_norm": [], "cv": [],
               "a_gate": []}
    t_start = time.time()
    info = {"action_gate": torch.tensor(0.0)}
    for step in range(args.max_steps):
        z_t_seq, chunks_seq, z_next_seq = sample_episode_window(
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
            z_t = z_t_seq[t].unsqueeze(0)       # [1, 2048]
            chunk_t = chunks_seq[t].unsqueeze(0)  # [1, 16, 7]
            z_next = z_next_seq[t].unsqueeze(0)   # [1, 2048]
            h_goal, z_pred, tangent, info = substrate.step(h_goal, z_t, chunk_t)
            target = z_next.detach()              # JEPA stop-grad on target
            if args.loss == "smooth_l1":
                loss_t = F.smooth_l1_loss(z_pred, target, beta=args.smooth_l1_beta)
            else:
                loss_t = F.mse_loss(z_pred, target)
            pred_loss_sum = pred_loss_sum + loss_t
            with torch.no_grad():
                if args.loss == "smooth_l1":
                    naive_loss_t = F.smooth_l1_loss(z_t, target, beta=args.smooth_l1_beta)
                else:
                    naive_loss_t = F.mse_loss(z_t, target)
                naive_loss_sum += float(naive_loss_t)
                tang_norm_sum += float(info["tangent_norm"])
                cv_sum += float(info["metric_cv"])

        avg_pred = pred_loss_sum / T
        if args.variance_reg > 0:
            # Anti-collapse: penalize low std of tangent across this episode
            # (encourages substrate to use action conditioning instead of constant output)
            avg_pred = avg_pred  # placeholder, would need per-dim std
        opt.zero_grad()
        avg_pred.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(substrate.parameters(), args.max_grad_norm)
        opt.step()

        rolling["pred"].append(float(avg_pred))
        rolling["naive"].append(naive_loss_sum / T)
        rolling["ratio"].append(float(avg_pred) / max(naive_loss_sum / T, 1e-8))
        rolling["tang_norm"].append(tang_norm_sum / T)
        rolling["cv"].append(cv_sum / T)
        rolling["a_gate"].append(float(info["action_gate"]))
        for k in rolling:
            rolling[k] = rolling[k][-50:]

        if step % args.log_every == 0:
            wall = time.time() - t_start
            print(
                f"step {step:>5}  pred={np.mean(rolling['pred']):.5f}  "
                f"naive={np.mean(rolling['naive']):.5f}  "
                f"ratio={np.mean(rolling['ratio']):.3f}  "
                f"tang_norm={np.mean(rolling['tang_norm']):.3f}  "
                f"cv={np.mean(rolling['cv']):.3f}  "
                f"a_gate={np.mean(rolling['a_gate']):.3f}  "
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
    print(f"\n[jepa] saved → {args.output}", flush=True)


if __name__ == "__main__":
    main()
