"""Train X-attn JEPA-LGT: per-token next-state prediction in bb_features space.

Per turn t in episode:
  bb_t [seq, 2048], chunk_t [16, 7], z_goal [2048] → substrate predicts bb_pred [seq, 2048]
  L = smooth_L1(bb_pred, sg(bb_next))                  # per-token

Per-token JEPA — substrate must learn how EACH token evolves, not just the average.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# DGX Spark sm121 — disable broken FMHA kernels, force math SDPA
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from liquid_goal_tracker_jepa_xattn import JEPA_LGT_XAttn  # type: ignore

torch.set_float32_matmul_precision("high")


def load_triples(path):
    d = np.load(path, allow_pickle=True)
    return {
        "bb_t": d["bb_t"], "chunks": d["chunks"], "bb_next": d["bb_next"],
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
    bb_t = torch.from_numpy(data["bb_t"][window]).to(device)       # [T, seq, 2048]
    chunks = torch.from_numpy(data["chunks"][window]).to(device)   # [T, 16, 7]
    bb_next = torch.from_numpy(data["bb_next"][window]).to(device) # [T, seq, 2048]
    z_goal = torch.from_numpy(data["z_goal"][window]).to(device)   # [T, 2048]
    return bb_t, chunks, z_goal, bb_next


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/tmp/libero_jepa_xattn_triples.npz")
    p.add_argument("--output", default="/tmp/lgt_jepa_xattn.pt")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--max_turns_per_ep", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=400)
    p.add_argument("--d_substrate", type=int, default=128)
    p.add_argument("--K_belief", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--tangent_scale", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--loss", choices=["mse", "smooth_l1"], default="smooth_l1")
    p.add_argument("--smooth_l1_beta", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[xattn] device={device}, output={args.output}", flush=True)

    data = load_triples(args.data)
    seq_len = data["bb_t"].shape[1]
    z_vl_dim = data["bb_t"].shape[2]
    action_horizon = data["chunks"].shape[1]
    action_dim = data["chunks"].shape[2]
    print(f"[xattn] {len(data['bb_t'])} triples, seq_len={seq_len}, "
          f"z_vl_dim={z_vl_dim}, chunk={action_horizon}x{action_dim}",
          flush=True)
    delta_norm = np.linalg.norm(
        (data["bb_next"] - data["bb_t"]).reshape(-1, z_vl_dim), axis=1).mean()
    bb_norm = np.linalg.norm(
        data["bb_t"].reshape(-1, z_vl_dim), axis=1).mean()
    print(f"[xattn] per-token |bb_next - bb_t|={delta_norm:.3f}, "
          f"|bb_t|={bb_norm:.3f}", flush=True)

    episodes = build_episode_index(data["ep_id"], data["suite"])
    print(f"[xattn] {len(episodes)} episodes, mean turns/ep = "
          f"{np.mean([len(e) for e in episodes]):.1f}", flush=True)

    substrate = JEPA_LGT_XAttn(
        z_vl_dim=z_vl_dim, action_dim=action_dim, horizon=action_horizon,
        d=args.d_substrate, K=args.K_belief, n_heads=args.n_heads,
        tangent_scale=args.tangent_scale,
    ).to(device)
    n_params = sum(p.numel() for p in substrate.parameters())
    print(f"[xattn] substrate params: {n_params:,}, "
          f"tangent_scale={args.tangent_scale}", flush=True)
    opt = torch.optim.AdamW(substrate.parameters(),
                             lr=args.lr, weight_decay=args.weight_decay)

    rolling = {"pred": [], "naive": [], "ratio": [], "tn_per_token": [],
               "tn_total": [], "cv": [], "g_gate": [], "a_gate": []}
    t_start = time.time()
    info = {"goal_gate": torch.tensor(0.0), "action_gate": torch.tensor(0.0)}
    for step in range(args.max_steps):
        bb_t_seq, chunks_seq, z_goal_seq, bb_next_seq = sample_episode_window(
            data, episodes, args.max_turns_per_ep, device)
        T = bb_t_seq.shape[0]
        if T < 1:
            continue
        h_goal = substrate.init_state(1, device)
        pred_loss_sum = torch.tensor(0.0, device=device)
        naive_loss_sum = 0.0
        tn_pt_sum = 0.0
        tn_total_sum = 0.0
        cv_sum = 0.0
        for t in range(T):
            bb_t = bb_t_seq[t].unsqueeze(0)        # [1, seq, 2048]
            chunk_t = chunks_seq[t].unsqueeze(0)    # [1, 16, 7]
            z_goal_t = z_goal_seq[t].unsqueeze(0)   # [1, 2048]
            bb_next = bb_next_seq[t].unsqueeze(0)   # [1, seq, 2048]
            h_goal, bb_pred, tangent, info = substrate.step(
                h_goal, bb_t, z_goal_t, chunk_t)
            target = bb_next.detach()
            if args.loss == "smooth_l1":
                loss_t = F.smooth_l1_loss(bb_pred, target, beta=args.smooth_l1_beta)
                naive_t = F.smooth_l1_loss(bb_t, target, beta=args.smooth_l1_beta)
            else:
                loss_t = F.mse_loss(bb_pred, target)
                naive_t = F.mse_loss(bb_t, target)
            pred_loss_sum = pred_loss_sum + loss_t
            with torch.no_grad():
                naive_loss_sum += float(naive_t)
                tn_pt_sum += float(info["tangent_norm_per_token"])
                tn_total_sum += float(info["tangent_total_norm"])
                cv_sum += float(info["metric_cv"])

        avg_pred = pred_loss_sum / T
        opt.zero_grad()
        avg_pred.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(substrate.parameters(), args.max_grad_norm)
        opt.step()

        rolling["pred"].append(float(avg_pred))
        rolling["naive"].append(naive_loss_sum / T)
        rolling["ratio"].append(float(avg_pred) / max(naive_loss_sum / T, 1e-8))
        rolling["tn_per_token"].append(tn_pt_sum / T)
        rolling["tn_total"].append(tn_total_sum / T)
        rolling["cv"].append(cv_sum / T)
        rolling["g_gate"].append(float(info["goal_gate"]))
        rolling["a_gate"].append(float(info["action_gate"]))
        for k in rolling:
            rolling[k] = rolling[k][-50:]

        if step % args.log_every == 0:
            wall = time.time() - t_start
            print(
                f"step {step:>5}  pred={np.mean(rolling['pred']):.5f}  "
                f"naive={np.mean(rolling['naive']):.5f}  "
                f"ratio={np.mean(rolling['ratio']):.3f}  "
                f"tn_pt={np.mean(rolling['tn_per_token']):.3f}  "
                f"tn_total={np.mean(rolling['tn_total']):.2f}  "
                f"cv={np.mean(rolling['cv']):.3f}  "
                f"g={np.mean(rolling['g_gate']):.2f}  "
                f"a={np.mean(rolling['a_gate']):.2f}  "
                f"T={T}  wall={wall:.0f}s", flush=True)

        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            torch.save({
                "substrate_state_dict": substrate.state_dict(),
                "args": vars(args), "z_vl_dim": z_vl_dim, "seq_len": seq_len,
                "action_dim": action_dim, "horizon": action_horizon, "step": step,
            }, args.output.replace(".pt", f"_step{step}.pt"))

    torch.save({
        "substrate_state_dict": substrate.state_dict(),
        "args": vars(args), "z_vl_dim": z_vl_dim, "seq_len": seq_len,
        "action_dim": action_dim, "horizon": action_horizon, "step": args.max_steps,
    }, args.output)
    print(f"\n[xattn] saved → {args.output}", flush=True)


if __name__ == "__main__":
    main()
