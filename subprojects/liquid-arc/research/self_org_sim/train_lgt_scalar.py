"""Train scalar-output JEPA-LGT to predict (steps_remaining, goal_distance).

Targets derived per-triple from existing data:
  - steps_remaining = ep_length - step_in_ep
  - goal_distance   = ||z_goal - z_t||₂ (computed at data-prep time, normalized)

Loss:
  L = α · MSE(pred_steps, steps_remaining_norm)
    + β · MSE(pred_goaldist, goal_dist_norm)

Held-out validation + early stop, same as antimemo trainer.
Success criterion: R² on held-out steps_remaining > 0.5 (substrate learned
to estimate remaining time better than predicting the mean).
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

from liquid_goal_tracker_scalar import JEPA_LGT_Scalar  # type: ignore

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


def compute_targets(data, episodes):
    """For each triple index i, compute (steps_remaining, goal_distance)."""
    n = len(data["z_t"])
    steps_rem = np.zeros(n, dtype=np.float32)
    goal_dist = np.zeros(n, dtype=np.float32)
    for ep_idxs in episodes:
        L = len(ep_idxs)
        for pos, i in enumerate(ep_idxs):
            steps_rem[i] = float(L - 1 - pos)
            goal_dist[i] = float(np.linalg.norm(
                data["z_goal"][i] - data["z_t"][i]))
    return steps_rem, goal_dist


def sample_episode_window(data, episodes, max_turns, targets, device):
    ep_idx = np.random.randint(len(episodes))
    ep = episodes[ep_idx]
    if len(ep) <= max_turns:
        window = ep
    else:
        s = np.random.randint(0, len(ep) - max_turns + 1)
        window = ep[s:s + max_turns]
    z_t = torch.from_numpy(data["z_t"][window]).to(device)
    chunks = torch.from_numpy(data["chunks"][window]).to(device)
    z_goal = torch.from_numpy(data["z_goal"][window]).to(device)
    steps_rem = torch.from_numpy(targets["steps"][window]).to(device)
    goal_dist = torch.from_numpy(targets["dist"][window]).to(device)
    return z_t, chunks, z_goal, steps_rem, goal_dist


@torch.no_grad()
def held_out_metrics(substrate, data, episodes, targets, device,
                      steps_mean, steps_std, dist_mean, dist_std):
    """Compute R² for steps and goal_distance on held-out."""
    preds_steps, true_steps = [], []
    preds_dist, true_dist = [], []
    for ep_idxs in episodes:
        h_goal = substrate.init_state(1, device)
        for i in ep_idxs:
            z_t = torch.from_numpy(data["z_t"][i]).to(device).unsqueeze(0)
            chunk_t = torch.from_numpy(data["chunks"][i]).to(device).unsqueeze(0)
            z_goal = torch.from_numpy(data["z_goal"][i]).to(device).unsqueeze(0)
            h_goal, p_steps_n, aux, _ = substrate.step(h_goal, z_t, z_goal, chunk_t)
            p_dist_n = aux["pred_goaldist"]
            # Un-normalize
            p_steps = float(p_steps_n) * steps_std + steps_mean
            p_dist = float(p_dist_n) * dist_std + dist_mean
            preds_steps.append(p_steps)
            true_steps.append(float(targets["steps"][i]))
            preds_dist.append(p_dist)
            true_dist.append(float(targets["dist"][i]))
    preds_steps = np.array(preds_steps); true_steps = np.array(true_steps)
    preds_dist = np.array(preds_dist); true_dist = np.array(true_dist)
    r2_steps = 1.0 - ((preds_steps - true_steps) ** 2).sum() / max(
        ((true_steps - true_steps.mean()) ** 2).sum(), 1e-8)
    r2_dist = 1.0 - ((preds_dist - true_dist) ** 2).sum() / max(
        ((true_dist - true_dist.mean()) ** 2).sum(), 1e-8)
    mae_steps = float(np.mean(np.abs(preds_steps - true_steps)))
    mae_dist = float(np.mean(np.abs(preds_dist - true_dist)))
    return {
        "r2_steps": float(r2_steps), "mae_steps": mae_steps,
        "r2_dist": float(r2_dist), "mae_dist": mae_dist,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/tmp/libero_jepa_goal_triples_v3.npz")
    p.add_argument("--held_out_data", default="/tmp/libero_jepa_held_out_triples.npz")
    p.add_argument("--output", default="/tmp/lgt_scalar.pt")
    p.add_argument("--max_steps", type=int, default=6000)
    p.add_argument("--max_turns_per_ep", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--validate_every", type=int, default=200)
    p.add_argument("--early_stop_patience", type=int, default=8)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--d_substrate", type=int, default=64)
    p.add_argument("--K_belief", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--alpha_steps", type=float, default=1.0)
    p.add_argument("--beta_dist", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[scalar] device={device}", flush=True)

    train_data = load_triples(args.data)
    val_data = load_triples(args.held_out_data)
    z_vl_dim = train_data["z_t"].shape[1]
    action_horizon = train_data["chunks"].shape[1]
    action_dim = train_data["chunks"].shape[2]

    train_episodes = build_episode_index(train_data["ep_id"], train_data["suite"])
    val_episodes = build_episode_index(val_data["ep_id"], val_data["suite"])

    train_steps, train_dist = compute_targets(train_data, train_episodes)
    val_steps, val_dist = compute_targets(val_data, val_episodes)
    # Normalize using train stats
    steps_mean, steps_std = float(train_steps.mean()), float(train_steps.std()) + 1e-6
    dist_mean, dist_std = float(train_dist.mean()), float(train_dist.std()) + 1e-6
    print(f"[scalar] train: {len(train_data['z_t'])} triples, "
          f"steps_mean={steps_mean:.2f} std={steps_std:.2f}, "
          f"dist_mean={dist_mean:.2f} std={dist_std:.2f}", flush=True)
    print(f"[scalar] val: {len(val_data['z_t'])} triples, "
          f"{len(val_episodes)} episodes", flush=True)

    train_targets = {
        "steps": ((train_steps - steps_mean) / steps_std).astype(np.float32),
        "dist": ((train_dist - dist_mean) / dist_std).astype(np.float32),
    }
    val_targets_raw = {"steps": val_steps, "dist": val_dist}

    substrate = JEPA_LGT_Scalar(
        z_vl_dim=z_vl_dim, action_dim=action_dim, horizon=action_horizon,
        d=args.d_substrate, K=args.K_belief,
    ).to(device)
    n_params = sum(pp.numel() for pp in substrate.parameters())
    print(f"[scalar] substrate params: {n_params:,}", flush=True)
    opt = torch.optim.AdamW(substrate.parameters(),
                             lr=args.lr, weight_decay=args.weight_decay)

    # Baseline R² for predicting mean
    print(f"[scalar] baseline (predict-mean) val R² = 0.000 by definition", flush=True)
    init_metrics = held_out_metrics(substrate, val_data, val_episodes,
                                      val_targets_raw, device,
                                      steps_mean, steps_std,
                                      dist_mean, dist_std)
    print(f"[scalar] INIT held-out: r2_steps={init_metrics['r2_steps']:.4f} "
          f"mae_steps={init_metrics['mae_steps']:.2f}  "
          f"r2_dist={init_metrics['r2_dist']:.4f} "
          f"mae_dist={init_metrics['mae_dist']:.2f}", flush=True)

    best_r2 = init_metrics["r2_steps"]
    best_step = 0
    consec_worse = 0
    rolling = {"steps_loss": [], "dist_loss": []}
    t_start = time.time()

    for step in range(args.max_steps):
        z_t_seq, chunks_seq, z_goal_seq, steps_rem_seq, goal_dist_seq = sample_episode_window(
            train_data, train_episodes, args.max_turns_per_ep, train_targets, device)
        T = z_t_seq.shape[0]
        if T < 1:
            continue
        h_goal = substrate.init_state(1, device)
        steps_losses = []
        dist_losses = []
        for t in range(T):
            z_t = z_t_seq[t].unsqueeze(0)
            chunk_t = chunks_seq[t].unsqueeze(0)
            z_goal = z_goal_seq[t].unsqueeze(0)
            target_steps = steps_rem_seq[t].unsqueeze(0)
            target_dist = goal_dist_seq[t].unsqueeze(0)
            h_goal, p_steps, aux, _ = substrate.step(h_goal, z_t, z_goal, chunk_t)
            l_steps = F.mse_loss(p_steps, target_steps)
            l_dist = F.mse_loss(aux["pred_goaldist"], target_dist)
            steps_losses.append(l_steps)
            dist_losses.append(l_dist)
        loss = (args.alpha_steps * torch.stack(steps_losses).mean()
                + args.beta_dist * torch.stack(dist_losses).mean())
        opt.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(substrate.parameters(), args.max_grad_norm)
        opt.step()

        rolling["steps_loss"].append(float(torch.stack(steps_losses).mean()))
        rolling["dist_loss"].append(float(torch.stack(dist_losses).mean()))
        for k in rolling: rolling[k] = rolling[k][-50:]

        if step % args.log_every == 0:
            wall = time.time() - t_start
            print(f"step {step:>5}  steps_loss={np.mean(rolling['steps_loss']):.5f}  "
                  f"dist_loss={np.mean(rolling['dist_loss']):.5f}  "
                  f"wall={wall:.0f}s", flush=True)

        if step > 0 and step % args.validate_every == 0:
            m = held_out_metrics(substrate, val_data, val_episodes,
                                   val_targets_raw, device,
                                   steps_mean, steps_std, dist_mean, dist_std)
            print(f"  [VAL step {step}] r2_steps={m['r2_steps']:.4f} "
                  f"mae_steps={m['mae_steps']:.2f}  "
                  f"r2_dist={m['r2_dist']:.4f} mae_dist={m['mae_dist']:.2f}  "
                  f"(best={best_r2:.4f} @step{best_step})", flush=True)
            if m["r2_steps"] > best_r2:
                best_r2 = m["r2_steps"]
                best_step = step
                consec_worse = 0
                torch.save({
                    "substrate_state_dict": substrate.state_dict(),
                    "args": vars(args), "z_vl_dim": z_vl_dim,
                    "action_dim": action_dim, "horizon": action_horizon,
                    "step": step, "val_r2_steps": m["r2_steps"],
                    "val_r2_dist": m["r2_dist"],
                    "steps_mean": steps_mean, "steps_std": steps_std,
                    "dist_mean": dist_mean, "dist_std": dist_std,
                }, args.output.replace(".pt", "_best.pt"))
            else:
                consec_worse += 1
                if consec_worse >= args.early_stop_patience:
                    print(f"  [EARLY STOP] r2_steps worsened for {consec_worse} "
                          f"validations. Best: {best_r2:.4f} @step{best_step}",
                          flush=True)
                    break

        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            torch.save({
                "substrate_state_dict": substrate.state_dict(),
                "args": vars(args), "z_vl_dim": z_vl_dim,
                "action_dim": action_dim, "horizon": action_horizon,
                "step": step,
                "steps_mean": steps_mean, "steps_std": steps_std,
                "dist_mean": dist_mean, "dist_std": dist_std,
            }, args.output.replace(".pt", f"_step{step}.pt"))

    torch.save({
        "substrate_state_dict": substrate.state_dict(),
        "args": vars(args), "z_vl_dim": z_vl_dim,
        "action_dim": action_dim, "horizon": action_horizon,
        "step": args.max_steps,
        "best_r2_steps": best_r2, "best_step": best_step,
        "steps_mean": steps_mean, "steps_std": steps_std,
        "dist_mean": dist_mean, "dist_std": dist_std,
    }, args.output)
    print(f"\n[scalar] saved → {args.output}", flush=True)
    print(f"[scalar] best r2_steps: {best_r2:.4f} @step{best_step}", flush=True)


if __name__ == "__main__":
    main()
