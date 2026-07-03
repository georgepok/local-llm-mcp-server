"""Train proprio JEPA-LGT — multi-target prediction with proprioceptive input.

Targets:
  goal_distance (regression, normalized)
  gripper_moving (binary, "did gripper qpos change > eps between turn t and t+1")
  state_delta (regression, ||state8_next - state8_t||)

Held-out validation + early stop on a composite metric:
  composite_score = r2_dist + bce_acc_gripper + r2_state_delta
(higher is better; if it stagnates → stop)
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

from liquid_goal_tracker_proprio import JEPA_LGT_Proprio  # type: ignore

torch.set_float32_matmul_precision("high")


def load_triples(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def build_episode_index(ep_id, suite):
    keys = [(str(s), int(e)) for s, e in zip(suite, ep_id)]
    eps = {}
    for i, k in enumerate(keys):
        eps.setdefault(k, []).append(i)
    return [sorted(idxs) for idxs in eps.values()]


def compute_targets(data, episodes, gripper_thresh=0.005):
    """For each triple: (goal_dist, gripper_moving_label, state_delta)."""
    n = len(data["z_t"])
    goal_dist = np.zeros(n, dtype=np.float32)
    gripper_moving = np.zeros(n, dtype=np.float32)
    state_delta = np.zeros(n, dtype=np.float32)
    for i in range(n):
        goal_dist[i] = float(np.linalg.norm(data["z_goal"][i] - data["z_t"][i]))
        s_t = data["state8_t"][i]
        s_n = data["state8_next"][i]
        # Gripper qpos columns: indices 6 and 7 of state8
        g_delta = float(np.abs(s_n[6:8] - s_t[6:8]).mean())
        gripper_moving[i] = 1.0 if g_delta > gripper_thresh else 0.0
        state_delta[i] = float(np.linalg.norm(s_n - s_t))
    return {"dist": goal_dist, "gripper_moving": gripper_moving,
            "state_delta": state_delta}


@torch.no_grad()
def held_out_metrics(substrate, data, episodes, targets_raw, device,
                      dist_mean, dist_std, state_delta_mean, state_delta_std):
    preds_d, true_d = [], []
    preds_g_logit, true_g = [], []
    preds_s, true_s = [], []
    for ep_idxs in episodes:
        h_goal = substrate.init_state(1, device)
        for i in ep_idxs:
            z_t = torch.from_numpy(data["z_t"][i]).to(device).unsqueeze(0)
            chunk_t = torch.from_numpy(data["chunks"][i]).to(device).unsqueeze(0)
            z_goal = torch.from_numpy(data["z_goal"][i]).to(device).unsqueeze(0)
            state8 = torch.from_numpy(data["state8_t"][i]).to(device).unsqueeze(0)
            h_goal, p_d_n, aux, _ = substrate.step(
                h_goal, z_t, z_goal, chunk_t, state8)
            p_d = float(p_d_n) * dist_std + dist_mean
            p_g = float(aux["pred_gripper_moving_logit"])
            p_s = float(aux["pred_state_delta"]) * state_delta_std + state_delta_mean
            preds_d.append(p_d); true_d.append(float(targets_raw["dist"][i]))
            preds_g_logit.append(p_g); true_g.append(float(targets_raw["gripper_moving"][i]))
            preds_s.append(p_s); true_s.append(float(targets_raw["state_delta"][i]))
    preds_d = np.array(preds_d); true_d = np.array(true_d)
    preds_g_logit = np.array(preds_g_logit); true_g = np.array(true_g)
    preds_s = np.array(preds_s); true_s = np.array(true_s)

    def r2(t, p):
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        return float(1.0 - ss_res / max(ss_tot, 1e-8))

    r2_d = r2(true_d, preds_d)
    r2_s = r2(true_s, preds_s)
    # Binary accuracy (logit > 0 = positive)
    p_g_bin = (preds_g_logit > 0).astype(np.float32)
    acc_g = float((p_g_bin == true_g).mean())
    pos_rate = float(true_g.mean())
    return {
        "r2_dist": r2_d, "mae_dist": float(np.abs(preds_d - true_d).mean()),
        "r2_state_delta": r2_s,
        "acc_gripper": acc_g, "gripper_pos_rate": pos_rate,
        "composite": r2_d + acc_g + r2_s,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--held_out_data", required=True)
    p.add_argument("--output", default="/tmp/lgt_proprio.pt")
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
    p.add_argument("--alpha_dist", type=float, default=1.0)
    p.add_argument("--alpha_gripper", type=float, default=1.0)
    p.add_argument("--alpha_state_delta", type=float, default=0.5)
    p.add_argument("--gripper_thresh", type=float, default=0.005)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[proprio] device={device}", flush=True)

    train_data = load_triples(args.data)
    val_data = load_triples(args.held_out_data)
    z_vl_dim = train_data["z_t"].shape[1]
    action_horizon = train_data["chunks"].shape[1]
    action_dim = train_data["chunks"].shape[2]
    state_dim = train_data["state8_t"].shape[1]
    print(f"[proprio] z_vl_dim={z_vl_dim} action={action_horizon}x{action_dim} "
          f"state_dim={state_dim}", flush=True)

    train_episodes = build_episode_index(train_data["ep_id"], train_data["suite"])
    val_episodes = build_episode_index(val_data["ep_id"], val_data["suite"])
    train_targets = compute_targets(train_data, train_episodes,
                                       gripper_thresh=args.gripper_thresh)
    val_targets = compute_targets(val_data, val_episodes,
                                     gripper_thresh=args.gripper_thresh)

    dist_mean, dist_std = float(train_targets["dist"].mean()), float(train_targets["dist"].std()) + 1e-6
    sd_mean, sd_std = float(train_targets["state_delta"].mean()), float(train_targets["state_delta"].std()) + 1e-6
    print(f"[proprio] dist: mean={dist_mean:.2f} std={dist_std:.2f}", flush=True)
    print(f"[proprio] state_delta: mean={sd_mean:.4f} std={sd_std:.4f}", flush=True)
    print(f"[proprio] gripper_moving train pos_rate: "
          f"{float(train_targets['gripper_moving'].mean()):.3f}", flush=True)

    train_dist_n = ((train_targets["dist"] - dist_mean) / dist_std).astype(np.float32)
    train_sd_n = ((train_targets["state_delta"] - sd_mean) / sd_std).astype(np.float32)

    substrate = JEPA_LGT_Proprio(
        z_vl_dim=z_vl_dim, action_dim=action_dim, horizon=action_horizon,
        state_dim=state_dim, d=args.d_substrate, K=args.K_belief,
    ).to(device)
    n_params = sum(pp.numel() for pp in substrate.parameters())
    print(f"[proprio] substrate params: {n_params:,}", flush=True)
    opt = torch.optim.AdamW(substrate.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

    init_m = held_out_metrics(substrate, val_data, val_episodes,
                                val_targets, device,
                                dist_mean, dist_std, sd_mean, sd_std)
    print(f"[proprio] INIT held-out: r2_dist={init_m['r2_dist']:+.3f} "
          f"acc_gripper={init_m['acc_gripper']:.3f} (pos_rate={init_m['gripper_pos_rate']:.3f})  "
          f"r2_state_delta={init_m['r2_state_delta']:+.3f}  "
          f"composite={init_m['composite']:+.3f}", flush=True)

    best_score = init_m["composite"]
    best_step = 0
    consec_worse = 0
    rolling = {"d": [], "g": [], "s": []}
    t_start = time.time()

    for step in range(args.max_steps):
        # Sample episode window
        ep = train_episodes[np.random.randint(len(train_episodes))]
        if len(ep) > args.max_turns_per_ep:
            s = np.random.randint(0, len(ep) - args.max_turns_per_ep + 1)
            window = ep[s:s + args.max_turns_per_ep]
        else:
            window = ep
        h_goal = substrate.init_state(1, device)
        d_losses = []
        g_losses = []
        s_losses = []
        for i in window:
            z_t = torch.from_numpy(train_data["z_t"][i]).to(device).unsqueeze(0)
            chunk_t = torch.from_numpy(train_data["chunks"][i]).to(device).unsqueeze(0)
            z_goal = torch.from_numpy(train_data["z_goal"][i]).to(device).unsqueeze(0)
            state8 = torch.from_numpy(train_data["state8_t"][i]).to(device).unsqueeze(0)
            t_d = torch.tensor([train_dist_n[i]], device=device)
            t_g = torch.tensor([train_targets["gripper_moving"][i]], device=device)
            t_s = torch.tensor([train_sd_n[i]], device=device)
            h_goal, p_d, aux, _ = substrate.step(h_goal, z_t, z_goal, chunk_t, state8)
            l_d = F.mse_loss(p_d, t_d)
            l_g = F.binary_cross_entropy_with_logits(
                aux["pred_gripper_moving_logit"], t_g)
            l_s = F.mse_loss(aux["pred_state_delta"], t_s)
            d_losses.append(l_d); g_losses.append(l_g); s_losses.append(l_s)
        loss = (args.alpha_dist * torch.stack(d_losses).mean()
                + args.alpha_gripper * torch.stack(g_losses).mean()
                + args.alpha_state_delta * torch.stack(s_losses).mean())
        opt.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(substrate.parameters(),
                                            args.max_grad_norm)
        opt.step()

        rolling["d"].append(float(torch.stack(d_losses).mean()))
        rolling["g"].append(float(torch.stack(g_losses).mean()))
        rolling["s"].append(float(torch.stack(s_losses).mean()))
        for k in rolling: rolling[k] = rolling[k][-50:]

        if step % args.log_every == 0:
            print(f"step {step:>5}  d_loss={np.mean(rolling['d']):.4f}  "
                  f"g_loss={np.mean(rolling['g']):.4f}  "
                  f"s_loss={np.mean(rolling['s']):.4f}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)

        if step > 0 and step % args.validate_every == 0:
            m = held_out_metrics(substrate, val_data, val_episodes,
                                   val_targets, device,
                                   dist_mean, dist_std, sd_mean, sd_std)
            print(f"  [VAL step {step}] r2_dist={m['r2_dist']:+.3f}  "
                  f"acc_gripper={m['acc_gripper']:.3f}  "
                  f"r2_state_delta={m['r2_state_delta']:+.3f}  "
                  f"composite={m['composite']:+.3f}  "
                  f"(best={best_score:+.3f} @step{best_step})", flush=True)
            if m["composite"] > best_score:
                best_score = m["composite"]; best_step = step
                consec_worse = 0
                torch.save({
                    "substrate_state_dict": substrate.state_dict(),
                    "args": vars(args), "z_vl_dim": z_vl_dim,
                    "action_dim": action_dim, "horizon": action_horizon,
                    "state_dim": state_dim, "step": step,
                    "val_metrics": m,
                    "dist_mean": dist_mean, "dist_std": dist_std,
                    "sd_mean": sd_mean, "sd_std": sd_std,
                }, args.output.replace(".pt", "_best.pt"))
            else:
                consec_worse += 1
                if consec_worse >= args.early_stop_patience:
                    print(f"  [EARLY STOP] composite worsened for {consec_worse} "
                          f"validations. Best: {best_score:+.3f} @step{best_step}",
                          flush=True)
                    break

        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            torch.save({
                "substrate_state_dict": substrate.state_dict(),
                "args": vars(args), "z_vl_dim": z_vl_dim,
                "action_dim": action_dim, "horizon": action_horizon,
                "state_dim": state_dim, "step": step,
                "dist_mean": dist_mean, "dist_std": dist_std,
                "sd_mean": sd_mean, "sd_std": sd_std,
            }, args.output.replace(".pt", f"_step{step}.pt"))

    torch.save({
        "substrate_state_dict": substrate.state_dict(),
        "args": vars(args), "z_vl_dim": z_vl_dim,
        "action_dim": action_dim, "horizon": action_horizon,
        "state_dim": state_dim, "step": args.max_steps,
        "best_score": best_score, "best_step": best_step,
        "dist_mean": dist_mean, "dist_std": dist_std,
        "sd_mean": sd_mean, "sd_std": sd_std,
    }, args.output)
    print(f"\n[proprio] saved → {args.output}", flush=True)
    print(f"[proprio] best composite: {best_score:+.3f} @step{best_step}", flush=True)


if __name__ == "__main__":
    main()
