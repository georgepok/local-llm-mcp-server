"""Train GoalImageSubstrate via BPTT over expert episodes.

For each expert episode:
  goal_features = goal_features_table[(suite, task_id)]   # static for episode
  h_goal = init_state()
  for t in 0..N stepping by exec_horizon:
    obs_features = v10_encoder(imgs[t], wrists[t], states[t])
    chunk = v10.sample(...) at this state (same distribution as inference)
    h_goal, progress, goal_reached_logit, info = substrate.step(
        h_goal, obs_features, state, chunk, goal_features, prev_chunk
    )
    target_progress = t / (N - 1)
    target_goal_reached = float(t >= N - 16)
    loss_t = MSE(progress, target_progress) + BCE(goal_reached_logit, target_goal_reached)
  backprop

Truncated BPTT length = max_turns_per_ep turns.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from distill_groot_flow import LiquidFlowPolicy  # type: ignore
from goal_image_substrate import GoalImageSubstrate  # type: ignore

torch.set_float32_matmul_precision("high")


SUITE_DIRS_DEFAULT = [
    "/home/pokazge/datasets/libero-10-expert-v1",
    "/home/pokazge/datasets/libero-goal-expert-v1",
    "/home/pokazge/datasets/libero-object-expert-v1",
    "/home/pokazge/datasets/libero-spatial-expert-v1",
]

SUITE_NAME_FROM_DIR = {
    "libero-10-expert-v1": "libero_10",
    "libero-object-expert-v1": "libero_object",
    "libero-goal-expert-v1": "libero_goal",
    "libero-spatial-expert-v1": "libero_spatial",
}


def load_episode_data(suite_dirs):
    memmaps = {}
    episodes = []
    for sd in suite_dirs:
        sd_p = Path(sd)
        if not sd_p.exists():
            print(f"[load] skip {sd}: not found")
            continue
        suite_name = SUITE_NAME_FROM_DIR.get(sd_p.name)
        if suite_name is None:
            print(f"[load] unknown suite dir {sd_p.name}")
            continue

        idx = np.load(sd_p / "index.npz")
        starts = idx["episode_starts"]
        lengths = idx["episode_lengths"]
        task_indices = idx["task_indices"]
        success = idx.get("success_per_episode", np.ones(len(lengths), dtype=bool))
        n_total = int(idx["n_total"])
        img_size = int(idx["img_size"])

        labels = np.load(sd_p / "labels_index.npz")
        sample_idx = labels["sample_idx"]
        n_samples = int(labels["n_samples"])

        imgs = np.memmap(sd_p / "imgs.dat", dtype=np.uint8, mode="r",
                         shape=(n_total, img_size, img_size, 3))
        wrists = np.memmap(sd_p / "wrists.dat", dtype=np.uint8, mode="r",
                           shape=(n_total, img_size, img_size, 3))
        states = np.memmap(sd_p / "states.dat", dtype=np.float32, mode="r",
                           shape=(n_total, 8))
        chunks = np.memmap(sd_p / "teacher_chunks.dat", dtype=np.float32, mode="r",
                           shape=(n_samples, 16, 7))
        memmaps[str(sd_p)] = {
            "imgs": imgs, "wrists": wrists, "states": states, "chunks": chunks,
            "starts": starts, "lengths": lengths, "task_indices": task_indices,
            "success": success, "n_total": n_total, "img_size": img_size,
            "sample_idx": sample_idx, "suite_name": suite_name,
        }

        for ep_i in range(len(lengths)):
            if not bool(success[ep_i]):
                continue
            mask = sample_idx[:, 0] == ep_i
            if not mask.any():
                continue
            ep_samples = np.where(mask)[0]
            order = np.argsort(sample_idx[ep_samples, 1])
            ep_samples_sorted = ep_samples[order]
            episodes.append({
                "suite": str(sd_p),
                "suite_name": suite_name,
                "ep": int(ep_i),
                "task_id": int(task_indices[ep_i]),
                "sample_indices": ep_samples_sorted,
                "length": int(lengths[ep_i]),
            })
    print(f"[load] {len(episodes)} successful episodes across {len(memmaps)} suites")
    return episodes, memmaps


def sample_episode_subsequence(ep, memmaps, exec_horizon=8, max_turns=40,
                                 target_img_size=224):
    mm = memmaps[ep["suite"]]
    samples = ep["sample_indices"]
    n_avail_turns = len(samples) // exec_horizon  # number of full-stride turns
    turn_idxs = np.arange(0, len(samples), exec_horizon)
    if len(turn_idxs) > max_turns:
        start = np.random.randint(0, len(turn_idxs) - max_turns + 1)
        turn_idxs = turn_idxs[start:start + max_turns]
    turn_samples = samples[turn_idxs]
    # Absolute time-index of each turn within the episode (for progress target)
    abs_t = mm["sample_idx"][turn_samples, 1]
    ep_total_len = ep["length"]

    ep_idx = ep["ep"]
    global_idxs = mm["starts"][ep_idx] + abs_t

    imgs = np.array(mm["imgs"][global_idxs])
    wrists = np.array(mm["wrists"][global_idxs])
    states = np.array(mm["states"][global_idxs])
    chunks = np.array(mm["chunks"][turn_samples])

    if imgs.shape[1] != target_img_size:
        imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()
        wrists_t = torch.from_numpy(wrists).permute(0, 3, 1, 2).float()
        imgs_t = F.interpolate(imgs_t, size=(target_img_size, target_img_size),
                                mode="bilinear", align_corners=False)
        wrists_t = F.interpolate(wrists_t, size=(target_img_size, target_img_size),
                                  mode="bilinear", align_corners=False)
        imgs = imgs_t.permute(0, 2, 3, 1).byte().numpy()
        wrists = wrists_t.permute(0, 2, 3, 1).byte().numpy()

    return {
        "imgs": torch.from_numpy(imgs),
        "wrists": torch.from_numpy(wrists),
        "states": torch.from_numpy(states),
        "chunks": torch.from_numpy(chunks),
        "abs_t": torch.from_numpy(abs_t.astype(np.int64)),
        "ep_total_len": ep_total_len,
        "suite_name": ep["suite_name"],
        "task_id": ep["task_id"],
    }


def collate_episodes(eps_data, goal_features_per_suite, device):
    min_T = min(ep["chunks"].shape[0] for ep in eps_data)
    imgs = torch.stack([ep["imgs"][:min_T] for ep in eps_data]).to(device)
    wrists = torch.stack([ep["wrists"][:min_T] for ep in eps_data]).to(device)
    states = torch.stack([ep["states"][:min_T] for ep in eps_data]).to(device)
    chunks = torch.stack([ep["chunks"][:min_T] for ep in eps_data]).to(device)
    abs_t = torch.stack([ep["abs_t"][:min_T] for ep in eps_data]).to(device)
    ep_total_lens = torch.tensor([ep["ep_total_len"] for ep in eps_data],
                                   dtype=torch.float32, device=device)
    # Goal features per episode
    goal_feats = torch.stack([
        torch.from_numpy(
            goal_features_per_suite[ep["suite_name"]][ep["task_id"]]
        ).float()
        for ep in eps_data
    ]).to(device)
    return imgs, wrists, states, chunks, abs_t, ep_total_lens, goal_feats, min_T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite_dirs", default=",".join(SUITE_DIRS_DEFAULT), type=str)
    p.add_argument("--v10_ckpt", required=True, type=str)
    p.add_argument("--goal_features", default="/tmp/goal_features.npz", type=str)
    p.add_argument("--output", default="/tmp/goal_image_substrate.pt")
    p.add_argument("--batch_episodes", type=int, default=4)
    p.add_argument("--max_turns_per_ep", type=int, default=24)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=200,
                   help="Save intermediate checkpoint every N steps for crash recovery")
    p.add_argument("--target_img_size", type=int, default=224)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--d_substrate", type=int, default=128)
    p.add_argument("--K_belief", type=int, default=8)
    p.add_argument("--progress_loss_weight", type=float, default=1.0)
    p.add_argument("--goal_reached_loss_weight", type=float, default=1.0)
    p.add_argument("--gr_pos_weight", type=float, default=1.0,
                   help="pos_weight for goal_reached BCE — counters class imbalance "
                        "(positives are ~6% of turns; weight 15-20 balances)")
    p.add_argument("--no_chunk", action="store_true",
                   help="Train substrate without chunk input — pure visual goal-distance. "
                        "Recommended to avoid v10/GR00T chunk distribution shift at inference.")
    args = p.parse_args()

    suite_dirs = [d.strip() for d in args.suite_dirs.split(",") if d.strip()]
    episodes, memmaps = load_episode_data(suite_dirs)

    # Load goal features
    gf = np.load(args.goal_features)
    goal_features_per_suite = {
        suite: gf[f"{suite}_features"]  # [10, 384]
        for suite in ["libero_10", "libero_object", "libero_goal", "libero_spatial"]
        if f"{suite}_features" in gf.files
    }
    d_goal = next(iter(goal_features_per_suite.values())).shape[1]
    print(f"[gi] loaded goal features for suites: {list(goal_features_per_suite.keys())}, d_goal={d_goal}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Frozen v10 (encoder + flow head for chunk sampling)
    print(f"[gi] loading frozen v10 from {args.v10_ckpt}")
    v10_ckpt = torch.load(args.v10_ckpt, map_location=device, weights_only=False)
    sa = v10_ckpt["args"]
    halt_mode = "learned" if sa["policy"] == "liquid_halt" else "none"
    v10 = LiquidFlowPolicy(
        state_dim=8, action_dim=7, action_horizon=sa["action_horizon"],
        d=sa["d"], d_vis=sa["d"], img_size=sa["img_size"],
        k_max=sa["k"], halt_mode=halt_mode, min_steps=sa["halting_min_steps"],
        n_tasks=sa["n_tasks"], d_task=sa["d_task"],
        head_d=sa["head_d"], head_layers=sa["head_layers"],
        head_heads=sa["head_heads"],
        n_task_heads=sa.get("n_task_heads", 0),
        z_groot_dim=sa.get("z_groot_dim", 0),
        gated_mixture=sa.get("gated_mixture", False),
        z_channel_dims=sa.get("z_channel_dims", None),
        query_bank=sa.get("use_query_bank", False),
    ).to(device)
    sd = v10_ckpt.get("policy", v10_ckpt.get("model", v10_ckpt))
    own = v10.state_dict()
    loaded = 0
    for k, v in sd.items():
        kk = k.replace("_orig_mod.", "")
        if kk in own and own[kk].shape == v.shape:
            own[kk].copy_(v); loaded += 1
    print(f"[gi] loaded {loaded}/{len(own)} v10 tensors (frozen)")
    v10.eval()
    for pp in v10.parameters():
        pp.requires_grad = False

    gi = GoalImageSubstrate(
        d_obs=sa["d"], d_state=8, d_chunk=16 * 7,
        d_goal=d_goal, d=args.d_substrate, K=args.K_belief, action_horizon=16,
        use_chunk=not args.no_chunk,
    ).to(device)
    print(f"[gi] use_chunk={not args.no_chunk}")
    n_params = sum(p.numel() for p in gi.parameters())
    print(f"[gi] goal-image substrate params: {n_params:,}")
    opt = torch.optim.AdamW(gi.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

    t_start = time.time()
    step = 0
    while step < args.max_steps:
        ep_indices = np.random.choice(len(episodes), args.batch_episodes, replace=False)
        eps_data = [sample_episode_subsequence(
            episodes[i], memmaps,
            exec_horizon=8, max_turns=args.max_turns_per_ep,
            target_img_size=args.target_img_size,
        ) for i in ep_indices]
        imgs, wrists, states, chunks, abs_t, ep_total_lens, goal_feats, T = collate_episodes(
            eps_data, goal_features_per_suite, device,
        )

        B = imgs.shape[0]
        imgs_f = imgs.float() / 255.0
        wrists_f = wrists.float() / 255.0
        imgs_bt = imgs_f.reshape(B*T, *imgs_f.shape[2:]).permute(0, 3, 1, 2).contiguous()
        wrists_bt = wrists_f.reshape(B*T, *wrists_f.shape[2:]).permute(0, 3, 1, 2).contiguous()
        states_bt = states.reshape(B*T, -1)

        with torch.no_grad():
            cond_bt, _ = v10.encoder(imgs_bt, wrists_bt, states_bt)
            cond_t = cond_bt.reshape(B, T, -1)
            v10_chunk_bt = v10.sample(imgs_bt, wrists_bt, states_bt,
                                        task_id=None, n_steps=10)
            v10_chunks_t = v10_chunk_bt.reshape(B, T, 16, 7).float()

        h_goal = gi.init_state(B, device)
        prev_chunk = None
        total_loss = torch.tensor(0.0, device=device)
        progress_err_sum = 0.0
        gr_acc_sum = 0.0
        gr_pos_recall_sum = 0.0
        gr_pos_recall_n = 0
        cv_sum = 0.0
        for turn in range(T):
            obs_t = cond_t[:, turn]
            state_t = states[:, turn].float()
            chunk_t = v10_chunks_t[:, turn]
            target_progress = (abs_t[:, turn].float() / (ep_total_lens - 1).clamp(min=1)).clamp(0, 1)
            target_goal_reached = (abs_t[:, turn].float() >= (ep_total_lens - 16)).float()

            h_goal, progress, goal_reached_logit, info = gi.step(
                h_goal, obs_t, state_t, chunk_t, goal_feats,
                prev_groot_chunk=prev_chunk,
            )
            prev_chunk = chunk_t.detach()

            loss_p = F.mse_loss(progress, target_progress)
            pos_weight_t = torch.tensor(args.gr_pos_weight, device=device)
            loss_gr = F.binary_cross_entropy_with_logits(
                goal_reached_logit, target_goal_reached,
                pos_weight=pos_weight_t,
            )
            loss_t = (args.progress_loss_weight * loss_p
                       + args.goal_reached_loss_weight * loss_gr)
            total_loss = total_loss + loss_t

            with torch.no_grad():
                progress_err_sum += (progress - target_progress).abs().mean().item()
                pred_gr = (goal_reached_logit > 0).float()
                gr_acc_sum += (pred_gr == target_goal_reached).float().mean().item()
                # Track POSITIVE recall (TP/(TP+FN)) separately — accuracy hides class imbalance
                pos_mask = target_goal_reached > 0.5
                if pos_mask.any():
                    gr_pos_recall_sum += (pred_gr[pos_mask] == 1).float().mean().item()
                    gr_pos_recall_n += 1
                cv_sum += float(info["metric_cv"])

        avg_loss = total_loss / T

        # NaN detection — abort BEFORE backward to preserve last good state
        if not torch.isfinite(avg_loss):
            print(f"[gi] NaN detected at step {step} — aborting; last good ckpt preserved")
            break

        opt.zero_grad()
        avg_loss.backward()
        # Check gradients for NaN
        grad_has_nan = False
        for p in gi.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grad_has_nan = True
                break
        if grad_has_nan:
            print(f"[gi] NaN gradient at step {step} — skipping step, NOT updating weights")
            opt.zero_grad()
            step += 1
            continue
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(gi.parameters(), args.max_grad_norm)
        opt.step()

        if step % args.log_every == 0:
            gr_pos_rec = gr_pos_recall_sum / max(gr_pos_recall_n, 1)
            print(f"step {step:>5}  loss={float(avg_loss):.4f}  T={T}  "
                  f"prog_err={progress_err_sum/T:.3f}  "
                  f"gr_acc={gr_acc_sum/T:.3f} gr_pos_rec={gr_pos_rec:.3f} "
                  f"cv={cv_sum/T:.3f}  wall={time.time()-t_start:.0f}s")

        # Save intermediate checkpoint
        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            ckpt_path = args.output.replace(".pt", f"_step{step}.pt")
            torch.save({
                "state_dict": gi.state_dict(), "args": vars(args),
                "d_obs": sa["d"], "d_goal": d_goal,
                "K_belief": args.K_belief, "d_substrate": args.d_substrate,
                "use_chunk": not args.no_chunk,
                "step": step,
            }, ckpt_path)
            print(f"[gi] ckpt → {ckpt_path}")
        step += 1

    torch.save({
        "state_dict": gi.state_dict(), "args": vars(args),
        "d_obs": sa["d"], "d_goal": d_goal,
        "K_belief": args.K_belief, "d_substrate": args.d_substrate,
        "use_chunk": not args.no_chunk,
        "step": step,
    }, args.output)
    print(f"\n[gi] saved final → {args.output}")


if __name__ == "__main__":
    main()
