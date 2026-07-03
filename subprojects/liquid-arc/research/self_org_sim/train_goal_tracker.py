"""Train SubstrateGoalTracker via BPTT over expert episodes.

Key difference from previous trainers: data is processed as EPISODE SEQUENCES,
not random samples. The substrate's h_goal carries across all turns in an
episode. Loss is accumulated turn-by-turn, then backprop through the whole
sequence (truncated BPTT for long episodes).

For each episode:
  h_goal = goal_tracker.init_state()
  for t in 0..episode_length, stepping by exec_horizon:
    obs_t = imgs[t], wrists[t], states[t]
    expert_chunk_t = teacher_chunks[t]  # this is the EXPERT chunk (proxy for GR00T)
    h_goal, gripper_logits, info = goal_tracker.step(h_goal, obs_features, state, expert_chunk_t)
    target = (expert_chunk_t[..., -1] < 0).float()  # P(open) labels
    loss_t = focal_BCE(gripper_logits, target)
  total_loss = sum(loss_t)
  backprop

Truncated BPTT: detach h_goal every `truncate_every` turns to bound memory.
"""
from __future__ import annotations
import argparse
import json
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

from distill_groot_flow import LiquidEncoder, LiquidFlowPolicy  # type: ignore
from substrate_goal_tracker import SubstrateGoalTracker  # type: ignore

torch.set_float32_matmul_precision("high")


SUITE_DIRS_DEFAULT = [
    "/home/pokazge/datasets/libero-10-expert-v1",
    "/home/pokazge/datasets/libero-goal-expert-v1",
    "/home/pokazge/datasets/libero-object-expert-v1",
    "/home/pokazge/datasets/libero-spatial-expert-v1",
]


def load_episode_data(suite_dirs):
    """For each suite, load: episode_starts, episode_lengths, imgs.dat, wrists.dat,
    states.dat memmaps, plus teacher_chunks.dat memmap.

    Returns:
        episodes: list of dicts, one per episode. Each has:
            suite, ep_global, n_turns, sample_indices (per turn within suite)
        memmaps: dict suite_dir → {imgs, wrists, states, chunks, n_total, img_size}
    """
    memmaps = {}
    episodes = []
    for sd in suite_dirs:
        sd_p = Path(sd)
        if not sd_p.exists():
            print(f"[load] skip {sd}: not found")
            continue
        idx = np.load(sd_p / "index.npz")
        starts = idx["episode_starts"]
        lengths = idx["episode_lengths"]
        n_total = int(idx["n_total"])
        img_size = int(idx["img_size"])

        labels = np.load(sd_p / "labels_index.npz")
        sample_idx = labels["sample_idx"]  # [n_samples, 3] = (ep, t, task)
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
            "starts": starts, "lengths": lengths, "n_total": n_total,
            "img_size": img_size, "sample_idx": sample_idx,
        }

        # Group samples by episode
        for ep_i in range(len(lengths)):
            mask = sample_idx[:, 0] == ep_i
            if not mask.any():
                continue
            ep_samples = np.where(mask)[0]
            order = np.argsort(sample_idx[ep_samples, 1])
            ep_samples_sorted = ep_samples[order]
            episodes.append({
                "suite": str(sd_p),
                "ep": int(ep_i),
                "sample_indices": ep_samples_sorted,
            })
    print(f"[load] {len(episodes)} episodes across {len(memmaps)} suites")
    return episodes, memmaps


def sample_episode_subsequence(ep, memmaps, exec_horizon=8, max_turns=40,
                                 target_img_size=224):
    """Extract a subsequence of turns from an episode at exec_horizon stride.

    Returns dict of tensors:
        imgs: [T, 224, 224, 3] uint8
        wrists: [T, 224, 224, 3] uint8
        states: [T, 8] float32
        chunks: [T, 16, 7] float32
    """
    mm = memmaps[ep["suite"]]
    samples = ep["sample_indices"]
    # Subsample at exec_horizon stride
    turn_idxs = np.arange(0, len(samples), exec_horizon)
    if len(turn_idxs) > max_turns:
        # Random window of max_turns
        start = np.random.randint(0, len(turn_idxs) - max_turns + 1)
        turn_idxs = turn_idxs[start:start + max_turns]
    turn_samples = samples[turn_idxs]

    # For each turn, look up (ep, t) → global_idx
    ep_idx = ep["ep"]
    ts = mm["sample_idx"][turn_samples, 1]  # t within episode
    global_idxs = mm["starts"][ep_idx] + ts  # global index in imgs/wrists/states

    imgs = np.array(mm["imgs"][global_idxs])      # [T, H, W, 3]
    wrists = np.array(mm["wrists"][global_idxs])
    states = np.array(mm["states"][global_idxs])
    chunks = np.array(mm["chunks"][turn_samples])  # [T, 16, 7]

    # Resize if needed (use torch F.interpolate to avoid cv2 dependency)
    if imgs.shape[1] != target_img_size:
        imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()  # [T, 3, H, W]
        wrists_t = torch.from_numpy(wrists).permute(0, 3, 1, 2).float()
        imgs_t = torch.nn.functional.interpolate(imgs_t, size=(target_img_size, target_img_size),
                                                  mode="bilinear", align_corners=False)
        wrists_t = torch.nn.functional.interpolate(wrists_t, size=(target_img_size, target_img_size),
                                                    mode="bilinear", align_corners=False)
        imgs = imgs_t.permute(0, 2, 3, 1).byte().numpy()
        wrists = wrists_t.permute(0, 2, 3, 1).byte().numpy()

    return {
        "imgs": torch.from_numpy(imgs),
        "wrists": torch.from_numpy(wrists),
        "states": torch.from_numpy(states),
        "chunks": torch.from_numpy(chunks),
    }


def collate_episodes(eps_data, device):
    """Stack a batch of episodes to common length (truncate to min)."""
    min_T = min(ep["chunks"].shape[0] for ep in eps_data)
    imgs = torch.stack([ep["imgs"][:min_T] for ep in eps_data]).to(device)
    wrists = torch.stack([ep["wrists"][:min_T] for ep in eps_data]).to(device)
    states = torch.stack([ep["states"][:min_T] for ep in eps_data]).to(device)
    chunks = torch.stack([ep["chunks"][:min_T] for ep in eps_data]).to(device)
    return imgs, wrists, states, chunks, min_T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite_dirs", default=",".join(SUITE_DIRS_DEFAULT), type=str)
    p.add_argument("--v10_ckpt", required=True, type=str,
                   help="v10-DEMO checkpoint (frozen encoder for obs features)")
    p.add_argument("--output", default="/tmp/goal_tracker.pt")
    p.add_argument("--batch_episodes", type=int, default=4,
                   help="Number of episodes per BPTT batch")
    p.add_argument("--max_turns_per_ep", type=int, default=24,
                   help="Truncated BPTT length")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--target_img_size", type=int, default=224)
    p.add_argument("--false_close_weight", type=float, default=3.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--d_substrate", type=int, default=128)
    p.add_argument("--K_belief", type=int, default=8)
    args = p.parse_args()

    suite_dirs = [d.strip() for d in args.suite_dirs.split(",") if d.strip()]
    episodes, memmaps = load_episode_data(suite_dirs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Frozen v10 FULL POLICY (encoder + flow head + gripper head) ===
    # CRITICAL: we use v10's predicted chunks (not expert chunks) as the chunk
    # input to the substrate. This ensures the substrate is trained on the
    # SAME chunk distribution it sees at inference, fixing the previous
    # train/inference distribution shift that caused noisy GT predictions.
    print(f"[gt] loading frozen v10 FULL POLICY from {args.v10_ckpt}")
    v10_ckpt = torch.load(args.v10_ckpt, map_location=device, weights_only=False)
    sa = v10_ckpt["args"]
    halt_mode = "learned" if sa["policy"] == "liquid_halt" else "none"
    v10 = LiquidFlowPolicy(
        state_dim=8, action_dim=7, action_horizon=sa["action_horizon"],
        d=sa["d"], d_vis=sa["d"], img_size=sa["img_size"],
        k_max=sa["k"], halt_mode=halt_mode,
        min_steps=sa["halting_min_steps"],
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
    print(f"[gt] loaded {loaded}/{len(own)} v10 tensors (frozen, full policy)")
    v10.eval()
    for pp in v10.parameters():
        pp.requires_grad = False
    v10_encoder = v10.encoder  # alias for the cond extraction path

    # === Goal Tracker ===
    gt = SubstrateGoalTracker(
        d_obs=sa["d"], d_state=8, d_chunk=16 * 7,
        d=args.d_substrate, K=args.K_belief, action_horizon=16,
    ).to(device)
    n_params = sum(p.numel() for p in gt.parameters())
    print(f"[gt] goal tracker params: {n_params:,}")
    opt = torch.optim.AdamW(gt.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_records = []
    t_start = time.time()
    step = 0
    while step < args.max_steps:
        # Sample batch of episodes
        ep_indices = np.random.choice(len(episodes), args.batch_episodes, replace=False)
        eps_data = [sample_episode_subsequence(
            episodes[i], memmaps,
            exec_horizon=8, max_turns=args.max_turns_per_ep,
            target_img_size=args.target_img_size,
        ) for i in ep_indices]
        imgs, wrists, states, chunks, T = collate_episodes(eps_data, device)

        B = imgs.shape[0]
        # Convert imgs to float and CHW
        imgs_f = imgs.float() / 255.0
        wrists_f = wrists.float() / 255.0
        # Shape [B, T, H, W, 3] → [B*T, 3, H, W] then back
        BT = B * T
        imgs_bt = imgs_f.reshape(B*T, *imgs_f.shape[2:]).permute(0, 3, 1, 2).contiguous()
        wrists_bt = wrists_f.reshape(B*T, *wrists_f.shape[2:]).permute(0, 3, 1, 2).contiguous()
        states_bt = states.reshape(B*T, -1)
        chunks_t = chunks  # [B, T, 16, 7]

        # Compute obs features from frozen v10 encoder, one big batch
        with torch.no_grad():
            cond_bt, _ = v10_encoder(imgs_bt, wrists_bt, states_bt)
            cond_t = cond_bt.reshape(B, T, -1)  # [B, T, d_obs]

            # v10 predicted chunks per turn — used as substrate's "chunk" input.
            # Same distribution as inference (vs the previous bug that fed expert
            # chunks during training but v10 chunks at inference).
            v10_chunk_bt = v10.sample(
                imgs_bt, wrists_bt, states_bt, task_id=None, n_steps=10,
            )  # [BT, 16, 7]
            v10_chunks_t = v10_chunk_bt.reshape(B, T, 16, 7).float()

        # Sequential BPTT through episode
        h_goal = gt.init_state(B, device)
        prev_chunk = None
        total_loss = torch.tensor(0.0, device=device)
        acc_open_total = 0.0
        acc_close_total = 0.0
        n_acc_open = 0
        n_acc_close = 0
        cv_total = 0.0
        for turn in range(T):
            obs_t = cond_t[:, turn]                       # [B, d_obs]
            state_t = states[:, turn].float()             # [B, 8]
            chunk_input_t = v10_chunks_t[:, turn]         # [B, 16, 7]  ← v10's prediction
            expert_chunk_t = chunks_t[:, turn]            # [B, 16, 7]  ← for labels only

            h_goal, gripper_logits, info = gt.step(
                h_goal, obs_t, state_t, chunk_input_t,
                prev_groot_chunk=prev_chunk,
            )
            prev_chunk = chunk_input_t.detach()  # next turn's diff baseline

            # Loss target: predict the EXACT OVERRIDE LABEL the cclamp40 fix uses.
            # override_needed[k] = 1 iff (expert is open AT k) AND (v10 wants close AT k).
            # This trains substrate to fire exactly when v10 has drifted from goal.
            # If expert and v10 agree → no override needed; substrate should output 0.
            # If expert open + v10 close → override needed; substrate should output 1.
            # If expert close → substrate should output 0 (don't keep gripper open).
            expert_open = (expert_chunk_t[..., -1] < 0).float()         # [B, 16]
            v10_close = (chunk_input_t[..., -1] > 0).float()             # [B, 16]
            target_open = expert_open * v10_close  # override label [B, 16]
            base = F.binary_cross_entropy_with_logits(gripper_logits, target_open, reduction="none")
            with torch.no_grad():
                model_says_close = (gripper_logits < 0).float()
                penalty = target_open * model_says_close  # expert open + model close = penalty
                weights = 1.0 + (args.false_close_weight - 1.0) * penalty
            loss_t = (base * weights).mean()
            total_loss = total_loss + loss_t

            with torch.no_grad():
                pred_open = (gripper_logits > 0).float()
                open_mask = target_open == 1
                close_mask = target_open == 0
                if open_mask.any():
                    acc_open_total += (pred_open[open_mask] == 1).float().sum().item()
                    n_acc_open += open_mask.sum().item()
                if close_mask.any():
                    acc_close_total += (pred_open[close_mask] == 0).float().sum().item()
                    n_acc_close += close_mask.sum().item()
                cv_total += float(info["metric_cv"])

        # Backprop through full episode
        avg_loss = total_loss / T
        opt.zero_grad()
        avg_loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(gt.parameters(), args.max_grad_norm)
        opt.step()

        if step % args.log_every == 0:
            acc_open = acc_open_total / max(n_acc_open, 1)
            acc_close = acc_close_total / max(n_acc_close, 1)
            cv_mean = cv_total / max(T, 1)
            rec = {
                "step": step, "wall_s": time.time() - t_start,
                "loss": float(avg_loss),
                "T_episode": T,
                "acc_open": acc_open, "acc_close": acc_close,
                "cv_mean": cv_mean,
            }
            log_records.append(rec)
            print(f"step {step:>5}  loss={rec['loss']:.4f}  T={T}  "
                  f"acc_open={acc_open:.3f}  acc_close={acc_close:.3f}  "
                  f"cv={cv_mean:.3f}  wall={rec['wall_s']:.0f}s")
        step += 1

    torch.save({
        "state_dict": gt.state_dict(), "args": vars(args),
        "d_obs": sa["d"], "K_belief": args.K_belief, "d_substrate": args.d_substrate,
    }, args.output)
    print(f"\n[gt] saved → {args.output}")


if __name__ == "__main__":
    main()
