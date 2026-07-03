"""Train GoalImageResidualSubstrate.

Per-sample training: substrate sees (obs, state, GR00T_chunk, goal_features)
and predicts delta. Target is clamped (expert_chunk - GR00T_chunk)[xyz/rpy].

Data: pre-collected GR00T chunks at /tmp/groot_chunks_<suite>.npz +
expert chunks from /home/pokazge/datasets/libero-*-expert-v1/teacher_chunks.dat.

Note: this is per-sample not per-episode (no BPTT through h_goal across episode).
Each turn is independent — substrate's h_goal resets per sample. This is OK
because the correction is a per-step local adjustment, not a long-horizon belief.
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

from distill_groot_flow import LiquidFlowPolicy  # type: ignore
from goal_image_residual_substrate import GoalImageResidualSubstrate  # type: ignore

torch.set_float32_matmul_precision("high")


SUITES = ["libero_10", "libero_object", "libero_goal", "libero_spatial"]


def load_all_samples(suites, dataset_root, groot_dir, goal_features):
    """Load (img, wrist, state8, expert_chunk, groot_chunk, goal_feat) per sample.

    Returns a list of dicts.
    """
    samples = []
    for suite in suites:
        suite_short = suite.replace("libero_", "")
        sd = Path(dataset_root) / f"libero-{suite_short}-expert-v1"
        if not sd.exists():
            print(f"  skip {suite}")
            continue
        idx = np.load(sd / "index.npz")
        starts = idx["episode_starts"]
        lengths = idx["episode_lengths"]
        task_indices = idx["task_indices"]
        n_total = int(idx["n_total"])
        img_size = int(idx["img_size"])
        labels = np.load(sd / "labels_index.npz")
        sample_idx_full = labels["sample_idx"]
        n_samples_full = int(labels["n_samples"])

        imgs = np.memmap(sd / "imgs.dat", dtype=np.uint8, mode="r",
                         shape=(n_total, img_size, img_size, 3))
        wrists = np.memmap(sd / "wrists.dat", dtype=np.uint8, mode="r",
                           shape=(n_total, img_size, img_size, 3))
        states = np.memmap(sd / "states.dat", dtype=np.float32, mode="r",
                           shape=(n_total, 8))
        teacher_chunks = np.memmap(sd / "teacher_chunks.dat", dtype=np.float32, mode="r",
                                     shape=(n_samples_full, 16, 7))

        gc = np.load(Path(groot_dir) / f"groot_chunks_{suite}.npz")
        gc_chunks = gc["groot_chunks"]            # [N, 16, 7]
        gc_sample_idx = gc["sample_idx"]          # [N] indices into sample_idx_full
        gc_meta = gc["meta"]                       # [N, 3] (ep, t, task)

        gf = goal_features.get(suite)
        if gf is None:
            print(f"  no goal features for {suite}")
            continue

        for i in range(len(gc_chunks)):
            si = int(gc_sample_idx[i])
            ep, t, task_id = gc_meta[i]
            gi = int(starts[ep]) + int(t)
            samples.append({
                "suite": suite,
                "global_idx": gi,
                "imgs_mem": imgs,
                "wrists_mem": wrists,
                "states_mem": states,
                "expert_chunk": teacher_chunks[si].copy(),  # [16, 7]
                "groot_chunk": gc_chunks[i],                  # [16, 7]
                "goal_feat": gf[int(task_id)],                # [384]
                "task_id": int(task_id),
            })
    print(f"[res] loaded {len(samples)} samples")
    return samples


def make_batch(samples, indices, device, target_img_size=224):
    imgs = []
    wrists = []
    states = []
    expert_chunks = []
    groot_chunks = []
    goal_feats = []
    for i in indices:
        s = samples[i]
        gi = s["global_idx"]
        img = np.array(s["imgs_mem"][gi])
        wri = np.array(s["wrists_mem"][gi])
        st = np.array(s["states_mem"][gi])
        if img.shape[0] != target_img_size:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
            wri_t = torch.from_numpy(wri).permute(2, 0, 1).float().unsqueeze(0)
            img_t = F.interpolate(img_t, size=(target_img_size, target_img_size),
                                    mode="bilinear", align_corners=False)
            wri_t = F.interpolate(wri_t, size=(target_img_size, target_img_size),
                                    mode="bilinear", align_corners=False)
            img = img_t.squeeze(0).permute(1, 2, 0).byte().numpy()
            wri = wri_t.squeeze(0).permute(1, 2, 0).byte().numpy()
        imgs.append(img)
        wrists.append(wri)
        states.append(st)
        expert_chunks.append(s["expert_chunk"])
        groot_chunks.append(s["groot_chunk"])
        goal_feats.append(s["goal_feat"])
    imgs_t = torch.from_numpy(np.stack(imgs)).to(device).float().permute(0, 3, 1, 2) / 255.0
    wrists_t = torch.from_numpy(np.stack(wrists)).to(device).float().permute(0, 3, 1, 2) / 255.0
    states_t = torch.from_numpy(np.stack(states)).to(device).float()
    expert_chunks_t = torch.from_numpy(np.stack(expert_chunks)).to(device).float()
    groot_chunks_t = torch.from_numpy(np.stack(groot_chunks)).to(device).float()
    goal_feats_t = torch.from_numpy(np.stack(goal_feats)).to(device).float()
    return imgs_t, wrists_t, states_t, expert_chunks_t, groot_chunks_t, goal_feats_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default="/home/pokazge/datasets")
    p.add_argument("--groot_dir", default="/tmp")
    p.add_argument("--goal_features", default="/tmp/goal_features.npz")
    p.add_argument("--v10_ckpt", required=True,
                   help="v10 ckpt — used ONLY for its frozen encoder (cond from img+wrist+state)")
    p.add_argument("--output", default="/tmp/residual_substrate.pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=300)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_delta", type=float, default=0.05)
    p.add_argument("--d_substrate", type=int, default=128)
    p.add_argument("--K_belief", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load goal features
    gf_npz = np.load(args.goal_features)
    goal_features = {
        s: gf_npz[f"{s}_features"]
        for s in SUITES
        if f"{s}_features" in gf_npz.files
    }
    d_goal = next(iter(goal_features.values())).shape[1]
    print(f"[res] goal features: {list(goal_features.keys())}, d_goal={d_goal}")

    # Frozen v10 encoder (only encoder used; no flow head needed)
    print(f"[res] loading v10 encoder from {args.v10_ckpt}")
    v10_ck = torch.load(args.v10_ckpt, map_location=device, weights_only=False)
    sa = v10_ck["args"]
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
    sd = v10_ck.get("policy", v10_ck.get("model", v10_ck))
    own = v10.state_dict()
    loaded = 0
    for k, v in sd.items():
        kk = k.replace("_orig_mod.", "")
        if kk in own and own[kk].shape == v.shape:
            own[kk].copy_(v); loaded += 1
    print(f"[res] v10 loaded {loaded}/{len(own)}")
    v10.eval()
    for pp in v10.parameters():
        pp.requires_grad = False

    samples = load_all_samples(SUITES, args.dataset_root, args.groot_dir, goal_features)
    if len(samples) < args.batch_size:
        raise SystemExit(f"only {len(samples)} samples; need at least {args.batch_size}")

    sub = GoalImageResidualSubstrate(
        d_obs=sa["d"], d_state=8, d_chunk=16 * 7, d_goal=d_goal,
        d=args.d_substrate, K=args.K_belief, action_horizon=16,
        action_dim_residual=6, max_delta=args.max_delta,
    ).to(device)
    n_params = sum(p.numel() for p in sub.parameters())
    print(f"[res] residual substrate params: {n_params:,}")
    opt = torch.optim.AdamW(sub.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    t_start = time.time()
    for step in range(args.max_steps):
        idxs = np.random.choice(len(samples), args.batch_size, replace=True)
        imgs, wrists, states, expert_chunks, groot_chunks, goal_feats = make_batch(
            samples, idxs, device,
        )
        with torch.no_grad():
            cond, _ = v10.encoder(imgs, wrists, states)
        h_goal = sub.init_state(args.batch_size, device)
        h_goal, delta, info = sub.step(
            h_goal, cond, states, groot_chunks, goal_feats,
        )
        # Target: delta we'd need to apply to GR00T's xyz/rpy to match expert
        # Clip the target to substrate's tanh×max_delta range to avoid pushing
        # against the bound
        target_delta = (expert_chunks[..., :6] - groot_chunks[..., :6])
        target_delta = target_delta.clamp(-args.max_delta * 2, args.max_delta * 2)
        loss = F.mse_loss(delta, target_delta)
        opt.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(sub.parameters(), args.max_grad_norm)
        opt.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                naive_loss = F.mse_loss(torch.zeros_like(delta), target_delta).item()
                improvement = (naive_loss - loss.item()) / max(naive_loss, 1e-6)
            print(f"step {step:>5}  loss={float(loss):.5f}  naive={naive_loss:.5f}  "
                  f"improvement={improvement:.2%}  delta_norm={float(info['delta_norm']):.4f}  "
                  f"cv={float(info['metric_cv']):.3f}  wall={time.time()-t_start:.0f}s")

        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            ckpt_path = args.output.replace(".pt", f"_step{step}.pt")
            torch.save({
                "state_dict": sub.state_dict(), "args": vars(args),
                "d_obs": sa["d"], "d_goal": d_goal,
                "K_belief": args.K_belief, "d_substrate": args.d_substrate,
                "max_delta": args.max_delta, "step": step,
            }, ckpt_path)
            print(f"[res] ckpt → {ckpt_path}")

    torch.save({
        "state_dict": sub.state_dict(), "args": vars(args),
        "d_obs": sa["d"], "d_goal": d_goal,
        "K_belief": args.K_belief, "d_substrate": args.d_substrate,
        "max_delta": args.max_delta,
    }, args.output)
    print(f"\n[res] saved final → {args.output}")


if __name__ == "__main__":
    main()
