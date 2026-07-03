"""Build episodic memory bank from GR00T-in-sim trajectories.

For each (obs, action_chunk) pair, compute the Liquid encoder's vision+state
fused embedding and store (embedding, action_chunk). At deployment, the
student retrieves nearest-neighbor entries by cosine similarity and blends
with its own prediction.

Run on Spark in main venv (CUDA torch):
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
  python build_memory_bank.py \\
    --student_ckpt /tmp/distill_flow_notext_v1/step_030000.pt \\
    --source_dirs /home/pokazge/datasets/groot-in-sim-iter2-720,/home/pokazge/datasets/groot-sim-spatial \\
    --out_path /home/pokazge/datasets/memory_bank_v1.npz
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot_flow import LiquidFlowPolicy

print = functools.partial(print, flush=True)
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
except Exception:
    pass


def load_flow_policy(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa = ckpt["args"]
    halt_mode = "learned" if sa["policy"] == "liquid_halt" else "none"
    model = LiquidFlowPolicy(
        state_dim=8, action_dim=7, action_horizon=sa["action_horizon"],
        d=sa["d"], d_vis=sa["d"], img_size=sa["img_size"], k_max=sa["k"],
        halt_mode=halt_mode, min_steps=sa["halting_min_steps"],
        n_tasks=sa["n_tasks"], d_task=sa["d_task"],
        head_d=sa["head_d"], head_layers=sa["head_layers"], head_heads=sa["head_heads"],
        n_task_heads=sa.get("n_task_heads", 0),
    ).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own = model.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
    model.eval()
    return model, sa


@torch.no_grad()
def encode_batch(model, imgs, wrists, states, device, target_size):
    """Returns d-dim post-encoder features for batch (vision+state fused, BEFORE ODE)."""
    from PIL import Image
    if imgs.shape[1] != target_size:
        # Resize all imgs in batch via PIL
        imgs_r = np.stack([np.array(Image.fromarray(x).resize((target_size, target_size)), dtype=np.uint8) for x in imgs])
        wrists_r = np.stack([np.array(Image.fromarray(x).resize((target_size, target_size)), dtype=np.uint8) for x in wrists])
    else:
        imgs_r = imgs; wrists_r = wrists
    img_t = torch.from_numpy(imgs_r).to(device).float().permute(0, 3, 1, 2) / 255.0
    wri_t = torch.from_numpy(wrists_r).to(device).float().permute(0, 3, 1, 2) / 255.0
    st_t = torch.from_numpy(states).to(device).float()
    # Use the encoder's pre-ODE fused feature (after self.fuse(...))
    # but we want POST-ODE since that captures the conditioning the head sees
    cond, _ = model.encoder(img_t, wri_t, st_t, task_id=None)
    # Normalize for cosine similarity
    cond_normed = F.normalize(cond, dim=-1)
    return cond_normed.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", required=True, type=str)
    p.add_argument("--source_dirs", required=True, type=str,
                   help="Comma-separated list of source dirs (groot-in-sim style)")
    p.add_argument("--out_path", required=True, type=str)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_per_source", type=int, default=0,
                   help="Cap entries per source (0 = all)")
    p.add_argument("--success_only", action="store_true",
                   help="Only include samples from trajectories with success=True")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sa = load_flow_policy(Path(args.student_ckpt), device)
    target_size = sa["img_size"]
    d_feat = sa["d"]
    K = sa["action_horizon"]
    print(f"Model loaded: d={d_feat}, img_size={target_size}, action_horizon={K}")

    all_features = []
    all_actions = []
    all_meta = []  # source name + (ep, t)

    sources = [s.strip() for s in args.source_dirs.split(",") if s.strip()]
    for src in sources:
        src_dir = Path(src)
        idx = np.load(src_dir / "index.npz")
        starts = idx["episode_starts"]; lengths = idx["episode_lengths"]
        n_total = int(idx["n_total"])
        img_size = int(idx["img_size"])
        success_arr = idx.get("success_per_episode")
        # Has teacher_chunks?
        chunks_path = src_dir / "teacher_chunks.dat"
        if not chunks_path.exists():
            print(f"  WARN: {src} has no teacher_chunks.dat, skipping")
            continue
        # Load samples meta
        labels_idx = np.load(src_dir / "labels_index.npz") if (src_dir / "labels_index.npz").exists() else None
        if labels_idx is not None:
            sample_idx = labels_idx["sample_idx"]
            n_samples = int(labels_idx["n_samples"])
        else:
            # Build sequentially
            sample_idx = []
            for ep_i in range(len(lengths)):
                for t in range(int(lengths[ep_i])):
                    sample_idx.append([ep_i, t, int(idx["task_indices"][ep_i])])
            sample_idx = np.array(sample_idx, dtype=np.int64)
            n_samples = len(sample_idx)

        imgs = np.memmap(src_dir / "imgs.dat", dtype=np.uint8, mode="r",
                         shape=(n_total, img_size, img_size, 3))
        wrists = np.memmap(src_dir / "wrists.dat", dtype=np.uint8, mode="r",
                           shape=(n_total, img_size, img_size, 3))
        states = np.memmap(src_dir / "states.dat", dtype=np.float32, mode="r",
                           shape=(n_total, 8))
        chunks = np.memmap(chunks_path, dtype=np.float32, mode="r",
                           shape=(n_samples, K, 7))

        # Filter to success-only trajectories if requested
        if args.success_only and success_arr is not None:
            keep_mask = np.array([bool(success_arr[int(sample_idx[j][0])]) for j in range(n_samples)])
            kept_idx = np.where(keep_mask)[0]
            sample_idx = sample_idx[kept_idx]
            chunks_filtered = chunks[kept_idx]
            n_samples = len(sample_idx)
            print(f"  {src_dir.name}: success-only filter kept {n_samples} samples")
            chunks = chunks_filtered

        cap = min(n_samples, args.max_per_source) if args.max_per_source > 0 else n_samples
        print(f"  {src_dir.name}: {n_samples} samples, processing {cap}")

        t0 = time.time()
        for start in range(0, cap, args.batch_size):
            end = min(start + args.batch_size, cap)
            B = end - start
            batch_imgs = []
            batch_wrists = []
            batch_states = []
            batch_actions = []
            batch_meta = []
            for j in range(start, end):
                ep_i, t, ti = sample_idx[j]
                global_idx = int(starts[int(ep_i)]) + int(t)
                batch_imgs.append(np.array(imgs[global_idx]))
                batch_wrists.append(np.array(wrists[global_idx]))
                batch_states.append(np.array(states[global_idx]))
                batch_actions.append(np.array(chunks[j]))
                ep_success = bool(success_arr[int(ep_i)]) if success_arr is not None else None
                batch_meta.append((src_dir.name, int(ep_i), int(t), int(ti), ep_success))

            batch_imgs = np.stack(batch_imgs)
            batch_wrists = np.stack(batch_wrists)
            batch_states = np.stack(batch_states)
            feats = encode_batch(model, batch_imgs, batch_wrists, batch_states, device, target_size)
            all_features.append(feats)
            all_actions.append(np.stack(batch_actions))
            all_meta.extend(batch_meta)

            if (start + B) % (args.batch_size * 10) == 0:
                elapsed = time.time() - t0
                rate = (start + B) / elapsed if elapsed > 0 else 0
                print(f"    {start + B}/{cap}  rate={rate:.1f}/s")

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    print(f"\nMemory bank: {features.shape[0]} entries, feat_dim={features.shape[1]}, action shape={actions.shape}")
    print(f"  features: {features.nbytes/1e6:.1f} MB")
    print(f"  actions:  {actions.nbytes/1e6:.1f} MB")

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        features=features,
        actions=actions,
        meta_source=np.array([m[0] for m in all_meta]),
        meta_ep=np.array([m[1] for m in all_meta], dtype=np.int64),
        meta_t=np.array([m[2] for m in all_meta], dtype=np.int64),
        meta_task=np.array([m[3] for m in all_meta], dtype=np.int64),
        meta_success=np.array([m[4] if m[4] is not None else -1 for m in all_meta], dtype=np.int64),
        ckpt=str(args.student_ckpt), d_feat=np.int64(features.shape[1]),
        action_horizon=np.int64(actions.shape[1]),
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
