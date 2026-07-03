"""Build v11 episodic memory bank from libero-{suite}-expert-v1 datasets.

For each frame in each suite, store:
  - feature [776-d]: DINOv2(agent) || DINOv2(wrist) || L2-norm(state[8])
  - action [16, 7]: teacher_chunks (GR00T's predicted action chunk)
  - metadata: suite_idx, task_idx, ep_idx, step_idx, success_flag

Retrieval feature uses RAW DINOv2 CLS (pre-projection) so the bank is
student-independent and reusable across future checkpoints.

Run on Spark in main venv (CUDA torch):
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  export LD_LIBRARY_PATH=/home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
  python build_memory_bank_v11.py \
    --suites libero_10,libero_goal,libero_object,libero_spatial \
    --dataset_root /home/pokazge/datasets \
    --out_path /home/pokazge/datasets/memory_bank_v11.npz \
    --batch_size 128
"""
from __future__ import annotations

import argparse
import functools
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

print = functools.partial(print, flush=True)
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
except Exception:
    pass

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SUITE_TO_IDX = {
    "libero_10": 0,
    "libero_goal": 1,
    "libero_object": 2,
    "libero_spatial": 3,
}


def load_dinov2(device):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", verbose=False
        )
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval().to(device)
    return backbone


@torch.no_grad()
def dinov2_features(backbone, imgs_uint8, mean, std, device):
    """imgs_uint8: [B, H, W, 3] uint8 → [B, 384] DINOv2 CLS features."""
    x = torch.from_numpy(imgs_uint8).to(device).float().permute(0, 3, 1, 2) / 255.0
    if x.shape[-1] != 224 or x.shape[-2] != 224:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    feats = backbone(x)  # [B, 384]
    return feats


def state_norm_feature(states):
    """states: [B, 8] → [B, 8] each dim L2-normalized over the bank (post-hoc).

    Returns raw states here; we'll L2-normalize the full 776-d feature at end.
    """
    return torch.from_numpy(states).float()


def process_suite(suite_name, dataset_root, backbone, mean, std, device, batch_size):
    suite_dir = Path(dataset_root) / f"libero-{suite_name.replace('libero_', '')}-expert-v1"
    if not suite_dir.exists():
        # Try without the prefix-replacement (libero_10 -> libero-10)
        suite_dir = Path(dataset_root) / f"libero-{suite_name.split('_')[-1]}-expert-v1"
    if not suite_dir.exists():
        print(f"  [SKIP] {suite_name} not found at {suite_dir}")
        return None

    idx = np.load(suite_dir / "index.npz")
    starts = idx["episode_starts"]
    lengths = idx["episode_lengths"]
    task_indices = idx["task_indices"]
    success_per_episode = idx.get("success_per_episode")
    n_total = int(idx["n_total"])
    img_size = int(idx["img_size"])
    n_episodes = len(lengths)

    # Discover action shape from teacher_chunks
    chunks_path = suite_dir / "teacher_chunks.dat"
    if not chunks_path.exists():
        print(f"  [SKIP] {suite_name}: no teacher_chunks.dat")
        return None

    # labels_index maps each sample to (ep, t)
    labels_path = suite_dir / "labels_index.npz"
    if labels_path.exists():
        labels_idx = np.load(labels_path)
        sample_idx = labels_idx["sample_idx"]   # [n_samples, 3] (ep, t, task)
        n_samples = int(labels_idx["n_samples"])
    else:
        rows = []
        for ep_i in range(n_episodes):
            ti = int(task_indices[ep_i])
            for t in range(int(lengths[ep_i])):
                rows.append((ep_i, t, ti))
        sample_idx = np.array(rows, dtype=np.int64)
        n_samples = len(sample_idx)

    # Action horizon discoverable from chunks file size
    K = 16
    A = 7
    chunks_filesize = chunks_path.stat().st_size
    expected = n_samples * K * A * 4
    if chunks_filesize != expected:
        # Try to infer K from filesize
        K = chunks_filesize // (n_samples * A * 4)
        print(f"  {suite_name}: inferred K={K} from filesize {chunks_filesize}")

    imgs = np.memmap(suite_dir / "imgs.dat", dtype=np.uint8, mode="r",
                     shape=(n_total, img_size, img_size, 3))
    wrists = np.memmap(suite_dir / "wrists.dat", dtype=np.uint8, mode="r",
                       shape=(n_total, img_size, img_size, 3))
    states = np.memmap(suite_dir / "states.dat", dtype=np.float32, mode="r",
                       shape=(n_total, 8))
    chunks = np.memmap(chunks_path, dtype=np.float32, mode="r",
                       shape=(n_samples, K, A))

    print(f"\n=== {suite_name}: {n_samples} samples, {n_episodes} episodes, img_size={img_size} ===")

    all_feats = []
    all_actions = []
    all_meta_task = []
    all_meta_ep = []
    all_meta_t = []
    all_meta_success = []

    suite_idx_int = SUITE_TO_IDX[suite_name]
    t0 = time.time()
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        B = end - start
        # Gather batch
        ep_ts = sample_idx[start:end]   # [B, 3] (ep, t, task)
        global_idxs = np.array([
            int(starts[int(ep_ts[j][0])]) + int(ep_ts[j][1]) for j in range(B)
        ])
        batch_imgs = np.array(imgs[global_idxs])      # [B, H, W, 3]
        batch_wrists = np.array(wrists[global_idxs]) # [B, H, W, 3]
        batch_states = np.array(states[global_idxs]) # [B, 8]
        batch_chunks = np.array(chunks[start:end])    # [B, K, A]

        # DINOv2 features for both cameras
        agent_feats = dinov2_features(backbone, batch_imgs, mean, std, device)   # [B, 384]
        wrist_feats = dinov2_features(backbone, batch_wrists, mean, std, device) # [B, 384]
        # Concat with state (raw — we'll normalize the full feature at the end)
        st_t = torch.from_numpy(batch_states).to(device).float()
        feat = torch.cat([agent_feats, wrist_feats, st_t], dim=-1)  # [B, 776]
        feat_np = feat.cpu().numpy().astype(np.float32)

        all_feats.append(feat_np)
        all_actions.append(batch_chunks.astype(np.float32))
        all_meta_task.append(ep_ts[:, 2].astype(np.int64))
        all_meta_ep.append(ep_ts[:, 0].astype(np.int64))
        all_meta_t.append(ep_ts[:, 1].astype(np.int64))
        if success_per_episode is not None:
            succ = np.array([int(success_per_episode[int(ep_ts[j][0])]) for j in range(B)], dtype=np.int64)
        else:
            succ = np.full(B, -1, dtype=np.int64)
        all_meta_success.append(succ)

        if (start + B) % (batch_size * 20) == 0 or end == n_samples:
            elapsed = time.time() - t0
            rate = (start + B) / max(elapsed, 1e-3)
            eta = (n_samples - (start + B)) / max(rate, 1e-3)
            print(f"  {start + B}/{n_samples}  {rate:.0f}/s  eta {eta:.0f}s")

    return {
        "features": np.concatenate(all_feats, axis=0),
        "actions": np.concatenate(all_actions, axis=0),
        "task": np.concatenate(all_meta_task, axis=0),
        "ep": np.concatenate(all_meta_ep, axis=0),
        "t": np.concatenate(all_meta_t, axis=0),
        "success": np.concatenate(all_meta_success, axis=0),
        "suite_idx": np.full(sum(len(f) for f in all_feats), suite_idx_int, dtype=np.int64),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suites", default="libero_10,libero_goal,libero_object,libero_spatial")
    p.add_argument("--dataset_root", default="/home/pokazge/datasets")
    p.add_argument("--out_path", default="/home/pokazge/datasets/memory_bank_v11.npz")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--success_only", action="store_true",
                   help="If set, drop samples from failed episodes")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading DINOv2-small (frozen)...")
    backbone = load_dinov2(device)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)
    print(f"DINOv2 ready.")

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]

    all_results = {}
    total_samples = 0
    for suite in suites:
        r = process_suite(suite, args.dataset_root, backbone, mean, std, device, args.batch_size)
        if r is not None:
            all_results[suite] = r
            total_samples += len(r["features"])

    # Concatenate all suites
    all_feats = np.concatenate([r["features"] for r in all_results.values()], axis=0)
    all_actions = np.concatenate([r["actions"] for r in all_results.values()], axis=0)
    all_task = np.concatenate([r["task"] for r in all_results.values()], axis=0)
    all_ep = np.concatenate([r["ep"] for r in all_results.values()], axis=0)
    all_t = np.concatenate([r["t"] for r in all_results.values()], axis=0)
    all_success = np.concatenate([r["success"] for r in all_results.values()], axis=0)
    all_suite = np.concatenate([r["suite_idx"] for r in all_results.values()], axis=0)

    if args.success_only:
        keep = all_success > 0
        print(f"\nsuccess_only filter: keeping {keep.sum()}/{len(keep)} entries")
        all_feats = all_feats[keep]
        all_actions = all_actions[keep]
        all_task = all_task[keep]
        all_ep = all_ep[keep]
        all_t = all_t[keep]
        all_success = all_success[keep]
        all_suite = all_suite[keep]

    # L2-normalize features so cosine sim is a dot product at retrieval time
    feat_norms = np.linalg.norm(all_feats, axis=-1, keepdims=True)
    feat_norms[feat_norms < 1e-6] = 1.0
    all_feats = all_feats / feat_norms

    print(f"\nTotal bank: {len(all_feats)} entries, feat_dim={all_feats.shape[1]}, "
          f"action shape={all_actions.shape}")
    print(f"  features: {all_feats.nbytes/1e6:.1f} MB")
    print(f"  actions:  {all_actions.nbytes/1e6:.1f} MB")
    print(f"  per-suite counts:")
    for suite_name, idx in SUITE_TO_IDX.items():
        n_in_suite = int((all_suite == idx).sum())
        print(f"    {suite_name}: {n_in_suite}")

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        features=all_feats,
        actions=all_actions,
        suite_idx=all_suite,
        task=all_task,
        ep=all_ep,
        t=all_t,
        success=all_success,
        suite_names=np.array(list(SUITE_TO_IDX.keys()), dtype=object),
        d_feat=np.int64(all_feats.shape[1]),
        K=np.int64(all_actions.shape[1]),
        A=np.int64(all_actions.shape[2]),
    )
    print(f"\nSaved to {out} ({out.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
