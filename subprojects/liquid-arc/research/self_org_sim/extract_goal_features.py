"""Extract per-task DINOv2 goal features from LIBERO expert demos.

For each (suite, task_id), take the last K=5 frames of each successful expert
episode, run DINOv2 ViT-S/14, average to get one 384-d goal feature per task.

Output: /tmp/goal_features.npz with arrays:
  - libero_10_features:      [10, 384]
  - libero_object_features:  [10, 384]
  - libero_goal_features:    [10, 384]
  - libero_spatial_features: [10, 384]
  - libero_10_n_demos:       [10] count of successful demos averaged
  - (same _n_demos suffix per suite)

Substrate uses this as the explicit goal carrier: at episode start (or sub-task
transition in chained eval), look up goal_features[(suite, task_id)] and feed
it into the substrate as static conditioning.
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


torch.set_float32_matmul_precision("high")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUITES = ["libero_10", "libero_object", "libero_goal", "libero_spatial"]
DATASET_ROOT = Path("/home/pokazge/datasets")
LAST_K_FRAMES = 5


def load_dinov2(device):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                    verbose=False)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval().to(device)
    return backbone


@torch.no_grad()
def dinov2_features(backbone, imgs_uint8, mean, std, device):
    """imgs_uint8: [B, H, W, 3] uint8 → [B, 384] CLS features."""
    x = torch.from_numpy(imgs_uint8).to(device).float().permute(0, 3, 1, 2) / 255.0
    if x.shape[-1] != 224 or x.shape[-2] != 224:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    return backbone(x)  # [B, 384]


def extract_suite(suite_name, backbone, mean, std, device, batch_size=32):
    suite_short = suite_name.replace("libero_", "")
    suite_dir = DATASET_ROOT / f"libero-{suite_short}-expert-v1"
    if not suite_dir.exists():
        print(f"  [SKIP] {suite_name} not at {suite_dir}")
        return None, None

    idx = np.load(suite_dir / "index.npz")
    starts = idx["episode_starts"]
    lengths = idx["episode_lengths"]
    task_indices = idx["task_indices"]
    success = idx.get("success_per_episode", np.ones(len(lengths), dtype=bool))
    n_total = int(idx["n_total"])
    img_size = int(idx["img_size"])
    n_episodes = len(lengths)

    imgs = np.memmap(suite_dir / "imgs.dat", dtype=np.uint8, mode="r",
                     shape=(n_total, img_size, img_size, 3))

    unique_tasks = sorted(set(int(t) for t in task_indices))
    n_tasks = len(unique_tasks)
    print(f"  {suite_name}: {n_episodes} episodes, {n_tasks} tasks, "
          f"{success.sum()} successful")

    # Gather last-K frames per successful episode
    all_frame_global_idxs = []
    all_frame_task = []
    for ep_i in range(n_episodes):
        if not bool(success[ep_i]):
            continue
        ep_len = int(lengths[ep_i])
        ep_start = int(starts[ep_i])
        last_k = min(LAST_K_FRAMES, ep_len)
        for k in range(last_k):
            all_frame_global_idxs.append(ep_start + ep_len - 1 - k)
            all_frame_task.append(int(task_indices[ep_i]))

    all_frame_global_idxs = np.array(all_frame_global_idxs, dtype=np.int64)
    all_frame_task = np.array(all_frame_task, dtype=np.int64)
    print(f"  extracting {len(all_frame_global_idxs)} end-frames...")

    # Batched DINOv2 feature extraction
    feats_all = np.zeros((len(all_frame_global_idxs), 384), dtype=np.float32)
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    for i in range(0, len(all_frame_global_idxs), batch_size):
        idxs = all_frame_global_idxs[i:i + batch_size]
        batch = np.array(imgs[idxs])  # [B, H, W, 3]
        feats = dinov2_features(backbone, batch, mean_t, std_t, device)
        feats_all[i:i + batch_size] = feats.cpu().numpy()

    # Average per task
    task_means = np.zeros((n_tasks, 384), dtype=np.float32)
    task_counts = np.zeros((n_tasks,), dtype=np.int64)
    for ti_idx, ti in enumerate(unique_tasks):
        mask = all_frame_task == ti
        if mask.any():
            task_means[ti_idx] = feats_all[mask].mean(axis=0)
            task_counts[ti_idx] = int(mask.sum())
        else:
            print(f"    [WARN] task {ti}: 0 successful demos")

    return task_means, task_counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/goal_features.npz", type=str)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[goal] device={device}")
    backbone = load_dinov2(device)
    print(f"[goal] DINOv2 ViT-S/14 loaded")
    mean = IMAGENET_MEAN
    std = IMAGENET_STD

    save_dict = {}
    for suite in SUITES:
        task_means, task_counts = extract_suite(
            suite, backbone, mean, std, device, args.batch_size,
        )
        if task_means is not None:
            save_dict[f"{suite}_features"] = task_means
            save_dict[f"{suite}_n_demos"] = task_counts

    np.savez(args.out, **save_dict)
    print(f"\n[goal] saved → {args.out}")
    for k, v in save_dict.items():
        if k.endswith("_features"):
            print(f"  {k}: shape={v.shape}, norm_mean={np.linalg.norm(v, axis=1).mean():.3f}")
        else:
            print(f"  {k}: total demos={int(v.sum())}, min/max={int(v.min())}/{int(v.max())}")


if __name__ == "__main__":
    main()
