"""Diagnose v10-DEMO's libero_object failure (13% vs libero_spatial 70%).

Compares two drift-obs npz files (libero_object + libero_spatial) to identify
WHAT structural difference in those suites makes object so much harder.

Hypotheses tested:
  H1: libero_object tasks are MORE visually similar to each other than
      libero_spatial tasks → DINOv2-feature inter-task similarity comparison
  H2: Action predictions across libero_object tasks converge to similar chunks
      (model can't disambiguate task) → predicted-chunk inter-task similarity
  H3: Failed rollouts diverge from successful ones early → action prediction
      divergence vs episode step, comparing successful vs failed rollouts
  H4: Gripper timing mismatch (closes at wrong physical state) → analyze
      gripper-closing point per rollout, compare timing across success/fail

Outputs a printed summary; no plots (visual inspection happens via numbers).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dinov2(device):
    import warnings
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
def dinov2_features(model, imgs_uint8, device):
    # imgs_uint8: [N, H, W, 3]
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)
    feats = []
    BS = 64
    for i in range(0, len(imgs_uint8), BS):
        x = torch.from_numpy(imgs_uint8[i:i+BS]).to(device).float().permute(0, 3, 1, 2) / 255.0
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        f = model(x)  # [B, 384] CLS
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def analyze_suite(npz_path: Path, dino, device, suite_label: str):
    print(f"\n{'='*70}")
    print(f"=== {suite_label}: {npz_path} ===")
    print(f"{'='*70}")
    data = np.load(npz_path, allow_pickle=True)
    imgs = data["imgs"]
    wrists = data["wrists"]
    states = data["states"]
    sim_ids = data["sim_ids"]
    rollout_idxs = data["rollout_idxs"]
    rollout_success = data["rollout_success"]
    rollout_n_steps = data["rollout_n_steps"]
    pred_chunks = data["pred_chunks"]
    groot_chunks = data["groot_chunks"]
    steps = data["steps"]
    N = len(imgs)
    print(f"N entries: {N}")
    n_sims = len(np.unique(sim_ids))
    print(f"sims: {n_sims}")
    print(f"rollouts: {len(np.unique(np.stack([sim_ids, rollout_idxs], 1), axis=0))}")

    # Per-sim success rate
    per_sim_success = {}
    for s in np.unique(sim_ids):
        mask = sim_ids == s
        outcomes = {}
        for r in np.unique(rollout_idxs[mask]):
            ro_mask = mask & (rollout_idxs == r)
            outcomes[int(r)] = int(rollout_success[ro_mask].max())
        per_sim_success[int(s)] = outcomes
    succ_per_sim = {s: sum(o.values()) / max(len(o), 1) for s, o in per_sim_success.items()}
    print(f"per-sim success: {succ_per_sim}")
    overall_succ = np.mean([v for v in succ_per_sim.values()])
    print(f"overall success rate: {overall_succ:.1%}")

    # === H1: inter-task DINOv2 similarity ===
    print(f"\n--- H1: DINOv2-feature inter-task similarity (do tasks look the same?) ---")
    # Sample one obs per sim at a similar progression point (step 0-50)
    early_idx = []
    for s in np.unique(sim_ids):
        s_mask = (sim_ids == s) & (steps <= 50)
        if s_mask.any():
            early_idx.append(np.where(s_mask)[0][0])
    early_idx = np.array(early_idx)
    print(f"sampling early obs per sim: {len(early_idx)} entries")
    early_imgs = imgs[early_idx]
    early_wrists = wrists[early_idx]
    dino_feat_img = dinov2_features(dino, early_imgs, device)
    dino_feat_wrist = dinov2_features(dino, early_wrists, device)
    dino_feat = np.concatenate([dino_feat_img, dino_feat_wrist], axis=-1)
    dino_feat = dino_feat / (np.linalg.norm(dino_feat, axis=-1, keepdims=True) + 1e-8)
    sim_matrix = dino_feat @ dino_feat.T
    off_diag = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
    print(f"agent+wrist CLS cosine similarity across tasks:")
    print(f"  mean: {off_diag.mean():.3f}   median: {np.median(off_diag):.3f}")
    print(f"  min:  {off_diag.min():.3f}   max:   {off_diag.max():.3f}")
    print(f"  std:  {off_diag.std():.3f}")

    # === H2: inter-task predicted-chunk similarity at step 0 ===
    print(f"\n--- H2: predicted-chunk inter-task similarity ---")
    # Sample one pred_chunk per sim at step 0-8 (first chunk decision)
    pred_chunks_per_sim = []
    sims_with_chunks = []
    for s in np.unique(sim_ids):
        s_mask = (sim_ids == s) & (steps <= 8)
        if not s_mask.any():
            continue
        idx = np.where(s_mask)[0][0]
        pc = pred_chunks[idx]
        if not np.isnan(pc).any():
            pred_chunks_per_sim.append(pc)
            sims_with_chunks.append(int(s))
    if len(pred_chunks_per_sim) > 1:
        pred_chunks_arr = np.stack(pred_chunks_per_sim)  # [n_sims, K, A]
        # Flatten to per-sim vector and compute cosine sim
        pc_flat = pred_chunks_arr.reshape(len(pred_chunks_arr), -1)
        pc_norm = pc_flat / (np.linalg.norm(pc_flat, axis=-1, keepdims=True) + 1e-8)
        pc_sim = pc_norm @ pc_norm.T
        off_diag_pc = pc_sim[np.triu_indices(len(pc_sim), k=1)]
        print(f"predicted-chunk cosine similarity across tasks (at first chunk decision):")
        print(f"  mean: {off_diag_pc.mean():.3f}   median: {np.median(off_diag_pc):.3f}")
        print(f"  std:  {off_diag_pc.std():.3f}")
        # Per-action-dim mean across tasks (xyz, rpy, gripper)
        per_dim = pred_chunks_arr.mean(axis=(0, 1))  # average over (sims, chunk_K)
        print(f"  per-action-dim mean of predicted chunks: {per_dim.round(3)}")
    else:
        print(f"  insufficient samples")

    # === H3: action divergence vs episode step, for successful vs failed rollouts ===
    print(f"\n--- H3: predicted-chunk magnitude over episode step (success vs fail) ---")
    # Bin by step and split by success
    bins = [(0, 50), (50, 150), (150, 300), (300, 500), (500, 720)]
    for lo, hi in bins:
        mask = (steps >= lo) & (steps < hi)
        if not mask.any():
            continue
        suc_mask = mask & (rollout_success == 1)
        fail_mask = mask & (rollout_success == 0)
        if not (suc_mask.any() and fail_mask.any()):
            continue
        # Use mean action magnitude of executed chunk[0] across step bin
        suc_pc = pred_chunks[suc_mask]
        fail_pc = pred_chunks[fail_mask]
        valid_suc = ~np.isnan(suc_pc).any(axis=(1, 2))
        valid_fail = ~np.isnan(fail_pc).any(axis=(1, 2))
        if not (valid_suc.any() and valid_fail.any()):
            continue
        # action[0, :3] xyz magnitude
        suc_xyz_norm = np.linalg.norm(suc_pc[valid_suc][:, 0, :3], axis=-1).mean()
        fail_xyz_norm = np.linalg.norm(fail_pc[valid_fail][:, 0, :3], axis=-1).mean()
        suc_grip = suc_pc[valid_suc][:, 0, -1].mean()
        fail_grip = fail_pc[valid_fail][:, 0, -1].mean()
        print(f"  step[{lo:>3},{hi:>3}): "
              f"|xyz| suc={suc_xyz_norm:.3f} fail={fail_xyz_norm:.3f}  "
              f"grip suc={suc_grip:+.3f} fail={fail_grip:+.3f}  "
              f"n_suc={valid_suc.sum()} n_fail={valid_fail.sum()}")

    # === H4: gripper transition timing ===
    print(f"\n--- H4: gripper transition step (when does grip flip sign?) ---")
    for s in np.unique(sim_ids):
        for r in np.unique(rollout_idxs[sim_ids == s]):
            ro_mask = (sim_ids == s) & (rollout_idxs == r)
            if not ro_mask.any():
                continue
            ro_pc = pred_chunks[ro_mask]
            ro_steps = steps[ro_mask]
            ro_succ = bool(rollout_success[ro_mask].max())
            valid = ~np.isnan(ro_pc).any(axis=(1, 2))
            if not valid.any():
                continue
            # Track action[0] gripper sign across chunk decisions
            grip0 = ro_pc[valid][:, 0, -1]
            steps_valid = ro_steps[valid]
            sign_changes = np.where(np.diff(np.sign(grip0)) != 0)[0]
            n_flips = len(sign_changes)
            first_close_step = -1
            for k, g in zip(steps_valid, grip0):
                if g < -0.1:
                    first_close_step = int(k)
                    break
            print(f"  sim{s:>2} r{r}: succ={ro_succ}  n_chunks={len(grip0)}  "
                  f"grip_flips={n_flips}  first_close@step={first_close_step}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object_npz", required=True, type=str)
    p.add_argument("--spatial_npz", required=True, type=str)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diag] device={device}")
    print(f"[diag] loading DINOv2-small...")
    dino = load_dinov2(device)

    analyze_suite(Path(args.object_npz), dino, device, "libero_object (v10-DEMO 13%)")
    analyze_suite(Path(args.spatial_npz), dino, device, "libero_spatial (v10-DEMO 70%)")

    print("\n=== INTER-SUITE COMPARISON ===")
    print("If H1 holds: libero_object's mean inter-task DINOv2 sim > libero_spatial's")
    print("If H2 holds: libero_object's mean pred_chunk inter-task sim > libero_spatial's")
    print("If H3 holds: failed-rollout pred_chunks diverge from successful-rollout ones early")
    print("If H4 holds: gripper-close timing varies more in libero_object than libero_spatial")


if __name__ == "__main__":
    main()
