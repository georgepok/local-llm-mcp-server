"""Offline GR00T-N1.7-LIBERO inference to compute real z_vl + z_vl_bank for all
frames in a libero-*-expert-v1 dataset.

Replaces the zeros placeholders in z_vl.dat and z_vl_bank.dat with real GR00T
features so multi-suite distillation has the same task-disambiguation signal
across all suites (matching the existing groot-sim-tempquery-v1 format).

Run on Spark in main GR00T venv (NOT inside fgn-train):
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
  HF_HOME=/home/pokazge/hf_cache HF_TOKEN=... python generate_libero_zvl.py \\
    --dataset_dir /home/pokazge/datasets/libero-spatial-expert-v1 \\
    --suite_dir /home/pokazge/Isaac-GR00T/external_dependencies/LIBERO/libero/datasets/libero_spatial \\
    --teacher_path /home/pokazge/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_10 \\
    --batch_size 8

Notes:
  - Uses GR00T-N1.7-LIBERO-libero_10 ckpt (the one we have); it's the most
    capable variant. Even though it scores 4% on libero_spatial action-wise,
    its scene+language fusion (z_vl) is still informative — the failure is in
    action generation, not in scene understanding.
  - Reads original 128×128 imgs from libero hdf5 and resizes to 256×256 for
    GR00T (its expected resolution).
  - Captures z_vl (mean-pooled vl_embeds, 2048-d truncated to 1024 to match
    dataset format) and z_vl_bank (DiT depth trajectory, 4 × 1024).
  - Writes in-place to dataset_dir's z_vl.dat and z_vl_bank.dat.
"""

from __future__ import annotations

import argparse
import functools
import time
import types
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

print = functools.partial(print, flush=True)


def task_name_from_filename(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return stem.replace("_", " ")


def resize_imgs(arr: np.ndarray, target_size: int) -> np.ndarray:
    """[T, H, W, 3] uint8 → [T, target_size, target_size, 3] uint8."""
    if arr.shape[1] == target_size and arr.shape[2] == target_size:
        return arr.astype(np.uint8)
    out = np.empty((arr.shape[0], target_size, target_size, 3), dtype=np.uint8)
    for i in range(arr.shape[0]):
        im = Image.fromarray(arr[i])
        im = im.resize((target_size, target_size), Image.BILINEAR)
        out[i] = np.asarray(im, dtype=np.uint8)
    return out


def install_capture_patch(policy):
    """Monkey-patch action_head.get_action_with_features to capture per-batch
    z_vl + traj_model_output. Mirrors groot_server.py but supports B>1."""
    ah = policy.model.action_head

    @torch.no_grad()
    def get_action_with_capture(
        self, backbone_features, state_features, embodiment_id,
        backbone_output, action_input, options=None,
    ):
        from transformers.feature_extraction_utils import BatchFeature
        vl_embeds = backbone_features
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            (batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype, device=device,
        )
        N = self.num_inference_timesteps
        dt = 1.0 / N
        traj_model_output_per_step = []  # list of [B, hidden] per step
        for t_idx in range(N):
            t_cont = t_idx / float(N)
            t_disc = int(t_cont * self.num_timestep_buckets)
            t_tensor = torch.full((batch_size,), t_disc, device=device)
            action_features = self.action_encoder(actions, t_tensor, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
            sa_embs = torch.cat((state_features, action_features), dim=1)
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs, encoder_hidden_states=vl_embeds,
                    timestep=t_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs, encoder_hidden_states=vl_embeds,
                    timestep=t_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon:]
            # Capture per-batch mean-pooled DiT hidden state at this depth.
            # model_output: [B, seq_len, hidden] → mean over seq_len → [B, hidden]
            traj_model_output_per_step.append(
                model_output.float().mean(dim=1).detach().cpu().numpy()
            )
            actions = actions + dt * pred_velocity
        # Stack: [N=4, B, hidden] → transpose to [B, 4, hidden]
        traj_model_output = np.transpose(
            np.stack(traj_model_output_per_step), (1, 0, 2)
        ).astype(np.float32)

        # vl_embeds: [B, seq_len, 2048] → mean-pool → [B, 2048]
        z_vl_pooled = vl_embeds.float().mean(dim=1).detach().cpu().numpy().astype(np.float32)

        self._batch_captures = {
            "z_vl": z_vl_pooled,                 # [B, 2048]
            "z_vl_bank": traj_model_output,      # [B, 4, hidden]
        }

        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        })

    ah.get_action_with_features = types.MethodType(get_action_with_capture, ah)
    print("[patch] action_head.get_action_with_features installed")


def build_groot_obs_batch(imgs_256: np.ndarray, wrists_256: np.ndarray,
                          states_8: np.ndarray, languages: list) -> dict:
    """Build batched obs dict for GR00T. Shapes:
      imgs_256, wrists_256: [B, H, W, 3] uint8
      states_8: [B, 8] float32
      languages: list of B strings
    """
    B = imgs_256.shape[0]
    state_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    state_slots = {
        "x": (0, 1), "y": (1, 2), "z": (2, 3),
        "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6),
        "gripper": (6, 8),
    }
    obs = {"video": {}, "state": {}, "language": {}}
    obs["video"]["image"] = imgs_256[:, None, ...]   # [B, T=1, H, W, 3]
    obs["video"]["wrist_image"] = wrists_256[:, None, ...]
    for k in state_keys:
        lo, hi = state_slots[k]
        obs["state"][k] = states_8[:, lo:hi].astype(np.float32)[:, None, :]  # [B, T=1, dim]
    obs["language"]["annotation.human.action.task_description"] = [[lang] for lang in languages]
    return obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True, type=str,
                   help="Existing libero-*-expert-v1 dataset to update in place.")
    p.add_argument("--suite_dir", required=True, type=str,
                   help="Original libero hdf5 suite directory (for full-res imgs).")
    p.add_argument("--teacher_path", required=True, type=str)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_demos_per_task", type=int, default=0,
                   help="0 = use all 50 demos (matching existing converter). >0 = subsample.")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    suite_dir = Path(args.suite_dir)
    teacher_path = Path(args.teacher_path)
    assert dataset_dir.exists(), f"missing {dataset_dir}"
    assert suite_dir.exists(), f"missing {suite_dir}"

    # Load existing dataset metadata
    idx = np.load(dataset_dir / "index.npz")
    n_total = int(idx["n_total"])
    z_vl_dim = int(idx["z_vl_dim"])
    K = int(idx["query_bank_K"])
    print(f"[load] {dataset_dir.name}: n_total={n_total:,}  z_vl_dim={z_vl_dim}  K={K}")

    # Open output memmaps in r+w mode (modify in place)
    z_vl_mm = np.memmap(dataset_dir / "z_vl.dat", dtype=np.float32, mode="r+",
                         shape=(n_total, z_vl_dim))
    z_vl_bank_mm = np.memmap(dataset_dir / "z_vl_bank.dat", dtype=np.float32, mode="r+",
                              shape=(n_total, K, z_vl_dim))

    # Load policy
    print(f"[gr00t] loading policy from {teacher_path}")
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path=str(teacher_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    install_capture_patch(policy)

    # Iterate suite hdf5 files in same order as the converter (alphabetical)
    hdf5_files = sorted(suite_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise SystemExit(f"No .hdf5 files in {suite_dir}")
    print(f"[scan] {len(hdf5_files)} task files")

    cursor = 0
    t_start = time.time()
    n_processed = 0
    for ti, path in enumerate(hdf5_files):
        lang = task_name_from_filename(path)
        with h5py.File(path, "r") as f:
            data = f["data"]
            demos = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
            if args.max_demos_per_task > 0:
                demos = demos[: args.max_demos_per_task]
            for di, dname in enumerate(demos):
                demo = data[dname]
                T = demo["actions"].shape[0]
                # Resize imgs to 256x256 for GR00T
                imgs_256 = resize_imgs(demo["obs/agentview_rgb"][:], 256)
                wrists_256 = resize_imgs(demo["obs/eye_in_hand_rgb"][:], 256)
                ee_pos = demo["obs/ee_pos"][:]
                ee_ori = demo["obs/ee_ori"][:]
                gripper = demo["obs/gripper_states"][:]
                states_8 = np.concatenate([ee_pos, ee_ori, gripper], axis=-1).astype(np.float32)

                # Process timesteps in batches
                for start in range(0, T, args.batch_size):
                    end = min(start + args.batch_size, T)
                    B = end - start
                    obs = build_groot_obs_batch(
                        imgs_256[start:end],
                        wrists_256[start:end],
                        states_8[start:end],
                        [lang] * B,
                    )
                    _, _ = policy.get_action(obs)
                    cap = policy.model.action_head._batch_captures
                    # cap["z_vl"]: [B, 2048] → truncate to [B, z_vl_dim]
                    z_vl_mm[cursor + start:cursor + end] = cap["z_vl"][:, :z_vl_dim]
                    # cap["z_vl_bank"]: [B, 4, hidden]
                    z_vl_bank_mm[cursor + start:cursor + end] = cap["z_vl_bank"][:, :, :z_vl_dim]
                cursor += T
                n_processed += T
                if di % 10 == 0:
                    elapsed = time.time() - t_start
                    rate = n_processed / max(elapsed, 1e-3)
                    eta = (n_total - n_processed) / max(rate, 1e-3)
                    print(f"[{ti}/{len(hdf5_files)-1}] task={lang[:50]:50s}  "
                          f"demo {di}/{len(demos)-1}  done {n_processed:,}/{n_total:,}  "
                          f"rate={rate:.1f}/s  ETA={eta/60:.1f}min  "
                          f"elapsed={elapsed/60:.1f}min")
        # Flush after each task (defensive)
        z_vl_mm.flush()
        z_vl_bank_mm.flush()

    if cursor != n_total:
        print(f"[partial] processed {cursor:,}/{n_total:,} samples (rest stay as zeros)")
    print(f"[done] processed {cursor:,} samples in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
