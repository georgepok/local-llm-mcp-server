"""Convert LIBERO official expert demonstrations (HDF5) into the memmap dataset
format consumed by TeacherLabelDataset (in distill_groot.py).

Usage (Spark):
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python prep_libero_expert.py \\
    --suite_dir /home/pokazge/Isaac-GR00T/external_dependencies/LIBERO/libero/datasets/libero_spatial \\
    --out_dir /home/pokazge/datasets/libero-spatial-expert-v1 \\
    --img_size 96 --action_horizon 16

For each suite (libero_spatial, libero_object, libero_goal, libero_10): one
output directory. Samples carry IMG/WRIST/STATE/CHUNK directly from the human-
demo HDF5; z_vl/z_bank are emitted as ZEROS (no GR00T involvement). Liquid
trains as direct imitation on these new suites; the mixed dataset (existing
groot-sim-tempquery-v1 with z_vl + new libero-*-expert-v1 with zeros) gives
the model both regimes — "I have GR00T advice" and "I don't" — under the
existing z_groot_drop_prob pressure-landscape mechanism.

Output files (compatible with TeacherLabelDataset):
  imgs.dat            uint8   [N, img_size, img_size, 3]
  wrists.dat          uint8   [N, img_size, img_size, 3]
  states.dat          float32 [N, 8]
  teacher_chunks.dat  float32 [N, action_horizon, 7]
  z_vl.dat            float32 [N, 2048]    (zeros)
  z_state.dat         float32 [N, 1536]    (zeros)
  z_motor.dat         float32 [N, 7]       (zeros)
  z_vl_bank.dat       float32 [N, 4, 1024] (zeros, K=4 depth bank)
  delta_s_bank.dat    float32 [N, 4, 4]    (identity per sample)
  index.npz           episode_starts, episode_lengths, task_indices, n_total,
                      img_size, success_per_episode, z_vl_dim, z_state_dim,
                      query_bank_K, query_channel, query_dim, depth_indices
  labels_index.npz    sample_idx [N, 3] = (ep_i, t, task_idx), n_samples,
                      action_horizon
  task_languages.json  {task_idx: language_string}
  summary.json         metadata
"""

from __future__ import annotations

import argparse
import functools
import json
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

print = functools.partial(print, flush=True)


# Constants matching existing groot-sim-tempquery-v1 format. Note: in that
# dataset, index.npz reports z_vl_dim=1024 (DiT hidden_dim, used for both z_vl.dat
# and z_vl_bank.dat by the loader). The actual Qwen3-VL embedding dim is 2048
# but the loader truncates to z_vl_dim. We match the loader's expectation here.
Z_VL_DIM = 1024  # MUST match what the loader reads from index.npz["z_vl_dim"]
Z_STATE_DIM = 1536
Z_MOTOR_DIM = 7
QUERY_BANK_K = 4
HIDDEN_DIM = 1024
QUERY_DIM = 4  # depth channel: identity over 4 depths
DEPTH_INDICES = np.array([0, 1, 2, 3], dtype=np.int64)


def task_name_from_filename(path: Path) -> str:
    """Extract task language from filename like
    'pick_up_the_black_bowl_..._demo.hdf5' → 'pick up the black bowl ...'."""
    stem = path.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return stem.replace("_", " ")


def build_state8(obs: dict) -> np.ndarray:
    """[ee_pos(3), ee_ori(3), gripper_states(2)] → [8]. Per timestep."""
    return np.concatenate([
        obs["ee_pos"][:],   # [T, 3]
        obs["ee_ori"][:],   # [T, 3]
        obs["gripper_states"][:],  # [T, 2]
    ], axis=-1).astype(np.float32)


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


def chunk_actions(actions: np.ndarray, horizon: int) -> np.ndarray:
    """[T, 7] → [T, horizon, 7]. Pad final chunks by repeating last action."""
    T = actions.shape[0]
    out = np.empty((T, horizon, 7), dtype=np.float32)
    for t in range(T):
        end = min(t + horizon, T)
        n = end - t
        out[t, :n] = actions[t:end]
        if n < horizon:
            out[t, n:] = actions[T - 1]  # pad with last action
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite_dir", required=True, type=str,
                   help="Path to libero_spatial/libero_object/libero_goal/libero_10 dir.")
    p.add_argument("--out_dir", required=True, type=str)
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--max_demos_per_task", type=int, default=0,
                   help="0 = use all 50 demos per task. >0 = subsample.")
    p.add_argument("--keep_z_vl", action="store_true",
                   help="If set, do NOT touch existing z_vl.dat / z_vl_bank.dat / "
                        "z_state.dat / z_motor.dat / delta_s_bank.dat. Useful when "
                        "regenerating imgs.dat/wrists.dat/states.dat/teacher_chunks.dat "
                        "at a different img_size while preserving GR00T-derived z_vl.")
    args = p.parse_args()

    suite_dir = Path(args.suite_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(suite_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise SystemExit(f"No .hdf5 files in {suite_dir}")
    print(f"[scan] {len(hdf5_files)} task files in {suite_dir}")

    # First pass: count total samples per task and across all tasks.
    task_languages: dict[int, str] = {}
    all_episodes: list[tuple[int, int, int, str]] = []   # (task_idx, demo_idx, length, file_idx)
    n_total = 0
    for ti, path in enumerate(hdf5_files):
        lang = task_name_from_filename(path)
        task_languages[ti] = lang
        with h5py.File(path, "r") as f:
            data = f["data"]
            demos = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
            if args.max_demos_per_task > 0:
                demos = demos[: args.max_demos_per_task]
            for di, dname in enumerate(demos):
                T = data[dname]["actions"].shape[0]
                all_episodes.append((ti, di, T, dname))
                n_total += T
        print(f"[scan]  task {ti}: {lang!r}  {len(demos)} demos  "
              f"sum T={sum(T for _, _, T, _ in all_episodes if _==ti or False)}")

    print(f"[scan] total {len(all_episodes)} episodes, {n_total:,} timesteps")

    # Pre-allocate memmaps. When --keep_z_vl, skip the GR00T-derived ones (assume
    # they exist from a prior run; we only regenerate inputs/targets at new img_size).
    imgs_mm = np.memmap(out_dir / "imgs.dat", dtype=np.uint8, mode="w+",
                         shape=(n_total, args.img_size, args.img_size, 3))
    wrists_mm = np.memmap(out_dir / "wrists.dat", dtype=np.uint8, mode="w+",
                           shape=(n_total, args.img_size, args.img_size, 3))
    states_mm = np.memmap(out_dir / "states.dat", dtype=np.float32, mode="w+",
                           shape=(n_total, 8))
    chunks_mm = np.memmap(out_dir / "teacher_chunks.dat", dtype=np.float32, mode="w+",
                           shape=(n_total, args.action_horizon, 7))
    if args.keep_z_vl:
        print("[keep] preserving existing z_vl.dat / z_vl_bank.dat / etc.")
        z_vl_mm = z_state_mm = z_motor_mm = z_vl_bank_mm = delta_bank_mm = None
    else:
        z_vl_mm = np.memmap(out_dir / "z_vl.dat", dtype=np.float32, mode="w+",
                             shape=(n_total, Z_VL_DIM))
        z_state_mm = np.memmap(out_dir / "z_state.dat", dtype=np.float32, mode="w+",
                                shape=(n_total, Z_STATE_DIM))
        z_motor_mm = np.memmap(out_dir / "z_motor.dat", dtype=np.float32, mode="w+",
                                shape=(n_total, Z_MOTOR_DIM))
        z_vl_bank_mm = np.memmap(out_dir / "z_vl_bank.dat", dtype=np.float32, mode="w+",
                                  shape=(n_total, QUERY_BANK_K, HIDDEN_DIM))
        delta_bank_mm = np.memmap(out_dir / "delta_s_bank.dat", dtype=np.float32, mode="w+",
                                   shape=(n_total, QUERY_BANK_K, QUERY_DIM))

    # zeros are already memmap-initialized to 0; just need delta_bank = identity.
    identity_K = np.eye(QUERY_BANK_K, dtype=np.float32)  # [K, K]
    # Note: query_dim = 4 (depth channel) so identity is [4, 4]. delta_bank shape is [N, 4, 4].

    # Second pass: stream samples in.
    episode_starts = np.zeros(len(all_episodes), dtype=np.int64)
    episode_lengths = np.zeros(len(all_episodes), dtype=np.int64)
    task_indices_arr = np.zeros(len(all_episodes), dtype=np.int64)
    success_per_episode = np.ones(len(all_episodes), dtype=np.int64)  # all expert demos succeed

    # Sort episodes by file (ti) so we open each file once.
    ep_by_task: dict[int, list[tuple[int, int, int, str]]] = {}
    for tup in all_episodes:
        ep_by_task.setdefault(tup[0], []).append(tup)

    cursor = 0
    ep_global_idx = 0
    for ti in sorted(ep_by_task.keys()):
        path = hdf5_files[ti]
        with h5py.File(path, "r") as f:
            data = f["data"]
            for (ti_, di, T, dname) in ep_by_task[ti]:
                assert ti_ == ti
                demo = data[dname]
                # Resize images
                imgs = resize_imgs(demo["obs/agentview_rgb"][:], args.img_size)
                wrists = resize_imgs(demo["obs/eye_in_hand_rgb"][:], args.img_size)
                # Build state8
                state8 = build_state8(demo["obs"])
                # Action chunks
                actions = demo["actions"][:].astype(np.float32)
                chunks = chunk_actions(actions, args.action_horizon)

                # Write to memmaps
                imgs_mm[cursor:cursor + T] = imgs
                wrists_mm[cursor:cursor + T] = wrists
                states_mm[cursor:cursor + T] = state8
                chunks_mm[cursor:cursor + T] = chunks
                if delta_bank_mm is not None:
                    # Identity bank query per sample
                    delta_bank_mm[cursor:cursor + T] = identity_K[None].repeat(T, axis=0)
                # z_vl, z_state, z_motor, z_vl_bank are already zeros-initialized
                # (or preserved from prior run when --keep_z_vl).

                episode_starts[ep_global_idx] = cursor
                episode_lengths[ep_global_idx] = T
                task_indices_arr[ep_global_idx] = ti
                cursor += T
                ep_global_idx += 1
        print(f"[write] task {ti}/{len(hdf5_files)-1}  cursor={cursor:,}/{n_total:,}")

    assert cursor == n_total, f"cursor mismatch {cursor} vs {n_total}"

    # Flush memmaps
    for mm in (imgs_mm, wrists_mm, states_mm, chunks_mm,
                z_vl_mm, z_state_mm, z_motor_mm, z_vl_bank_mm, delta_bank_mm):
        if mm is not None:
            mm.flush()

    # Build sample_idx for labels_index. Each sample is (ep_global_idx, t, task_idx).
    sample_idx = np.zeros((n_total, 3), dtype=np.int64)
    cursor = 0
    for ep_i in range(len(all_episodes)):
        T = int(episode_lengths[ep_i])
        ti = int(task_indices_arr[ep_i])
        for t in range(T):
            sample_idx[cursor + t] = [ep_i, t, ti]
        cursor += T

    np.savez(out_dir / "index.npz",
             episode_starts=episode_starts,
             episode_lengths=episode_lengths,
             task_indices=task_indices_arr,
             n_total=np.int64(n_total),
             img_size=np.int64(args.img_size),
             success_per_episode=success_per_episode,
             z_vl_dim=np.int64(Z_VL_DIM),
             z_state_dim=np.int64(Z_STATE_DIM),
             query_bank_K=np.int64(QUERY_BANK_K),
             query_channel=np.array("depth"),
             query_dim=np.int64(QUERY_DIM),
             depth_indices=DEPTH_INDICES,
             )
    np.savez(out_dir / "labels_index.npz",
             sample_idx=sample_idx,
             n_samples=np.int64(n_total),
             action_horizon=np.int64(args.action_horizon),
             )

    with (out_dir / "task_languages.json").open("w") as fout:
        json.dump({str(k): v for k, v in task_languages.items()}, fout, indent=2)

    summary = {
        "suite": str(suite_dir.name),
        "n_episodes": len(all_episodes),
        "n_samples": int(n_total),
        "n_tasks": len(hdf5_files),
        "img_size": args.img_size,
        "action_horizon": args.action_horizon,
        "z_vl_dim": Z_VL_DIM,
        "z_state_dim": Z_STATE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "query_bank_K": QUERY_BANK_K,
        "query_channel": "depth",
        "depth_indices": DEPTH_INDICES.tolist(),
        "success_rate": 1.0,
        "z_groot_is_zeros": True,  # IMPORTANT — flag that z_vl/z_bank are placeholders
        "size_gb": float(
            sum((out_dir / f).stat().st_size for f in [
                "imgs.dat", "wrists.dat", "states.dat", "teacher_chunks.dat",
                "z_vl.dat", "z_state.dat", "z_motor.dat", "z_vl_bank.dat",
                "delta_s_bank.dat",
            ]) / 1e9),
    }
    with (out_dir / "summary.json").open("w") as fout:
        json.dump(summary, fout, indent=2)
    print(f"[done] {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
