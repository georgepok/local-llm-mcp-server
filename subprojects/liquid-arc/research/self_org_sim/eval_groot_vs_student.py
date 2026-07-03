"""Head-to-head eval: GR00T-N1.7 teacher vs Liquid student on the SAME val episodes.

Reports for each policy:
  - MSE / MAE on action chunks vs ground-truth demonstrator actions
  - Per-task breakdown (10 LIBERO_10 tasks)
  - Inference Hz (samples/sec end-to-end)
  - Per-step latency stats (p50, p90)

Run on Spark (after a distill_groot.py training run produces a student checkpoint):
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
  HF_HOME=/home/pokazge/hf_cache HF_TOKEN=hf_... \\
  python eval_groot_vs_student.py \\
    --raw_data_root /home/pokazge/datasets/libero-10-r-raw/libero-10-r \\
    --decoded_data_dir /home/pokazge/datasets/libero-10-r-decoded \\
    --teacher_path /home/pokazge/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_10 \\
    --student_ckpt /tmp/distill_groot_v1/step_030000.pt \\
    --val_episodes 37 --max_steps_per_ep 100 --action_horizon 16
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Local: same dir as distill_groot.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot import LiquidStudent, LiberoMemmapDataset

print = functools.partial(print, flush=True)
torch.set_float32_matmul_precision("high")


def load_student(ckpt_path: Path, device: torch.device) -> LiquidStudent:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sargs = ckpt["args"]
    halt_mode = "learned" if sargs["policy"] == "liquid_halt" else "none"
    student = LiquidStudent(
        state_dim=8,
        action_dim=7,
        action_horizon=sargs["action_horizon"],
        d=sargs["d"],
        d_vis=sargs["d"],
        img_size=sargs["img_size"],
        k_max=sargs["k"],
        halt_mode=halt_mode,
        min_steps=sargs["halting_min_steps"],
    ).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own_sd = student.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in own_sd and own_sd[k].shape == v.shape:
            own_sd[k].copy_(v)
            loaded += 1
    print(f"[student] loaded {loaded} tensors from {ckpt_path} (step={ckpt.get('step')})")
    student.eval()
    return student, sargs


@torch.no_grad()
def eval_student(
    student: LiquidStudent,
    decoded_dir: Path,
    val_indices: list[int],
    action_horizon: int,
    device: torch.device,
    max_steps_per_ep: int,
    chunks_at_every_step: bool = False,
):
    """Eval student on val episodes. For each step (0, K, 2K, ...) up to max_steps_per_ep,
    predict an action chunk and compare to the next K ground-truth actions.

    Returns: per-episode list of dicts with mse/mae/timing/task_idx.
    """
    ds = LiberoMemmapDataset(decoded_dir, action_horizon=action_horizon, episode_indices=val_indices)
    starts = ds.starts
    lengths = ds.lengths
    task_indices = ds.task_indices

    results = []
    # Warm-up the student once (compile / cudnn)
    dummy_img = torch.zeros(1, 3, ds.img_size, ds.img_size, device=device)
    dummy_state = torch.zeros(1, 8, device=device)
    for _ in range(2):
        _ = student(dummy_img, dummy_img, dummy_state)
    torch.cuda.synchronize()

    for ep_i in val_indices:
        ep_start = int(starts[ep_i])
        ep_len = int(lengths[ep_i])
        n_steps = min(ep_len, max_steps_per_ep)
        # Inference points: every action_horizon steps
        if chunks_at_every_step:
            ts = list(range(n_steps))
        else:
            ts = list(range(0, n_steps, action_horizon))

        sq_total = 0.0
        abs_total = 0.0
        n_pairs = 0
        per_step_times = []

        for t in ts:
            img = torch.from_numpy(np.array(ds.imgs[ep_start + t])).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            wrist = torch.from_numpy(np.array(ds.wrists[ep_start + t])).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            state = torch.from_numpy(np.array(ds.states[ep_start + t])).to(device).float().unsqueeze(0)

            # Build target chunk
            end = min(t + action_horizon, ep_len)
            target = np.array(ds.actions[ep_start + t:ep_start + end])
            if len(target) < action_horizon:
                pad = np.tile(target[-1:], (action_horizon - len(target), 1))
                target = np.concatenate([target, pad], axis=0)
            target_t = torch.from_numpy(target).to(device).float().unsqueeze(0)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred, _ = student(img, wrist, state)
            torch.cuda.synchronize()
            per_step_times.append(time.perf_counter() - t0)

            diff = pred - target_t
            sq_total += diff.pow(2).mean().item()
            abs_total += diff.abs().mean().item()
            n_pairs += 1

        results.append({
            "episode": ep_i,
            "task": int(task_indices[ep_i]),
            "n_inferences": n_pairs,
            "mse": sq_total / max(n_pairs, 1),
            "mae": abs_total / max(n_pairs, 1),
            "median_latency_ms": float(np.median(per_step_times) * 1000),
            "p90_latency_ms": float(np.percentile(per_step_times, 90) * 1000),
        })
    return results


def eval_teacher(
    teacher_path: Path,
    raw_data_root: Path,
    val_indices: list[int],
    action_horizon: int,
    device: torch.device,
    max_steps_per_ep: int,
    chunks_at_every_step: bool = False,
):
    """Eval GR00T teacher on val episodes by reading parquet directly.

    libero-r stores images inline (PIL Image dtype), not mp4 — so we bypass
    LeRobotEpisodeLoader and construct observation dicts manually.
    """
    import io
    from PIL import Image
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    print(f"[teacher] loading from {teacher_path}")
    embodiment_tag = EmbodimentTag.LIBERO_PANDA
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=str(teacher_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    modality_configs = policy.get_modality_config()
    state_keys = modality_configs["state"].modality_keys
    action_keys = modality_configs["action"].modality_keys
    video_keys = modality_configs["video"].modality_keys
    language_keys = modality_configs["language"].modality_keys
    print(f"[teacher] state_keys={state_keys}, action_keys={action_keys}, "
          f"video_keys={video_keys}, lang_keys={language_keys}, device hint={device}")

    # libero-r raw doesn't ship modality.json. Use the canonical LIBERO_PANDA
    # slot layout (matches demo_data/libero_demo/meta/modality.json).
    state_slots = {
        "x":       {"start": 0, "end": 1},
        "y":       {"start": 1, "end": 2},
        "z":       {"start": 2, "end": 3},
        "roll":    {"start": 3, "end": 4},
        "pitch":   {"start": 4, "end": 5},
        "yaw":     {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 8},
    }
    # In libero-r, parquet columns are 'image' (workspace) and 'wrist_image'
    # which correspond to the modality.video.{image, wrist_image} keys.
    video_orig_keys = {
        "image":       "observation.images.image",
        "wrist_image": "observation.images.wrist_image",
    }
    # Load tasks.jsonl for language strings
    tasks_map = {}
    with open(raw_data_root / "meta" / "tasks.jsonl") as f:
        for line in f:
            d = json.loads(line)
            tasks_map[d["task_index"]] = d["task"]

    def _decode_jpeg(b: bytes) -> np.ndarray:
        return np.array(Image.open(io.BytesIO(b)), dtype=np.uint8)  # (H, W, 3)

    def _build_obs(row, task_index: int):
        """Build one-step GR00T observation dict from a parquet row."""
        state_full = np.asarray(row["state"], dtype=np.float32)  # [8]
        new_obs = {"video": {}, "state": {}, "language": {}}
        # video: (1, 1, H, W, 3) — batch + time dims
        for k in video_keys:
            field = "image" if video_orig_keys[k] == "observation.images.image" else "wrist_image"
            img = _decode_jpeg(row[field]["bytes"])     # (H, W, 3)
            new_obs["video"][k] = img[None, None, ...]  # (1, 1, H, W, 3)
        # state: (1, 1, slot_size)
        for k in state_keys:
            slot = state_slots[k]
            slot_arr = state_full[slot["start"]:slot["end"]].astype(np.float32)
            new_obs["state"][k] = slot_arr[None, None, :]  # (1, 1, D)
        # language: nested list shape [[str]]
        for lk in language_keys:
            new_obs["language"][lk] = [[tasks_map[task_index]]]
        return new_obs

    def _parse_action(action_dict):
        out = {f"action.{k}": action_dict[k][0] for k in action_dict}
        chunks = []
        for j in range(action_horizon):
            row = np.concatenate([
                np.atleast_1d(np.atleast_1d(out[f"action.{k}"])[j]) for k in action_keys
            ], axis=0)
            chunks.append(row)
        return np.stack(chunks)  # [K, action_dim]

    results = []
    # Warmup using first val episode
    ep0 = pd.read_parquet(raw_data_root / "data" / "chunk-000" / f"episode_{val_indices[0]:06d}.parquet")
    task0 = int(ep0["task_index"].iloc[0])
    parsed0 = _build_obs(ep0.iloc[0], task0)
    for _ in range(2):
        _ = policy.get_action(parsed0)
    torch.cuda.synchronize()

    for ep_i in val_indices:
        ep_path = raw_data_root / "data" / "chunk-000" / f"episode_{ep_i:06d}.parquet"
        traj = pd.read_parquet(ep_path)
        ep_len = len(traj)
        task_idx = int(traj["task_index"].iloc[0])
        n_steps = min(ep_len, max_steps_per_ep)
        ts = list(range(n_steps)) if chunks_at_every_step else list(range(0, n_steps, action_horizon))

        gt_actions = np.stack([
            np.array(traj.iloc[t]["actions"]).astype(np.float32) for t in range(ep_len)
        ])

        sq_total, abs_total = 0.0, 0.0
        n_pairs = 0
        per_step_times = []
        for t in ts:
            parsed_obs = _build_obs(traj.iloc[t], task_idx)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            action_chunk_dict, _ = policy.get_action(parsed_obs)
            torch.cuda.synchronize()
            per_step_times.append(time.perf_counter() - t0)
            pred_chunk = _parse_action(action_chunk_dict)

            end = min(t + action_horizon, ep_len)
            target = gt_actions[t:end]
            if len(target) < action_horizon:
                pad = np.tile(target[-1:], (action_horizon - len(target), 1))
                target = np.concatenate([target, pad], axis=0)

            diff = pred_chunk - target
            sq_total += float((diff ** 2).mean())
            abs_total += float(np.abs(diff).mean())
            n_pairs += 1

        results.append({
            "episode": ep_i,
            "task": task_idx,
            "n_inferences": n_pairs,
            "mse": sq_total / max(n_pairs, 1),
            "mae": abs_total / max(n_pairs, 1),
            "median_latency_ms": float(np.median(per_step_times) * 1000),
            "p90_latency_ms": float(np.percentile(per_step_times, 90) * 1000),
        })
        print(f"  [teacher] ep {ep_i:3d} (task {task_idx})  mse={results[-1]['mse']:.5f}  "
              f"mae={results[-1]['mae']:.5f}  median={results[-1]['median_latency_ms']:.1f}ms")

    return results


def summarize(results: list[dict], label: str):
    n_total_infers = sum(r["n_inferences"] for r in results)
    avg_mse = np.mean([r["mse"] for r in results])
    avg_mae = np.mean([r["mae"] for r in results])
    avg_lat = np.mean([r["median_latency_ms"] for r in results])
    p90_lat = np.mean([r["p90_latency_ms"] for r in results])
    hz = 1000.0 / avg_lat if avg_lat > 0 else float("nan")
    print(f"\n=== {label} ===")
    print(f"  episodes: {len(results)}, total inferences: {n_total_infers}")
    print(f"  avg MSE: {avg_mse:.5f}")
    print(f"  avg MAE: {avg_mae:.5f}")
    print(f"  median latency: {avg_lat:.1f} ms  ({hz:.1f} Hz)")
    print(f"  p90 latency: {p90_lat:.1f} ms")
    print(f"  per-task breakdown:")
    by_task = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)
    for t in sorted(by_task):
        rs = by_task[t]
        print(f"    task {t}: n_eps={len(rs)}  mse={np.mean([r['mse'] for r in rs]):.5f}  "
              f"mae={np.mean([r['mae'] for r in rs]):.5f}")
    return {
        "label": label,
        "n_episodes": len(results),
        "n_inferences": n_total_infers,
        "avg_mse": float(avg_mse),
        "avg_mae": float(avg_mae),
        "median_latency_ms": float(avg_lat),
        "hz": float(hz),
        "p90_latency_ms": float(p90_lat),
        "per_episode": results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_data_root", required=True, type=str,
                   help="Path to LeRobot-format LIBERO root (for GR00T teacher)")
    p.add_argument("--decoded_data_dir", required=True, type=str,
                   help="Path to memmap dataset (for student)")
    p.add_argument("--teacher_path", default="", type=str,
                   help="Path to GR00T-N1.7-LIBERO checkpoint dir; '' to skip teacher")
    p.add_argument("--student_ckpt", default="", type=str,
                   help="Path to trained student .pt checkpoint; '' to skip student")
    p.add_argument("--val_episodes", type=int, default=37,
                   help="Number of last episodes to use as val (matches train split)")
    p.add_argument("--max_steps_per_ep", type=int, default=100)
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--chunks_at_every_step", action="store_true",
                   help="Eval chunk at every t (slower but matches training distribution)")
    p.add_argument("--out_json", default="", type=str)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx = np.load(Path(args.decoded_data_dir) / "index.npz")
    n_eps = len(idx["episode_lengths"])
    val_indices = list(range(n_eps - args.val_episodes, n_eps))
    print(f"val episode indices: {val_indices[:3]}...{val_indices[-3:]} (n={len(val_indices)})")

    summaries = {}
    if args.student_ckpt:
        student, _ = load_student(Path(args.student_ckpt), device)
        student_results = eval_student(
            student, Path(args.decoded_data_dir), val_indices,
            args.action_horizon, device, args.max_steps_per_ep,
            chunks_at_every_step=args.chunks_at_every_step,
        )
        summaries["student"] = summarize(student_results, "STUDENT (Liquid)")

    if args.teacher_path:
        teacher_results = eval_teacher(
            Path(args.teacher_path), Path(args.raw_data_root), val_indices,
            args.action_horizon, device, args.max_steps_per_ep,
            chunks_at_every_step=args.chunks_at_every_step,
        )
        summaries["teacher"] = summarize(teacher_results, "TEACHER (GR00T-N1.7)")

    if "student" in summaries and "teacher" in summaries:
        s = summaries["student"]; t = summaries["teacher"]
        print("\n=== HEAD-TO-HEAD ===")
        mse_ratio = s["avg_mse"] / t["avg_mse"]
        mae_ratio = s["avg_mae"] / t["avg_mae"]
        speedup = s["hz"] / t["hz"]
        print(f"  MSE student/teacher = {mse_ratio:.2f}x ({'student worse' if mse_ratio>1 else 'student better'})")
        print(f"  MAE student/teacher = {mae_ratio:.2f}x")
        print(f"  speed: student {s['hz']:.1f} Hz vs teacher {t['hz']:.1f} Hz = {speedup:.2f}x faster")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summaries, indent=2))
        print(f"\nResults saved to {args.out_json}")


if __name__ == "__main__":
    main()
