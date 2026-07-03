"""Verify the Liquid student learned input-conditional behavior, not just statistics.

Compares student val MSE against three trivial baselines:
  - global_mean: predict the global average action across all training data
  - per_task_mean: predict the per-task average action (uses task_index)
  - last_state: predict zero (delta-EEF actions are mostly small steps from current state)
  - random_perm: predict the action of a randomly-shuffled timestep (noise floor)

A meaningful student should beat ALL trivial baselines by a wide margin.

Run on Spark:
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
  python verify_student.py \\
    --decoded_data_dir /home/pokazge/datasets/libero-10-r-decoded \\
    --student_ckpt /tmp/distill_groot_v1/step_030000.pt \\
    --val_episodes 37
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot import LiquidStudent, LiberoMemmapDataset, collate_batch

print = functools.partial(print, flush=True)


def chunk_pad(actions: np.ndarray, t: int, K: int, ep_len: int) -> np.ndarray:
    end = min(t + K, ep_len)
    chunk = actions[t:end]
    if len(chunk) < K:
        pad = np.tile(chunk[-1:], (K - len(chunk), 1))
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk


def compute_baselines(decoded_dir: Path, train_indices, val_indices, action_horizon=16):
    """Compute baseline action chunks for each val sample.

    Returns dict of (label -> [N_val, K, A] predictions) and ground-truth [N_val, K, A].
    """
    idx = np.load(decoded_dir / "index.npz")
    starts = idx["episode_starts"]
    lengths = idx["episode_lengths"]
    task_indices = idx["task_indices"]
    n_total = int(idx["n_total"])

    actions = np.memmap(decoded_dir / "actions.dat", dtype=np.float32, mode="r",
                        shape=(n_total, 7))

    # Stats from TRAIN data only
    train_action_rows = []
    train_action_per_task = {}
    for ep_i in train_indices:
        s, n = int(starts[ep_i]), int(lengths[ep_i])
        ep_acts = np.array(actions[s:s + n])
        train_action_rows.append(ep_acts)
        ti = int(task_indices[ep_i])
        train_action_per_task.setdefault(ti, []).append(ep_acts)
    train_action_all = np.concatenate(train_action_rows, axis=0)
    global_mean = train_action_all.mean(axis=0)  # [7]
    print(f"Train action global mean: {global_mean}")
    print(f"Train action global std:  {train_action_all.std(axis=0)}")
    per_task_mean = {ti: np.concatenate(v, axis=0).mean(axis=0) for ti, v in train_action_per_task.items()}

    # Build val ground-truth chunks
    targets = []
    pred_global_mean = []
    pred_per_task = []
    pred_last = []   # repeat first action of chunk (i.e., predict no movement = current pose stays)
    pred_zero = []   # predict zero action
    sample_meta = []

    rng = np.random.default_rng(0)
    val_action_pool = []
    for ep_i in val_indices:
        s, n = int(starts[ep_i]), int(lengths[ep_i])
        val_action_pool.append(np.array(actions[s:s + n]))
    val_action_pool = np.concatenate(val_action_pool, axis=0)  # [N_val_total, 7]

    pred_random = []  # random val action repeated across the chunk

    for ep_i in val_indices:
        s, n = int(starts[ep_i]), int(lengths[ep_i])
        ep_acts = np.array(actions[s:s + n])
        ti = int(task_indices[ep_i])
        # Inference points: every action_horizon steps
        ts = list(range(0, n, action_horizon))
        for t in ts:
            tgt = chunk_pad(ep_acts, t, action_horizon, n)
            targets.append(tgt)
            pred_global_mean.append(np.tile(global_mean, (action_horizon, 1)))
            pred_per_task.append(np.tile(per_task_mean[ti], (action_horizon, 1)))
            pred_zero.append(np.zeros_like(tgt))
            # "last" baseline: replay the action at this timestep K times
            current_action = ep_acts[t]
            pred_last.append(np.tile(current_action, (action_horizon, 1)))
            # random: one random sample from val pool
            r = rng.integers(0, len(val_action_pool))
            pred_random.append(np.tile(val_action_pool[r], (action_horizon, 1)))
            sample_meta.append((ep_i, t, ti))

    targets = np.stack(targets)               # [N, K, 7]
    pred_global_mean = np.stack(pred_global_mean)
    pred_per_task = np.stack(pred_per_task)
    pred_zero = np.stack(pred_zero)
    pred_last = np.stack(pred_last)
    pred_random = np.stack(pred_random)
    return {
        "global_mean": pred_global_mean,
        "per_task_mean": pred_per_task,
        "zero_action": pred_zero,
        "current_action_repeated": pred_last,
        "random_train_action": pred_random,
    }, targets, sample_meta


def eval_student(ckpt_path: Path, decoded_dir: Path, val_indices, action_horizon, device):
    """Run student on all (ep_i, t) sample points; return predictions [N, K, 7]."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa = ckpt["args"]
    halt_mode = "learned" if sa["policy"] == "liquid_halt" else "none"
    student = LiquidStudent(d=sa["d"], d_vis=sa["d"], img_size=sa["img_size"],
                            k_max=sa["k"], halt_mode=halt_mode,
                            min_steps=sa["halting_min_steps"],
                            action_horizon=sa["action_horizon"]).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own = student.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
    student.eval()

    ds = LiberoMemmapDataset(decoded_dir, action_horizon=action_horizon, episode_indices=val_indices)
    starts = ds.starts
    lengths = ds.lengths

    preds = []
    with torch.no_grad():
        for ep_i in val_indices:
            s = int(starts[ep_i]); n = int(lengths[ep_i])
            for t in range(0, n, action_horizon):
                img = torch.from_numpy(np.array(ds.imgs[s + t])).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                wri = torch.from_numpy(np.array(ds.wrists[s + t])).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                st = torch.from_numpy(np.array(ds.states[s + t])).to(device).float().unsqueeze(0)
                pred, _ = student(img, wri, st)
                preds.append(pred[0].cpu().numpy())
    return np.stack(preds)


def metrics(pred, target):
    """Per-channel and overall MSE/MAE."""
    diff = pred - target
    mse = float((diff ** 2).mean())
    mae = float(np.abs(diff).mean())
    per_ch_mse = (diff ** 2).mean(axis=(0, 1))  # [7]
    per_ch_mae = np.abs(diff).mean(axis=(0, 1))  # [7]
    return mse, mae, per_ch_mse, per_ch_mae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoded_data_dir", required=True, type=str)
    p.add_argument("--student_ckpt", required=True, type=str)
    p.add_argument("--val_episodes", type=int, default=37)
    p.add_argument("--action_horizon", type=int, default=16)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoded_dir = Path(args.decoded_data_dir)
    idx = np.load(decoded_dir / "index.npz")
    n_eps = len(idx["episode_lengths"])
    val_indices = list(range(n_eps - args.val_episodes, n_eps))
    train_indices = list(range(n_eps - args.val_episodes))

    print(f"\nComputing baselines on {len(train_indices)} train + {len(val_indices)} val episodes...")
    baselines, targets, _meta = compute_baselines(decoded_dir, train_indices, val_indices, args.action_horizon)
    print(f"Total val sample points: {len(targets)}")

    print(f"\nRunning student inference on val samples...")
    student_preds = eval_student(Path(args.student_ckpt), decoded_dir, val_indices,
                                 args.action_horizon, device)
    assert student_preds.shape == targets.shape, f"shape mismatch {student_preds.shape} vs {targets.shape}"

    ch_names = ["dx", "dy", "dz", "dRoll", "dPitch", "dYaw", "grip"]
    print("\n" + "=" * 80)
    print(f"{'PREDICTOR':<28} {'MSE':>10} {'MAE':>10}    PER-CHANNEL MAE")
    print("=" * 80)

    rows = []
    for label, pred in [
        ("baseline:zero_action", baselines["zero_action"]),
        ("baseline:global_mean", baselines["global_mean"]),
        ("baseline:per_task_mean", baselines["per_task_mean"]),
        ("baseline:current_repeated", baselines["current_action_repeated"]),
        ("baseline:random_action", baselines["random_train_action"]),
        ("STUDENT (Liquid)", student_preds),
    ]:
        mse, mae, ch_mse, ch_mae = metrics(pred, targets)
        per_ch = "  ".join(f"{c}={v:.3f}" for c, v in zip(ch_names, ch_mae))
        print(f"{label:<28} {mse:>10.5f} {mae:>10.5f}    {per_ch}")
        rows.append((label, mse, mae))

    print("\n" + "=" * 80)
    student_mse = rows[-1][1]
    print("STUDENT vs BASELINES (lower MSE = better):")
    for label, mse, _ in rows[:-1]:
        ratio = mse / student_mse
        verdict = "STUDENT WINS" if ratio > 1.0 else "STUDENT LOSES"
        print(f"  {label:<28} MSE = {mse:.5f}  →  {ratio:.2f}× student   ({verdict})")

    # Overall verdict
    best_baseline = min(rows[:-1], key=lambda r: r[1])
    print(f"\nBest trivial baseline: {best_baseline[0]} (MSE={best_baseline[1]:.5f})")
    print(f"Student MSE: {student_mse:.5f}")
    if student_mse < best_baseline[1] * 0.5:
        print(f"=> STUDENT IS LEARNING SIGNIFICANTLY (≥2× better than best trivial baseline)")
    elif student_mse < best_baseline[1]:
        print(f"=> Student modestly beats best baseline ({best_baseline[1]/student_mse:.2f}× better)")
    else:
        print(f"=> WARNING: student does NOT beat trivial baseline; possibly memorized statistics only")


if __name__ == "__main__":
    main()
