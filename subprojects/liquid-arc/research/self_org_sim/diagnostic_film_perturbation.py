"""Micro-diagnostic: does the FiLM mechanism in distill_groot_flow CAN make z_vl
content matter? v7 trained with zero-init z_vl_film. Loss didn't push it
nonzero — vision alone discriminates libero tasks. So FiLM stayed inert.

Question: if we manually set FiLM weights to non-zero, does z_vl content
suddenly affect Liquid's chunk output? If YES → v8 should non-zero-init FiLM
+ force vision-z_vl reliance. If NO → FiLM injection point itself is wrong.

Reads samples directly from existing dataset shards; no libero env, no
groot_server needed. Compatible with concurrent z_vl regen jobs.

Usage:
  source /home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/bin/activate
  cd /home/pokazge/liquid-arc/research/self_org_sim
  python diagnostic_film_perturbation.py --ckpt /tmp/distill_multisuite_v7/step_016000.pt
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot_flow import LiquidFlowPolicy

print = functools.partial(print, flush=True)


def load_policy(ckpt_path: Path, device):
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
        z_groot_dim=sa.get("z_groot_dim", 0),
        gated_mixture=sa.get("gated_mixture", False),
        z_channel_dims=sa.get("z_channel_dims", None),
        query_bank=sa.get("use_query_bank", False),
        query_dim=sa.get("query_dim", 8),
        cadence_head=sa.get("cadence_head", False),
    ).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own = model.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
    return model, ckpt


def load_sample(dataset_dir: Path, sample_idx: int):
    idx = np.load(dataset_dir / "index.npz")
    n_total = int(idx["n_total"])
    img_size = int(idx["img_size"])
    z_vl_dim = int(idx["z_vl_dim"])
    K = int(idx["query_bank_K"])
    qd = int(idx["query_dim"])

    # mmap views
    imgs = np.memmap(dataset_dir / "imgs.dat", dtype=np.uint8, mode="r",
                     shape=(n_total, img_size, img_size, 3))
    wrists = np.memmap(dataset_dir / "wrists.dat", dtype=np.uint8, mode="r",
                       shape=(n_total, img_size, img_size, 3))
    states = np.memmap(dataset_dir / "states.dat", dtype=np.float32, mode="r",
                       shape=(n_total, 8))
    z_vl_bank = np.memmap(dataset_dir / "z_vl_bank.dat", dtype=np.float32, mode="r",
                          shape=(n_total, K, z_vl_dim))
    delta_bank = np.memmap(dataset_dir / "delta_s_bank.dat", dtype=np.float32, mode="r",
                           shape=(n_total, K, qd))
    img = np.array(imgs[sample_idx])
    wri = np.array(wrists[sample_idx])
    state = np.array(states[sample_idx])
    bank = np.array(z_vl_bank[sample_idx])
    delta = np.array(delta_bank[sample_idx])
    task_idx = int(idx["task_indices"][np.searchsorted(idx["episode_starts"], sample_idx, side="right") - 1])
    return img, wri, state, bank, delta, task_idx


def first_sample_per_task(dataset_dir: Path):
    """Return one sample_idx per unique task_idx (first episode of each)."""
    idx = np.load(dataset_dir / "index.npz")
    starts = idx["episode_starts"]
    tasks = idx["task_indices"]
    seen = {}
    for ep, t in enumerate(tasks):
        if int(t) not in seen:
            seen[int(t)] = int(starts[ep])
    return seen


def preprocess(img_raw, target_size):
    if img_raw.shape[0] != target_size:
        # quick nearest-neighbor resize via numpy stride trick
        from PIL import Image
        img_pil = Image.fromarray(img_raw).resize((target_size, target_size))
        return np.array(img_pil)
    return img_raw


def run_swap_test(model, img_t, wri_t, st_t, bank_A_t, bank_B_t, bank_zero_t,
                   delta_t, n_samples=4):
    chunks_A, chunks_B, chunks_C = [], [], []
    with torch.no_grad():
        for i in range(n_samples):
            torch.manual_seed(42 + i)
            cA = model.sample(img_t, wri_t, st_t, task_id=None, n_steps=10,
                               z_bank=bank_A_t, delta_bank=delta_t)[0].cpu().numpy()
            torch.manual_seed(42 + i)
            cB = model.sample(img_t, wri_t, st_t, task_id=None, n_steps=10,
                               z_bank=bank_B_t, delta_bank=delta_t)[0].cpu().numpy()
            torch.manual_seed(42 + i)
            cC = model.sample(img_t, wri_t, st_t, task_id=None, n_steps=10,
                               z_bank=bank_zero_t, delta_bank=delta_t)[0].cpu().numpy()
            chunks_A.append(cA); chunks_B.append(cB); chunks_C.append(cC)
    chunks_A = np.stack(chunks_A); chunks_B = np.stack(chunks_B); chunks_C = np.stack(chunks_C)
    A_var = float(chunks_A.std(axis=0).mean())
    A_B = float(np.mean((chunks_A.mean(0) - chunks_B.mean(0)) ** 2))
    A_C = float(np.mean((chunks_A.mean(0) - chunks_C.mean(0)) ** 2))
    return A_var, A_B, A_C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--dataset_dir", default="/home/pokazge/datasets/libero-10-expert-v1", type=str)
    p.add_argument("--task_a", type=int, default=0)
    p.add_argument("--task_b", type=int, default=3)
    p.add_argument("--n_samples", type=int, default=4)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_policy(Path(args.ckpt), device)
    sa = ckpt["args"]; target_size = sa["img_size"]
    model.eval()

    enc = model.encoder if hasattr(model, "encoder") else model
    if not hasattr(enc, "z_vl_film") or enc.z_vl_film is None:
        print(f"[error] checkpoint has no z_vl_film module — wrong checkpoint")
        return
    print(f"[ok] loaded {args.ckpt}")
    print(f"[ok] z_vl_film: weight_norm={enc.z_vl_film.weight.norm().item():.4f} "
          f"bias_norm={enc.z_vl_film.bias.norm().item():.4f}")

    # Load one sample per task; pick task_a as the held image+state, task_b for swap
    dataset_dir = Path(args.dataset_dir)
    task_to_sample = first_sample_per_task(dataset_dir)
    print(f"[dataset] {dataset_dir.name}: {len(task_to_sample)} tasks → first samples = {task_to_sample}")

    sample_a = task_to_sample[args.task_a]
    sample_b = task_to_sample[args.task_b]
    img_a, wri_a, st_a, bank_a, delta_a, ti_a = load_sample(dataset_dir, sample_a)
    _, _, _, bank_b, _, ti_b = load_sample(dataset_dir, sample_b)
    print(f"[A] task_idx={ti_a} sample={sample_a}  bank_norm={np.linalg.norm(bank_a):.2f}")
    print(f"[B] task_idx={ti_b} sample={sample_b}  bank_norm={np.linalg.norm(bank_b):.2f}")
    print(f"[delta] ||bank_A - bank_B|| = {np.linalg.norm(bank_a - bank_b):.3f}")

    # Hold image/state from A throughout; swap only the bank
    img_p = preprocess(img_a, target_size)
    wri_p = preprocess(wri_a, target_size)
    img_t = torch.from_numpy(img_p).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    wri_t = torch.from_numpy(wri_p).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    st_t = torch.from_numpy(st_a).to(device).float().unsqueeze(0)
    delta_t = torch.from_numpy(delta_a).to(device).float().unsqueeze(0)
    bank_A_t = torch.from_numpy(bank_a).to(device).float().unsqueeze(0)
    bank_B_t = torch.from_numpy(bank_b).to(device).float().unsqueeze(0)
    bank_zero_t = torch.zeros_like(bank_A_t)

    orig_w = enc.z_vl_film.weight.data.clone()
    orig_b = enc.z_vl_film.bias.data.clone()

    print(f"\n=== FiLM weight perturbation sweep ===")
    print(f"sigma | wnorm | A_var   | MSE(A,B)  | MSE(A,C)  | content_signal")
    print(f"------|-------|---------|-----------|-----------|---------------")
    for sigma in [0.0, 0.01, 0.05, 0.1, 0.3, 1.0]:
        if sigma == 0.0:
            enc.z_vl_film.weight.data.copy_(orig_w)
        else:
            torch.manual_seed(0)
            new_w = torch.randn_like(orig_w) * sigma
            enc.z_vl_film.weight.data.copy_(new_w)
        enc.z_vl_film.bias.data.copy_(orig_b)
        wn = enc.z_vl_film.weight.norm().item()

        A_var, A_B, A_C = run_swap_test(model, img_t, wri_t, st_t,
                                         bank_A_t, bank_B_t, bank_zero_t,
                                         delta_t, n_samples=args.n_samples)
        signal = A_B / max(A_var, 1e-6)
        flag = "  ← strong" if signal > 5 else ("  ← detectable" if signal > 2 else "")
        print(f"{sigma:5.2f} | {wn:5.2f} | {A_var:.5f} | {A_B:.6f} | {A_C:.6f} | {signal:6.2f}x {flag}")

    print(f"\n=== INTERPRETATION ===")
    print(f"  - MSE(A,B) growing with sigma → FiLM CAN make z_vl content matter; v8 should non-zero-init FiLM + train pressure")
    print(f"  - MSE(A,B) stays flat → FiLM injection point itself is wrong; need to inject z_vl earlier (e.g., into vis_enc)")


if __name__ == "__main__":
    main()
