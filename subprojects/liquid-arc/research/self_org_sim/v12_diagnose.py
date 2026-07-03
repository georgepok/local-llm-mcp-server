"""Systematic v12 diagnosis: find what's actually broken between training and eval.

Tests:
  1. Training-sample prediction MSE — does v12 reproduce its own training targets?
  2. Cond statistics — what does the encoder produce? Is it informative?
  3. Sampling variance — for same cond, are samples consistent (suggests cond carries signal)
     or random (suggests cond is uninformative)?
  4. Per-step error analysis — where in the 16-step chunk does prediction diverge from target?
  5. Compare side-by-side with v10-DEMO ckpt on identical training samples.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_v12(ckpt_path, device):
    from distill_groot_v12 import V12Policy
    from liquid_arc_substrate_libero import make_v12_config
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = make_v12_config()
    for k, v in ckpt["config"].items():
        if hasattr(config, k):
            setattr(config, k, v)
    model = V12Policy(config, action_horizon=16, action_dim=7, state_dim=8,
                     head_d=256, head_layers=4, head_heads=4, use_goal_img=False).to(device)
    sd = ckpt["policy"]
    own = model.state_dict()
    for k, v in sd.items():
        kk = k.replace("_orig_mod.", "")
        if kk in own and own[kk].shape == v.shape:
            own[kk].copy_(v)
    model.eval()
    return model, ckpt.get("step", "?")


def load_v10(ckpt_path, device):
    from distill_groot_flow import LiquidFlowPolicy
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
        gripper_head=sa.get("gripper_head", False),
        pretrained_vision=sa.get("pretrained_vision", ""),
    ).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own = model.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
    model.eval()
    return model, sa


def load_samples(n=20, suite_dir="/home/pokazge/datasets/libero-10-expert-v1"):
    suite_dir = Path(suite_dir)
    idx = np.load(suite_dir / "index.npz")
    n_total = int(idx["n_total"])
    img_size = int(idx["img_size"])
    imgs = np.memmap(suite_dir / "imgs.dat", dtype=np.uint8, mode="r",
                     shape=(n_total, img_size, img_size, 3))
    wrists = np.memmap(suite_dir / "wrists.dat", dtype=np.uint8, mode="r",
                       shape=(n_total, img_size, img_size, 3))
    states = np.memmap(suite_dir / "states.dat", dtype=np.float32, mode="r",
                       shape=(n_total, 8))
    lbl = np.load(suite_dir / "labels_index.npz")
    sample_idx = lbl["sample_idx"]
    n_samples = int(lbl["n_samples"])
    chunks = np.memmap(suite_dir / "teacher_chunks.dat", dtype=np.float32, mode="r",
                       shape=(n_samples, 16, 7))

    # Sample n evenly spaced indices
    test_idxs = np.linspace(0, n_samples - 1, n, dtype=int)
    samples = []
    from PIL import Image
    for s_idx in test_idxs:
        ep_i, t, _ = sample_idx[s_idx]
        global_idx = int(idx["episode_starts"][int(ep_i)]) + int(t)
        img224 = np.array(Image.fromarray(np.array(imgs[global_idx])).resize((224, 224)), dtype=np.uint8)
        wri224 = np.array(Image.fromarray(np.array(wrists[global_idx])).resize((224, 224)), dtype=np.uint8)
        st = np.array(states[global_idx])
        chunk = np.array(chunks[s_idx])
        samples.append((img224, wri224, st, chunk, int(s_idx)))
    return samples


def to_gpu(img_np, wri_np, st_np, device):
    img_t = torch.from_numpy(img_np).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    wri_t = torch.from_numpy(wri_np).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    st_t = torch.from_numpy(st_np).to(device).float().unsqueeze(0)
    return img_t, wri_t, st_t


@torch.no_grad()
def predict_v12(model, img_t, wri_t, st_t, n_steps=10, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    out = model.encode(img_t, wri_t, st_t)
    cond = out["cond"]
    x = torch.randn(1, 16, 7, device=cond.device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = torch.full((1,), i * dt, device=cond.device)
        v = model.velocity(x, t_val, cond)
        x = x + dt * v
    return x[0].cpu().numpy(), cond[0].cpu().numpy()


@torch.no_grad()
def predict_v10(model, img_t, wri_t, st_t, n_steps=10, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    chunk = model.sample(img_t, wri_t, st_t, task_id=None, n_steps=n_steps)
    return chunk[0].cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n=== 1. Loading models ===")
    v12_model, v12_step = load_v12("/tmp/distill_v12/step_008000.pt", device)
    v10_model, v10_sa = load_v10("/tmp/distill_v10_goal/step_008000.pt", device)
    print(f"  v12 step: {v12_step}")
    print(f"  v10 d={v10_sa['d']} step=8000")

    print("\n=== 2. Loading 20 training samples ===")
    samples = load_samples(n=20)

    # --- 3. Per-sample chunk prediction comparison ---
    print("\n=== 3. Per-sample teacher-MSE: v12 vs v10 ===")
    print("idx | v12_mse | v10_mse | v12_xyz | v10_xyz | v12_first | v10_first | target_first")
    v12_mses, v10_mses = [], []
    v12_xyzs, v10_xyzs = [], []
    v12_first_norms, v10_first_norms, target_first_norms = [], [], []
    for i, (img224, wri224, st, target_chunk, sample_idx) in enumerate(samples):
        img_t, wri_t, st_t = to_gpu(img224, wri224, st, device)
        v12_chunk, _ = predict_v12(v12_model, img_t, wri_t, st_t, seed=42)
        v10_chunk = predict_v10(v10_model, img_t, wri_t, st_t, seed=42)
        v12_mse = float(np.mean((v12_chunk - target_chunk) ** 2))
        v10_mse = float(np.mean((v10_chunk - target_chunk) ** 2))
        v12_xyz = float(np.mean((v12_chunk[:, :3] - target_chunk[:, :3]) ** 2))
        v10_xyz = float(np.mean((v10_chunk[:, :3] - target_chunk[:, :3]) ** 2))
        v12_first = np.linalg.norm(v12_chunk[0, :6])
        v10_first = np.linalg.norm(v10_chunk[0, :6])
        t_first = np.linalg.norm(target_chunk[0, :6])
        v12_mses.append(v12_mse)
        v10_mses.append(v10_mse)
        v12_xyzs.append(v12_xyz)
        v10_xyzs.append(v10_xyz)
        v12_first_norms.append(v12_first)
        v10_first_norms.append(v10_first)
        target_first_norms.append(t_first)
        if i < 5:
            print(f"{sample_idx:6d} | {v12_mse:.4f} | {v10_mse:.4f} | {v12_xyz:.4f} | {v10_xyz:.4f} | "
                  f"{v12_first:.3f}  | {v10_first:.3f}  | {t_first:.3f}")

    print(f"\n  v12 mean total MSE: {np.mean(v12_mses):.4f} ± {np.std(v12_mses):.4f}")
    print(f"  v10 mean total MSE: {np.mean(v10_mses):.4f} ± {np.std(v10_mses):.4f}")
    print(f"  v12 mean xyz MSE:   {np.mean(v12_xyzs):.4f}")
    print(f"  v10 mean xyz MSE:   {np.mean(v10_xyzs):.4f}")
    print(f"  v12 mean first6 norm: {np.mean(v12_first_norms):.3f}")
    print(f"  v10 mean first6 norm: {np.mean(v10_first_norms):.3f}")
    print(f"  target mean first6 norm: {np.mean(target_first_norms):.3f}")

    # --- 4. Per-step MSE distribution ---
    print("\n=== 4. Per-timestep MSE pattern ===")
    v12_per_step = np.zeros(16)
    v10_per_step = np.zeros(16)
    for img224, wri224, st, target_chunk, _ in samples:
        img_t, wri_t, st_t = to_gpu(img224, wri224, st, device)
        v12_chunk, _ = predict_v12(v12_model, img_t, wri_t, st_t, seed=42)
        v10_chunk = predict_v10(v10_model, img_t, wri_t, st_t, seed=42)
        v12_per_step += np.mean((v12_chunk - target_chunk) ** 2, axis=-1)
        v10_per_step += np.mean((v10_chunk - target_chunk) ** 2, axis=-1)
    v12_per_step /= len(samples)
    v10_per_step /= len(samples)
    print("step | v12 mse | v10 mse | ratio")
    for k in range(16):
        ratio = v12_per_step[k] / max(v10_per_step[k], 1e-6)
        print(f"  {k:2d} | {v12_per_step[k]:.4f}  | {v10_per_step[k]:.4f}  | {ratio:.2f}x")

    # --- 5. Sampling variance: does cond carry signal? ---
    print("\n=== 5. Sampling variance test (does cond carry info?) ===")
    # Same cond, different noise → should give similar chunks if cond encodes the task
    img224, wri224, st, target_chunk, _ = samples[0]
    img_t, wri_t, st_t = to_gpu(img224, wri224, st, device)
    v12_chunks_diff_noise = np.stack([
        predict_v12(v12_model, img_t, wri_t, st_t, seed=s)[0] for s in range(5)
    ])
    v10_chunks_diff_noise = np.stack([
        predict_v10(v10_model, img_t, wri_t, st_t, seed=s) for s in range(5)
    ])
    v12_var = float(v12_chunks_diff_noise.var(axis=0).mean())
    v10_var = float(v10_chunks_diff_noise.var(axis=0).mean())
    v12_mean_chunk = v12_chunks_diff_noise.mean(axis=0)
    v10_mean_chunk = v10_chunks_diff_noise.mean(axis=0)
    print(f"  v12 chunk variance across 5 noise seeds: {v12_var:.5f}  (low ↔ cond encodes task)")
    print(f"  v10 chunk variance across 5 noise seeds: {v10_var:.5f}")
    print(f"  v12 mean chunk[0]: {v12_mean_chunk[0]}")
    print(f"  v10 mean chunk[0]: {v10_mean_chunk[0]}")
    print(f"  target  chunk[0]: {target_chunk[0]}")

    # --- 6. Cond statistics ---
    print("\n=== 6. Cond vector statistics ===")
    v12_conds = []
    for img224, wri224, st, _, _ in samples:
        img_t, wri_t, st_t = to_gpu(img224, wri224, st, device)
        _, cond = predict_v12(v12_model, img_t, wri_t, st_t, seed=0)
        v12_conds.append(cond)
    v12_conds = np.stack(v12_conds)  # [N, d]
    cond_mean = v12_conds.mean(axis=0)
    cond_std = v12_conds.std(axis=0)
    # Pairwise distance between conds (different samples should produce different cond)
    diffs = v12_conds[:, None] - v12_conds[None, :]
    pairwise_dist = float(np.sqrt((diffs ** 2).sum(axis=-1)).mean())
    print(f"  v12 cond shape: {v12_conds.shape}")
    print(f"  v12 cond global mean: {cond_mean.mean():.4f}, global std: {cond_std.mean():.4f}")
    print(f"  v12 cond mean pairwise distance: {pairwise_dist:.4f}")
    print(f"    (low ↔ cond collapses; should be O(sqrt(d)) ≈ 16 for d=256 random)")

if __name__ == "__main__":
    main()
