"""Diagnose v15 at step_015000 — why did it regress from train metrics to closed-loop?

Probes:
  1. Sensitivity: how much does cond change if we zero z_bank / z_state / state?
     If substrate ignores GR00T features (learned to lean on latent slots due
     to 20% dropout pressure), zeroing them changes cond little.
  2. Per-position attention weights from the attention pool — which positions dominate?
  3. xyz_corr, grip_corr, grip_mean at step 15K on held-out batch.
  4. Cond variance across distinct tasks (does cond discriminate at all?)
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from liquid_arc_substrate_v15 import V15Encoder, make_v15_config
from distill_groot_v15 import V15Policy, TeacherLabelDataset

import os
CKPT = os.environ.get("CKPT", "/tmp/distill_v15/step_015000.pt")
DATASETS = [
    "/home/pokazge/datasets/libero-10-expert-v1",
    "/home/pokazge/datasets/libero-spatial-expert-v1",
    "/home/pokazge/datasets/libero-object-expert-v1",
    "/home/pokazge/datasets/libero-goal-expert-v1",
]
N_BATCH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[diag] loading {CKPT} on {DEVICE}")
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
sd = ckpt["policy"]
z_vl_dim = sd["encoder.zbank_proj.weight"].shape[1]
z_state_dim = sd["encoder.zstate_proj.weight"].shape[1]
K_latent = int(ckpt["args"]["K_latent"])
print(f"[diag] z_vl_dim={z_vl_dim}, z_state_dim={z_state_dim}, K_latent={K_latent}")

model = V15Policy(
    config=make_v15_config(d_model=768),
    state_dim=8, z_vl_dim=z_vl_dim, z_state_dim=z_state_dim,
    K_bank=4, K_latent=K_latent,
).to(DEVICE)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"[diag] loaded with {len(missing)} missing, {len(unexpected)} unexpected")
model.eval()

# Load one batch per suite, pick a few obs per suite
samples = []
for d in DATASETS:
    ds = TeacherLabelDataset(d, d, return_z_groot="z_state", use_query_bank=True)
    # Random N/4 entries per suite
    idx = np.random.choice(len(ds), N_BATCH // len(DATASETS), replace=False)
    for i in idx:
        samples.append((ds[i], d.split("/")[-1]))

# Stack
states, chunks, z_states, z_banks = [], [], [], []
suites = []
for (_imgs, _wrists, st, ch, _ti, zs, zb, _db), suite in samples:
    states.append(st)
    chunks.append(ch)
    z_states.append(zs)
    z_banks.append(zb)
    suites.append(suite)

states_t = torch.stack([s if isinstance(s, torch.Tensor) else torch.from_numpy(s) for s in states]).to(DEVICE).float()
chunks_t = torch.stack([c if isinstance(c, torch.Tensor) else torch.from_numpy(c) for c in chunks]).to(DEVICE).float()
z_states_t = torch.stack([z if isinstance(z, torch.Tensor) else torch.from_numpy(z) for z in z_states]).to(DEVICE).float()
z_banks_t = torch.stack([z if isinstance(z, torch.Tensor) else torch.from_numpy(z) for z in z_banks]).to(DEVICE).float()
print(f"[diag] batch shape: states {states_t.shape}, chunks {chunks_t.shape}, z_states {z_states_t.shape}, z_banks {z_banks_t.shape}")

# === 1. Sensitivity: zero each input, measure cond change ===
print("\n=== sensitivity (cond L2 change when input is zeroed) ===")
with torch.no_grad():
    out_full = model.encode(z_banks_t, z_states_t, states_t)
    cond_full = out_full["cond"]
    attn_full = out_full["attn_weights"]  # [B, K]
    print(f"cond_full: mean L2={cond_full.norm(dim=-1).mean():.3f}, std L2={cond_full.norm(dim=-1).std():.3f}")
    print(f"per-position attention (mean over batch): {attn_full.mean(dim=0).cpu().numpy().round(3)}")
    print(f"position labels: bank0,bank1,bank2,bank3,zstate,state,latent0,latent1")

    for label, mod in [("zero z_bank", "z_bank"), ("zero z_state", "z_state"), ("zero state", "state")]:
        zb = torch.zeros_like(z_banks_t) if mod == "z_bank" else z_banks_t
        zs = torch.zeros_like(z_states_t) if mod == "z_state" else z_states_t
        st = torch.zeros_like(states_t) if mod == "state" else states_t
        out = model.encode(zb, zs, st)
        delta = (out["cond"] - cond_full).norm(dim=-1).mean()
        rel = delta / (cond_full.norm(dim=-1).mean() + 1e-8)
        print(f"  {label}: ΔL2={delta:.3f} (rel {rel*100:.1f}% of cond magnitude)")

# === 2. Cross-task cond variance ===
print("\n=== cross-suite cond variance ===")
with torch.no_grad():
    out = model.encode(z_banks_t, z_states_t, states_t)
    cond = out["cond"]
    by_suite = {}
    for c, s in zip(cond, suites):
        by_suite.setdefault(s, []).append(c)
    means = {}
    for s, cs in by_suite.items():
        cs_t = torch.stack(cs)
        means[s] = cs_t.mean(dim=0)
        print(f"  {s}: n={len(cs)}, mean L2={cs_t.norm(dim=-1).mean():.3f}, std L2={cs_t.norm(dim=-1).std():.3f}")
    print("\n  pairwise mean-cond L2 distances:")
    keys = list(means.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            d = (means[keys[i]] - means[keys[j]]).norm()
            print(f"    {keys[i]} ↔ {keys[j]}: {d:.3f}")
    # Total cond variance vs within-suite variance
    total_var = cond.var(dim=0).sum()
    within_var = sum(torch.stack(cs).var(dim=0).sum() for cs in by_suite.values()) / len(by_suite)
    print(f"\n  total cond variance: {total_var:.3f}")
    print(f"  within-suite variance: {within_var:.3f}")
    print(f"  between-suite ratio: {(total_var - within_var) / total_var * 100:.1f}% (higher = more task-discriminative)")

# === 3. xyz_corr + grip_corr at step 15K ===
print("\n=== offline action prediction at step 15K (n_steps=10 sampling) ===")
with torch.no_grad():
    out = model.encode(z_banks_t, z_states_t, states_t)
    cond = out["cond"]
    K, A = chunks_t.shape[1], chunks_t.shape[2]
    x = torch.randn_like(chunks_t)
    dt = 1.0 / 10
    for i in range(10):
        t_val = torch.full((chunks_t.shape[0],), i * dt, device=DEVICE)
        v = model.velocity(x, t_val, cond)
        x = x + dt * v
    pred = x
    target = chunks_t
    # First action (action[0]) is what gets executed
    pred_xyz = pred[:, 0, :3].cpu().numpy()
    target_xyz = target[:, 0, :3].cpu().numpy()
    pred_grip = pred[:, 0, -1].cpu().numpy()
    target_grip = target[:, 0, -1].cpu().numpy()
    xyz_corr = np.corrcoef(pred_xyz.flatten(), target_xyz.flatten())[0, 1]
    grip_corr = np.corrcoef(pred_grip, target_grip)[0, 1]
    print(f"  xyz_corr (action[0] x,y,z, all flat): {xyz_corr:.3f}")
    print(f"  grip_corr (action[0] gripper): {grip_corr:.3f}")
    print(f"  target grip_mean: {target_grip.mean():.3f}  pred grip_mean: {pred_grip.mean():.3f}")
    print(f"  target grip sign +1 frac: {(target_grip > 0).mean():.3f}")
    print(f"  pred grip sign +1 frac:   {(pred_grip > 0).mean():.3f}")

print("\n=== done ===")
