"""V9 path A: offline-train forward_pred_head against V7b's frozen encoder.

Solves the cold-start failure mode of V9. Without this, forward_pred_head is
zero-initialized at deployment — first ~89 SGD updates per episode are noise,
destabilizing V7b's frozen weights and producing K=0 regression.

Two phases:
1. Pre-compute V7b cond for every sample (one encoder forward per sample).
2. Train forward_pred_head: predict cond_{t+1} from (cond_t, chunk_t.mean).
   Use only same-episode adjacent pairs.

Saves a new checkpoint at --out_ckpt that's V7b's state_dict + the trained
forward_pred_head weights, ready for deployment-time adaptation.

Run on Spark in main venv (has CUDA torch):
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
  python train_forward_head.py \\
    --base_ckpt /tmp/distill_s1s2_v7b/step_008000.pt \\
    --data_dir /home/pokazge/datasets/groot-sim-tempquery-v1 \\
    --out_ckpt /tmp/distill_s1s2_v9/step_008000.pt \\
    --max_steps 5000
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot import TeacherLabelDataset, collate_batch
from distill_groot_flow import LiquidFlowPolicy

print = functools.partial(print, flush=True)
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
except Exception:
    pass


def load_v7b_with_forward_head(ckpt_path: Path, device):
    """Load V7b checkpoint into a model with forward_model=True so the head exists."""
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
        forward_model=True,  # KEY: instantiate forward_pred_head
    ).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v); loaded += 1
    print(f"[load] {loaded}/{len(own)} tensors from {ckpt_path}")
    return model, ckpt


@torch.no_grad()
def precompute_conds(model, dataset, device, batch_size=256, num_workers=8):
    """Run V7b encoder over the entire dataset, store cond per sample."""
    model.eval()
    n = len(dataset)
    d = model.encoder.fuse.out_features
    conds = np.zeros((n, d), dtype=np.float32)

    # Iterate in order so we can index by sample idx directly.
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_batch,
        pin_memory=True, drop_last=False,
    )

    print(f"[precompute] {n} samples through V7b encoder...")
    cur = 0
    t0 = time.time()
    for bi, batch in enumerate(loader):
        # Unpack (8-tuple for V7 dataset).
        if len(batch) == 8:
            imgs, wrists, states, chunks, _, z_groot, z_bank, delta_bank = batch
        elif len(batch) == 6:
            imgs, wrists, states, chunks, _, z_groot = batch
            z_bank = delta_bank = None
        else:
            raise RuntimeError(f"unexpected batch size {len(batch)}")
        imgs = imgs.to(device, non_blocking=True)
        wrists = wrists.to(device, non_blocking=True)
        states = states.to(device, non_blocking=True)
        z_groot = z_groot.to(device, non_blocking=True) if z_groot is not None else None
        z_bank = z_bank.to(device, non_blocking=True) if z_bank is not None else None
        delta_bank = delta_bank.to(device, non_blocking=True) if delta_bank is not None else None
        cond, _ = model.forward_encoder(
            imgs, wrists, states, task_id=None,
            z_groot=z_groot, z_bank=z_bank, delta_bank=delta_bank,
        )
        b = cond.shape[0]
        conds[cur:cur + b] = cond.float().cpu().numpy()
        cur += b
        if bi % 5 == 0:
            print(f"  batch {bi+1}/{len(loader)}  cur={cur}/{n}  ({time.time()-t0:.1f}s)")

    print(f"[precompute] done. {cur} conds, dim={d}, took {time.time()-t0:.1f}s")
    return conds


def build_pair_indices(dataset):
    """Indices i where sample i and i+1 are in the same episode."""
    si = dataset.sample_idx  # [N, 3] = (ep_i, t, ti)
    pairs = []
    for i in range(len(si) - 1):
        if int(si[i + 1][0]) == int(si[i][0]):
            pairs.append(i)
    pairs = np.array(pairs, dtype=np.int64)
    print(f"[pairs] {len(pairs)} same-episode adjacent pairs (out of {len(si)} samples)")
    return pairs


def train_forward_head(model, conds, chunks, pair_idx, device,
                       max_steps=5000, batch_size=256, lr=1e-3):
    """Train just the forward_pred_head on (cond_t, chunk_t.mean) → cond_{t+1} pairs."""
    head = model.encoder.forward_pred_head
    # Re-init: zero-init was for online safety; for offline supervised we want
    # a fresh start with default initialization.
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    head.train()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=lr * 0.05,
    )

    # Move to device
    conds_t = torch.from_numpy(conds).to(device)
    # chunks shape [N, 16, 7]; we use mean over time → [N, 7]
    chunks_mean = torch.from_numpy(chunks.mean(axis=1)).to(device)

    print(f"[train head] {max_steps} steps, batch={batch_size}, lr={lr}, "
          f"head_params={sum(p.numel() for p in head.parameters()):,}")
    t0 = time.time()
    losses = []
    for step in range(max_steps):
        # Sample batch of pair indices
        sel = np.random.choice(pair_idx, size=batch_size, replace=False)
        sel_t = torch.from_numpy(sel).to(device)
        sel_next_t = sel_t + 1

        cond_t = conds_t[sel_t]
        cond_t1 = conds_t[sel_next_t]
        chunk_mean_t = chunks_mean[sel_t]

        x = torch.cat([cond_t, chunk_mean_t], dim=-1)
        pred = head(x)
        loss = F.mse_loss(pred, cond_t1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if step % 100 == 0:
            recent = np.mean(losses[-100:])
            print(f"  step {step:5d}/{max_steps}  loss={loss.item():.5f}  "
                  f"avg100={recent:.5f}  lr={scheduler.get_last_lr()[0]:.4e}  "
                  f"({time.time()-t0:.1f}s)")
    print(f"[train head] done. final avg100 loss = {np.mean(losses[-100:]):.5f}")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", required=True, type=str)
    p.add_argument("--data_dir", required=True, type=str)
    p.add_argument("--extra_data_dirs", default="", type=str,
                   help="Comma-separated additional dataset paths (e.g. libero-*-expert-v1) "
                        "to include in forward-head training. Each used as both data_dir "
                        "and teacher_labels_dir; pair indices are built per-dataset to "
                        "respect episode boundaries within each suite.")
    p.add_argument("--out_ckpt", required=True, type=str)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # Load V7b with forward head
    model, ckpt = load_v7b_with_forward_head(Path(args.base_ckpt), device)
    sa = ckpt["args"]

    # Build per-suite datasets and concatenate. Pair indices are built within
    # each dataset (so we never pair the last sample of suite A with the first
    # of suite B) then offset by cumulative sample count.
    all_data_dirs = [args.data_dir]
    if args.extra_data_dirs:
        all_data_dirs.extend(p.strip() for p in args.extra_data_dirs.split(",") if p.strip())

    all_conds: list = []
    all_chunks: list = []
    all_pair_indices: list = []
    cumulative_offset = 0
    for data_dir in all_data_dirs:
        print(f"\n[dataset] processing {data_dir}")
        ds = TeacherLabelDataset(
            Path(data_dir), Path(data_dir),
            action_horizon=sa["action_horizon"],
            return_task_id=False,
            target_img_size=sa["img_size"],
            return_z_groot=sa.get("use_z_groot", "z_vl"),
            cadence_dropout=None,
            use_query_bank=sa.get("use_query_bank", False),
        )
        print(f"  samples: {len(ds)}")

        conds_d = precompute_conds(model, ds, device,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)
        pair_idx_d = build_pair_indices(ds)
        n_samples_d = ds.n_samples
        chunks_d = np.memmap(Path(data_dir) / "teacher_chunks.dat",
                             dtype=np.float32, mode="r",
                             shape=(n_samples_d, sa["action_horizon"], 7))
        all_conds.append(conds_d)
        all_chunks.append(np.array(chunks_d))
        all_pair_indices.append(pair_idx_d + cumulative_offset)
        cumulative_offset += n_samples_d

    conds = np.concatenate(all_conds, axis=0)
    chunks_arr = np.concatenate(all_chunks, axis=0)
    pair_idx = np.concatenate(all_pair_indices, axis=0)
    print(f"\n[combined] conds={conds.shape}  chunks={chunks_arr.shape}  pairs={pair_idx.shape}")

    # Phase 2: train the forward head
    model = train_forward_head(
        model, conds, chunks_arr, pair_idx, device,
        max_steps=args.max_steps, batch_size=args.batch_size, lr=args.lr,
    )

    # Save the new checkpoint: V7b state_dict + forward_pred_head trained weights.
    out = Path(args.out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_state = model.state_dict()
    ckpt_out = {
        "step": ckpt.get("step"),
        "policy": new_state,
        "opt": ckpt.get("opt"),
        "args": {**sa, "forward_model": True},
    }
    torch.save(ckpt_out, out)
    print(f"[save] {out}  ({sum(v.numel() for v in new_state.values()):,} total params)")


if __name__ == "__main__":
    main()
