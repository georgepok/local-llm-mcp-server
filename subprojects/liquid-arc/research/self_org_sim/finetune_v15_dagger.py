"""DAgger fine-tune for v15: mix expert dataset + Liquid-drift relabeled dataset.

Usage:
    python finetune_v15_dagger.py \
        --resume /tmp/distill_v15/step_015000.pt \
        --drift_npz /tmp/v15_drift_object_relabeled.npz \
        --expert_dir /home/pokazge/datasets/libero-object-expert-v1 \
        --output_dir /tmp/distill_v15_dagger \
        --max_steps 2000 \
        --drift_frac 0.5 \
        --z_groot_drop_prob 0.0
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from distill_groot import TeacherLabelDataset  # type: ignore
from distill_groot_v15 import (
    V15Policy, flow_matching_loss, gripper_bce_loss,
    collect_aux_losses, build_optimizer,
)  # type: ignore
from liquid_arc_substrate_v15 import make_v15_config  # type: ignore


class DriftDataset(torch.utils.data.Dataset):
    """Wraps the relabeled drift .npz to emit the same 8-tuple as TeacherLabelDataset."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.states = d["states"].astype(np.float32)
        self.chunks = d["teacher_chunks"].astype(np.float32)
        self.z_states = d["z_state"].astype(np.float32)
        self.z_banks = d["z_vl_bank"].astype(np.float32)
        self.N = len(self.states)
        print(f"[drift] {self.N} samples; states {self.states.shape}, chunks {self.chunks.shape}, "
              f"z_state {self.z_states.shape}, z_bank {self.z_banks.shape}")
        # Stats
        print(f"[drift] action xyz norm mean={np.linalg.norm(self.chunks[..., :3], axis=-1).mean():.3f}")
        print(f"[drift] grip mean={self.chunks[..., -1].mean():.3f}, sign+={(self.chunks[..., -1] > 0).mean():.3f}")

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        # 8-tuple matching TeacherLabelDataset(return_z_groot="z_state", use_query_bank=True).
        # TeacherLabelDataset emits imgs as HWC uint8 at target_img_size=224.
        return (
            np.zeros((224, 224, 3), dtype=np.uint8),  # img placeholder HWC (unused)
            np.zeros((224, 224, 3), dtype=np.uint8),  # wrist placeholder HWC (unused)
            self.states[i],                           # state
            self.chunks[i],                           # teacher chunk
            0,                                        # task_id placeholder
            self.z_states[i],                         # z_state
            self.z_banks[i],                          # z_bank
            np.zeros((4, 4), dtype=np.float32),       # delta_bank placeholder (unused)
        )


def make_mixed_loader(expert_ds, drift_ds, batch_size, drift_frac, num_workers):
    """Yields batches with drift_frac of batch from drift_ds, rest from expert_ds."""
    n_drift = int(round(batch_size * drift_frac))
    n_expert = batch_size - n_drift
    print(f"[mix] per-batch: {n_expert} expert + {n_drift} drift")

    expert_loader = DataLoader(
        expert_ds, batch_size=n_expert, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    drift_loader = DataLoader(
        drift_ds, batch_size=n_drift, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    def mix_iter():
        e_it = iter(expert_loader)
        d_it = iter(drift_loader)
        while True:
            try:
                eb = next(e_it)
            except StopIteration:
                e_it = iter(expert_loader); eb = next(e_it)
            try:
                db = next(d_it)
            except StopIteration:
                d_it = iter(drift_loader); db = next(d_it)
            # 8 fields each, concatenate along batch dim
            merged = []
            for i, (e, d) in enumerate(zip(eb, db)):
                if isinstance(e, torch.Tensor) and isinstance(d, torch.Tensor):
                    if e.dtype != d.dtype:
                        d = d.to(e.dtype)
                    merged.append(torch.cat([e, d], dim=0))
                else:
                    # Mismatched types (e.g. task_id is int) — use expert's only since unused
                    merged.append(e if isinstance(e, torch.Tensor) else torch.tensor([0]))
            yield merged
    return mix_iter()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", required=True, type=str)
    p.add_argument("--drift_npz", required=True, type=str)
    p.add_argument("--expert_dir", required=True, type=str,
                   help="libero-{suite}-expert-v1 path (decoded+teacher_labels in one)")
    p.add_argument("--output_dir", default="/tmp/distill_v15_dagger", type=str)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Lower than initial 3e-4 — fine-tune from converged ckpt.")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gripper_bce_weight", type=float, default=1.0)
    p.add_argument("--drift_frac", type=float, default=0.5,
                   help="Fraction of each batch sampled from drift dataset (0.0=pure expert, 1.0=pure drift)")
    p.add_argument("--z_groot_drop_prob", type=float, default=0.0,
                   help="Recommended 0.0 — earlier 0.2 dropout suppressed z_bank usage")
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--K_latent", type=int, default=2)
    p.add_argument("--aux_loss_scale", type=float, default=0.1)
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dagger] output: {out_dir}")

    config = make_v15_config(
        d_model=args.d_model, K_total=4 + 2 + args.K_latent,
        n_ode_steps=16, ode_steps_min=12, ode_steps_max=20,
        tau_freeze_steps=5000, integration_time=2.0, cold_start=True,
        aux_loss_scale=args.aux_loss_scale,
    )

    # === Data ===
    expert_ds = TeacherLabelDataset(
        Path(args.expert_dir), Path(args.expert_dir),
        action_horizon=16, target_img_size=224,
        return_goal_img=False, return_z_groot="z_state",
        use_query_bank=True, z_groot_drop_prob=args.z_groot_drop_prob,
    )
    drift_ds = DriftDataset(args.drift_npz)

    # === Model ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V15Policy(
        config, action_horizon=16, action_dim=7, state_dim=8,
        head_d=256, head_layers=4, head_heads=4,
        z_vl_dim=1024, z_state_dim=1536,
        K_bank=4, K_latent=args.K_latent,
    ).to(device)

    print(f"[dagger] resuming from {args.resume}")
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    sd = ckpt["policy"]; own = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        kk = k.replace("_orig_mod.", "")
        if kk in own and own[kk].shape == v.shape:
            own[kk].copy_(v); loaded += 1
    print(f"  loaded {loaded}/{len(own)} tensors")
    start_step = int(ckpt.get("step", 0))
    print(f"  resumed step={start_step} -> finetune for {args.max_steps} more steps")

    optimizer = build_optimizer(model, config, args)

    mix_iter = make_mixed_loader(expert_ds, drift_ds, args.batch_size, args.drift_frac, args.num_workers)

    log_records = []
    t_start = time.time()
    for step in range(args.max_steps):
        batch = next(mix_iter)
        _imgs, _wrists, states, chunks, _ti, z_states, z_banks, _delta_bank = batch
        states = states.to(device, non_blocking=True).float()
        chunks = chunks.to(device, non_blocking=True).float()
        z_states = z_states.to(device, non_blocking=True).float()
        z_banks = z_banks.to(device, non_blocking=True).float()

        out = model.encode(z_banks, z_states, states)
        cond = out["cond"]
        flow_loss = flow_matching_loss(model, cond, chunks)
        gripper_loss = gripper_bce_loss(model, cond, chunks)
        aux = collect_aux_losses(out, config)
        loss = flow_loss + args.gripper_bce_weight * gripper_loss
        for name, l in aux.items():
            if isinstance(l, torch.Tensor):
                if name == "criticality":
                    loss = loss + config.criticality_loss_lambda * l
                elif name == "curvature_diversity":
                    loss = loss + config.curvature_diversity_lambda * l
                elif name == "tau_quality":
                    loss = loss + config.tau_quality_lambda * l
                elif name == "ponder":
                    loss = loss + l

        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step % args.log_every == 0:
            rec = {
                "step": step, "wall_s": time.time() - t_start,
                "loss": float(loss.detach()),
                "flow_mse": float(flow_loss.detach()),
                "gripper_bce": float(gripper_loss.detach()),
                "metric_cv": float(out["metric_cv"].detach()),
                "tau_avg": float(out["tau_avg"].detach()),
                "actual_steps": int(out["actual_steps"]),
            }
            log_records.append(rec)
            print(f"step {step:>5}  loss={rec['loss']:.4f}  flow={rec['flow_mse']:.4f}  "
                  f"grip={rec['gripper_bce']:.4f}  cv={rec['metric_cv']:.3f}  "
                  f"τ={rec['tau_avg']:.2f}  steps={rec['actual_steps']}  wall={rec['wall_s']:.0f}s")
        if step > 0 and step % args.save_every == 0:
            ckpt_path = out_dir / f"step_{start_step + step:06d}.pt"
            torch.save({
                "policy": model.state_dict(), "step": start_step + step,
                "config": config.__dict__, "args": vars(args),
            }, ckpt_path)
            print(f"  [ckpt] {ckpt_path}")

    final_step = start_step + args.max_steps
    final_ckpt = out_dir / f"step_{final_step:06d}.pt"
    torch.save({
        "policy": model.state_dict(), "step": final_step,
        "config": config.__dict__, "args": vars(args),
    }, final_ckpt)
    print(f"\n[dagger] done; final ckpt: {final_ckpt}")
    (out_dir / "dagger_log.json").write_text(json.dumps(log_records, indent=2))


if __name__ == "__main__":
    main()
