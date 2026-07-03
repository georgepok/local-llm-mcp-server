"""v16 trainer: substrate AS denoiser (flow head).

  encoder (flat MLP) → cond [B, d]
  decoder (substrate over K=16 action positions) → v_pred [B, K, A]
  gripper_head(cond) → gripper logits  (kept for binary inductive bias)

Loss: standard flow matching MSE + gripper BCE + substrate aux losses (criticality,
curvature_diversity, tau_quality, ponder/halting) — same as v15 but the substrate
is now organizing action-chunk temporal positions instead of feature positions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from distill_groot import TeacherLabelDataset  # type: ignore
from liquid_arc_substrate_v16 import V16Encoder, V16Decoder, make_v16_config  # type: ignore

# Aux loss helpers (canonical LiquidARC SoC stabilizers)
from liquid_arc.sustained_criticality import (  # type: ignore
    compute_criticality_loss,
    compute_curvature_diversity_loss,
    compute_tau_quality_loss,
)

torch.set_float32_matmul_precision("high")  # per project memory: ~30% perf


class V16Policy(nn.Module):
    def __init__(self, config, action_horizon=16, action_dim=7, state_dim=8,
                 z_vl_dim=1024, z_state_dim=1536, K_bank=4):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.encoder = V16Encoder(
            d=config.d_model, z_vl_dim=z_vl_dim, z_state_dim=z_state_dim,
            K_bank=K_bank, state_dim=state_dim,
        )
        self.decoder = V16Decoder(
            config, action_horizon=action_horizon, action_dim=action_dim,
        )
        # Gripper head — predicts gripper logits from cond. Keeps the binary
        # inductive bias that prevents gripper collapse (per project memory).
        self.gripper_head = nn.Sequential(
            nn.Linear(config.d_model, 128), nn.SiLU(),
            nn.Linear(128, action_horizon),
        )

    def encode(self, z_bank, z_state, state):
        return self.encoder(z_bank, z_state, state)

    def velocity(self, noisy_chunk, t, cond):
        out = self.decoder(noisy_chunk, t, cond)
        return out["v_pred"], out

    def gripper_logits(self, cond):
        return self.gripper_head(cond)

    def geo_parameters(self):
        return self.decoder.geo_parameters()

    def other_parameters(self):
        geo_ids = {id(p) for p in self.geo_parameters()}
        return [p for p in self.parameters() if id(p) not in geo_ids]


def flow_matching_loss(model, cond, target_chunk):
    B, K, A = target_chunk.shape
    t = torch.rand(B, device=target_chunk.device)
    noise = torch.randn_like(target_chunk)
    x_t = t.view(-1, 1, 1) * target_chunk + (1 - t.view(-1, 1, 1)) * noise
    v_target = target_chunk - noise
    v_pred, dec_out = model.velocity(x_t, t, cond)
    loss = F.mse_loss(v_pred, v_target)
    return loss, dec_out


def gripper_bce_loss(model, cond, target_chunk):
    gripper_target = (target_chunk[..., -1] > 0).float()
    gripper_logits = model.gripper_logits(cond)
    return F.binary_cross_entropy_with_logits(gripper_logits, gripper_target)


def collect_aux_losses(dec_out, config):
    losses = {}
    g = dec_out["g"]; h_final = dec_out["h_final"]
    tau = dec_out["tau"]; t_diff_param = dec_out["t_diff_param"]
    if config.criticality_loss_enabled and tau is not None and t_diff_param is not None:
        loss, _ = compute_criticality_loss(
            h_final, g, tau, t_diff_param,
            target_ratio=config.criticality_target_ratio,
            d_sq_target=config.criticality_D_sq_target,
        )
        losses["criticality"] = loss
    if config.curvature_diversity_loss_enabled:
        losses["curvature_diversity"] = compute_curvature_diversity_loss(
            g, cv_floor=config.curvature_cv_floor,
            cv_ceiling=config.curvature_cv_ceiling,
        )
    if config.tau_quality_loss_enabled and tau is not None:
        mean_target = config.tau_mean_target
        if mean_target <= 0:
            mean_target = config.integration_time / config.n_ode_steps * 16
        losses["tau_quality"] = compute_tau_quality_loss(
            tau, mean_target=mean_target,
            log_spread_target=config.tau_log_spread_target,
        )
    if config.halting_enabled and "ponder_cost" in dec_out:
        losses["ponder"] = dec_out["ponder_cost"].mean() * config.halting_ponder_lambda
    return losses


def build_optimizer(model, config, args):
    geo_params = model.geo_parameters()
    other_params = model.other_parameters()
    print(f"  geo params: {sum(p.numel() for p in geo_params):,}")
    print(f"  other params: {sum(p.numel() for p in other_params):,}")
    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr},
            {"params": geo_params, "lr": args.lr * config.structural_lr_ratio},
        ],
        weight_decay=args.weight_decay,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoded_dirs", required=True, type=str)
    p.add_argument("--teacher_labels_dirs", required=True, type=str)
    p.add_argument("--output_dir", default="/tmp/distill_v16", type=str)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=15000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--target_img_size", type=int, default=224)
    p.add_argument("--gripper_bce_weight", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--aux_loss_scale", type=float, default=0.1)
    p.add_argument("--z_groot_drop_prob", type=float, default=0.0,
                   help="No dropout by default — v15 dropout backfired by suppressing z usage")
    p.add_argument("--resume", default="", type=str)
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[v16] output: {out_dir}")

    config = make_v16_config(
        d_model=args.d_model, K_total=16,
        n_ode_steps=16, ode_steps_min=12, ode_steps_max=20,
        tau_freeze_steps=5000, integration_time=2.0, cold_start=True,
        aux_loss_scale=args.aux_loss_scale,
    )
    print(f"[v16] config: d={config.d_model}, K=16 action positions, halting={config.halting_enabled}")

    # === Data ===
    decoded_dirs = [Path(d.strip()) for d in args.decoded_dirs.split(",") if d.strip()]
    labels_dirs = [Path(d.strip()) for d in args.teacher_labels_dirs.split(",") if d.strip()]
    datasets = []
    z_vl_dim = 1024
    z_state_dim = 1536
    for dd, ld in zip(decoded_dirs, labels_dirs):
        ds = TeacherLabelDataset(
            dd, ld, action_horizon=16,
            target_img_size=args.target_img_size,
            return_goal_img=False,
            return_z_groot="z_state",
            use_query_bank=True,
            z_groot_drop_prob=args.z_groot_drop_prob,
        )
        datasets.append(ds)
    train_ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"[v16] samples: {len(train_ds)}, drop_prob={args.z_groot_drop_prob}")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # === Model ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V16Policy(
        config, action_horizon=16, action_dim=7, state_dim=8,
        z_vl_dim=z_vl_dim, z_state_dim=z_state_dim, K_bank=4,
    ).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[v16] params: {n_total:,} total")

    optimizer = build_optimizer(model, config, args)
    print(f"[v16] geo LR: {args.lr * config.structural_lr_ratio:.2e}")

    step = 0
    log_records = []
    if args.resume:
        print(f"[v16] resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ckpt["policy"]; own = model.state_dict()
        loaded = 0
        for k, v in sd.items():
            kk = k.replace("_orig_mod.", "")
            if kk in own and own[kk].shape == v.shape:
                own[kk].copy_(v); loaded += 1
        step = int(ckpt.get("step", 0))
        print(f"  loaded {loaded}/{len(own)} tensors, resuming step={step}")

    t_start = time.time()
    data_iter = iter(train_loader)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        _imgs, _wrists, states, chunks, _ti, z_states, z_banks, _delta_bank = batch
        states = states.to(device, non_blocking=True).float()
        chunks = chunks.to(device, non_blocking=True).float()
        z_states = z_states.to(device, non_blocking=True).float()
        z_banks = z_banks.to(device, non_blocking=True).float()

        cond = model.encode(z_banks, z_states, states)

        flow_loss, dec_out = flow_matching_loss(model, cond, chunks)
        gripper_loss = gripper_bce_loss(model, cond, chunks)
        aux = collect_aux_losses(dec_out, config)

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
                "metric_cv": float(dec_out["metric_cv"].detach()),
                "avg_kappa": float(dec_out["avg_kappa"].detach()),
                "tau_avg": float(dec_out["tau_avg"].detach()),
                "tau_log_std": float(dec_out["tau_log_std"].detach()),
                "actual_steps": int(dec_out["actual_steps"]),
            }
            log_records.append(rec)
            print(f"step {step:>5}  loss={rec['loss']:.4f}  flow={rec['flow_mse']:.4f}  "
                  f"grip={rec['gripper_bce']:.4f}  cv={rec['metric_cv']:.3f}  "
                  f"|κ|={rec['avg_kappa']:.4f}  τ={rec['tau_avg']:.3f}±{rec['tau_log_std']:.3f}  "
                  f"steps={rec['actual_steps']}  wall={rec['wall_s']:.0f}s")
        step += 1
        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_path = out_dir / f"step_{step:06d}.pt"
            torch.save({
                "policy": model.state_dict(), "step": step,
                "config": config.__dict__, "args": vars(args),
            }, ckpt_path)
            print(f"  [ckpt] {ckpt_path}")
            (out_dir / "log.json").write_text(json.dumps(log_records, indent=2))

    print(f"\n[v16] training complete: {step} steps, {time.time() - t_start:.1f}s")
    (out_dir / "log.json").write_text(json.dumps(log_records, indent=2))


if __name__ == "__main__":
    main()
