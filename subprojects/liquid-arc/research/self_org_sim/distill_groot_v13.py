"""v13 — v10's fan-in + canonical LiquidARC substrate on K=8 virtual positions.

Difference from v12: substrate operates on K=8 derived positions (each = cond_in + pos_embed[k]),
not on 256 raw DINOv2 patches. Inputs are CLS tokens (DINOv2's own attention-pooled global)
not patches. No lossy spatial pool after substrate. d=768 (matches v10's capacity).

Per the v12 failure analysis: the substrate-on-patches → mean-pool pipeline destroyed gripper-
discriminative features. v13 substrate operates on a summary representation that's already
been task-discriminatively encoded by v10's fan-in.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot import TeacherLabelDataset
from distill_groot_flow import FlowMatchingHead
from liquid_arc_substrate_v13 import V13Encoder, make_v13_config

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
from liquid_arc.sustained_criticality import (  # type: ignore
    compute_criticality_loss,
    compute_curvature_diversity_loss,
    compute_tau_quality_loss,
)

print = functools.partial(print, flush=True)
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
except Exception:
    pass


class V13Policy(nn.Module):
    """v13 encoder + flow head + gripper head."""

    def __init__(self, config, action_horizon=16, action_dim=7, state_dim=8,
                 head_d=256, head_layers=4, head_heads=4,
                 use_goal_img=True, z_vl_dim=1024, K_virtual=8):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.use_goal_img = use_goal_img
        self.z_vl_dim = z_vl_dim

        self.encoder = V13Encoder(
            config=config, K_virtual=K_virtual, state_dim=state_dim,
            z_vl_dim=z_vl_dim, use_goal_img=use_goal_img,
        )
        self.flow_head = FlowMatchingHead(
            d_cond=self.encoder.d_out,
            action_horizon=action_horizon,
            action_dim=action_dim,
            d_model=head_d, n_layers=head_layers, n_heads=head_heads,
        )
        self.gripper_head = nn.Sequential(
            nn.Linear(self.encoder.d_out, 128),
            nn.SiLU(),
            nn.Linear(128, action_horizon),
        )

    def encode(self, img, wrist_img, state, goal_img=None, z_vl=None,
               n_ode_steps_override=None, ablate_substrate=False):
        return self.encoder(img, wrist_img, state, goal_img=goal_img,
                            z_vl=z_vl, n_ode_steps_override=n_ode_steps_override,
                            ablate_substrate=ablate_substrate)

    def velocity(self, noisy_chunk, t, cond):
        return self.flow_head(noisy_chunk, t, cond)

    def gripper_logits(self, cond):
        return self.gripper_head(cond)

    def geo_parameters(self):
        return self.encoder.geo_parameters()

    def other_parameters(self):
        geo_ids = {id(p) for p in self.geo_parameters()}
        return [p for p in self.parameters() if id(p) not in geo_ids]


def flow_matching_loss(model, cond, target_chunk):
    B, K, A = target_chunk.shape
    t = torch.rand(B, device=target_chunk.device)
    noise = torch.randn_like(target_chunk)
    x_t = t.view(-1, 1, 1) * target_chunk + (1 - t.view(-1, 1, 1)) * noise
    v_target = target_chunk - noise
    v_pred = model.velocity(x_t, t, cond)
    return F.mse_loss(v_pred, v_target)


def gripper_bce_loss(model, cond, target_chunk):
    gripper_target = (target_chunk[..., -1] > 0).float()
    gripper_logits = model.gripper_logits(cond)
    return F.binary_cross_entropy_with_logits(gripper_logits, gripper_target)


def collect_aux_losses(out, config):
    losses = {}
    g = out["g"]
    h_final = out["h_final"]
    tau = out["tau"]
    t_diff_param = out["t_diff_param"]

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
    if config.halting_enabled and "ponder_cost" in out:
        losses["ponder"] = out["ponder_cost"].mean() * config.halting_ponder_lambda
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
    p.add_argument("--output_dir", default="/tmp/distill_v13", type=str)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=15000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--target_img_size", type=int, default=224)
    p.add_argument("--use_goal_img", action="store_true", default=False)
    p.add_argument("--use_z_vl", action="store_true",
                   help="Inject GR00T z_vl via z_vl_bank mean (1024-d compatible)")
    p.add_argument("--z_vl_drop_prob", type=float, default=0.2)
    p.add_argument("--gripper_bce_weight", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--d_model", type=int, default=768,
                   help="Encoder/cond dim. v10 uses 768 — match for capacity.")
    p.add_argument("--K_virtual", type=int, default=8,
                   help="Number of virtual positions for substrate to operate on.")
    p.add_argument("--aux_loss_scale", type=float, default=0.1,
                   help="Multiplier on SoC aux losses. 0 = disable.")
    p.add_argument("--resume", default="", type=str)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[v13] output: {out_dir}")

    config = make_v13_config(
        d_model=args.d_model, K_virtual=args.K_virtual,
        n_ode_steps=16, ode_steps_min=12, ode_steps_max=20,
        tau_freeze_steps=5000, integration_time=2.0, cold_start=True,
        aux_loss_scale=args.aux_loss_scale,
    )
    print(f"[v13] config: d={config.d_model}, K_virtual={args.K_virtual}, "
          f"halting={config.halting_enabled}, struct_τ={config.structural_tau_enabled}")

    # === Data ===
    decoded_dirs = [Path(d.strip()) for d in args.decoded_dirs.split(",") if d.strip()]
    labels_dirs = [Path(d.strip()) for d in args.teacher_labels_dirs.split(",") if d.strip()]
    datasets = []
    z_vl_dim = 0
    for dd, ld in zip(decoded_dirs, labels_dirs):
        ds = TeacherLabelDataset(
            dd, ld, action_horizon=16,
            target_img_size=args.target_img_size,
            return_goal_img=args.use_goal_img,
            return_z_groot=("z_vl" if args.use_z_vl else ""),
            use_query_bank=args.use_z_vl,
            z_groot_drop_prob=args.z_vl_drop_prob if args.use_z_vl else 0.0,
        )
        if args.use_z_vl and len(datasets) == 0:
            z_vl_dim = int(sum(ds.z_dims)) if hasattr(ds, "z_dims") else 1024
        datasets.append(ds)
    if args.use_z_vl:
        print(f"[v13] z_vl from bank-mean: dim={z_vl_dim}, drop={args.z_vl_drop_prob}")
    train_ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"[v13] samples: {len(train_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # === Model ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V13Policy(
        config, action_horizon=16, action_dim=7, state_dim=8,
        head_d=256, head_layers=4, head_heads=4,
        use_goal_img=args.use_goal_img,
        z_vl_dim=z_vl_dim,
        K_virtual=args.K_virtual,
    ).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[v13] params: {n_total:,} total, {n_train:,} trainable")

    optimizer = build_optimizer(model, config, args)
    print(f"[v13] geo LR: {args.lr * config.structural_lr_ratio:.2e}")

    step = 0
    log_records = []
    if args.resume:
        print(f"[v13] resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ckpt["policy"]
        own = model.state_dict()
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

        z_groots = None
        goal_imgs = None
        if args.use_z_vl:
            imgs, wrists, states, chunks, _ti, _z_groot_old, z_bank, _delta_bank = batch
            z_groots = z_bank.float().mean(dim=1)
        elif args.use_goal_img:
            imgs, wrists, states, chunks, goal_imgs = batch
        else:
            imgs, wrists, states, chunks = batch

        imgs = imgs.to(device, non_blocking=True).float().permute(0, 3, 1, 2) / 255.0
        wrists = wrists.to(device, non_blocking=True).float().permute(0, 3, 1, 2) / 255.0
        states = states.to(device, non_blocking=True).float()
        chunks = chunks.to(device, non_blocking=True).float()
        if args.use_goal_img and goal_imgs is not None:
            goal_imgs = goal_imgs.to(device, non_blocking=True).float().permute(0, 3, 1, 2) / 255.0
        if args.use_z_vl and z_groots is not None:
            z_groots = z_groots.to(device, non_blocking=True)

        out = model.encode(imgs, wrists, states, goal_img=goal_imgs, z_vl=z_groots)
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
                "step": step, "loss": float(loss.detach()),
                "flow_mse": float(flow_loss.detach()),
                "gripper_bce": float(gripper_loss.detach()),
                "metric_cv": float(out["metric_cv"].detach()),
                "avg_kappa": float(out["avg_kappa"].detach()),
                "tau_avg": float(out["tau_avg"].detach()),
                "tau_log_std": float(out["tau_log_std"].detach()),
                "actual_steps": int(out["actual_steps"]),
                "wall_s": round(time.time() - t_start, 1),
            }
            for k in ("criticality", "curvature_diversity", "tau_quality", "ponder"):
                if k in aux and isinstance(aux[k], torch.Tensor):
                    rec[k] = float(aux[k].detach())
            log_records.append(rec)
            print(f"step {step:5d}  loss={rec['loss']:.4f}  flow={rec['flow_mse']:.4f}  "
                  f"grip={rec['gripper_bce']:.4f}  cv={rec['metric_cv']:.3f}  "
                  f"|κ|={rec['avg_kappa']:.4f}  τ={rec['tau_avg']:.3f}±{rec['tau_log_std']:.3f}  "
                  f"steps={rec['actual_steps']}  wall={rec['wall_s']}s")

        if (step + 1) % args.save_every == 0 or (step + 1) == args.max_steps:
            ckpt_path = out_dir / f"step_{step+1:06d}.pt"
            torch.save({
                "step": step + 1, "policy": model.state_dict(),
                "config": config.__dict__, "args": vars(args),
            }, ckpt_path)
            (out_dir / "log.json").write_text(json.dumps(log_records, indent=2))
            print(f"  [ckpt] {ckpt_path}")

        step += 1

    print(f"[v13] training complete: {step} steps, {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
