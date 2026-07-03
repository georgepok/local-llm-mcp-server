"""v12 — train a real LiquidARC student on LIBERO via flow-matching distillation.

This is the v11 design's promise made architecturally real: vision tokens
(DINOv2 patches) flow through canonical ContinuousDynamics with full SoC
machinery (MetricNet + TauNet + heat-kernel SDPA + halting + structural_τ
+ criticality + diversity + tau_quality + cold-start ReZero + PonderNet deep
supervision + KL prior + step-conditional FiLM Tier 1 + randomized [12,20]
ODE depth).

Pipeline:
  TeacherLabelDataset (4 suites mixed) →
    LiquidARCVisionEncoder (substrate-B) →
      FlowMatchingHead (continuous actions, MSE on velocity) +
      gripper_head (binary, BCE) →
  aux losses: flow_mse + gripper_bce + criticality + curvature_diversity
              + tau_quality + ponder_cost + KL prior
  optimizer: AdamW with two param groups (geo @ 0.1×, content @ 1×)

Run on Spark in main venv (CUDA):
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
  python distill_groot_v12.py \
    --decoded_dirs /home/pokazge/datasets/libero-{10,goal,object,spatial}-expert-v1 \
    --teacher_labels_dirs /home/pokazge/datasets/libero-{10,goal,object,spatial}-expert-v1 \
    --output_dir /tmp/distill_v12 \
    --batch_size 64 --max_steps 15000
"""
from __future__ import annotations

import argparse
import functools
import json
import math
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
from liquid_arc_substrate_libero import (
    LiquidARCVisionEncoder, make_v12_config, IMAGENET_MEAN, IMAGENET_STD,
)

# Canonical LiquidARC SoC losses
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


# -------------------- model -------------------------------------------------- #


class V12Policy(nn.Module):
    """LiquidARC encoder + flow matching head + gripper head."""

    def __init__(self, config, action_horizon=16, action_dim=7, state_dim=8,
                 head_d=256, head_layers=4, head_heads=4,
                 use_goal_img=True, z_vl_dim=0):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.use_goal_img = use_goal_img
        self.z_vl_dim = z_vl_dim

        self.encoder = LiquidARCVisionEncoder(
            config=config, state_dim=state_dim, use_goal_img=use_goal_img,
            z_vl_dim=z_vl_dim,
        )
        self.flow_head = FlowMatchingHead(
            d_cond=self.encoder.d_out,
            action_horizon=action_horizon,
            action_dim=action_dim,
            d_model=head_d,
            n_layers=head_layers,
            n_heads=head_heads,
        )
        # Separate sigmoid gripper head (14.8 collapse fix) — outputs K binary logits
        self.gripper_head = nn.Sequential(
            nn.Linear(self.encoder.d_out, 128),
            nn.SiLU(),
            nn.Linear(128, action_horizon),
        )

    def encode(self, img, wrist_img, state, goal_img=None, z_vl=None,
               n_ode_steps_override=None):
        return self.encoder(img, wrist_img, state, goal_img=goal_img,
                            z_vl=z_vl,
                            n_ode_steps_override=n_ode_steps_override)

    def velocity(self, noisy_chunk, t, cond):
        return self.flow_head(noisy_chunk, t, cond)

    def gripper_logits(self, cond):
        return self.gripper_head(cond)

    def geo_parameters(self):
        return self.encoder.geo_parameters()

    def other_parameters(self):
        geo_ids = {id(p) for p in self.geo_parameters()}
        return [p for p in self.parameters() if id(p) not in geo_ids]


# -------------------- training step ----------------------------------------- #


def flow_matching_loss(model, cond, target_chunk):
    """Rectified flow MSE on velocity."""
    B, K, A = target_chunk.shape
    t = torch.rand(B, device=target_chunk.device)
    noise = torch.randn_like(target_chunk)
    x_t = t.view(-1, 1, 1) * target_chunk + (1 - t.view(-1, 1, 1)) * noise
    v_target = target_chunk - noise  # rectified flow target velocity
    v_pred = model.velocity(x_t, t, cond)
    return F.mse_loss(v_pred, v_target)


def gripper_bce_loss(model, cond, target_chunk):
    """Binary BCE on gripper channel (last dim of action). +1 → 1, -1 → 0."""
    # gripper in [-1, +1] → binary {0, 1}
    gripper_target = (target_chunk[..., -1] > 0).float()  # [B, K]
    gripper_logits = model.gripper_logits(cond)
    return F.binary_cross_entropy_with_logits(gripper_logits, gripper_target)


def collect_aux_losses(out, config):
    """Compute canonical SoC stabilizer losses from the encoder output dict.

    Canonical signatures:
      compute_criticality_loss(h, g, tau, t_diff_param, target_ratio, n_pairs, d_sq_target)
        → (loss, diagnostics)
      compute_curvature_diversity_loss(g, cv_floor, cv_ceiling, n_bins) → loss
      compute_tau_quality_loss(tau, mean_target, log_spread_target) → loss
    """
    losses = {}
    g = out["g"]                  # [B, N, d]
    h_final = out["h_final"]      # [B, N, d]
    tau = out["tau"]              # [B, N, 1] or None
    t_diff_param = out["t_diff_param"]

    # 1. Criticality: D²/(4τ) → 18
    if config.criticality_loss_enabled and tau is not None and t_diff_param is not None:
        loss, _ = compute_criticality_loss(
            h_final, g, tau, t_diff_param,
            target_ratio=config.criticality_target_ratio,
            d_sq_target=config.criticality_D_sq_target,
        )
        losses["criticality"] = loss

    # 2. Curvature diversity: CV band [floor, ceiling]
    if config.curvature_diversity_loss_enabled:
        losses["curvature_diversity"] = compute_curvature_diversity_loss(
            g,
            cv_floor=config.curvature_cv_floor,
            cv_ceiling=config.curvature_cv_ceiling,
        )

    # 3. Tau quality: mean(τ)→1, std(log τ)→target
    if config.tau_quality_loss_enabled and tau is not None:
        # Canonical's tau_mean_target=0 means "auto" — translate to actual target
        mean_target = config.tau_mean_target
        if mean_target <= 0:
            mean_target = config.integration_time / config.n_ode_steps * 16
        losses["tau_quality"] = compute_tau_quality_loss(
            tau,
            mean_target=mean_target,
            log_spread_target=config.tau_log_spread_target,
        )

    # 4. Ponder cost (encourages early halting)
    if config.halting_enabled and "ponder_cost" in out:
        losses["ponder"] = out["ponder_cost"].mean() * config.halting_ponder_lambda

    return losses


# -------------------- training loop ----------------------------------------- #


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
    p.add_argument("--decoded_dirs", required=True, type=str,
                   help="comma-separated dirs containing imgs/wrists/states .dat")
    p.add_argument("--teacher_labels_dirs", required=True, type=str,
                   help="comma-separated dirs containing teacher_chunks.dat + labels_index.npz")
    p.add_argument("--output_dir", default="/tmp/distill_v12", type=str)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=15000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--target_img_size", type=int, default=224)
    p.add_argument("--use_goal_img", action="store_true", default=False,
                   help="DISABLED by default for v12 — substrate-test priority. "
                        "Per catalog 14.9, goal_img is bimodal across suites.")
    p.add_argument("--gripper_bce_weight", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--criticality_weight_multiplier", type=float, default=1.0,
                   help="multiplier on top of config.criticality_loss_lambda")
    p.add_argument("--resume", default="", type=str,
                   help="Path to ckpt to resume from. Loads weights + step counter.")
    p.add_argument("--use_z_vl", action="store_true",
                   help="v12.2: include GR00T's z_vl in cond. Catalog 14.5 finding: "
                        "all working LIBERO students use z_vl (it's the task-phase signal).")
    p.add_argument("--z_vl_drop_prob", type=float, default=0.2,
                   help="Stage-1 pressure-landscape: random z_vl dropout (zeroed) at training.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[v12] output dir: {out_dir}")

    # === Config ===
    config = make_v12_config(
        d_model=256, d_metric=64, d_ffn=512,
        n_ode_steps=16, ode_steps_min=12, ode_steps_max=20,
        tau_freeze_steps=5000, integration_time=2.0, cold_start=True,
    )
    print(f"[v12] config: d={config.d_model}, halting={config.halting_enabled}, "
          f"rezero={config.rezero_enabled}, struct_τ={config.structural_tau_enabled}, "
          f"step_embed={config.step_embed_enabled}")

    # === Data ===
    decoded_dirs = [Path(d.strip()) for d in args.decoded_dirs.split(",") if d.strip()]
    labels_dirs = [Path(d.strip()) for d in args.teacher_labels_dirs.split(",") if d.strip()]
    assert len(decoded_dirs) == len(labels_dirs)

    datasets = []
    z_vl_dim = 0
    # v12.3: use z_vl_bank (depth-subsampled traj_model_output) — compatible
    # between training data and current GR00T's traj_model_output at inference.
    use_bank_as_z_vl = args.use_z_vl  # v12.3: when --use_z_vl, load bank
    for dd, ld in zip(decoded_dirs, labels_dirs):
        ds = TeacherLabelDataset(
            dd, ld, action_horizon=16,
            target_img_size=args.target_img_size,
            return_goal_img=args.use_goal_img,
            return_z_groot=("z_vl" if use_bank_as_z_vl else ""),
            use_query_bank=use_bank_as_z_vl,  # v12.3: load z_vl_bank.dat
            z_groot_drop_prob=args.z_vl_drop_prob if use_bank_as_z_vl else 0.0,
        )
        if use_bank_as_z_vl and len(datasets) == 0:
            # Bank entries are [K, z_vl_dim] — we mean-pool over K at training-time.
            # z_vl_dim from bank.
            z_vl_dim = int(sum(ds.z_dims)) if hasattr(ds, "z_dims") else 0
        datasets.append(ds)
    if args.use_z_vl:
        print(f"[v12.3] using z_vl_bank (mean over K=4 depth samples), z_vl_dim={z_vl_dim}, drop_prob={args.z_vl_drop_prob}")
    train_ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"[v12] total samples: {len(train_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # === Model ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V12Policy(config, action_horizon=16, action_dim=7, state_dim=8,
                      head_d=256, head_layers=4, head_heads=4,
                      use_goal_img=args.use_goal_img,
                      z_vl_dim=z_vl_dim).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[v12] params: {n_total:,} total, {n_train:,} trainable")

    optimizer = build_optimizer(model, config, args)
    print(f"[v12] geo LR: {args.lr * config.structural_lr_ratio:.2e} "
          f"({config.structural_lr_ratio}× base)")

    # === Resume from checkpoint if specified ===
    step = 0
    log_records = []
    if args.resume:
        print(f"[v12] resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ckpt["policy"]
        own = model.state_dict()
        loaded = 0
        for k, v in sd.items():
            kk = k.replace("_orig_mod.", "")
            if kk in own and own[kk].shape == v.shape:
                own[kk].copy_(v); loaded += 1
        step = int(ckpt.get("step", 0))
        print(f"  loaded {loaded}/{len(own)} tensors, resuming at step={step}")

    # === Training loop ===
    t_start = time.time()
    data_iter = iter(train_loader)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Batch unpack: TeacherLabelDataset returns variable tuples depending on flags.
        # With use_z_vl (v12.3): use_query_bank=True returns 8-tuple
        #   (img, wrist, state, chunk, ti_field, z_groot, z_bank, delta_bank)
        #   We use z_bank (the depth bank) — mean-pool over K to get z_vl signal.
        z_groots = None
        if args.use_z_vl:
            imgs, wrists, states, chunks, _ti, _z_groot_old, z_bank, _delta_bank = batch
            # z_bank shape: [B, K, dim]. Mean-pool over K → [B, dim]
            z_groots = z_bank.float().mean(dim=1)
            goal_imgs = None
        elif args.use_goal_img:
            imgs, wrists, states, chunks, goal_imgs = batch
        else:
            imgs, wrists, states, chunks = batch
            goal_imgs = None

        imgs = imgs.to(device, non_blocking=True).float().permute(0, 3, 1, 2) / 255.0
        wrists = wrists.to(device, non_blocking=True).float().permute(0, 3, 1, 2) / 255.0
        states = states.to(device, non_blocking=True).float()
        chunks = chunks.to(device, non_blocking=True).float()
        if args.use_goal_img and goal_imgs is not None:
            goal_imgs = goal_imgs.to(device, non_blocking=True).float().permute(0, 3, 1, 2) / 255.0
        if args.use_z_vl and z_groots is not None:
            z_groots = z_groots.to(device, non_blocking=True)

        # Forward encoder
        out = model.encode(imgs, wrists, states, goal_img=goal_imgs, z_vl=z_groots)
        cond = out["cond"]

        # Losses
        flow_loss = flow_matching_loss(model, cond, chunks)
        gripper_loss = gripper_bce_loss(model, cond, chunks)
        aux = collect_aux_losses(out, config)

        loss = flow_loss + args.gripper_bce_weight * gripper_loss
        for name, l in aux.items():
            if name.startswith("_"):
                continue
            if isinstance(l, torch.Tensor):
                if name == "criticality":
                    loss = loss + config.criticality_loss_lambda * args.criticality_weight_multiplier * l
                elif name == "curvature_diversity":
                    loss = loss + config.curvature_diversity_lambda * l
                elif name == "tau_quality":
                    loss = loss + config.tau_quality_lambda * l
                elif name == "ponder":
                    loss = loss + l  # already weighted

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Log
        if step % args.log_every == 0:
            rec = {
                "step": step,
                "loss": float(loss.detach()),
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
            print(f"step {step:5d}  loss={rec['loss']:.4f}  "
                  f"flow={rec['flow_mse']:.4f}  grip={rec['gripper_bce']:.4f}  "
                  f"crit={rec.get('criticality', 0):.4f}  "
                  f"cv={rec['metric_cv']:.3f}  |κ|={rec['avg_kappa']:.4f}  "
                  f"τ={rec['tau_avg']:.3f}±{rec['tau_log_std']:.3f}  "
                  f"steps={rec['actual_steps']}  wall={rec['wall_s']}s")

        # Checkpoint
        if (step + 1) % args.save_every == 0 or (step + 1) == args.max_steps:
            ckpt_path = out_dir / f"step_{step+1:06d}.pt"
            torch.save({
                "step": step + 1,
                "policy": model.state_dict(),
                "config": config.__dict__,
                "args": vars(args),
            }, ckpt_path)
            (out_dir / "log.json").write_text(json.dumps(log_records, indent=2))
            print(f"  [ckpt] saved {ckpt_path}")

        step += 1

    print(f"[v12] training complete: {step} steps, {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
