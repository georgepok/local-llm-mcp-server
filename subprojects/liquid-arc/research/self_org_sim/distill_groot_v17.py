"""v17: v10 backbone + substrate gripper module + knowledge-driven training.

Design rationale (collected from this session's diagnostic work):
  1. v10's vision encoder + transformer flow head WORK for continuous control.
     Don't replace what works → encoder + xyz/rpy head unchanged from v10.
  2. Gripper decision is structurally different (binary state with critical
     timing transitions, not continuous motion). Substrate's distinctive
     capabilities (halting, geometric routing, SoC) map to this task structure.
     → gripper head replaced by ContinuousDynamics over K=4 belief positions.
  3. Pre-validation showed substrate gripper head beats MLP head on offline
     gripper accuracy (94.0% vs 91.7%, +2.35pp). Real gain especially on the
     close class (+3.6pp) — the diagnosed failure-mode-relevant axis.
  4. Focal BCE training pressure on the diagnosed failure mode:
     penalize "model says close while expert says open" 3x.

Architecture:
  Encoder (v10's LiquidEncoder): img + wrist + state8 → fused ODE with z_vl
    FiLM → cond [B, d=768]
  XYZ/RPY Head (v10's FlowMatchingHead): flow velocity for all 7 action dims
    (gripper dim is overridden at inference but trained via flow MSE for
    consistency)
  Gripper Substrate Head (NEW):
    cond → 4 belief-position projections + pos_embed
    ContinuousDynamics: 8-step Euler ODE with halting
    Per-position readout: target_state → MLP → 16 logits (one per chunk pos)
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
from distill_groot_flow import LiquidEncoder, FlowMatchingHead  # type: ignore

from liquid_arc.config import LiquidARCConfig  # type: ignore
from liquid_arc.dynamics import ContinuousDynamics  # type: ignore
from liquid_arc.solver import euler_solve_halting  # type: ignore
from liquid_arc.context_pool import ContextPool  # type: ignore
from liquid_arc.sustained_criticality import (  # type: ignore
    compute_criticality_loss,
    compute_curvature_diversity_loss,
)

torch.set_float32_matmul_precision("high")


def make_gripper_substrate_config(d=128):
    """Small substrate tuned for binary state maintenance."""
    return LiquidARCConfig(
        d_model=d, d_metric=32, d_ffn=128, max_seq_len=4,
        n_ode_steps=8, ode_steps_min=4, ode_steps_max=12,
        integration_time=1.0,
        tau_min=0.3, tau_max=1.0, t_diffusion_init=0.5,
        routing_mode="metric",
        tau_freeze_steps=2000,
        halting_enabled=True, halting_min_steps=2,
        halting_ponder_lambda=0.001 * 0.1,
        rezero_enabled=True, rezero_gate_init=-3.0,
        metric_bias_init_std=0.1,
        deep_supervision_enabled=False,
        ponder_kl_lambda=0.0,
        criticality_loss_enabled=True,
        criticality_loss_lambda=0.0001,
        criticality_target_ratio=18.0, criticality_D_sq_target=30.0,
        curvature_diversity_loss_enabled=True,
        curvature_diversity_lambda=0.0001,
        curvature_cv_floor=1.5, curvature_cv_ceiling=8.0,
        tau_quality_loss_enabled=False,
        tau_mean_target=0.0, tau_log_spread_target=0.4,
        step_embed_enabled=False,
        step_conditional_operator=False,
        structural_tau_enabled=False,
        norm_ref=20.0, norm_lambda=0.1,
        base_lr=3e-4, structural_lr_ratio=0.1,
        warmup_steps=200, weight_decay=0.01,
        use_torch_compile=False,
    )


class GripperSubstrateHead(nn.Module):
    """Substrate over K=4 belief positions, outputs 16 gripper logits per chunk.

    Position roles (semantic init via separate projections):
      0: approach_belief
      1: contact_belief
      2: hold_belief
      3: target_state (output position)
    """

    def __init__(self, d_in=768, d_substrate=128, K_belief=4, action_horizon=16):
        super().__init__()
        self.d_in = d_in
        self.d = d_substrate
        self.K_belief = K_belief
        self.action_horizon = action_horizon

        self.config = make_gripper_substrate_config(d=d_substrate)

        # Per-position projections from cond
        self.pos_projs = nn.ModuleList([
            nn.Linear(d_in, d_substrate) for _ in range(K_belief)
        ])
        for proj in self.pos_projs:
            nn.init.normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)
        self.pos_embed = nn.Parameter(torch.zeros(K_belief, d_substrate))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Canonical substrate
        self.context_pool = ContextPool(self.config)
        self.dynamics = ContinuousDynamics(self.config)

        # Readout: target_state → 16 logits (one per chunk position)
        self.readout = nn.Sequential(
            nn.Linear(d_substrate, d_substrate), nn.SiLU(),
            nn.Linear(d_substrate, action_horizon),
        )

    def forward(self, cond):
        # cond: [B, d_in]
        B = cond.shape[0]
        h0 = torch.stack([proj(cond) for proj in self.pos_projs], dim=1)  # [B, K, d]
        h0 = h0 + self.pos_embed.unsqueeze(0)
        context = self.context_pool(h0, None)
        self.dynamics.set_context(context, mask=None)
        if self.training:
            lo = int(self.config.ode_steps_min)
            hi = int(self.config.ode_steps_max)
            n_steps = int(torch.randint(lo, hi + 1, (1,)).item())
        else:
            n_steps = int(self.config.n_ode_steps)
        self.dynamics.set_n_steps(n_steps)
        T = float(self.config.integration_time)
        out = euler_solve_halting(
            self.dynamics, h0, (0.0, T), n_steps,
            min_steps=self.config.halting_min_steps,
        )
        if isinstance(out, tuple) and len(out) >= 3:
            h, ponder_cost, _ = out[0], out[1], out[2]
        else:
            h = out
            ponder_cost = torch.zeros(B, device=h.device)

        # Readout from target_state position (index 3)
        logits = self.readout(h[:, 3])  # [B, action_horizon]

        # Diagnostics
        g = self.dynamics.compute_metric_diag(h0)
        metric_cv = g.std() / (g.mean() + 1e-8)
        return {
            "logits": logits,
            "h_final": h, "h0": h0, "g": g,
            "ponder_cost": ponder_cost,
            "metric_cv": metric_cv,
        }


class V17Policy(nn.Module):
    def __init__(self, encoder_kwargs, head_kwargs, gripper_d=128, gripper_K=4,
                 action_horizon=16, action_dim=7, d=768):
        super().__init__()
        self.encoder = LiquidEncoder(**encoder_kwargs)
        self.flow_head = FlowMatchingHead(d_cond=d, action_horizon=action_horizon,
                                          action_dim=action_dim, **head_kwargs)
        self.gripper_substrate = GripperSubstrateHead(
            d_in=d, d_substrate=gripper_d, K_belief=gripper_K,
            action_horizon=action_horizon,
        )
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    def encode(self, img, wrist, state, z_groot=None, z_bank=None, delta_bank=None):
        cond, info = self.encoder(img, wrist, state,
                                  z_groot=z_groot, z_bank=z_bank, delta_bank=delta_bank)
        return cond, info

    def velocity(self, noisy_chunk, t, cond):
        return self.flow_head(noisy_chunk, t, cond)

    def gripper_logits(self, cond):
        return self.gripper_substrate(cond)


def flow_matching_loss(model, cond, target_chunk):
    """Flow MSE on first 6 action dims (xyz, rpy). Gripper handled separately."""
    B, K, A = target_chunk.shape
    t = torch.rand(B, device=target_chunk.device)
    noise = torch.randn_like(target_chunk)
    v_target = target_chunk - noise
    x_t = t.view(-1, 1, 1) * target_chunk + (1 - t.view(-1, 1, 1)) * noise
    v_pred = model.velocity(x_t, t, cond)
    # MSE on first 6 dims only
    loss_xyz = F.mse_loss(v_pred[..., :6], v_target[..., :6])
    return loss_xyz


def focal_gripper_loss(model, cond, target_chunk, false_close_weight: float = 3.0):
    """Focal BCE on gripper from substrate head.

    Weight `false_close_weight` × on samples where expert=open AND model=close.
    Directly targets the diagnosed v10 failure mode.
    """
    gripper_target_signs = target_chunk[..., -1]  # [B, K] ∈ {-1, +1} or near
    gripper_target_open = (gripper_target_signs < 0).float()  # 1 if expert says OPEN
    sub_out = model.gripper_logits(cond)
    logits = sub_out["logits"]  # [B, K]
    # Predicted: P(open) = sigmoid(logit)
    # If logit < 0 → model says close
    base = F.binary_cross_entropy_with_logits(logits, gripper_target_open, reduction="none")
    with torch.no_grad():
        model_says_close = (logits < 0).float()
        # Failure mode: expert open (=1) AND model says close → upweight
        penalty_mask = gripper_target_open * model_says_close
        weights = 1.0 + (false_close_weight - 1.0) * penalty_mask
    loss = (base * weights).mean()
    return loss, sub_out


def build_optimizer(model, args, gripper_substrate_cfg):
    """Two param groups: encoder+head get base_lr; gripper substrate's geo
    params get base_lr * gripper_geo_ratio (slow geometric warm-up)."""
    geo_params = []
    geo_ids = set()
    sub = model.gripper_substrate
    dyn = sub.dynamics
    for mod in [dyn.metric_net_linear1, dyn.metric_net_linear2_diag]:
        for p in mod.parameters():
            geo_params.append(p); geo_ids.add(id(p))
    if hasattr(dyn, "tau_net_linear1"):
        for mod in [dyn.tau_net_linear1, dyn.tau_net_linear2]:
            for p in mod.parameters():
                geo_params.append(p); geo_ids.add(id(p))
    if hasattr(dyn, "rezero_gate_logit"):
        geo_params.append(dyn.rezero_gate_logit); geo_ids.add(id(dyn.rezero_gate_logit))
    for p in sub.context_pool.parameters():
        geo_params.append(p); geo_ids.add(id(p))
    geo_params.append(sub.pos_embed); geo_ids.add(id(sub.pos_embed))

    other_params = [p for p in model.parameters() if id(p) not in geo_ids]

    print(f"  geo params (substrate dyn): {sum(p.numel() for p in geo_params):,}")
    print(f"  other params: {sum(p.numel() for p in other_params):,}")
    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr},
            {"params": geo_params, "lr": args.lr * gripper_substrate_cfg.structural_lr_ratio},
        ],
        weight_decay=args.weight_decay,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoded_dirs", required=True)
    p.add_argument("--teacher_labels_dirs", required=True)
    p.add_argument("--output_dir", default="/tmp/distill_v17")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=12000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--target_img_size", type=int, default=224)
    p.add_argument("--gripper_bce_weight", type=float, default=1.0)
    p.add_argument("--false_close_weight", type=float, default=3.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--d", type=int, default=768)
    p.add_argument("--gripper_d", type=int, default=128)
    p.add_argument("--gripper_K", type=int, default=4)
    p.add_argument("--resume", default="", type=str,
                   help="Resume from v10-DEMO checkpoint (loads encoder + flow_head, "
                        "leaves new gripper substrate at init)")
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[v17] output: {out_dir}")

    # === Data ===
    decoded_dirs = [Path(d.strip()) for d in args.decoded_dirs.split(",") if d.strip()]
    labels_dirs = [Path(d.strip()) for d in args.teacher_labels_dirs.split(",") if d.strip()]
    datasets = []
    for dd, ld in zip(decoded_dirs, labels_dirs):
        ds = TeacherLabelDataset(
            dd, ld, action_horizon=16,
            target_img_size=args.target_img_size,
            return_goal_img=False,
            return_z_groot="",
            use_query_bank=False,
        )
        datasets.append(ds)
    train_ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"[v17] samples: {len(train_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # === Model ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_kwargs = dict(
        state_dim=8, d=args.d, d_vis=args.d, img_size=args.target_img_size,
        k_max=16, halt_mode="learned", min_steps=4, dt=0.5,
        n_tasks=0, d_task=32, z_groot_dim=0,
    )
    head_kwargs = dict(d_model=256, n_layers=4, n_heads=4, d_t=64)
    model = V17Policy(
        encoder_kwargs=encoder_kwargs, head_kwargs=head_kwargs,
        gripper_d=args.gripper_d, gripper_K=args.gripper_K,
        action_horizon=16, action_dim=7, d=args.d,
    ).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[v17] params: {n_total:,} total")

    # Optional: warm-start from v10-DEMO checkpoint (encoder + flow_head)
    if args.resume:
        print(f"[v17] warm-starting encoder + flow_head from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ckpt.get("model", ckpt.get("policy", ckpt))
        own = model.state_dict()
        loaded = 0
        for k, v in sd.items():
            kk = k.replace("_orig_mod.", "")
            if kk in own and own[kk].shape == v.shape:
                own[kk].copy_(v); loaded += 1
        print(f"[v17] loaded {loaded}/{len(own)} tensors (gripper substrate left at init)")

    optimizer = build_optimizer(model, args, model.gripper_substrate.config)

    step = 0
    log_records = []
    t_start = time.time()
    data_iter = iter(train_loader)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # TeacherLabelDataset returns (imgs, wrists, states, chunks) without z_groot/bank
        imgs, wrists, states, chunks = batch[0], batch[1], batch[2], batch[3]
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        wrists = wrists.to(device, non_blocking=True).float() / 255.0
        # HWC → CHW
        imgs = imgs.permute(0, 3, 1, 2).contiguous()
        wrists = wrists.permute(0, 3, 1, 2).contiguous()
        states = states.to(device, non_blocking=True).float()
        chunks = chunks.to(device, non_blocking=True).float()

        cond, enc_info = model.encode(imgs, wrists, states)

        flow_loss = flow_matching_loss(model, cond, chunks)
        gripper_loss, sub_out = focal_gripper_loss(model, cond, chunks,
                                                   false_close_weight=args.false_close_weight)

        # Substrate aux losses (small lambdas)
        sub_cfg = model.gripper_substrate.config
        aux_crit = 0.0
        aux_curv = 0.0
        if sub_cfg.criticality_loss_enabled and "g" in sub_out:
            # Criticality needs tau and t_diff_param — gripper substrate doesn't expose
            # these directly. Use a softer proxy: curvature_diversity is enough here.
            pass
        if sub_cfg.curvature_diversity_loss_enabled and "g" in sub_out:
            aux_curv = compute_curvature_diversity_loss(
                sub_out["g"], cv_floor=sub_cfg.curvature_cv_floor,
                cv_ceiling=sub_cfg.curvature_cv_ceiling,
            )

        loss = flow_loss + args.gripper_bce_weight * gripper_loss
        if isinstance(aux_curv, torch.Tensor):
            loss = loss + sub_cfg.curvature_diversity_lambda * aux_curv
        ponder = sub_out.get("ponder_cost", torch.zeros(1, device=device))
        loss = loss + sub_cfg.halting_ponder_lambda * ponder.mean()

        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                # Per-class gripper accuracy on this batch
                grip_target_signs = chunks[..., -1]
                grip_target_open = (grip_target_signs < 0).float()
                logits = sub_out["logits"]
                pred_open = (logits > 0).float()
                acc_open = (pred_open[grip_target_open == 1] == 1).float().mean() if (grip_target_open == 1).any() else torch.tensor(0.0)
                acc_close = (pred_open[grip_target_open == 0] == 0).float().mean() if (grip_target_open == 0).any() else torch.tensor(0.0)
                cv = float(sub_out["metric_cv"].detach())
            rec = {
                "step": step, "wall_s": time.time() - t_start,
                "loss": float(loss.detach()),
                "flow_mse": float(flow_loss.detach()),
                "gripper_focal_bce": float(gripper_loss.detach()),
                "acc_open": float(acc_open),
                "acc_close": float(acc_close),
                "sub_cv": cv,
            }
            log_records.append(rec)
            print(f"step {step:>5}  loss={rec['loss']:.4f}  flow={rec['flow_mse']:.4f}  "
                  f"grip={rec['gripper_focal_bce']:.4f}  acc_open={rec['acc_open']:.3f}  "
                  f"acc_close={rec['acc_close']:.3f}  cv={rec['sub_cv']:.3f}  wall={rec['wall_s']:.0f}s")
        step += 1
        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_path = out_dir / f"step_{step:06d}.pt"
            torch.save({
                "model": model.state_dict(), "step": step,
                "args": vars(args),
            }, ckpt_path)
            print(f"  [ckpt] {ckpt_path}")
            (out_dir / "log.json").write_text(json.dumps(log_records, indent=2))

    print(f"\n[v17] training complete: {step} steps, {time.time() - t_start:.1f}s")
    (out_dir / "log.json").write_text(json.dumps(log_records, indent=2))


if __name__ == "__main__":
    main()
