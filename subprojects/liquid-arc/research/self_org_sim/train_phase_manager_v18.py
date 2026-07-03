"""v18 Substrate Phase Manager: uses GR00T chunk + v10 cond + state as input.

DINOv2 features REMOVED from input — they alone capped both MLP and substrate
at 17% closed-loop. The strongest signal we identified this session is GR00T's
predicted chunk (which is what cclamp40 used to get 37%). Substrate's job is
to FUSE GR00T's chunk + v10's cond + state into a robust phase belief, then
output a gripper-override decision per chunk position.

Architecture:
  Substrate K=6 belief positions, d=128:
    pos 0: GR00T near-term intent  (chunk[0:4] flat → linear)
    pos 1: GR00T far-term intent   (chunk[12:16] flat → linear)
    pos 2: scene-grounded belief   (v10 cond → linear)
    pos 3: proprio belief          (state8 + staleness → linear)
    pos 4: integration scratch     (init from mean of inputs)
    pos 5: target output           (init from cond)

  ContinuousDynamics: 8 Euler steps, halting per-position, metric+tau learned.

  Output: pos 5 → MLP → 16 logits → P(open) per chunk position.

Training data construction:
  For each expert sample s with (img, wrist, state, expert_chunk):
    cond_s = v10_frozen.encode(img, wrist, state)  # 768-d
    grip_target = (expert_chunk[..., -1] < 0).float()  # [16] — 1 if expert open
  Substrate input: (expert_chunk, cond_s, state, 0)
  Label: grip_target [16]
  Loss: focal BCE with 3x penalty when expert=open AND prediction says close.
"""
from __future__ import annotations
import argparse
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
from distill_groot_flow import LiquidEncoder  # type: ignore

from liquid_arc.config import LiquidARCConfig  # type: ignore
from liquid_arc.dynamics import ContinuousDynamics  # type: ignore
from liquid_arc.solver import euler_solve_halting  # type: ignore
from liquid_arc.context_pool import ContextPool  # type: ignore

torch.set_float32_matmul_precision("high")


def make_pm_config(d=128):
    return LiquidARCConfig(
        d_model=d, d_metric=32, d_ffn=128, max_seq_len=6,
        n_ode_steps=8, ode_steps_min=4, ode_steps_max=12,
        integration_time=1.0,
        tau_min=0.3, tau_max=1.0, t_diffusion_init=0.5,
        routing_mode="metric",
        tau_freeze_steps=2000,
        halting_enabled=True, halting_min_steps=2,
        halting_ponder_lambda=0.0001,
        rezero_enabled=True, rezero_gate_init=-3.0,
        metric_bias_init_std=0.1,
        deep_supervision_enabled=False, ponder_kl_lambda=0.0,
        criticality_loss_enabled=False,
        curvature_diversity_loss_enabled=True,
        curvature_diversity_lambda=0.0001,
        curvature_cv_floor=1.5, curvature_cv_ceiling=8.0,
        tau_quality_loss_enabled=False,
        step_embed_enabled=False,
        step_conditional_operator=False,
        structural_tau_enabled=False,
        norm_ref=20.0, norm_lambda=0.1,
        base_lr=3e-4, structural_lr_ratio=0.1,
        warmup_steps=200, weight_decay=0.01,
        use_torch_compile=False,
    )


class PhaseManagerV18(nn.Module):
    def __init__(self, cond_dim=768, state_dim=8, action_horizon=16, action_dim=7,
                 d_substrate=128):
        super().__init__()
        self.K_belief = 6
        self.d = d_substrate
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.config = make_pm_config(d=d_substrate)

        # Per-position projections from different input sources
        chunk_seg_dim = 4 * action_dim  # 4 chunk positions × 7 dims = 28
        self.proj_near = nn.Linear(chunk_seg_dim, d_substrate)
        self.proj_far = nn.Linear(chunk_seg_dim, d_substrate)
        self.proj_cond = nn.Linear(cond_dim, d_substrate)
        self.proj_state = nn.Linear(state_dim + 1, d_substrate)  # +1 for staleness
        self.proj_integ = nn.Linear(d_substrate, d_substrate)  # operates on mean
        self.proj_target = nn.Linear(cond_dim, d_substrate)

        for proj in [self.proj_near, self.proj_far, self.proj_cond, self.proj_state,
                     self.proj_integ, self.proj_target]:
            nn.init.normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)

        self.pos_embed = nn.Parameter(torch.zeros(self.K_belief, d_substrate))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.context_pool = ContextPool(self.config)
        self.dynamics = ContinuousDynamics(self.config)

        # Readout: target position → 16 logits
        self.readout = nn.Sequential(
            nn.Linear(d_substrate, d_substrate), nn.SiLU(),
            nn.Linear(d_substrate, action_horizon),
        )

    def forward(self, groot_chunk, cond, state, staleness=None):
        # groot_chunk: [B, K=16, 7]
        # cond: [B, d_cond]
        # state: [B, 8]
        # staleness: [B] or None
        B = cond.shape[0]
        device = cond.device

        if staleness is None:
            staleness = torch.zeros(B, device=device)
        state_aug = torch.cat([state, staleness.unsqueeze(-1)], dim=-1)

        near = groot_chunk[:, :4].reshape(B, -1)  # [B, 28]
        far = groot_chunk[:, 12:16].reshape(B, -1)  # [B, 28]

        h0_list = [
            self.proj_near(near),
            self.proj_far(far),
            self.proj_cond(cond),
            self.proj_state(state_aug),
        ]
        # Integration position: init from mean of inputs
        h0_mean = torch.stack(h0_list, dim=1).mean(dim=1)
        h0_list.append(self.proj_integ(h0_mean))
        # Target position: init from cond projection (separate)
        h0_list.append(self.proj_target(cond))

        h0 = torch.stack(h0_list, dim=1) + self.pos_embed.unsqueeze(0)  # [B, K=6, d]

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
        if isinstance(out, tuple):
            h = out[0]
        else:
            h = out

        # Readout from target position (index 5)
        logits = self.readout(h[:, 5])  # [B, K_chunk=16]

        g = self.dynamics.compute_metric_diag(h0)
        metric_cv = g.std() / (g.mean() + 1e-8)

        return {"logits": logits, "metric_cv": metric_cv, "h_final": h, "g": g}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoded_dirs", required=True)
    p.add_argument("--teacher_labels_dirs", required=True)
    p.add_argument("--v10_ckpt", required=True, type=str,
                   help="v10-DEMO checkpoint to extract cond from (frozen)")
    p.add_argument("--output", default="/tmp/phase_manager_v18.pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=8000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--target_img_size", type=int, default=224)
    p.add_argument("--false_close_weight", type=float, default=3.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--d_substrate", type=int, default=128)
    args = p.parse_args()

    # === Data ===
    decoded_dirs = [Path(d.strip()) for d in args.decoded_dirs.split(",") if d.strip()]
    labels_dirs = [Path(d.strip()) for d in args.teacher_labels_dirs.split(",") if d.strip()]
    datasets = []
    for dd, ld in zip(decoded_dirs, labels_dirs):
        ds = TeacherLabelDataset(
            dd, ld, action_horizon=16, target_img_size=args.target_img_size,
            return_goal_img=False, return_z_groot="", use_query_bank=False,
        )
        datasets.append(ds)
    train_ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"[pm_v18] samples: {len(train_ds)}")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=(args.num_workers > 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Frozen v10 for cond extraction ===
    print(f"[pm_v18] loading frozen v10 from {args.v10_ckpt}")
    v10_ckpt = torch.load(args.v10_ckpt, map_location=device, weights_only=False)
    sa = v10_ckpt["args"]
    halt_mode = "learned" if sa["policy"] == "liquid_halt" else "none"
    v10_encoder = LiquidEncoder(
        state_dim=8, d=sa["d"], d_vis=sa["d"], img_size=sa["img_size"],
        k_max=sa["k"], halt_mode=halt_mode, min_steps=sa["halting_min_steps"],
        dt=sa.get("dt", 0.5),
        n_tasks=sa["n_tasks"], d_task=sa["d_task"],
        z_groot_dim=sa.get("z_groot_dim", 0),
        query_bank=sa.get("use_query_bank", False),
    ).to(device)
    # Strip "encoder." prefix from policy state dict
    sd = v10_ckpt.get("model", v10_ckpt.get("policy", v10_ckpt))
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    own = v10_encoder.state_dict()
    loaded = 0
    for k, v in enc_sd.items():
        kk = k.replace("_orig_mod.", "")
        if kk in own and own[kk].shape == v.shape:
            own[kk].copy_(v); loaded += 1
    print(f"[pm_v18] loaded {loaded}/{len(own)} encoder tensors")
    v10_encoder.eval()
    for p_ in v10_encoder.parameters():
        p_.requires_grad = False

    # === Model ===
    pm = PhaseManagerV18(
        cond_dim=sa["d"], state_dim=8, action_horizon=16, action_dim=7,
        d_substrate=args.d_substrate,
    ).to(device)
    n_params = sum(p.numel() for p in pm.parameters())
    print(f"[pm_v18] phase manager params: {n_params:,}")

    opt = torch.optim.AdamW(pm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    log_records = []
    t_start = time.time()
    data_iter = iter(loader)
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); batch = next(data_iter)
        imgs, wrists, states, chunks = batch[0], batch[1], batch[2], batch[3]
        imgs = imgs.to(device).float() / 255.0
        wrists = wrists.to(device).float() / 255.0
        imgs = imgs.permute(0, 3, 1, 2).contiguous()
        wrists = wrists.permute(0, 3, 1, 2).contiguous()
        states = states.to(device).float()
        chunks = chunks.to(device).float()  # [B, 16, 7]

        # Frozen v10 cond
        with torch.no_grad():
            cond, _ = v10_encoder(imgs, wrists, states)

        # Substrate phase manager forward
        # At training time, teacher chunk IS the GR00T signal proxy
        out = pm(chunks, cond, states, staleness=None)
        logits = out["logits"]  # [B, 16]
        target_open = (chunks[..., -1] < 0).float()  # 1 if expert says open

        base = F.binary_cross_entropy_with_logits(logits, target_open, reduction="none")
        with torch.no_grad():
            model_says_close = (logits < 0).float()
            penalty_mask = target_open * model_says_close
            weights = 1.0 + (args.false_close_weight - 1.0) * penalty_mask
        loss = (base * weights).mean()

        # Substrate aux: curvature diversity
        # (skip the import; just monitor metric_cv)

        opt.zero_grad(); loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(pm.parameters(), args.max_grad_norm)
        opt.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                pred_open = (logits > 0).float()
                acc = (pred_open == target_open).float().mean().item()
                open_mask = target_open == 1
                close_mask = target_open == 0
                acc_open = (pred_open[open_mask] == 1).float().mean().item() if open_mask.any() else 0
                acc_close = (pred_open[close_mask] == 0).float().mean().item() if close_mask.any() else 0
                cv = float(out["metric_cv"])
            rec = {"step": step, "loss": float(loss), "acc": acc,
                   "acc_open": acc_open, "acc_close": acc_close, "cv": cv,
                   "wall_s": time.time() - t_start}
            log_records.append(rec)
            print(f"step {step:>5}  loss={rec['loss']:.4f}  acc={rec['acc']:.3f}  "
                  f"acc_open={rec['acc_open']:.3f}  acc_close={rec['acc_close']:.3f}  "
                  f"cv={rec['cv']:.3f}  wall={rec['wall_s']:.0f}s")
        step += 1

    torch.save({"state_dict": pm.state_dict(), "args": vars(args),
                "cond_dim": sa["d"], "d_substrate": args.d_substrate}, args.output)
    print(f"\n[pm_v18] saved → {args.output}")


if __name__ == "__main__":
    main()
