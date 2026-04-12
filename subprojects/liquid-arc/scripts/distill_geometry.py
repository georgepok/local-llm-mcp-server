#!/usr/bin/env python3
"""Cross-dimension geometry distillation for LiquidARC.

Trains a student LiquidARC at d=student_d by:
  1. Distilling attention patterns from a post-transition teacher (d=teacher_d)
  2. Training on ARC task loss (same data as original experiments)
  3. Using 100x LR ratio (geo slow, content fast)

The teacher's phase transition produced a geometric substrate (CV~6, structured
attention routing, tau differentiation). The student learns to reproduce those
patterns at a different dimension without going through its own phase transition.

Usage:
    python scripts/distill_geometry.py \
        --teacher_checkpoint output_30m/checkpoints/step_10000.pt \
        --teacher_d 768 \
        --student_d 2688 \
        --data_dir /workspace/fgn-v3/data/arc-repo/data \
        --output_dir output/distilled_2688 \
        --max_steps 10000
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel
from liquid_arc.dynamics import ContinuousDynamics
from liquid_arc.context_pool import ContextPool


# ═══════════════════════════════════════════════════════════════════
# Teacher: extract attention patterns and tau from frozen checkpoint
# ═══════════════════════════════════════════════════════════════════

def load_teacher(checkpoint_path: str, device: str = 'cuda',
                  max_seq_len: int = None) -> LiquidARCModel:
    """Load full teacher model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'config' in ckpt and hasattr(ckpt['config'], 'd_model'):
        config = ckpt['config']
    else:
        raise ValueError("Teacher checkpoint must contain config")

    if max_seq_len is not None:
        config.max_seq_len = max_seq_len

    model = LiquidARCModel(config).to(device)

    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("._orig_mod.", ".")
        k = k.replace('metric_net_linear2.', 'metric_net_linear2_diag.')
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n = sum(p.numel() for p in model.parameters())
    print(f"  Teacher: d={config.d_model}, {n/1e6:.2f}M params, "
          f"d_metric={config.d_metric}, d_ffn={config.d_ffn}")
    return model


def get_teacher_attention(teacher: LiquidARCModel, h: torch.Tensor) -> dict:
    """Extract teacher's attention pattern and tau from hidden state h.

    Args:
        teacher: frozen teacher model
        h: [B, N, d_teacher] — hidden state after ODE integration

    Returns:
        dict with attention [B, N, N], tau [B, N, 1], metric_cv
    """
    dyn = getattr(teacher.dynamics, '_orig_mod', teacher.dynamics)
    B, N, d = h.shape

    h_normed = dyn.norm_geo(h)
    context = teacher.context_pool(h)
    ctx_expanded = context.unsqueeze(1).expand(-1, N, -1)
    metric_input = torch.cat([h_normed, ctx_expanded], dim=-1)

    hidden = F.gelu(dyn.metric_net_linear1(metric_input))
    g_diag = F.softplus(dyn.metric_net_linear2_diag(hidden))

    sqrt_g = g_diag.sqrt()
    scaled_h = h_normed * sqrt_g

    diff = scaled_h.unsqueeze(2) - scaled_h.unsqueeze(1)
    D_sq = (diff ** 2).sum(dim=-1)

    t = F.softplus(dyn.t_diffusion)
    attention = F.softmax(-D_sq / (4 * t), dim=-1)

    tau_hidden = F.gelu(dyn.tau_net_linear1(h))
    tau = torch.sigmoid(dyn.tau_net_linear2(tau_hidden))
    tau = tau * (dyn.tau_max - dyn.tau_min) + dyn.tau_min

    cv = (g_diag.std() / (g_diag.mean() + 1e-8)).item()

    return {'attention': attention, 'tau': tau, 'metric_cv': cv}


# ═══════════════════════════════════════════════════════════════════
# Distillation losses
# ═══════════════════════════════════════════════════════════════════

def attention_distill_loss(student_attn, teacher_attn):
    """KL(teacher || student) on attention patterns."""
    student_log = (student_attn + 1e-8).log()
    return F.kl_div(student_log, teacher_attn + 1e-8, reduction='batchmean',
                    log_target=False)


def tau_distill_loss(student_tau, teacher_tau):
    """MSE on normalized tau (relative pattern, not absolute)."""
    def normalize(tau):
        mn = tau.min(dim=1, keepdim=True).values
        mx = tau.max(dim=1, keepdim=True).values
        return (tau - mn) / (mx - mn + 1e-8)
    return F.mse_loss(normalize(student_tau), normalize(teacher_tau))


def get_student_attention(student: LiquidARCModel, h: torch.Tensor) -> dict:
    """Extract student's attention pattern and tau (WITH gradients)."""
    dyn = getattr(student.dynamics, '_orig_mod', student.dynamics)
    B, N, d = h.shape

    h_normed = dyn.norm_geo(h)
    context = student.context_pool(h)
    ctx_expanded = context.unsqueeze(1).expand(-1, N, -1)
    metric_input = torch.cat([h_normed, ctx_expanded], dim=-1)

    hidden = F.gelu(dyn.metric_net_linear1(metric_input))
    g_diag = F.softplus(dyn.metric_net_linear2_diag(hidden))

    sqrt_g = g_diag.sqrt()
    scaled_h = h_normed * sqrt_g

    diff = scaled_h.unsqueeze(2) - scaled_h.unsqueeze(1)
    D_sq = (diff ** 2).sum(dim=-1)

    t = F.softplus(dyn.t_diffusion)
    attention = F.softmax(-D_sq / (4 * t), dim=-1)

    tau_hidden = F.gelu(dyn.tau_net_linear1(h))
    tau = torch.sigmoid(dyn.tau_net_linear2(tau_hidden))
    tau = tau * (dyn.tau_max - dyn.tau_min) + dyn.tau_min

    cv = (g_diag.std() / (g_diag.mean() + 1e-8))

    return {'attention': attention, 'tau': tau, 'metric_cv': cv}


# ═══════════════════════════════════════════════════════════════════
# Geometry parameter classification
# ═══════════════════════════════════════════════════════════════════

def split_parameters(model: LiquidARCModel):
    """Split parameters into geometric (slow LR) and content (fast LR)."""
    geo_names = ['metric_net', 'tau_net', 't_diffusion', 'alpha_logit',
                 'context_pool']
    geo_params = []
    content_params = []
    for name, p in model.named_parameters():
        if any(g in name for g in geo_names):
            geo_params.append(p)
        else:
            content_params.append(p)
    return geo_params, content_params


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cross-dimension geometry distillation")
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--teacher_d", type=int, default=768)
    parser.add_argument("--student_d", type=int, required=True)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to ARC data (arc-repo/data)")
    parser.add_argument("--output_dir", type=str, default="output/distilled")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--geo_lr", type=float, default=1e-4)
    parser.add_argument("--content_lr", type=float, default=1e-2)
    parser.add_argument("--attn_weight", type=float, default=1.0)
    parser.add_argument("--tau_weight", type=float, default=0.1)
    parser.add_argument("--task_weight", type=float, default=1.0,
                        help="Weight for ARC task CE loss")
    parser.add_argument("--real_arc_mix", type=float, default=0.3)
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Max sequence length (ARC avg is 440, max ~1375)")
    parser.add_argument("--distill_steps", type=int, default=500,
                        help="Steps with distillation loss active, then pure task training")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--resume_student", type=str, default=None,
                        help="Path to pre-trained student checkpoint (e.g. criticality-prepared)")
    parser.add_argument("--criticality", action='store_true', default=False,
                        help="Enable criticality loss (D²/4τ targeting)")
    parser.add_argument("--crit_lambda", type=float, default=0.005,
                        help="Criticality loss weight (gentler than d=768's 0.01)")
    parser.add_argument("--tau_quality", action='store_true', default=False,
                        help="Enable tau_quality_loss (replaces tau_var_loss)")
    parser.add_argument("--tau_quality_lambda", type=float, default=0.1,
                        help="Tau quality loss weight")
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # ── Load teacher ──
    print("═══ Loading teacher ═══")
    teacher = load_teacher(args.teacher_checkpoint, device,
                            max_seq_len=args.max_seq_len)
    teacher_config = teacher.config

    # ── Create student ──
    print(f"\n═══ Creating student (d={args.student_d}) ═══")
    # Scale metric bottleneck proportionally with d_model
    # Teacher: d_metric=192 for d=768 = 25% ratio
    # Student: maintain same ratio so MetricNet has proportional capacity
    teacher_metric_ratio = teacher_config.d_metric / teacher_config.d_model
    student_d_metric = max(int(args.student_d * teacher_metric_ratio), teacher_config.d_metric)
    print(f"  MetricNet bottleneck: {student_d_metric} "
          f"({student_d_metric/args.student_d*100:.0f}% of d, "
          f"teacher was {teacher_config.d_metric}/{teacher_config.d_model}="
          f"{teacher_metric_ratio*100:.0f}%)")

    # Scale D²/4τ target proportionally with dimension
    d_ratio = args.student_d / args.teacher_d
    crit_target = 18.0 * d_ratio  # d=768→18, d=2688→63

    student_config = LiquidARCConfig(
        d_model=args.student_d,
        d_metric=student_d_metric,
        d_metric_bottleneck=0,  # use d_metric directly
        metric_rank=teacher_config.metric_rank,
        d_ffn=args.student_d * 2,
        max_seq_len=args.max_seq_len,
        n_ode_steps=teacher_config.n_ode_steps,
        ode_steps_min=16,
        ode_steps_max=16,
        tau_min=0.1,
        tau_max=3.0,
        tau_freeze_steps=0,  # TauNet active from step 0 — must learn during distillation
        t_diffusion_init=teacher_config.t_diffusion_init,
        alpha_logit_init=teacher_config.alpha_logit_init,
        curvature_lambda=teacher_config.curvature_lambda,
        tau_var_lambda=0.0,  # DISABLED — replaced by tau_quality_loss
        cv_floor_target=teacher_config.cv_floor_target,
        cv_ceiling_target=teacher_config.cv_ceiling_target,
        cv_floor_lambda=teacher_config.cv_floor_lambda,
        use_torch_compile=False,  # d=2688 may exceed Triton limits
        dropout=teacher_config.dropout,
        # Sustained criticality — active from step 0
        criticality_loss_enabled=args.criticality,
        criticality_loss_lambda=args.crit_lambda,
        criticality_target_ratio=crit_target,
        criticality_D_sq_target=0.0,  # no D² anchor — let MetricNet find its scale
        tau_quality_loss_enabled=args.tau_quality,
        tau_quality_lambda=args.tau_quality_lambda,
        tau_mean_target=1.0,
        tau_log_spread_target=0.6,
    )
    print(f"  Criticality: enabled={args.criticality}, λ={args.crit_lambda}, "
          f"D²/4τ target={crit_target:.1f} (scaled {d_ratio:.1f}× from 18)")
    print(f"  Tau quality: enabled={args.tau_quality}, λ={args.tau_quality_lambda}")
    student = LiquidARCModel(student_config).to(device)
    student.dynamics.freeze_tau = False  # TauNet must be active during distillation

    # Resume from pre-trained student checkpoint (e.g. criticality-prepared)
    if args.resume_student:
        ckpt_s = torch.load(args.resume_student, map_location=device, weights_only=False)
        s_state = ckpt_s.get('model_state_dict', ckpt_s.get('model', ckpt_s))
        cleaned_s = {k.replace('._orig_mod.', '.'): v for k, v in s_state.items()}
        student.load_state_dict(cleaned_s, strict=False)
        print(f"  Resumed student from {args.resume_student} (step {ckpt_s.get('step', '?')})")

    n_params = sum(p.numel() for p in student.parameters())
    geo_params, content_params = split_parameters(student)
    n_geo = sum(p.numel() for p in geo_params)
    print(f"  Student: {n_params/1e6:.2f}M total, {n_geo/1e6:.2f}M geometric")
    print(f"  freeze_tau={student.dynamics.freeze_tau} tau_range=[{student_config.tau_min}, {student_config.tau_max}]")

    # ── Data ──
    print(f"\n═══ Loading data ═══")
    from liquid_arc.tasks.procedural import ProceduralARCTask
    train_task = ProceduralARCTask(seq_len=student_config.max_seq_len, augment=True)
    print(f"  Procedural: seq_len={student_config.max_seq_len}")

    real_arc_train = None
    eval_task = None
    if args.data_dir:
        try:
            from fgn.tasks.arc import ARCTask
            real_arc_train = ARCTask(
                seq_len=student_config.max_seq_len, data_dir=args.data_dir,
                split='train', augment=True)
            eval_task = ARCTask(
                seq_len=student_config.max_seq_len, data_dir=args.data_dir,
                split='eval', augment=False)
            print(f"  Real ARC: {args.real_arc_mix:.0%} mix, eval from eval split")
        except Exception as e:
            print(f"  ARC data unavailable ({e})")

    if eval_task is None:
        eval_task = ProceduralARCTask(seq_len=student_config.max_seq_len, augment=False)
        eval_task._seed_counter = 999999

    # ── Optimizer: 100x LR ratio ──
    optimizer = torch.optim.AdamW([
        {'params': geo_params, 'lr': args.geo_lr},
        {'params': content_params, 'lr': args.content_lr},
    ], weight_decay=0.01)
    print(f"\n  Geo LR: {args.geo_lr}, Content LR: {args.content_lr} "
          f"({args.content_lr/args.geo_lr:.0f}x ratio)")

    # ── Training ──
    print(f"\n═══ Training ({args.max_steps} steps) ═══")
    print(f"{'Step':>6} | {'loss':>8} | {'ce':>7} | {'attn_kl':>8} | "
          f"{'xform%':>7} | {'CV_s':>6} | {'CV_t':>6} | {'tok/s':>7}")
    print("-" * 80)

    student.train()
    t0 = time.time()
    metrics = {}
    step = 0

    while step < args.max_steps:
        # ── Generate batch ──
        use_arc = (real_arc_train is not None and
                   random.random() < args.real_arc_mix)
        try:
            if use_arc:
                _, _, meta = real_arc_train.generate_batch(
                    batch_size=args.batch_size, device=device)
            else:
                _, _, meta = train_task.generate_batch(
                    batch_size=args.batch_size, device=device)
        except Exception:
            continue

        # ── Student forward (ARC task) ──
        with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                enabled=(device == 'cuda')):
            result = student(
                colors=meta['colors'], xs=meta['xs'], ys=meta['ys'],
                roles=meta['roles'], sep_mask=meta['sep_mask'],
                sep_types=meta['sep_types'], target_mask=meta['target_mask'],
                target_labels=meta.get('target_labels'),
                grid_ids=meta.get('grid_ids'),
            )
            ce_loss = result['ce_loss']

        # ── Distillation or pure task training ──
        distilling = step <= args.distill_steps
        loss_attn = torch.tensor(0.0, device=device)
        loss_tau = torch.tensor(0.0, device=device)

        if distilling:
            with torch.no_grad():
                t_result = teacher(
                    colors=meta['colors'], xs=meta['xs'], ys=meta['ys'],
                    roles=meta['roles'], sep_mask=meta['sep_mask'],
                    sep_types=meta['sep_types'], target_mask=meta['target_mask'],
                    target_labels=meta.get('target_labels'),
                    grid_ids=meta.get('grid_ids'),
                )
                teacher_attn = get_teacher_attention(teacher, t_result['h_final'])

            student_attn = get_student_attention(student, result['h_final'])
            loss_attn = attention_distill_loss(
                student_attn['attention'], teacher_attn['attention'])
            loss_tau = tau_distill_loss(student_attn['tau'], teacher_attn['tau'])

        if distilling:
            loss = (args.task_weight * ce_loss +
                    args.attn_weight * loss_attn +
                    args.tau_weight * loss_tau)
        else:
            loss = ce_loss

        # Regularization always active
        curv_loss = result.get('curv_loss', torch.tensor(0.0))
        tau_var_loss = result.get('tau_var_loss', torch.tensor(0.0))
        cv_floor_loss = result.get('cv_floor_loss', torch.tensor(0.0))
        crit_loss = result.get('criticality_loss', torch.tensor(0.0))
        tau_q_loss = result.get('tau_quality_loss', torch.tensor(0.0))
        loss = loss + curv_loss + tau_var_loss + cv_floor_loss + crit_loss + tau_q_loss

        # Snapshot when distillation ends
        if step == args.distill_steps:
            snap_path = os.path.join(args.output_dir, "checkpoints",
                                     "distill_peak.pt")
            torch.save({
                'step': step,
                'model_state_dict': student.state_dict(),
                'config': student_config,
                'phase': 'distill_peak',
            }, snap_path)
            print(f"  ── DISTILLATION COMPLETE at step {step} ──")
            print(f"  Saved snapshot: {snap_path}")
            print(f"  Switching to pure task training (CE only)")

        # ── Backward ──
        optimizer.zero_grad()
        if torch.isnan(loss) or torch.isinf(loss):
            step += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        step += 1

        # ── Accumulate metrics ──
        with torch.no_grad():
            metrics['loss'] = metrics.get('loss', 0) + loss.item()
            metrics['ce'] = metrics.get('ce', 0) + ce_loss.item()
            metrics['attn_kl'] = metrics.get('attn_kl', 0) + loss_attn.item()
            metrics['xform'] = metrics.get('xform', 0) + result.get('transform_accuracy', 0)
            metrics['cv'] = metrics.get('cv', 0) + result.get('metric_cv', 0)
            tau_avg = result.get('tau_avg', 0)
            if isinstance(tau_avg, torch.Tensor): tau_avg = tau_avg.item()
            metrics['tau'] = metrics.get('tau', 0) + tau_avg
            metrics['crit_ratio'] = metrics.get('crit_ratio', 0) + result.get('crit_ratio', 0)
            metrics['log_tau_std'] = metrics.get('log_tau_std', 0) + result.get('log_tau_std', 0)
            metrics['n'] = metrics.get('n', 0) + 1

        # ── Logging ──
        if step % args.log_every == 0:
            n = max(metrics['n'], 1)
            elapsed = time.time() - t0
            tps = (step * args.batch_size * student_config.max_seq_len) / max(elapsed, 1)
            phase = "distill" if distilling else "task"
            crit_str = f" D²/4τ={metrics['crit_ratio']/n:.1f}" if metrics.get('crit_ratio') else ""
            lts_str = f" lτσ={metrics['log_tau_std']/n:.2f}" if metrics.get('log_tau_std') else ""
            print(f"{step:>6} [{phase:>7s}] | loss={metrics['loss']/n:>7.3f} "
                  f"ce={metrics['ce']/n:.3f} kl={metrics['attn_kl']/n:.3f} | "
                  f"xform={metrics['xform']/n*100:.1f}% "
                  f"CV={metrics['cv']/n:.2f} "
                  f"tau={metrics['tau']/n:.3f}{lts_str}{crit_str} | "
                  f"{tps:.0f} tok/s")
            metrics = {}

        # ── Eval ──
        if step % args.eval_every == 0:
            student.eval()
            eval_m = {'xform': 0, 'ce': 0, 'cv': 0, 'n': 0}
            with torch.no_grad():
                for _ in range(args.eval_batches):
                    try:
                        _, _, emeta = eval_task.generate_batch(
                            batch_size=args.batch_size, device=device)
                        er = student(
                            colors=emeta['colors'], xs=emeta['xs'], ys=emeta['ys'],
                            roles=emeta['roles'], sep_mask=emeta['sep_mask'],
                            sep_types=emeta['sep_types'],
                            target_mask=emeta['target_mask'],
                            target_labels=emeta.get('target_labels'),
                            grid_ids=emeta.get('grid_ids'),
                        )
                        eval_m['xform'] += er.get('transform_accuracy', 0)
                        eval_m['ce'] += er['ce_loss'].item()
                        eval_m['cv'] += er.get('metric_cv', 0)
                        eval_m['n'] += 1
                    except Exception:
                        pass

            ne = max(eval_m['n'], 1)
            print(f"  ── EVAL step {step}: xform={eval_m['xform']/ne*100:.1f}% "
                  f"CE={eval_m['ce']/ne:.3f} CV={eval_m['cv']/ne:.2f}")
            student.train()

        # ── Save ──
        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, "checkpoints",
                                     f"step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': student_config,
                'teacher_checkpoint': args.teacher_checkpoint,
                'teacher_d': args.teacher_d,
            }, ckpt_path)
            print(f"  → Saved {ckpt_path}")

    # ── Final eval ──
    print(f"\n═══ Final evaluation ═══")
    student.eval()
    eval_m = {'xform': 0, 'ce': 0, 'cv': 0, 'n': 0}
    with torch.no_grad():
        for _ in range(args.eval_batches * 2):
            try:
                _, _, emeta = eval_task.generate_batch(
                    batch_size=args.batch_size, device=device)
                er = student(
                    colors=emeta['colors'], xs=emeta['xs'], ys=emeta['ys'],
                    roles=emeta['roles'], sep_mask=emeta['sep_mask'],
                    sep_types=emeta['sep_types'],
                    target_mask=emeta['target_mask'],
                    target_labels=emeta.get('target_labels'),
                    grid_ids=emeta.get('grid_ids'),
                )
                eval_m['xform'] += er.get('transform_accuracy', 0)
                eval_m['ce'] += er['ce_loss'].item()
                eval_m['cv'] += er.get('metric_cv', 0)
                eval_m['n'] += 1
            except Exception:
                pass

    ne = max(eval_m['n'], 1)
    print(f"  Eval xform: {eval_m['xform']/ne*100:.1f}%")
    print(f"  Eval CE: {eval_m['ce']/ne:.3f}")
    print(f"  Eval CV: {eval_m['cv']/ne:.2f}")
    print(f"  (Teacher reference: 54.2% xform at d=768)")

    # Save final
    final_path = os.path.join(args.output_dir, "checkpoints", "final.pt")
    torch.save({
        'step': step,
        'model_state_dict': student.state_dict(),
        'config': student_config,
        'teacher_checkpoint': args.teacher_checkpoint,
        'teacher_d': args.teacher_d,
        'eval_xform': eval_m['xform'] / ne * 100,
        'eval_ce': eval_m['ce'] / ne,
    }, final_path)
    print(f"  Saved: {final_path}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} minutes")


if __name__ == '__main__':
    main()
