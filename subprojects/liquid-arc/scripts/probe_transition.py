#!/usr/bin/env python3
"""Probe the phase transition during geometry distillation.

Instruments the distillation process with detailed probes around the
critical event where attention KL collapses (step ~175) and geometry
rapidly develops. Records per-step snapshots of:

1. Attention pattern statistics (entropy, sparsity, eigenspectrum)
2. MetricNet internals (g_diag distribution, gradient norms)
3. TauNet outputs (per-position tau, tau gradient)
4. t_diffusion trajectory and gradient
5. D² distance distribution (before/after metric scaling)
6. ODE dynamics (dh/dt magnitude, residual vs routing balance)
7. Loss decomposition (which component drives the transition)
8. Weight change rates per module

Goal: identify the TRIGGER — what changes first, and what follows.

Usage:
    python scripts/probe_transition.py \
        --teacher_checkpoint output_30m/checkpoints/step_10000.pt \
        --student_d 2688 \
        --data_dir /workspace/fgn-v3/data/arc-repo/data \
        --output_dir output/transition_probe \
        --max_steps 500 --log_every 1
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel


def load_teacher(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = LiquidARCModel(config).to(device)
    sd = ckpt.get('model_state_dict', ckpt.get('model', {}))
    cleaned = {}
    for k, v in sd.items():
        k = k.replace("._orig_mod.", ".")
        k = k.replace('metric_net_linear2.', 'metric_net_linear2_diag.')
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_attention_and_internals(model, h):
    """Extract detailed attention internals from hidden state h."""
    dyn = getattr(model.dynamics, '_orig_mod', model.dynamics)
    B, N, d = h.shape

    h_normed = dyn.norm_geo(h)
    context = model.context_pool(h)
    ctx_expanded = context.unsqueeze(1).expand(-1, N, -1)
    metric_input = torch.cat([h_normed, ctx_expanded], dim=-1)

    # MetricNet internals
    hidden = F.gelu(dyn.metric_net_linear1(metric_input))
    g_diag = F.softplus(dyn.metric_net_linear2_diag(hidden))

    # Distance computation
    sqrt_g = g_diag.sqrt()
    scaled_h = h_normed * sqrt_g
    diff = scaled_h.unsqueeze(2) - scaled_h.unsqueeze(1)
    D_sq = (diff ** 2).sum(dim=-1)

    # Raw distances (without metric)
    raw_diff = h_normed.unsqueeze(2) - h_normed.unsqueeze(1)
    raw_D_sq = (raw_diff ** 2).sum(dim=-1)

    t = F.softplus(dyn.t_diffusion)
    attention = F.softmax(-D_sq / (4 * t), dim=-1)

    # TauNet
    tau_hidden = F.gelu(dyn.tau_net_linear1(h))
    tau = torch.sigmoid(dyn.tau_net_linear2(tau_hidden))
    tau = tau * (dyn.tau_max - dyn.tau_min) + dyn.tau_min

    alpha = torch.sigmoid(dyn.alpha_logit)

    # Attention eigenspectrum (on first batch)
    attn_0 = attention[0].detach()
    try:
        eigenvalues = torch.linalg.eigvalsh(attn_0)
    except Exception:
        eigenvalues = torch.zeros(N)

    # Attention entropy per row
    attn_entropy = -(attention * (attention + 1e-8).log()).sum(dim=-1)

    return {
        'attention': attention,
        'g_diag': g_diag,
        'g_mean': g_diag.mean().item(),
        'g_std': g_diag.std().item(),
        'g_cv': (g_diag.std() / (g_diag.mean() + 1e-8)).item(),
        'g_min': g_diag.min().item(),
        'g_max': g_diag.max().item(),
        'D_sq_mean': D_sq[D_sq > 0].mean().item() if (D_sq > 0).any() else 0,
        'D_sq_std': D_sq[D_sq > 0].std().item() if (D_sq > 0).any() else 0,
        'raw_D_sq_mean': raw_D_sq[raw_D_sq > 0].mean().item() if (raw_D_sq > 0).any() else 0,
        'metric_amplification': (D_sq[D_sq > 0].mean() / (raw_D_sq[raw_D_sq > 0].mean() + 1e-8)).item() if (D_sq > 0).any() else 0,
        't_diffusion': t.item(),
        'neg_D2_over_4t_mean': (-D_sq[D_sq > 0] / (4 * t)).mean().item() if (D_sq > 0).any() else 0,
        'attn_entropy_mean': attn_entropy.mean().item(),
        'attn_entropy_std': attn_entropy.std().item(),
        'attn_max_per_row': attention.max(dim=-1).values.mean().item(),
        'attn_diag_mean': attention[0].diag().mean().item() if N <= 512 else 0,
        'attn_top1_mass': attention.max(dim=-1).values.mean().item(),
        'attn_top5_mass': attention.topk(min(5, N), dim=-1).values.sum(dim=-1).mean().item(),
        'tau_mean': tau.mean().item(),
        'tau_std': tau.std().item(),
        'tau_min': tau.min().item(),
        'tau_max_val': tau.max().item(),
        'alpha': alpha.item(),
        'eigenvalue_max': eigenvalues[-1].item() if len(eigenvalues) > 0 else 0,
        'eigenvalue_ratio': (eigenvalues[-1] / (eigenvalues[-2] + 1e-8)).item() if len(eigenvalues) > 1 else 0,
    }


def get_gradient_norms(model):
    """Get per-module gradient norms."""
    norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            norms[name] = p.grad.norm().item()
    return norms


def get_weight_norms(model):
    """Get per-module weight norms."""
    norms = {}
    for name, p in model.named_parameters():
        norms[name] = p.norm().item()
    return norms


def split_parameters(model):
    geo_names = ['metric_net', 'tau_net', 't_diffusion', 'alpha_logit', 'context_pool']
    geo, content = [], []
    for name, p in model.named_parameters():
        if any(g in name for g in geo_names):
            geo.append(p)
        else:
            content.append(p)
    return geo, content


def main():
    parser = argparse.ArgumentParser(description="Probe phase transition")
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--student_d", type=int, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/transition_probe")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--geo_lr", type=float, default=1e-4)
    parser.add_argument("--content_lr", type=float, default=1e-2)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # Load teacher
    print("═══ Loading teacher ═══")
    teacher = load_teacher(args.teacher_checkpoint, device)
    teacher_config = teacher.config

    # Create student
    print(f"\n═══ Creating student (d={args.student_d}) ═══")
    teacher_metric_ratio = teacher_config.d_metric / teacher_config.d_model
    student_d_metric = max(int(args.student_d * teacher_metric_ratio), teacher_config.d_metric)

    student_config = LiquidARCConfig(
        d_model=args.student_d,
        d_metric=student_d_metric,
        d_metric_bottleneck=0,
        metric_rank=teacher_config.metric_rank,
        d_ffn=args.student_d * 2,
        max_seq_len=args.max_seq_len,
        n_ode_steps=teacher_config.n_ode_steps,
        ode_steps_min=16, ode_steps_max=16,
        tau_min=0.1, tau_max=3.0, tau_freeze_steps=0,
        t_diffusion_init=teacher_config.t_diffusion_init,
        alpha_logit_init=teacher_config.alpha_logit_init,
        curvature_lambda=teacher_config.curvature_lambda,
        tau_var_lambda=teacher_config.tau_var_lambda,
        cv_floor_target=teacher_config.cv_floor_target,
        cv_ceiling_target=teacher_config.cv_ceiling_target,
        cv_floor_lambda=teacher_config.cv_floor_lambda,
        use_torch_compile=False,
        dropout=teacher_config.dropout,
    )
    student = LiquidARCModel(student_config).to(device)
    student.dynamics.freeze_tau = False

    # Data
    print(f"\n═══ Loading data ═══")
    from liquid_arc.tasks.procedural import ProceduralARCTask
    train_task = ProceduralARCTask(seq_len=args.max_seq_len, augment=True)
    real_arc = None
    if args.data_dir:
        try:
            from fgn.tasks.arc import ARCTask
            real_arc = ARCTask(seq_len=args.max_seq_len, data_dir=args.data_dir,
                               split='train', augment=True)
        except Exception:
            pass

    # Optimizer
    geo_params, content_params = split_parameters(student)
    optimizer = torch.optim.AdamW([
        {'params': geo_params, 'lr': args.geo_lr},
        {'params': content_params, 'lr': args.content_lr},
    ], weight_decay=0.01)

    # Record initial weights for change tracking
    initial_weights = {n: p.data.clone() for n, p in student.named_parameters()}

    # Probe storage
    probe_log = []

    print(f"\n═══ Probing ({args.max_steps} steps, log every {args.log_every}) ═══")
    student.train()
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        # Generate batch
        use_real = real_arc is not None and random.random() < 0.3
        try:
            if use_real:
                _, _, meta = real_arc.generate_batch(batch_size=args.batch_size, device=device)
            else:
                _, _, meta = train_task.generate_batch(batch_size=args.batch_size, device=device)
        except Exception:
            continue

        # Student forward
        result = student(
            colors=meta['colors'], xs=meta['xs'], ys=meta['ys'],
            roles=meta['roles'], sep_mask=meta['sep_mask'],
            sep_types=meta['sep_types'], target_mask=meta['target_mask'],
            target_labels=meta.get('target_labels'),
            grid_ids=meta.get('grid_ids'),
        )
        ce_loss = result['ce_loss']

        # Teacher attention target
        with torch.no_grad():
            t_result = teacher(
                colors=meta['colors'], xs=meta['xs'], ys=meta['ys'],
                roles=meta['roles'], sep_mask=meta['sep_mask'],
                sep_types=meta['sep_types'], target_mask=meta['target_mask'],
                target_labels=meta.get('target_labels'),
                grid_ids=meta.get('grid_ids'),
            )
            t_internals = get_attention_and_internals(teacher, t_result['h_final'])

        # Student attention
        s_internals = get_attention_and_internals(student, result['h_final'])

        # Distillation loss
        s_log = (s_internals['attention'] + 1e-8).log()
        attn_kl = F.kl_div(s_log, t_internals['attention'] + 1e-8,
                           reduction='batchmean', log_target=False)

        loss = ce_loss + attn_kl
        loss = loss + result.get('curv_loss', 0) + result.get('tau_var_loss', 0)
        loss = loss + result.get('cv_floor_loss', 0)

        # Backward
        optimizer.zero_grad()
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            # NaN scrubbing
            for p in student.parameters():
                if p.grad is not None and p.grad.isnan().any():
                    p.grad.nan_to_num_(nan=0.0)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

        # ── Probe ──
        if step % args.log_every == 0:
            with torch.no_grad():
                grad_norms = get_gradient_norms(student)
                weight_changes = {}
                for n, p in student.named_parameters():
                    if n in initial_weights:
                        weight_changes[n] = (p.data - initial_weights[n]).norm().item()

                # Key gradient groups
                geo_grad = sum(v for k, v in grad_norms.items()
                              if any(g in k for g in ['metric_net', 'tau_net', 't_diffusion', 'alpha']))
                content_grad = sum(v for k, v in grad_norms.items()
                                  if not any(g in k for g in ['metric_net', 'tau_net', 't_diffusion', 'alpha', 'context_pool']))

                probe_entry = {
                    'step': step,
                    'ce_loss': ce_loss.item(),
                    'attn_kl': attn_kl.item(),
                    'total_loss': loss.item(),
                    'xform': result.get('transform_accuracy', 0),
                    # Student internals
                    **{f's_{k}': v for k, v in s_internals.items() if k != 'attention' and k != 'g_diag'},
                    # Teacher internals for comparison
                    't_g_cv': t_internals['g_cv'],
                    't_attn_entropy': t_internals['attn_entropy_mean'],
                    't_tau_mean': t_internals['tau_mean'],
                    't_tau_std': t_internals['tau_std'],
                    # Gradient analysis
                    'geo_grad_total': geo_grad,
                    'content_grad_total': content_grad,
                    'grad_t_diffusion': grad_norms.get('dynamics.t_diffusion', 0),
                    'grad_alpha': grad_norms.get('dynamics.alpha_logit', 0),
                    'grad_metric_l1': grad_norms.get('dynamics.metric_net_linear1.weight', 0),
                    'grad_metric_l2': grad_norms.get('dynamics.metric_net_linear2_diag.weight', 0),
                    'grad_tau_l1': grad_norms.get('dynamics.tau_net_linear1.weight', 0),
                    'grad_tau_l2': grad_norms.get('dynamics.tau_net_linear2.weight', 0),
                    # Weight change from init
                    'delta_t_diffusion': weight_changes.get('dynamics.t_diffusion', 0),
                    'delta_alpha': weight_changes.get('dynamics.alpha_logit', 0),
                    'delta_metric_l1': weight_changes.get('dynamics.metric_net_linear1.weight', 0),
                    'delta_tau_l1': weight_changes.get('dynamics.tau_net_linear1.weight', 0),
                }

                probe_log.append(probe_entry)

                # Console output
                print(f"  {step:>4d} | kl={attn_kl.item():.1f} ce={ce_loss.item():.3f} "
                      f"xform={result.get('transform_accuracy',0)*100:.1f}% | "
                      f"CV={s_internals['g_cv']:.2f} tau={s_internals['tau_mean']:.2f}±{s_internals['tau_std']:.3f} | "
                      f"t={s_internals['t_diffusion']:.2f} α={s_internals['alpha']:.3f} | "
                      f"ent={s_internals['attn_entropy_mean']:.2f} diag={s_internals['attn_diag_mean']:.3f} | "
                      f"D²={s_internals['D_sq_mean']:.0f} amp={s_internals['metric_amplification']:.1f}x | "
                      f"∇geo={geo_grad:.3f} ∇cont={content_grad:.3f}")

        # Save probe data periodically
        if step % 50 == 0:
            probe_path = os.path.join(args.output_dir, "probe_log.json")
            with open(probe_path, 'w') as f:
                json.dump(probe_log, f, indent=1,
                          default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    # Final save
    probe_path = os.path.join(args.output_dir, "probe_log.json")
    with open(probe_path, 'w') as f:
        json.dump(probe_log, f, indent=1, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"\nProbe data saved: {probe_path} ({len(probe_log)} entries)")
    print(f"Total time: {(time.time()-t0)/60:.1f} minutes")


if __name__ == '__main__':
    main()
