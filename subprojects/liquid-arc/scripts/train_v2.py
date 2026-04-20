"""LiquidARC v2 — Geometry Distillation Training Script.

Trains a new LiquidARCModel with structural tau, initialized from the
post-transition teacher's geometry (MetricNet/TauNet weights transferred).
The new model starts in the post-transition regime and preserves it via:
  1. Direct weight transfer from teacher (MetricNet, TauNet, context_pool)
  2. 100× slower LR for geometric parameters vs content parameters
  3. Explicit gradient scaling by mean structural tau after each backward pass

Usage:
    python scripts/train_v2.py \
      --config configs/liquid_arc_v2.yaml \
      --data_dir /workspace/fgn-v3/data/arc-repo/data \
      --teacher_checkpoint /workspace/liquid-arc/PRECIOUS_CHECKPOINTS/5m_post_transition.pt \
      --geometry_init weight_transfer \
      --output_dir output_v2/seeded \
      --max_steps 10000

    # Record teacher geometry first (optional, for comparison only):
    python scripts/record_geometry.py \
      --checkpoint /workspace/liquid-arc/PRECIOUS_CHECKPOINTS/5m_post_transition.pt \
      --config configs/liquid_arc_5m.yaml \
      --data_dir /workspace/fgn-v3/data/arc-repo/data \
      --output geometry_targets.pt
"""

import argparse
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

FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if not Path(FGN_ROOT).exists():
    FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, FGN_ROOT)

try:
    from torch.utils.tensorboard import SummaryWriter
    _tb_available = True
except ImportError:
    _tb_available = False


# ──────────────────── GEOMETRY DISTILLATION UTILITIES ────────────────────

def transfer_geometry_weights(student: LiquidARCModel, teacher_checkpoint: str,
                               device: str = 'cuda') -> int:
    """Copy MetricNet, TauNet, and context_pool weights from teacher to student.

    The student starts with exactly the teacher's geometry. Structural tau
    and the 100× LR ratio then preserve it during training while content
    parameters (FFN, W_v, W_o, embedding, output head) learn from scratch.

    Returns the number of parameters transferred.
    """
    ckpt = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}

    geo_keys = [
        'dynamics.metric_net_linear1.weight', 'dynamics.metric_net_linear1.bias',
        'dynamics.metric_net_linear2.weight', 'dynamics.metric_net_linear2.bias',
        'dynamics.tau_net_linear1.weight',    'dynamics.tau_net_linear1.bias',
        'dynamics.tau_net_linear2.weight',    'dynamics.tau_net_linear2.bias',
        'dynamics.t_diffusion',
        'dynamics.alpha_logit',
    ]

    student_params = dict(student.named_parameters())
    transferred = 0
    skipped = []

    for key in geo_keys:
        if key not in cleaned:
            skipped.append(f"{key} (missing in teacher)")
            continue
        param = student_params.get(key)
        if param is None:
            skipped.append(f"{key} (not in student)")
            continue
        if param.shape != cleaned[key].shape:
            skipped.append(f"{key} (shape mismatch: {param.shape} vs {cleaned[key].shape})")
            continue
        param.data.copy_(cleaned[key].to(device))
        transferred += 1

    # Also transfer context_pool (geometric infrastructure)
    cp_keys = [k for k in cleaned if k.startswith('context_pool.')]
    for key in cp_keys:
        param = student_params.get(key)
        if param is not None and param.shape == cleaned[key].shape:
            param.data.copy_(cleaned[key].to(device))
            transferred += 1

    if skipped:
        print(f"  Skipped {len(skipped)} keys: {skipped[:3]}{'...' if len(skipped) > 3 else ''}")
    print(f"  Transferred {transferred} geometry parameters from teacher")
    return transferred


def apply_structural_gradient_coupling(model: LiquidARCModel):
    """Scale geometric parameter gradients by inverse mean structural tau.

    High structural_tau → gradient scaled DOWN → geometry learns slower.
    Low structural_tau  → gradient unchanged  → geometry learns at normal rate.

    This EXPLICITLY creates the learning-time timescale separation that the
    inference-time tau creates at the activation level. Geometric parameters
    already get 100× lower LR; this provides additional per-position scaling.
    """
    # Handle torch.compile wrapper
    dyn = getattr(model, '_orig_mod', model).dynamics if hasattr(model, '_orig_mod') \
        else model.dynamics

    if not getattr(dyn, 'structural_tau_enabled', False):
        return

    with torch.no_grad():
        s_tau_mean = torch.sigmoid(dyn.structural_tau).mean()
        scale = 1.0 / (s_tau_mean.item() + 0.1)  # avoid div-by-zero

    geo_params = (
        list(dyn.metric_net_linear1.parameters()) +
        list(dyn.metric_net_linear2.parameters()) +
        list(dyn.tau_net_linear1.parameters()) +
        list(dyn.tau_net_linear2.parameters())
    )

    for p in geo_params:
        if p.grad is not None:
            p.grad.mul_(scale)


def build_optimizer(model: LiquidARCModel, base_lr: float,
                    structural_lr_ratio: float, weight_decay: float):
    """Three-group optimizer: structural (slow), content (normal), structural_tau (slow).

    structural_params: MetricNet, TauNet, t_diffusion, alpha_logit, context_pool
    structural_tau param: structural_tau (very slow — position-level geometry)
    content_params: everything else (FFN, W_v, W_o, embedding, head)
    """
    structural_param_ids = set()

    # Structural params: MetricNet, TauNet, t_diffusion, alpha_logit, context_pool
    # Handle _orig_mod. prefix from torch.compile
    structural_prefixes = [
        'dynamics.metric_net_linear1', 'dynamics.metric_net_linear2',
        'dynamics.tau_net_linear1', 'dynamics.tau_net_linear2',
        'context_pool.',
    ]
    structural_names = {'dynamics.t_diffusion', 'dynamics.alpha_logit'}

    structural_params = []
    for name, p in model.named_parameters():
        # Strip _orig_mod. prefix for matching
        clean_name = name.replace('_orig_mod.', '')
        if any(clean_name.startswith(pfx) for pfx in structural_prefixes) or \
           clean_name in structural_names:
            structural_params.append(p)
            structural_param_ids.add(id(p))

    # Structural tau param (position-level geometry)
    structural_tau_params = []
    for name, p in model.named_parameters():
        clean_name = name.replace('_orig_mod.', '')
        if clean_name == 'dynamics.structural_tau':
            structural_tau_params.append(p)
            structural_param_ids.add(id(p))

    content_params = [p for p in model.parameters()
                      if id(p) not in structural_param_ids]

    # structural_tau gets 10× higher LR than other structural params
    # (10× slower than content, not 100×) — needs to differentiate per-position
    s_tau_lr_ratio = structural_lr_ratio * 10.0  # 0.01 * 10 = 0.1

    param_groups = [
        {'params': content_params,        'lr': base_lr,                        'name': 'content'},
        {'params': structural_params,     'lr': base_lr * structural_lr_ratio,  'name': 'structural'},
        {'params': structural_tau_params, 'lr': base_lr * s_tau_lr_ratio,       'name': 'structural_tau'},
    ]

    optimizer = torch.optim.AdamW(
        [g for g in param_groups if g['params']],
        weight_decay=weight_decay,
    )

    n_content = sum(p.numel() for p in content_params)
    n_structural = sum(p.numel() for p in structural_params)
    n_s_tau = sum(p.numel() for p in structural_tau_params)
    print(f"  Optimizer groups: content={n_content:,}  structural={n_structural:,}  "
          f"structural_tau={n_s_tau:,}")
    print(f"  LR: content={base_lr:.2e}  structural={base_lr*structural_lr_ratio:.2e}  "
          f"structural_tau={base_lr*s_tau_lr_ratio:.2e}")

    return optimizer


def make_scheduler(optimizer, base_lr: float, warmup_steps: int, max_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────── TRAINING LOOP ────────────────────

def train(args):
    config = LiquidARCConfig.from_yaml(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ──
    model = LiquidARCModel(config).to(device)

    if args.use_torch_compile and config.use_torch_compile:
        print("torch.compile: compiling model...")
        model = torch.compile(model)
        print("torch.compile: done")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    # ── Geometry initialization ──
    if args.teacher_checkpoint:
        if args.geometry_init == 'weight_transfer':
            print(f"\nTransferring geometry from teacher: {args.teacher_checkpoint}")
            transfer_geometry_weights(model, args.teacher_checkpoint, device)
        elif args.geometry_init == 'statistical':
            if args.geometry_targets:
                print(f"Statistical init from targets: {args.geometry_targets}")
                targets = torch.load(args.geometry_targets, weights_only=False)
                # TODO: run Phase 3A init loop
                print("  (Statistical init not implemented, falling back to random init)")
            else:
                print("  --geometry_targets required for statistical init, skipping")

        # Verify geometry via a full forward pass (context must be set)
        with torch.no_grad():
            from liquid_arc.tasks.procedural import ProceduralARCTask
            _vtask = ProceduralARCTask(seq_len=config.max_seq_len)
            _, _, _vmeta = _vtask.generate_batch(batch_size=2, device=device)
            _vr = model(
                colors=_vmeta['colors'], xs=_vmeta['xs'], ys=_vmeta['ys'],
                roles=_vmeta['roles'], sep_mask=_vmeta['sep_mask'],
                sep_types=_vmeta['sep_types'], target_mask=_vmeta['target_mask'],
                target_labels=_vmeta.get('target_labels'),
                grid_ids=_vmeta.get('grid_ids'),
            )
            init_cv = _vr.get('metric_cv', 0)
            init_tau = _vr.get('tau_avg', 0)
        print(f"  Initialization check: CV={init_cv:.3f}, tau={init_tau:.3f}")
        if init_cv > 3.0:
            print("  ✓ Geometry in post-transition regime (CV > 3)")
        else:
            print("  ✗ WARNING: CV low — geometry may not have transferred correctly")

    # ── Optimizer + Scheduler ──
    base_lr = getattr(config, 'base_lr', 3e-4)
    structural_lr_ratio = getattr(config, 'structural_lr_ratio', 0.01)
    warmup_steps = getattr(config, 'warmup_steps', 500)
    weight_decay = getattr(config, 'weight_decay', 0.01)

    optimizer = build_optimizer(model, base_lr, structural_lr_ratio, weight_decay)
    scheduler = make_scheduler(optimizer, base_lr, warmup_steps, args.max_steps)

    # ── Data ──
    from liquid_arc.tasks.procedural import ProceduralARCTask
    train_task = ProceduralARCTask(seq_len=config.max_seq_len, augment=True)
    print(f"  Training: procedural tasks (seq_len={config.max_seq_len})")

    real_arc_train = None
    eval_task = None
    if args.data_dir:
        try:
            from fgn.tasks.arc import ARCTask
            real_arc_train = ARCTask(
                seq_len=config.max_seq_len, data_dir=args.data_dir,
                split='train', augment=True,
            )
            eval_task = ARCTask(
                seq_len=config.max_seq_len, data_dir=args.data_dir,
                split='eval', augment=False,
            )
            print(f"  Real ARC mix: {args.real_arc_mix_ratio:.0%} of batches")
        except Exception as e:
            print(f"  ARC data unavailable ({e}), using procedural only")

    if eval_task is None:
        eval_task = ProceduralARCTask(seq_len=config.max_seq_len, augment=False)
        eval_task._seed_counter = 999999

    # ── TensorBoard ──
    tb = None
    if _tb_available:
        tb = SummaryWriter(os.path.join(args.output_dir, 'tb'))

    # ── Training ──
    model.train()
    step = 0
    t0 = time.time()
    log_metrics = {}

    print(f"\nTraining for {args.max_steps} steps...")
    print(f"{'Step':>6} | {'loss':>8} | {'xform%':>7} | {'CV':>6} | {'tau':>6} | "
          f"{'s_tau':>6} | {'tok/s':>7}")
    print("-" * 70)

    while step < args.max_steps:
        # ── Batch ──
        use_arc = (real_arc_train is not None and
                   random.random() < args.real_arc_mix_ratio)
        try:
            if use_arc:
                _, _, meta = real_arc_train.generate_batch(
                    batch_size=args.batch_size, device=device)
            else:
                _, _, meta = train_task.generate_batch(
                    batch_size=args.batch_size, device=device)
        except Exception as e:
            print(f"  Batch error at step {step}: {e}")
            continue

        # ── Forward ──
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            result = model(
                colors=meta['colors'],
                xs=meta['xs'],
                ys=meta['ys'],
                roles=meta['roles'],
                sep_mask=meta['sep_mask'],
                sep_types=meta['sep_types'],
                target_mask=meta['target_mask'],
                target_labels=meta.get('target_labels'),
                grid_ids=meta.get('grid_ids'),
            )

            ce_loss = result['ce_loss']
            curv_loss = result.get('curv_loss', torch.tensor(0.0))
            tau_var_loss = result.get('tau_var_loss', torch.tensor(0.0))
            cv_floor_loss = result.get('cv_floor_loss', torch.tensor(0.0))

            loss = ce_loss + curv_loss + tau_var_loss + cv_floor_loss

        # ── Backward ──
        optimizer.zero_grad()
        loss.backward()
        apply_structural_gradient_coupling(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        # ── Metrics accumulation ──
        with torch.no_grad():
            log_metrics['loss'] = log_metrics.get('loss', 0) + loss.item()
            log_metrics['ce'] = log_metrics.get('ce', 0) + ce_loss.item()
            log_metrics['xform'] = log_metrics.get('xform', 0) + result.get('transform_accuracy', 0)
            log_metrics['cv'] = log_metrics.get('cv', 0) + result.get('metric_cv', 0)
            log_metrics['tau'] = log_metrics.get('tau', 0) + result.get('tau_avg', 0)
            log_metrics['n'] = log_metrics.get('n', 0) + 1

        # ── Logging ──
        if step % args.log_every == 0:
            n = max(log_metrics['n'], 1)
            avg_loss = log_metrics['loss'] / n
            avg_xform = log_metrics['xform'] / n * 100
            avg_cv = log_metrics['cv'] / n
            avg_tau = log_metrics['tau'] / n

            # Structural tau stats
            s_tau_mean = 0.0
            if hasattr(model, 'dynamics') and hasattr(model.dynamics, 'structural_tau'):
                with torch.no_grad():
                    s_tau_mean = torch.sigmoid(model.dynamics.structural_tau).mean().item()
            elif hasattr(model, '_orig_mod'):
                dyn = model._orig_mod.dynamics
                if hasattr(dyn, 'structural_tau'):
                    with torch.no_grad():
                        s_tau_mean = torch.sigmoid(dyn.structural_tau).mean().item()

            elapsed = time.time() - t0
            toks_per_sec = (step * args.batch_size * config.max_seq_len) / max(elapsed, 1)

            print(f"{step:>6} | {avg_loss:>8.4f} | {avg_xform:>6.1f}% | {avg_cv:>6.2f} | "
                  f"{avg_tau:>6.3f} | {s_tau_mean:>6.3f} | {toks_per_sec:>7.0f}")

            if tb:
                tb.add_scalar('loss/total', avg_loss, step)
                tb.add_scalar('loss/ce', avg_ce := log_metrics['ce'] / n, step)
                tb.add_scalar('accuracy/xform_train', avg_xform, step)
                tb.add_scalar('metric/cv', avg_cv, step)
                tb.add_scalar('metric/tau', avg_tau, step)
                tb.add_scalar('metric/structural_tau_mean', s_tau_mean, step)
                tb.add_scalar('lr', scheduler.get_last_lr()[0], step)

            log_metrics = {}

        # ── Eval ──
        if step % args.eval_every == 0:
            model.eval()
            eval_metrics = {'xform': 0, 'cv': 0, 'tau': 0, 'n': 0}

            with torch.no_grad():
                for _ in range(args.eval_batches):
                    try:
                        _, _, emeta = eval_task.generate_batch(
                            batch_size=args.batch_size, device=device)
                        er = model(
                            colors=emeta['colors'], xs=emeta['xs'], ys=emeta['ys'],
                            roles=emeta['roles'], sep_mask=emeta['sep_mask'],
                            sep_types=emeta['sep_types'], target_mask=emeta['target_mask'],
                            target_labels=emeta.get('target_labels'),
                            grid_ids=emeta.get('grid_ids'),
                        )
                        eval_metrics['xform'] += er.get('transform_accuracy', 0)
                        eval_metrics['cv'] += er.get('metric_cv', 0)
                        eval_metrics['tau'] += er.get('tau_avg', 0)
                        eval_metrics['n'] += 1
                    except Exception:
                        pass

            n_eval = max(eval_metrics['n'], 1)
            eval_xform = eval_metrics['xform'] / n_eval * 100
            eval_cv = eval_metrics['cv'] / n_eval
            eval_tau = eval_metrics['tau'] / n_eval

            print(f"  EVAL step={step}: xform={eval_xform:.1f}% CV={eval_cv:.2f} tau={eval_tau:.3f}")

            if tb:
                tb.add_scalar('accuracy/xform_eval', eval_xform, step)
                tb.add_scalar('metric/cv_eval', eval_cv, step)

            # Log structural_tau per-position variance (key diagnostic)
            if hasattr(model, 'dynamics') and hasattr(model.dynamics, 'structural_tau'):
                with torch.no_grad():
                    s_raw = model.dynamics.structural_tau
                    s_vals = torch.sigmoid(s_raw)
                    print(f"  structural_tau: mean={s_vals.mean():.3f} "
                          f"std={s_vals.std():.3f} "
                          f"min={s_vals.min():.3f} max={s_vals.max():.3f}")
                    if tb:
                        tb.add_scalar('metric/structural_tau_std', s_vals.std().item(), step)
                        tb.add_histogram('metric/structural_tau_dist', s_vals, step)
                # Gradient-reachability check: if structural_tau receives 0
                # gradient across training, the loss is not wired to it.
                # Must be read BEFORE optimizer.zero_grad() in the training
                # loop, but at eval boundary we still have post-backward grads.
                if s_raw.grad is not None:
                    grad_norm = s_raw.grad.norm().item()
                    print(f"  structural_tau.grad.norm={grad_norm:.3e}")
                    if tb:
                        tb.add_scalar('metric/structural_tau_grad_norm',
                                       grad_norm, step)
                else:
                    print("  structural_tau.grad: None (not yet populated or detached)")

            model.train()

        # ── Checkpoint ──
        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f'step_{step}.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(args.output_dir, 'final.pt')
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")

    if tb:
        tb.close()


# ──────────────────── ENTRY POINT ────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LiquidARC v2 Geometry Distillation')
    parser.add_argument('--config', required=True,
                        help='Path to config YAML (use configs/liquid_arc_v2.yaml)')
    parser.add_argument('--data_dir', default=None,
                        help='Path to ARC data directory (for real ARC mixing and eval)')
    parser.add_argument('--teacher_checkpoint', default=None,
                        help='Path to post-transition teacher checkpoint')
    parser.add_argument('--geometry_init', default='weight_transfer',
                        choices=['weight_transfer', 'statistical', 'none'],
                        help='Geometry initialization method')
    parser.add_argument('--geometry_targets', default=None,
                        help='Path to geometry_targets.pt (for statistical init)')
    parser.add_argument('--output_dir', default='output_v2/seeded')
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--eval_batches', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--use_torch_compile', action='store_true', default=False,
                        help='torch.compile the model (requires --config use_torch_compile: true)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--real_arc_mix_ratio', type=float, default=0.3)
    args = parser.parse_args()

    train(args)
