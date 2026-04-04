"""LiquidARC Fluid Metric — Training Script.

Trains a LiquidARCModel with the Fluid Metric architecture (wider diagonal
bottleneck + optional low-rank off-diagonal factors), initialized from a
post-transition teacher checkpoint via shape-compatible weight transfer.

Key differences from train_v2.py:
  - MetricNet shapes differ (teacher: metric_net_linear2 64-dim bottleneck;
    student: metric_net_linear2_diag 256-dim bottleneck) — direct transfer
    is skipped for MetricNet output layer; a 500-step metric calibration
    phase adjusts the new MetricNet to match the teacher's CV statistics.
  - Two optimizer groups (no structural_tau group): geometric (0.01x LR)
    and content (1x LR).
  - Extra diagnostics: D_cv (diagonal metric CV), L_norm (low-rank factor
    norm), L_rank_usage (active low-rank dims).
  - No structural_tau code (removed entirely).
  - Stage A only: ARC task data.

Usage:
    python scripts/train_fluid_metric.py \\
      --config configs/liquid_arc_fluid.yaml \\
      --data_dir /workspace/fgn-v3/data/arc-repo/data \\
      --teacher_checkpoint /workspace/liquid-arc/PRECIOUS_CHECKPOINTS/5m_post_transition.pt \\
      --output_dir output_fluid/run1 \\
      --max_steps 10000

    # Resume from fluid metric checkpoint:
    python scripts/train_fluid_metric.py \\
      --config configs/liquid_arc_fluid.yaml \\
      --resume output_fluid/run1/step_5000.pt \\
      --output_dir output_fluid/run1 \\
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


# ──────────────────── WEIGHT TRANSFER ────────────────────

def transfer_compatible_weights(student: LiquidARCModel, teacher_checkpoint: str,
                                 device: str = 'cuda') -> tuple[int, int]:
    """Transfer all shape-compatible weights from teacher to student.

    MetricNet output layer (metric_net_linear2 / metric_net_linear2_diag) is
    SKIPPED when shapes differ — the caller must run metric calibration after
    this function to align the new bottleneck with teacher CV statistics.

    Teacher keys named ``metric_net_linear2.*`` are remapped to
    ``metric_net_linear2_diag.*`` before shape matching.

    All other shape-compatible weights (TauNet, t_diffusion, alpha_logit,
    W_v, W_o, norms, context_pool, FFN, embedding, head) are transferred
    directly.

    Args:
        student: student model (fluid metric architecture).
        teacher_checkpoint: path to teacher .pt checkpoint.
        device: device for loading.

    Returns:
        (n_transferred, n_skipped) parameter tensors.
    """
    ckpt = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))

    # Strip torch.compile prefix and remap teacher's metric_net_linear2 ->
    # metric_net_linear2_diag so matching works against the student.
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        k = k.replace("._orig_mod.", ".")
        k = k.replace("metric_net_linear2.", "metric_net_linear2_diag.")
        cleaned[k] = v

    student_params = dict(student.named_parameters())
    transferred = 0
    skipped = []

    for key, student_param in student_params.items():
        # Strip _orig_mod. from student key for matching against cleaned teacher keys
        clean_key = key.replace("_orig_mod.", "")
        if clean_key not in cleaned:
            skipped.append(f"{clean_key} (missing in teacher)")
            continue
        teacher_tensor = cleaned[clean_key]
        if student_param.shape != teacher_tensor.shape:
            skipped.append(
                f"{key} (shape mismatch: student {tuple(student_param.shape)} "
                f"vs teacher {tuple(teacher_tensor.shape)})"
            )
            continue
        student_param.data.copy_(teacher_tensor.to(device))
        transferred += 1

    print(f"  Weight transfer: {transferred} tensors transferred, "
          f"{len(skipped)} skipped")
    if skipped:
        # Show up to 5 skips; shape mismatches are expected for metric_net_linear2_diag
        display = skipped[:5]
        if len(skipped) > 5:
            display.append(f"... and {len(skipped) - 5} more")
        for s in display:
            print(f"    skip: {s}")

    return transferred, len(skipped)


# ──────────────────── METRIC CALIBRATION ────────────────────

def calibrate_metric_net(model: LiquidARCModel, teacher_checkpoint: str,
                          data_dir: str | None, config: LiquidARCConfig,
                          device: str = 'cuda', n_steps: int = 500,
                          batch_size: int = 4) -> float:
    """Per-position metric distillation: match teacher's full g(h) field.

    Loads the teacher model (original diagonal architecture) and for each
    batch, runs both models' embedding + context_pool to get h0, then
    computes teacher's per-position g [B,N,d] and trains the student's
    wider MetricNet to reproduce it via MSE.

    Only ``metric_net_linear1`` and ``metric_net_linear2_diag`` are optimized.
    ``metric_net_linear2_lr`` stays at zero — low-rank terms should only
    develop from task pressure, not from the teacher (which has no rotational
    geometry to transfer).

    Args:
        model: student model (already has compatible weights transferred).
        teacher_checkpoint: path to teacher checkpoint.
        data_dir: path to ARC data dir (for real batches) or None.
        config: student config.
        device: device.
        n_steps: number of calibration steps.
        batch_size: batch size for calibration.

    Returns:
        Final student CV after calibration.
    """
    print(f"\nMetric distillation: {n_steps} steps, per-position g matching...")

    # ── Load teacher model ──
    # Teacher uses the original config (from checkpoint) with diagonal metric
    ckpt = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    teacher_cfg = ckpt.get('config', None)
    if teacher_cfg is None:
        # Fallback: use student config but with original metric dims
        teacher_cfg = LiquidARCConfig(
            d_model=config.d_model, d_metric=64, d_ffn=config.d_ffn,
            max_seq_len=config.max_seq_len, n_ode_steps=config.n_ode_steps,
            n_colors=config.n_colors, n_roles=config.n_roles,
            n_sep_types=config.n_sep_types, max_grid_size=config.max_grid_size,
            max_grids=config.max_grids,
        )
    elif not isinstance(teacher_cfg, LiquidARCConfig):
        teacher_cfg = LiquidARCConfig(**teacher_cfg)
    # Force teacher to diagonal-only (no fluid metric)
    teacher_cfg.d_metric_bottleneck = 0
    teacher_cfg.metric_rank = 0

    teacher = LiquidARCModel(teacher_cfg).to(device)
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    # Remap for backward compat
    cleaned = {k.replace('metric_net_linear2.', 'metric_net_linear2_diag.'): v
               for k, v in cleaned.items()}
    teacher.load_state_dict(cleaned, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"  Teacher loaded: {sum(p.numel() for p in teacher.parameters()):,} params")

    # ── Student calibration params ──
    student_model = getattr(model, '_orig_mod', model) if hasattr(model, '_orig_mod') else model
    student_dyn = student_model.dynamics

    calib_params = (
        list(student_dyn.metric_net_linear1.parameters()) +
        list(student_dyn.metric_net_linear2_diag.parameters())
    )
    calib_optimizer = torch.optim.AdamW(calib_params, lr=1e-3, weight_decay=0.0)

    # ── Data ──
    from liquid_arc.tasks.procedural import ProceduralARCTask
    calib_task = ProceduralARCTask(seq_len=config.max_seq_len, augment=True)

    real_arc_task = None
    if data_dir:
        try:
            from fgn.tasks.arc import ARCTask
            real_arc_task = ARCTask(
                seq_len=config.max_seq_len, data_dir=data_dir,
                split='train', augment=True,
            )
        except Exception:
            pass

    model.train()
    cv_history = []
    loss_history = []

    for step in range(1, n_steps + 1):
        use_arc = (real_arc_task is not None and random.random() < 0.3)
        try:
            if use_arc:
                _, _, meta = real_arc_task.generate_batch(batch_size=batch_size, device=device)
            else:
                _, _, meta = calib_task.generate_batch(batch_size=batch_size, device=device)
        except Exception as e:
            print(f"  Calibration batch error step {step}: {e}")
            continue

        fwd_kwargs = dict(
            colors=meta['colors'], xs=meta['xs'], ys=meta['ys'],
            roles=meta['roles'], sep_mask=meta['sep_mask'],
            sep_types=meta['sep_types'], target_mask=meta['target_mask'],
            target_labels=meta.get('target_labels'),
            grid_ids=meta.get('grid_ids'),
        )

        # ── Teacher: embed + compute per-position g target ──
        # Replicate the model's embedding path inline (no embed() method)
        PAD_COLOR = 0
        with torch.no_grad():
            colors_masked = meta['colors'].clone()
            colors_masked[meta['target_mask']] = PAD_COLOR

            h0_t = teacher.embedding(
                colors_masked, meta['xs'], meta['ys'], meta['roles'],
                meta['sep_mask'], meta['sep_types'],
                grid_ids=meta.get('grid_ids'),
            )
            ctx_t = teacher.context_pool(h0_t)
            teacher.dynamics.set_context(ctx_t)
            g_teacher = teacher.dynamics.compute_metric_diag(h0_t)  # [B, N, d]

        # ── Student: same embedding path, compute metric ──
        # Embedding/context_pool weights were transferred, so h0 should be close
        h0_s = student_model.embedding(
            colors_masked, meta['xs'], meta['ys'], meta['roles'],
            meta['sep_mask'], meta['sep_types'],
            grid_ids=meta.get('grid_ids'),
        )
        ctx_s = student_model.context_pool(h0_s)
        student_dyn.set_context(ctx_s)
        g_student = student_dyn.compute_metric_diag(h0_s)  # [B, N, d]

        # Per-position MSE on the metric field
        metric_loss = F.mse_loss(g_student, g_teacher.detach())

        calib_optimizer.zero_grad()
        metric_loss.backward()
        torch.nn.utils.clip_grad_norm_(calib_params, 1.0)
        calib_optimizer.step()

        student_cv = (g_student.std() / (g_student.mean() + 1e-8)).item()
        teacher_cv = (g_teacher.std() / (g_teacher.mean() + 1e-8)).item()
        cv_history.append(student_cv)
        loss_history.append(metric_loss.item())

        if step % 100 == 0 or step == n_steps:
            recent_cv = sum(cv_history[-20:]) / max(len(cv_history[-20:]), 1)
            recent_loss = sum(loss_history[-20:]) / max(len(loss_history[-20:]), 1)
            print(f"  Distill step {step}/{n_steps}: student_CV={recent_cv:.3f} "
                  f"teacher_CV={teacher_cv:.3f} metric_mse={recent_loss:.5f}")

    final_cv = sum(cv_history[-20:]) / max(len(cv_history[-20:]), 1)
    final_loss = sum(loss_history[-20:]) / max(len(loss_history[-20:]), 1)
    print(f"  Metric distillation done. CV={final_cv:.3f}, MSE={final_loss:.5f}")
    if final_cv > 3.0:
        print("  Geometry in post-transition regime (CV > 3)")
    else:
        print("  WARNING: CV low — metric field may not have transferred correctly")

    # ── Verify low-rank stays at zero ──
    if hasattr(student_dyn, 'metric_net_linear2_lr'):
        lr_norm = student_dyn.metric_net_linear2_lr.weight.norm().item()
        print(f"  Low-rank weight norm: {lr_norm:.6f} (should be ~0)")

    # Free teacher
    del teacher
    torch.cuda.empty_cache()

    return final_cv


# ──────────────────── OPTIMIZER ────────────────────

def build_optimizer(model: LiquidARCModel, base_lr: float, weight_decay: float):
    """Two-group optimizer: geometric (slow) and content (normal).

    Geometric group (0.01x LR):
        metric_net_linear1, metric_net_linear2_diag, metric_net_linear2_lr,
        tau_net_*, t_diffusion, alpha_logit, context_pool

    Content group (1x LR):
        everything else (FFN, W_v, W_o, embedding, head, norms)

    No structural_tau group.
    """
    geo_param_ids: set[int] = set()

    geo_prefixes = [
        'dynamics.metric_net_linear1',
        'dynamics.metric_net_linear2_diag',
        'dynamics.metric_net_linear2_lr',
        'dynamics.tau_net_linear1',
        'dynamics.tau_net_linear2',
        'context_pool.',
    ]
    geo_names = {'dynamics.t_diffusion', 'dynamics.alpha_logit'}

    geo_params = []
    for name, p in model.named_parameters():
        clean = name.replace('_orig_mod.', '')
        if any(clean.startswith(pfx) for pfx in geo_prefixes) or clean in geo_names:
            geo_params.append(p)
            geo_param_ids.add(id(p))

    content_params = [p for p in model.parameters() if id(p) not in geo_param_ids]

    geo_lr = base_lr * 0.01
    param_groups = [
        {'params': content_params, 'lr': base_lr,  'name': 'content'},
        {'params': geo_params,     'lr': geo_lr,    'name': 'geometric'},
    ]

    optimizer = torch.optim.AdamW(
        [g for g in param_groups if g['params']],
        weight_decay=weight_decay,
    )

    n_content = sum(p.numel() for p in content_params)
    n_geo = sum(p.numel() for p in geo_params)
    print(f"  Optimizer groups: content={n_content:,}  geometric={n_geo:,}")
    print(f"  LR: content={base_lr:.2e}  geometric={geo_lr:.2e}")

    return optimizer


def make_scheduler(optimizer, base_lr: float, warmup_steps: int, max_steps: int):
    """Cosine decay with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────── FLUID METRIC DIAGNOSTICS ────────────────────

def compute_fluid_metric_diagnostics(model: LiquidARCModel, h: torch.Tensor,
                                      prev_L_norm: float | None = None
                                      ) -> dict[str, float]:
    """Compute fluid metric-specific diagnostics from a hidden state sample.

    Args:
        model: student model (may be torch.compile wrapped).
        h: [B, N, d] sample hidden state (detached).
        prev_L_norm: L_norm from the previous eval (to detect growth).

    Returns:
        dict with keys: D_cv, L_norm, L_rank_usage, L_norm_grew
    """
    dyn = getattr(model, '_orig_mod', model).dynamics if hasattr(model, '_orig_mod') \
        else model.dynamics

    diag = {}
    with torch.no_grad():
        try:
            result = dyn.compute_metric(h)
            if isinstance(result, tuple):
                D, L = result  # D: [B,N,d], L: [B,N,d,rank]
                # Diagonal CV: coefficient of variation of the diagonal metric
                d_mean = D.mean()
                d_std = D.std()
                diag['D_cv'] = (d_std / (d_mean + 1e-8)).item()
                # Low-rank factor norm: mean per-position Frobenius norm
                diag['L_norm'] = L.reshape(-1).norm().item() / (D.shape[0] * D.shape[1])
                # Active rank: columns of L with mean norm > 1% of max column norm
                # L is [B,N,d,rank] — norm across (B,N,d) for each rank dim
                col_norms = L.reshape(-1, L.shape[-1]).norm(dim=0)  # [rank]
                max_col = col_norms.max().item()
                threshold = max(max_col * 0.01, 1e-6)
                diag['L_rank_usage'] = (col_norms > threshold).sum().item()
                # Growth flag (eval only)
                if prev_L_norm is not None:
                    diag['L_norm_grew'] = float(diag['L_norm'] > prev_L_norm * 1.05)
            else:
                D = result  # diagonal-only model
                d_mean = D.mean()
                d_std = D.std()
                diag['D_cv'] = (d_std / (d_mean + 1e-8)).item()
                diag['L_norm'] = 0.0
                diag['L_rank_usage'] = 0.0
                if prev_L_norm is not None:
                    diag['L_norm_grew'] = 0.0
        except Exception as e:
            diag['D_cv'] = 0.0
            diag['L_norm'] = 0.0
            diag['L_rank_usage'] = 0.0
            if prev_L_norm is not None:
                diag['L_norm_grew'] = 0.0
            print(f"  WARNING: fluid metric diagnostics failed: {e}")

    return diag


# ──────────────────── TRAINING LOOP ────────────────────

def train(args):
    config = LiquidARCConfig.from_yaml(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model (compile AFTER weight transfer + calibration) ──
    model = LiquidARCModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    # ── Resume OR fresh init (before compile) ──
    start_step = 0
    _resume_optimizer_state = None
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        # Strip torch.compile prefix and remap metric_net_linear2 -> metric_net_linear2_diag
        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("_orig_mod.", "")
            k = k.replace("metric_net_linear2.", "metric_net_linear2_diag.")
            cleaned[k] = v
        # Drop low-rank weights from checkpoint so model keeps fresh random init
        # (zero-init L from Stage A would be a gradient trap)
        cleaned = {k: v for k, v in cleaned.items()
                   if 'metric_net_linear2_lr' not in k}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            # Filter out expected missing keys (lr layer)
            real_missing = [k for k in missing if 'metric_net_linear2_lr' not in k]
            if real_missing:
                print(f"  Missing keys: {real_missing[:5]}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}")
        _resume_optimizer_state = ckpt.get('optimizer_state_dict')
        start_step = ckpt.get('step', 0)
        print(f"  Resumed from step {start_step} (L weights re-randomized)")

    elif args.teacher_checkpoint:
        # Fresh init from teacher weights
        print(f"\nTransferring weights from teacher: {args.teacher_checkpoint}")
        transfer_compatible_weights(model, args.teacher_checkpoint, device)

        # Metric calibration phase (500 steps)
        calibrate_metric_net(
            model, args.teacher_checkpoint,
            data_dir=args.data_dir,
            config=config,
            device=device,
            n_steps=500,
            batch_size=args.batch_size,
        )

        # Verification forward pass
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
        print(f"  Post-init check: CV={init_cv:.3f}")
        if init_cv > 3.0:
            print("  Geometry in post-transition regime (CV > 3)")
        else:
            print("  WARNING: CV low — geometry may not have calibrated correctly")

    else:
        print("No teacher checkpoint and no resume — training from random init")

    # ── torch.compile (after weight transfer + calibration) ──
    if args.use_torch_compile and config.use_torch_compile:
        print("torch.compile: compiling model...")
        model = torch.compile(model)
        print("torch.compile: done")

    # ── Optimizer + Scheduler ──
    base_lr = getattr(config, 'base_lr', 3e-4)
    warmup_steps = getattr(config, 'warmup_steps', 500)
    weight_decay = getattr(config, 'weight_decay', 0.01)

    optimizer = build_optimizer(model, base_lr, weight_decay)
    if _resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(_resume_optimizer_state)
            print("  Optimizer state restored from checkpoint")
        except Exception as e:
            print(f"  Optimizer state not restored: {e}")
    # NOTE: scheduler created AFTER text param group is added (see below)

    # ── Data (Stage A: ARC only) ──
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

    # ── Text data (Stage B: multi-domain) ──
    text_task = None
    text_embed_module = None
    text_head_module = None
    text_mix_ratio = args.text_mix_ratio
    text_loss_weight = args.text_loss_weight

    if text_mix_ratio > 0:
        from liquid_arc.tasks.text_task import TextTask, TextEmbedding, TextHead
        text_task = TextTask(seq_len=config.max_seq_len, split='train')
        text_embed_module = TextEmbedding(
            vocab_size=text_task.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        ).to(device)
        text_head_module = TextHead(
            d_model=config.d_model,
            vocab_size=text_task.vocab_size,
        ).to(device)
        # Add text modules to optimizer (content LR)
        text_params = list(text_embed_module.parameters()) + list(text_head_module.parameters())
        optimizer.add_param_group({'params': text_params, 'lr': base_lr, 'name': 'text'})
        n_text = sum(p.numel() for p in text_params)
        print(f"  Text enabled: mix={text_mix_ratio:.0%}, weight={text_loss_weight}, "
              f"vocab={text_task.vocab_size}, text_params={n_text:,}")

    # ── Scheduler (after all param groups added) ──
    scheduler = make_scheduler(optimizer, base_lr, warmup_steps, args.max_steps)
    if start_step > 0:
        for _ in range(start_step):
            scheduler.step()

    # ── TensorBoard ──
    tb = None
    if _tb_available:
        tb = SummaryWriter(os.path.join(args.output_dir, 'tb'))

    # ── Training ──
    model.train()
    step = start_step
    t0 = time.time()
    log_metrics: dict[str, float] = {}
    last_eval_L_norm: float | None = None
    _last_h_final = None  # cached from last ARC step for diagnostics

    print(f"\nTraining for {args.max_steps} steps (starting from step {start_step})...")
    print(f"{'Step':>6} | {'loss':>8} | {'xform%':>7} | {'CV':>6} | "
          f"{'D_cv':>6} | {'tau':>6} | {'L_norm':>7} | {'tok/s':>7}")
    print("-" * 80)

    # Access raw model for text forward (unwrap torch.compile)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    while step < args.max_steps:
        is_text_step = (text_task is not None and random.random() < text_mix_ratio)

        if is_text_step:
            # ── Text batch: embed → shared ODE → text head → CE ──
            try:
                text_input_ids, text_target_ids = text_task.generate_batch(
                    batch_size=args.batch_size, device=device)
            except Exception as e:
                print(f"  Text batch error at step {step}: {e}")
                continue

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                h0 = text_embed_module(text_input_ids)  # [B, T, d]
                context = raw_model.context_pool(h0)
                raw_model.dynamics.set_context(context, mask=None)
                raw_model.dynamics.set_n_steps(config.n_ode_steps)

                # Euler ODE integration (same dynamics as ARC)
                dt = 2.0 / config.n_ode_steps
                h = h0
                t_ode = 0.0
                for ode_step in range(config.n_ode_steps):
                    raw_model.dynamics.set_step_index(ode_step, config.n_ode_steps)
                    dh = raw_model.dynamics(t_ode, h)
                    h = h + dt * dh
                    t_ode += dt

                text_logits = text_head_module(h)  # [B, T, vocab]
                text_ce = F.cross_entropy(
                    text_logits.view(-1, text_logits.size(-1)),
                    text_target_ids.view(-1),
                )
                loss = text_loss_weight * text_ce

                # Text diagnostics
                text_ppl = torch.exp(text_ce).item()
                _cv_val = 0.0
                _tau_val = 0.0
                _xform_val = 0.0
                _loss_val = loss.item()
                _ce_val = text_ce.item()

        else:
            # ── ARC batch (same as Stage A) ──
            use_real_arc = (real_arc_train is not None and
                            random.random() < args.real_arc_mix_ratio)
            try:
                if use_real_arc:
                    _, _, meta = real_arc_train.generate_batch(
                        batch_size=args.batch_size, device=device)
                else:
                    _, _, meta = train_task.generate_batch(
                        batch_size=args.batch_size, device=device)
            except Exception as e:
                print(f"  Batch error at step {step}: {e}")
                continue

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

                _loss_val = loss.item()
                _ce_val = ce_loss.item()
                _xform_val = float(result.get('transform_accuracy', 0.0))
                _cv_val = float(result.get('metric_cv', 0.0))
                _tau_val = float(result.get('tau_avg', 0.0))
                _last_h_final = result.get('h_final')

        # ── Backward ──
        optimizer.zero_grad()
        loss.backward()
        # Scrub NaN gradients (pre-existing bfloat16 SDPA backward issue at d=768)
        all_params = list(model.parameters())
        if text_embed_module is not None:
            all_params += list(text_embed_module.parameters()) + list(text_head_module.parameters())
        for p in all_params:
            if p.grad is not None and p.grad.isnan().any():
                p.grad.nan_to_num_(nan=0.0)
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        # Debug first steps
        if step <= 3:
            src = "text" if is_text_step else "arc"
            print(f"  DEBUG step {step} [{src}]: loss={_loss_val:.4f} "
                  f"ce={_ce_val:.4f} cv={_cv_val:.3f} "
                  f"tau={_tau_val:.3f} xform={_xform_val:.3f}")

        # ── Metrics accumulation ──
        log_metrics['loss'] = log_metrics.get('loss', 0.0) + _loss_val
        log_metrics['ce'] = log_metrics.get('ce', 0.0) + _ce_val
        log_metrics['xform'] = log_metrics.get('xform', 0.0) + _xform_val
        log_metrics['cv'] = log_metrics.get('cv', 0.0) + _cv_val
        log_metrics['tau'] = log_metrics.get('tau', 0.0) + _tau_val
        log_metrics['n'] = log_metrics.get('n', 0.0) + 1.0

        # ── Logging ──
        if step % args.log_every == 0:
            n = max(log_metrics['n'], 1)
            avg_loss = log_metrics['loss'] / n
            avg_xform = log_metrics['xform'] / n * 100
            avg_cv = log_metrics['cv'] / n
            avg_tau = log_metrics['tau'] / n

            elapsed = time.time() - t0
            toks_per_sec = (
                (step - start_step) * args.batch_size * config.max_seq_len
            ) / max(elapsed, 1)

            # Fluid metric diagnostics (requires a hidden state sample)
            # We get these from the last forward result if the model exposes them,
            # otherwise we skip to avoid an extra forward pass every log step.
            d_cv_diag = 0.0
            l_norm = 0.0
            l_rank_usage = 0.0

            # Check if result contains hidden states for diagnostics (from last ARC step)
            _h_sample = _last_h_final
            if _h_sample is not None:
                _fd = compute_fluid_metric_diagnostics(model, _h_sample.detach())
                d_cv_diag = _fd.get('D_cv', 0.0)
                l_norm = _fd.get('L_norm', 0.0)
                l_rank_usage = _fd.get('L_rank_usage', 0.0)

            print(f"{step:>6} | {avg_loss:>8.4f} | {avg_xform:>6.1f}% | {avg_cv:>6.2f} | "
                  f"{d_cv_diag:>6.2f} | {avg_tau:>6.3f} | {l_norm:>7.4f} | {toks_per_sec:>7.0f}")

            if tb:
                tb.add_scalar('loss/total', avg_loss, step)
                tb.add_scalar('loss/ce', log_metrics['ce'] / n, step)
                tb.add_scalar('accuracy/xform_train', avg_xform, step)
                tb.add_scalar('metric/cv', avg_cv, step)
                tb.add_scalar('metric/D_cv', d_cv_diag, step)
                tb.add_scalar('metric/tau', avg_tau, step)
                tb.add_scalar('metric/L_norm', l_norm, step)
                tb.add_scalar('metric/L_rank_usage', l_rank_usage, step)
                tb.add_scalar('lr', scheduler.get_last_lr()[0], step)

            log_metrics = {}

        # ── Eval ──
        if step % args.eval_every == 0:
            model.eval()
            eval_metrics: dict[str, float] = {
                'xform': 0.0, 'cv': 0.0, 'tau': 0.0,
                'D_cv': 0.0, 'L_norm': 0.0, 'L_rank_usage': 0.0, 'n': 0.0,
            }

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
                        eval_metrics['xform'] += er.get('transform_accuracy', 0.0)
                        eval_metrics['cv'] += er.get('metric_cv', 0.0)
                        eval_metrics['tau'] += er.get('tau_avg', 0.0)

                        # Fluid metric diagnostics from eval batch
                        _h_eval = er.get('h_final')
                        if _h_eval is not None:
                            _fd = compute_fluid_metric_diagnostics(
                                model, _h_eval.detach(), prev_L_norm=last_eval_L_norm,
                            )
                            eval_metrics['D_cv'] += _fd.get('D_cv', 0.0)
                            eval_metrics['L_norm'] += _fd.get('L_norm', 0.0)
                            eval_metrics['L_rank_usage'] += _fd.get('L_rank_usage', 0.0)

                        eval_metrics['n'] += 1.0
                    except Exception:
                        pass

            n_eval = max(eval_metrics['n'], 1.0)
            eval_xform = eval_metrics['xform'] / n_eval * 100
            eval_cv = eval_metrics['cv'] / n_eval
            eval_tau = eval_metrics['tau'] / n_eval
            eval_D_cv = eval_metrics['D_cv'] / n_eval
            eval_L_norm = eval_metrics['L_norm'] / n_eval
            eval_L_rank = eval_metrics['L_rank_usage'] / n_eval

            # Detect L_norm growth
            l_norm_grew = ""
            if last_eval_L_norm is not None and eval_L_norm > last_eval_L_norm * 1.05:
                l_norm_grew = " [L_norm GREW]"
            last_eval_L_norm = eval_L_norm if eval_L_norm > 0 else last_eval_L_norm

            # Text eval: perplexity on validation split
            text_ppl_str = ""
            if text_task is not None and text_embed_module is not None:
                text_embed_module.eval()
                text_head_module.eval()
                text_losses = []
                with torch.no_grad():
                    for _ in range(5):  # 5 text eval batches
                        try:
                            t_ids, t_tgt = text_task.generate_batch(
                                batch_size=args.batch_size, device=device)
                            h0_t = text_embed_module(t_ids)
                            ctx_t = raw_model.context_pool(h0_t)
                            raw_model.dynamics.set_context(ctx_t, mask=None)
                            raw_model.dynamics.set_n_steps(config.n_ode_steps)
                            dt = 2.0 / config.n_ode_steps
                            h_t = h0_t
                            t_ode = 0.0
                            for ode_s in range(config.n_ode_steps):
                                raw_model.dynamics.set_step_index(ode_s, config.n_ode_steps)
                                dh = raw_model.dynamics(t_ode, h_t)
                                h_t = h_t + dt * dh
                                t_ode += dt
                            t_logits = text_head_module(h_t)
                            t_ce = F.cross_entropy(
                                t_logits.view(-1, t_logits.size(-1)),
                                t_tgt.view(-1),
                            )
                            text_losses.append(t_ce.item())
                        except Exception:
                            pass
                if text_losses:
                    avg_text_ce = sum(text_losses) / len(text_losses)
                    text_ppl = torch.exp(torch.tensor(avg_text_ce)).item()
                    text_ppl_str = f" text_ppl={text_ppl:.1f}"
                    if tb:
                        tb.add_scalar('text/perplexity_eval', text_ppl, step)
                        tb.add_scalar('text/ce_eval', avg_text_ce, step)
                text_embed_module.train()
                text_head_module.train()

            print(
                f"  EVAL step={step}: xform={eval_xform:.1f}% CV={eval_cv:.2f} "
                f"tau={eval_tau:.3f} D_cv={eval_D_cv:.2f} "
                f"L_norm={eval_L_norm:.4f} L_rank={eval_L_rank:.0f}"
                f"{l_norm_grew}{text_ppl_str}"
            )

            if tb:
                tb.add_scalar('accuracy/xform_eval', eval_xform, step)
                tb.add_scalar('metric/cv_eval', eval_cv, step)
                tb.add_scalar('metric/D_cv_eval', eval_D_cv, step)
                tb.add_scalar('metric/L_norm_eval', eval_L_norm, step)
                tb.add_scalar('metric/L_rank_usage_eval', eval_L_rank, step)

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

    # ── Final checkpoint ──
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
    parser = argparse.ArgumentParser(
        description='LiquidARC Fluid Metric Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', required=True,
                        help='Path to config YAML (e.g. configs/liquid_arc_fluid.yaml)')
    parser.add_argument('--data_dir', default=None,
                        help='Path to ARC data directory (real ARC mixing + eval)')
    parser.add_argument('--teacher_checkpoint', default=None,
                        help='Path to post-transition teacher checkpoint for weight transfer')
    parser.add_argument('--resume', default=None,
                        help='Path to fluid metric checkpoint to resume from')
    parser.add_argument('--output_dir', default='output_fluid/run1')
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--eval_batches', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--use_torch_compile', action='store_true', default=False,
                        help='torch.compile the model (requires config.use_torch_compile=true)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--real_arc_mix_ratio', type=float, default=0.3)
    parser.add_argument('--text_mix_ratio', type=float, default=0.0,
                        help='Fraction of steps that use text data (0=Stage A, 0.3=Stage B)')
    parser.add_argument('--text_loss_weight', type=float, default=0.1,
                        help='Weight for text CE loss relative to ARC loss')
    args = parser.parse_args()

    if args.resume and args.teacher_checkpoint:
        parser.error('--resume and --teacher_checkpoint are mutually exclusive')

    train(args)
