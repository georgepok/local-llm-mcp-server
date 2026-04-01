"""Latent Oracle training — topological knowledge distillation for LiquidARC.

3-phase distillation from precomputed oracle embeddings:
  Phase 0 (0–warmup): Projection head only (CE loss, align oracle→context space)
  Phase 1 (warmup–distill_end): Proj + ODE (CE + λ_κ·MSE(|κ|, κ_target))
  Phase 2 (distill_end–end): Same params (CE + 0.1·λ_κ·MSE, internalize priors)

Usage:
    python scripts/train.py \
        --config configs/latent_oracle.yaml \
        --ode_checkpoint /workspace/liquid-arc/output_30m/checkpoints/best.pt \
        --embeddings /workspace/latent-oracle/embeddings.pt \
        --data_dir /workspace/fgn-v3/data/arc \
        --output_dir /workspace/latent-oracle/output_v1
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch

# Path setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_LIQUID_ARC_ROOT = str(Path(__file__).resolve().parent.parent.parent / "liquid-arc")
if _LIQUID_ARC_ROOT not in sys.path:
    sys.path.insert(0, _LIQUID_ARC_ROOT)

_FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if _FGN_ROOT not in sys.path:
    sys.path.insert(0, _FGN_ROOT)

from latent_oracle.config import LatentOracleConfig
from latent_oracle.oracle_distill import OracleDistillLoss
from latent_oracle.oracle_hypernet import OracleHyperNet
from latent_oracle.projection import OracleProjectionHead
from latent_oracle.train_utils import OracleArcDataset, forward_with_oracle
from liquid_arc.model import LiquidARCModel


def create_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_phase(step: int, config: LatentOracleConfig) -> int:
    """Determine training phase from step count."""
    if step < config.warmup_steps:
        return 0
    elif step < config.distill_end_step:
        return 1
    else:
        return 2


def get_lambda_kappa(step: int, config: LatentOracleConfig) -> float:
    """Kappa distillation weight, decayed in Phase 2."""
    phase = get_phase(step, config)
    if phase == 0:
        return 0.0  # no kappa loss during projection warmup
    elif phase == 1:
        return config.lambda_kappa
    else:
        return config.lambda_kappa * config.lambda_kappa_decay


def get_lambda_distill(step: int, config: LatentOracleConfig) -> float:
    """Representation distillation weight with linear ramp."""
    if config.lambda_distill <= 0 or not config.similarity_path:
        return 0.0
    if step < config.distill_ramp_start:
        return 0.0
    if step >= config.distill_ramp_end:
        return config.lambda_distill
    # Linear ramp
    progress = (step - config.distill_ramp_start) / max(1, config.distill_ramp_end - config.distill_ramp_start)
    return config.lambda_distill * progress


def load_ode_checkpoint(path: str, model: LiquidARCModel, device: torch.device):
    """Load 5M LiquidARC checkpoint, stripping _orig_mod. prefix from torch.compile."""
    print(f"  Loading ODE checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("._orig_mod.", ".").replace("_orig_mod.", "")] = v
    model.load_state_dict(cleaned, strict=False)
    step = ckpt.get("step", 0)
    print(f"  Loaded at step {step}")
    return model


def evaluate_quick(model, projection, dataset, config, device,
                   hypernet=None, n_batches=10):
    """Quick eval: cell acc, xform acc, CE, kappa distill on eval split."""
    model.eval()
    projection.eval()
    if hypernet is not None:
        hypernet.eval()
    total_correct = 0
    total_cells = 0
    total_xform_correct = 0
    total_xform_cells = 0
    total_ce = 0.0
    total_kappa_distill = 0.0
    n_valid = 0

    with torch.no_grad():
        for _ in range(n_batches):
            try:
                oracle_embs, batch = dataset.sample_batch(
                    config.batch_size, "eval", device
                )
            except RuntimeError:
                continue

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                z_context, kappa_target = projection(oracle_embs)
                delta_W_o = hypernet(z_context) if hypernet is not None else None
                result = forward_with_oracle(
                    model, z_context, kappa_target,
                    colors=batch["colors"],
                    xs=batch["xs"],
                    ys=batch["ys"],
                    roles=batch["roles"],
                    sep_mask=batch["sep_mask"],
                    sep_types=batch["sep_types"],
                    target_mask=batch["target_mask"],
                    target_labels=batch["target_labels"],
                    context_mask=batch["context_mask"],
                    grid_ids=batch.get("grid_ids"),
                    target_input_colors=batch.get("target_input_colors"),
                    delta_W_o=delta_W_o,
                )

            cell_acc = result.get("cell_accuracy", torch.tensor(0.0))
            if isinstance(cell_acc, torch.Tensor):
                cell_acc = cell_acc.item()
            n_tgt = (batch["target_labels"] != -100).sum().item()
            total_correct += int(cell_acc * n_tgt)
            total_cells += n_tgt

            xform_acc = result.get("transform_accuracy", torch.tensor(0.0))
            if isinstance(xform_acc, torch.Tensor):
                xform_acc = xform_acc.item()
            n_xform = result.get("n_transform", torch.tensor(0))
            if isinstance(n_xform, torch.Tensor):
                n_xform = n_xform.item()
            total_xform_correct += int(xform_acc * n_xform)
            total_xform_cells += n_xform

            total_ce += result["ce_loss"].item()
            kd = result.get("kappa_distill_loss", torch.tensor(0.0))
            if isinstance(kd, torch.Tensor):
                kd = kd.item()
            total_kappa_distill += kd
            n_valid += 1

    cell_acc = total_correct / max(total_cells, 1)
    xform_acc = total_xform_correct / max(total_xform_cells, 1)
    avg_ce = total_ce / max(n_valid, 1)
    avg_kd = total_kappa_distill / max(n_valid, 1)
    return cell_acc, xform_acc, avg_ce, avg_kd


def save_checkpoint(model, projection, optimizer, config, step, path,
                    hypernet=None, extra=None):
    """Save checkpoint with model, projection head, hypernet, and optimizer."""
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "projection": projection.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    if hypernet is not None:
        ckpt["hypernet"] = hypernet.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config = LatentOracleConfig.from_yaml(args.config)
    config.oracle_embeddings_path = args.embeddings
    if args.similarity:
        config.similarity_path = args.similarity

    print(f"\n{'='*70}")
    print(f"Latent Oracle — Topological Knowledge Distillation")
    print(f"{'='*70}")

    # Create model
    model = LiquidARCModel(config).to(device)

    # Load pretrained 5M checkpoint
    if args.ode_checkpoint:
        model = load_ode_checkpoint(args.ode_checkpoint, model, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # Validate oracle embeddings
    emb_check = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    oracle_dim = emb_check["oracle_dim"]
    n_embs = emb_check["embeddings"].shape[0]
    assert oracle_dim == config.oracle_dim, \
        f"Oracle dim mismatch: embeddings={oracle_dim}, config={config.oracle_dim}"
    print(f"  Oracle embeddings: {n_embs} × {oracle_dim}")
    del emb_check

    # Create projection head
    projection = OracleProjectionHead(
        oracle_dim=config.oracle_dim,
        d_model=config.d_model,
        d_hidden=config.proj_d_hidden,
    ).to(device)
    n_proj = sum(p.numel() for p in projection.parameters())
    print(f"  Projection head: {n_proj:,} params")

    # Create Oracle HyperNet (task-specific W_o deltas) if enabled
    oracle_hypernet = None
    if config.oracle_hypernet_enabled:
        oracle_hypernet = OracleHyperNet(
            d_model=config.d_model,
            task_dim=config.hypernet_task_dim,
            rank=config.hypernet_rank,
            scale_init=config.hypernet_scale_init,
        ).to(device)
        n_hyper = sum(p.numel() for p in oracle_hypernet.parameters())
        print(f"  Oracle HyperNet: {n_hyper:,} params (task_dim={config.hypernet_task_dim}, "
              f"rank={config.hypernet_rank})")
    else:
        print(f"  Oracle HyperNet: DISABLED")

    # Dataset
    dataset = OracleArcDataset(
        embeddings_path=args.embeddings,
        data_dir=args.data_dir,
        max_seq_len=config.max_seq_len,
        similarity_path=config.similarity_path if hasattr(config, "similarity_path") else "",
    )

    # Representation distillation loss
    distill_loss_fn = None
    if config.similarity_path and dataset.has_similarities:
        distill_loss_fn = OracleDistillLoss()
        print(f"  Oracle distillation: ENABLED (λ={config.lambda_distill}, "
              f"ramp {config.distill_ramp_start}→{config.distill_ramp_end})")
    else:
        print(f"  Oracle distillation: DISABLED")

    # Output directory
    out_dir = args.output_dir
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    # Logging
    log_path = os.path.join(out_dir, "train.log")
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)

    import builtins
    _orig_print = builtins.print
    def _log_print(*a, **kw):
        msg = " ".join(str(x) for x in a)
        logger.info(msg)
    builtins.print = _log_print

    # Tensorboard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(out_dir, "logs"))
    except ImportError:
        writer = None
        print("  TensorBoard not available, skipping")

    # Optimizer: param groups
    # Group 0: projection head + hypernet (always trainable, same LR)
    # Group 1: ODE params (unfrozen in Phase 1+)
    # Group 2: other model params
    ode_params = list(model.dynamics.parameters()) + \
                 list(model.context_pool.parameters())
    ode_param_ids = {id(p) for p in ode_params}
    other_model_params = [p for p in model.parameters() if id(p) not in ode_param_ids]

    proj_params = list(projection.parameters())
    if oracle_hypernet is not None:
        proj_params += list(oracle_hypernet.parameters())

    optimizer = torch.optim.AdamW([
        {"params": proj_params, "lr": config.proj_lr, "name": "projection"},
        {"params": ode_params, "lr": config.ode_lr, "name": "ode"},
        {"params": other_model_params, "lr": config.ode_lr, "name": "other"},
    ], weight_decay=config.weight_decay)

    scheduler = create_scheduler(optimizer, config.warmup_steps, config.max_steps)

    # torch.compile dynamics
    if config.use_torch_compile and device.type == "cuda":
        model.dynamics = torch.compile(model.dynamics, mode="default", dynamic=True)
        print(f"  torch.compile: dynamics compiled")

    # Phase info
    print(f"\n  Training phases:")
    print(f"    Phase 0 (warmup):   steps 0–{config.warmup_steps} — projection only, CE")
    print(f"    Phase 1 (distill):  steps {config.warmup_steps}–{config.distill_end_step} "
          f"— proj+ODE, CE + λ_κ={config.lambda_kappa}")
    print(f"    Phase 2 (finetune): steps {config.distill_end_step}–{config.max_steps} "
          f"— same, λ_κ×{config.lambda_kappa_decay}")
    print()

    # Tau freeze from LiquidARC config
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model.dynamics.freeze_tau = False  # start with tau unfrozen (pretrained)

    model.train()
    projection.train()
    if oracle_hypernet is not None:
        oracle_hypernet.train()
    t0 = time.time()
    best_eval_xform = 0.0
    prev_phase = -1

    for step in range(config.max_steps):
        phase = get_phase(step, config)
        lk = get_lambda_kappa(step, config)

        # Phase transition logging
        if phase != prev_phase:
            phase_names = {0: "WARMUP (projection only)", 1: "DISTILL (proj + ODE)",
                           2: "FINE-TUNE (reduced κ weight)"}
            print(f"\n  >> PHASE {phase}: {phase_names[phase]} at step {step}\n")

            # Phase 0: freeze ODE params
            if phase == 0:
                for p in ode_params + other_model_params:
                    p.requires_grad = False
                for p in projection.parameters():
                    p.requires_grad = True
            # Phase 1+: unfreeze ODE
            elif phase >= 1 and prev_phase == 0:
                for p in ode_params + other_model_params:
                    p.requires_grad = True

            prev_phase = phase

        optimizer.zero_grad()

        # Sample batch with oracle embeddings
        try:
            oracle_embs, batch = dataset.sample_batch(
                config.batch_size, "train", device
            )
        except RuntimeError as e:
            print(f"  [WARN] Batch sampling failed: {e}")
            continue

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            # Project oracle embeddings
            z_context, kappa_target = projection(oracle_embs)

            # Compute HyperNet W_o delta (active in Phase 1+)
            delta_W_o = None
            if oracle_hypernet is not None and phase >= 1:
                delta_W_o = oracle_hypernet(z_context)

            # Forward with oracle context
            result = forward_with_oracle(
                model, z_context, kappa_target,
                colors=batch["colors"],
                xs=batch["xs"],
                ys=batch["ys"],
                roles=batch["roles"],
                sep_mask=batch["sep_mask"],
                sep_types=batch["sep_types"],
                target_mask=batch["target_mask"],
                target_labels=batch["target_labels"],
                context_mask=batch["context_mask"],
                grid_ids=batch.get("grid_ids"),
                target_input_colors=batch.get("target_input_colors"),
                lambda_kappa=lk,
                delta_W_o=delta_W_o,
            )

        # Representation distillation loss
        ld = get_lambda_distill(step, config)
        distill_result = {}
        if ld > 0 and distill_loss_fn is not None and "oracle_sim" in batch:
            # Use h0 (pre-ODE) for geometric supervision — same convention as geo_loss
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            h0 = result.get("h0", None)
            if h0 is not None:
                # compute_metric applies norm_geo internally, so pass raw h0
                g = raw_model.dynamics.compute_metric(h0)
                # For D² computation, we need norm_geo'd h (same space as g)
                h_normed = raw_model.dynamics.norm_geo(h0)
                t_diff = raw_model.dynamics.t_diffusion

                distill_result = distill_loss_fn(
                    h_normed, g, t_diff,
                    oracle_sim=batch["oracle_sim"],
                    cell_to_seq=batch["cell_to_seq"],
                    valid_mask=batch["sim_valid_mask"],
                )

        # Assemble loss
        loss = result["ce_loss"] + result["curv_loss"] + result["tau_var_loss"]
        loss = loss + result.get("cv_floor_loss", torch.tensor(0.0, device=device))

        if lk > 0:
            loss = loss + lk * result["kappa_distill_loss"]

        if ld > 0 and "distill_loss" in distill_result:
            loss = loss + ld * distill_result["distill_loss"]

        result["loss"] = loss
        result.update(distill_result)
        loss.backward()

        # Phase 0: zero ODE grads (safety — should already be frozen)
        if phase == 0:
            for p in ode_params + other_model_params:
                if p.grad is not None:
                    p.grad.zero_()

        all_params = list(model.parameters()) + list(projection.parameters())
        if oracle_hypernet is not None:
            all_params += list(oracle_hypernet.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, config.grad_clip)
        optimizer.step()
        scheduler.step()

        # Logging
        if step % config.log_every == 0:
            dt = time.time() - t0
            avg_n = batch.get("lengths", torch.tensor(config.max_seq_len)).float().mean().item()
            tok_s = config.batch_size * avg_n * (step + 1) / max(dt, 1e-6)

            cv_val = result["metric_cv"]
            if isinstance(cv_val, torch.Tensor):
                cv_val = cv_val.item()
            kappa_val = result["avg_kappa"]
            if isinstance(kappa_val, torch.Tensor):
                kappa_val = kappa_val.item()
            kt_val = result.get("kappa_target_mean", torch.tensor(0.0))
            if isinstance(kt_val, torch.Tensor):
                kt_val = kt_val.item()
            kd_val = result.get("kappa_distill_loss", torch.tensor(0.0))
            if isinstance(kd_val, torch.Tensor):
                kd_val = kd_val.item()

            cell_acc = result.get("cell_accuracy", torch.tensor(0.0))
            if isinstance(cell_acc, torch.Tensor):
                cell_acc = cell_acc.item()
            xform_acc = result.get("transform_accuracy", torch.tensor(0.0))
            if isinstance(xform_acc, torch.Tensor):
                xform_acc = xform_acc.item()

            # Distillation metrics
            distill_mse = result.get("distill_mse", torch.tensor(0.0))
            if isinstance(distill_mse, torch.Tensor):
                distill_mse = distill_mse.item()
            model_sim_mean = result.get("model_sim_mean", torch.tensor(0.0))
            if isinstance(model_sim_mean, torch.Tensor):
                model_sim_mean = model_sim_mean.item()

            distill_str = f", distill={distill_mse:.4f}(λ={ld:.3f})" if ld > 0 else ""

            # HyperNet delta diagnostics (only when active, Phase 1+)
            delta_norm_val = 0.0
            delta_max_val = 0.0
            if delta_W_o is not None:
                delta_norm_val = delta_W_o.detach().norm().item()
                delta_max_val = delta_W_o.detach().abs().max().item()
            hypernet_str = f", Δ_norm={delta_norm_val:.6f}, Δ_max={delta_max_val:.6f}" if delta_W_o is not None else ""

            print(f"  [step={step} P{phase}] loss={loss.item():.4f}, "
                  f"ce={result['ce_loss'].item():.4f}, "
                  f"κ_d={kd_val:.6f}(λ={lk:.3f}){distill_str}{hypernet_str}, "
                  f"|κ|={kappa_val:.4f}, κ_t={kt_val:.4f}, "
                  f"cv={cv_val:.4f}, "
                  f"cell={cell_acc:.4f}, xform={xform_acc:.4f}, "
                  f"tok/s={tok_s:.0f}")

            if writer:
                writer.add_scalar("loss/total", loss.item(), step)
                writer.add_scalar("loss/ce", result["ce_loss"].item(), step)
                writer.add_scalar("loss/kappa_distill", kd_val, step)
                writer.add_scalar("loss/rep_distill", distill_mse, step)
                writer.add_scalar("metric/cv", cv_val, step)
                writer.add_scalar("metric/kappa", kappa_val, step)
                writer.add_scalar("metric/kappa_target", kt_val, step)
                writer.add_scalar("metric/model_sim_mean", model_sim_mean, step)
                writer.add_scalar("accuracy/cell_train", cell_acc, step)
                writer.add_scalar("accuracy/xform_train", xform_acc, step)
                writer.add_scalar("phase", phase, step)
                writer.add_scalar("lambda_kappa", lk, step)
                writer.add_scalar("lambda_distill", ld, step)
                if delta_W_o is not None:
                    writer.add_scalar("hypernet/delta_norm", delta_norm_val, step)
                    writer.add_scalar("hypernet/delta_max", delta_max_val, step)

        # Eval
        if step > 0 and step % config.eval_every == 0:
            print(f"\n  --- Eval at step {step} ---")
            cell_acc, xform_acc, avg_ce, avg_kd = evaluate_quick(
                model, projection, dataset, config, device,
                hypernet=oracle_hypernet,
            )
            print(f"  Eval: cell={cell_acc:.4f}, xform={xform_acc:.4f}, "
                  f"CE={avg_ce:.4f}, κ_d={avg_kd:.6f}")

            if writer:
                writer.add_scalar("eval/cell_accuracy", cell_acc, step)
                writer.add_scalar("eval/xform_accuracy", xform_acc, step)
                writer.add_scalar("eval/ce", avg_ce, step)
                writer.add_scalar("eval/kappa_distill", avg_kd, step)

            if xform_acc > best_eval_xform:
                best_eval_xform = xform_acc
                save_checkpoint(
                    model, projection, optimizer, config, step,
                    os.path.join(out_dir, "checkpoints", "best.pt"),
                    hypernet=oracle_hypernet,
                    extra={"eval_xform": xform_acc},
                )
                print(f"  >> New best xform: {xform_acc:.4f}")

            model.train()
            projection.train()
            if oracle_hypernet is not None:
                oracle_hypernet.train()
            print()

        # Checkpoints
        if step > 0 and step % config.save_every == 0:
            save_checkpoint(
                model, projection, optimizer, config, step,
                os.path.join(out_dir, "checkpoints", f"step_{step}.pt"),
                hypernet=oracle_hypernet,
            )

    # Final save
    save_checkpoint(
        model, projection, optimizer, config, config.max_steps,
        os.path.join(out_dir, "checkpoints", "final.pt"),
        hypernet=oracle_hypernet,
    )
    print(f"\nTraining complete. Best eval xform: {best_eval_xform:.4f}")

    if writer:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description="Latent Oracle training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to latent_oracle.yaml config")
    parser.add_argument("--ode_checkpoint", type=str, default="",
                        help="Path to pretrained 5M LiquidARC checkpoint")
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to precomputed embeddings.pt")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ARC-AGI data directory")
    parser.add_argument("--output_dir", type=str, default="output_oracle",
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--similarity", type=str, default="",
                        help="Path to precomputed similarity_matrices.pt (rep distillation)")
    args = parser.parse_args()

    # Override config with CLI args
    if args.similarity:
        # Will be picked up via config after loading
        pass

    train(args)


if __name__ == "__main__":
    main()
