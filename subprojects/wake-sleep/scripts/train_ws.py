"""Wake-Sleep V2 training script for LiquidARC.

V2 changes:
- VQ-VAE encoder with EMA codebook + dead code restart
- AR decoder with teacher forcing during wake
- Hybrid sleep: 50% dreams + 50% real ARC
- W_o unfrozen during sleep (full WHERE+WHEN+WHAT plasticity)
- VQ-specific logging: codebook usage, commitment loss, recon loss

Phase 0: Wake pre-training (VQ-Encoder + AR Decoder learn from real ARC)
Phase 1: Alternating Wake-Sleep (ODE learns from dreams + real ARC)

Usage:
    python scripts/train_ws.py --config configs/wake_sleep_v2.yaml --data_dir data/arc
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch
# Disable cuDNN — container has version mismatch (9.17.1 vs PyTorch's 9.18.0).
# Only affects Conv2d in Encoder (small grids), negligible perf impact.
torch.backends.cudnn.enabled = False
from torch.utils.tensorboard import SummaryWriter

# Path setup: this script is in subprojects/wake-sleep/scripts/
WAKE_SLEEP_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, WAKE_SLEEP_ROOT)

LIQUID_ARC_ROOT = str(Path(__file__).resolve().parent.parent.parent / "liquid-arc")
sys.path.insert(0, LIQUID_ARC_ROOT)

FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent / "fgn-v3")
if not Path(FGN_ROOT).exists():
    FGN_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, FGN_ROOT)

from wake_sleep.config import WakeSleepConfig
from wake_sleep.wake_sleep import WakeSleepModel, extract_grid_pairs
from wake_sleep.dream_ttt import evaluate_dream_ttt
from liquid_arc.model import LiquidARCModel
from fgn.tasks.arc import load_arc_tasks


def freeze_all(*modules):
    for m in modules:
        for p in m.parameters():
            p.requires_grad = False


def unfreeze_all(*modules):
    for m in modules:
        for p in m.parameters():
            p.requires_grad = True


def save_checkpoint(ws_model, base_model, wake_opt, sleep_opt, config, step, path, extra=None):
    ckpt = {
        "step": step,
        "ws_model": {
            "encoder": ws_model.encoder.state_dict(),
            "decoder": ws_model.decoder.state_dict(),
            "z_to_context": ws_model.z_to_context.state_dict(),
        },
        "base_model": base_model.state_dict(),
        "wake_opt": wake_opt.state_dict(),
        "sleep_opt": sleep_opt.state_dict(),
        "config": config.__dict__,
        "concept_bank_count": ws_model.concept_bank.count,
        "version": "v2",
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(ws_model, base_model, wake_opt, sleep_opt, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # strict=False: tolerate new buffers (z_buffer etc.) missing from old checkpoints
    ws_model.encoder.load_state_dict(ckpt["ws_model"]["encoder"], strict=False)
    ws_model.decoder.load_state_dict(ckpt["ws_model"]["decoder"], strict=False)
    ws_model.z_to_context.load_state_dict(ckpt["ws_model"]["z_to_context"], strict=False)

    # Handle torch.compile _orig_mod prefix
    state = ckpt["base_model"]
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
    base_model.load_state_dict(cleaned)

    wake_opt.load_state_dict(ckpt["wake_opt"])
    sleep_opt.load_state_dict(ckpt["sleep_opt"])
    return ckpt["step"]


def train(args, config, device):
    print(f"\n{'='*70}")
    print(f"Wake-Sleep V2 Training (VQ-VAE + AR Decoder + Hybrid Sleep)")
    print(f"{'='*70}")

    # Load or create base ODE model
    base_model = LiquidARCModel(config).to(device)
    if args.ode_checkpoint:
        print(f"  Loading ODE checkpoint: {args.ode_checkpoint}")
        ckpt = torch.load(args.ode_checkpoint, map_location=device, weights_only=False)
        state = ckpt["model"]
        cleaned = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
        base_model.load_state_dict(cleaned)

    n_ode_params = sum(p.numel() for p in base_model.parameters())
    print(f"  ODE params: {n_ode_params:,}")

    # Create Wake-Sleep V2 wrapper
    ws_model = WakeSleepModel(config, base_model).to(device)

    n_enc = sum(p.numel() for p in ws_model.encoder.parameters())
    n_vq = ws_model.encoder.vq.n_embeddings * ws_model.encoder.vq.z_dim
    n_dec = sum(p.numel() for p in ws_model.decoder.parameters())
    n_proj = sum(p.numel() for p in ws_model.z_to_context.parameters())
    print(f"  VQ Encoder params: {n_enc:,} (codebook: {n_vq:,})")
    print(f"  AR Decoder params: {n_dec:,}")
    print(f"  z_to_context params: {n_proj:,}")
    print(f"  Total: {n_ode_params + n_enc + n_dec + n_proj:,}")
    print(f"  Codebook: K={config.ws_vq_n_embeddings}, z_dim={config.ws_z_dim}")
    print(f"  AR Decoder: {config.ws_ar_n_layers} layers, d={config.ws_ar_d_model}, "
          f"{config.ws_ar_n_heads} heads")
    print(f"  Sleep mix: {config.ws_real_arc_mix_ratio:.0%} real ARC, "
          f"{1-config.ws_real_arc_mix_ratio:.0%} dreams")
    print(f"  W_o unfreeze: {config.ws_unfreeze_wo}")

    # Output directory
    out_dir = args.output_dir
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    # Logging
    log_path = os.path.join(out_dir, "train.log")
    logger = logging.getLogger("train_ws")
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

    writer = SummaryWriter(os.path.join(out_dir, "logs"))

    # Separate optimizers
    wake_opt = torch.optim.AdamW(
        ws_model.wake_parameters(), lr=config.ws_wake_lr, weight_decay=0.01)
    sleep_opt = torch.optim.AdamW(
        ws_model.sleep_parameters(), lr=config.ws_sleep_lr, weight_decay=0.01)

    # Load real ARC dataset
    all_arc = load_arc_tasks(args.data_dir)
    arc_train_tasks = all_arc.get("train", [])
    arc_eval_tasks = all_arc.get("eval", [])
    print(f"  ARC train tasks: {len(arc_train_tasks)}")
    print(f"  ARC eval tasks: {len(arc_eval_tasks)}")

    if not arc_train_tasks:
        print("  ERROR: No ARC training tasks found!")
        return

    # Resume
    start_step = 0
    if args.resume:
        print(f"  Resuming from {args.resume}")
        start_step = load_checkpoint(ws_model, base_model, wake_opt, sleep_opt,
                                     args.resume, device)
        print(f"  Resumed at step {start_step}")

    # torch.compile the ODE dynamics (hot path)
    if config.use_torch_compile and device.type == "cuda":
        base_model.dynamics = torch.compile(base_model.dynamics, mode="default", dynamic=True)
        print(f"  torch.compile: dynamics compiled")

    # Unfreeze tau if past freeze period
    if hasattr(base_model.dynamics, 'freeze_tau'):
        base_model.dynamics.freeze_tau = False

    t0 = time.time()
    best_eval_xform = 0.0
    step = start_step
    dead_restart_counter = 0  # for periodic dead code restart

    # ──────────────────────────────────────────────────────────────
    # Phase 0: Wake pre-training (VQ-Encoder + AR Decoder only)
    # ──────────────────────────────────────────────────────────────
    if step < config.ws_wake_only_steps:
        print(f"\n  >> PHASE 0: Wake Pre-training (steps {step}-{config.ws_wake_only_steps})")
        print(f"     VQ-Encoder + AR Decoder learn concepts, ODE frozen")
        print(f"     Loss = recon_CE + beta * commitment_loss")

    # Number of tasks per wake step for meaningful entropy regularization
    wake_batch_n = min(args.batch_size, 16)  # N tasks batched through VQ

    while step < config.ws_wake_only_steps and step < args.max_steps:
        # Sample N tasks for batched VQ entropy
        tasks = [random.choice(arc_train_tasks) for _ in range(wake_batch_n)]
        task_pairs = [extract_grid_pairs(t, device) for t in tasks]

        freeze_all(base_model, ws_model.z_to_context)
        unfreeze_all(ws_model.encoder, ws_model.decoder)

        wake_opt.zero_grad()
        result = ws_model.wake_step_batched(task_pairs, device)
        result["wake_loss"].backward()
        torch.nn.utils.clip_grad_norm_(ws_model.wake_parameters(), args.grad_clip)
        wake_opt.step()

        # Periodic dead code restart (uses accumulated z_e buffer)
        dead_restart_counter += 1
        if dead_restart_counter >= config.ws_vq_dead_restart_every:
            dead_restart_counter = 0
            n_restarted = ws_model.encoder.vq.restart_dead_codes()
            if n_restarted > 0:
                print(f"  [step={step}] Dead codes restarted: {n_restarted}")

        # Logging
        if step % args.log_every == 0:
            bank_size = ws_model.concept_bank.size
            usage = result["codebook_usage"]
            print(f"  [step={step}] WAKE loss={result['wake_loss'].item():.4f}, "
                  f"recon={result['recon_loss'].item():.4f}, "
                  f"vq={result['vq_loss'].item():.4f}, "
                  f"z_norm={result['z_norm'].item():.3f}, "
                  f"cb_usage={usage:.4f}, bank={bank_size}")
            writer.add_scalar("wake/loss", result["wake_loss"].item(), step)
            writer.add_scalar("wake/recon_loss", result["recon_loss"].item(), step)
            writer.add_scalar("wake/vq_loss", result["vq_loss"].item(), step)
            writer.add_scalar("wake/z_norm", result["z_norm"].item(), step)
            writer.add_scalar("wake/codebook_usage", usage, step)
            writer.add_scalar("wake/bank_size", bank_size, step)

        # Checkpoint
        if step > 0 and step % args.save_every == 0:
            save_checkpoint(ws_model, base_model, wake_opt, sleep_opt, config, step,
                          os.path.join(out_dir, "checkpoints", f"step_{step}.pt"))

        step += 1

    # ──────────────────────────────────────────────────────────────
    # Phase 1: Alternating Wake-Sleep
    # ──────────────────────────────────────────────────────────────
    if step >= config.ws_wake_only_steps:
        print(f"\n  >> PHASE 1: Alternating Wake-Sleep (step {step})")
        print(f"     Wake: {config.ws_wake_steps} steps/cycle, "
              f"Sleep: {config.ws_sleep_steps} steps/cycle")
        print(f"     Sleep mix: {config.ws_real_arc_mix_ratio:.0%} real ARC")
        print(f"     Concept bank: {ws_model.concept_bank.size} concepts")

    cycle = 0
    while step < args.max_steps:
        cycle += 1

        # ── Wake cycle ──
        wake_losses = []
        wake_recon_losses = []
        wake_vq_losses = []
        for _ in range(config.ws_wake_steps):
            if step >= args.max_steps:
                break

            # Sample N tasks for batched VQ entropy
            tasks = [random.choice(arc_train_tasks) for _ in range(wake_batch_n)]
            task_pairs = [extract_grid_pairs(t, device) for t in tasks]

            freeze_all(base_model)
            unfreeze_all(ws_model.encoder, ws_model.decoder)

            wake_opt.zero_grad()
            result = ws_model.wake_step_batched(task_pairs, device)
            result["wake_loss"].backward()
            torch.nn.utils.clip_grad_norm_(ws_model.wake_parameters(), args.grad_clip)
            wake_opt.step()

            wake_losses.append(result["wake_loss"].item())
            wake_recon_losses.append(result["recon_loss"].item())
            wake_vq_losses.append(result["vq_loss"].item())

            # Dead code restart
            dead_restart_counter += 1
            if dead_restart_counter >= config.ws_vq_dead_restart_every:
                dead_restart_counter = 0
                n_restarted = ws_model.encoder.vq.restart_dead_codes()
                if n_restarted > 0:
                    print(f"  [step={step}] Dead codes restarted: {n_restarted}")

            if step % args.log_every == 0:
                usage = result["codebook_usage"]
                print(f"  [step={step}] WAKE loss={result['wake_loss'].item():.4f}, "
                      f"recon={result['recon_loss'].item():.4f}, "
                      f"vq={result['vq_loss'].item():.4f}, "
                      f"z_norm={result['z_norm'].item():.3f}, "
                      f"cb_usage={usage:.4f}, bank={ws_model.concept_bank.size}")
                writer.add_scalar("wake/loss", result["wake_loss"].item(), step)
                writer.add_scalar("wake/recon_loss", result["recon_loss"].item(), step)
                writer.add_scalar("wake/vq_loss", result["vq_loss"].item(), step)
                writer.add_scalar("wake/z_norm", result["z_norm"].item(), step)
                writer.add_scalar("wake/codebook_usage", usage, step)

            step += 1

        # ── Sleep cycle ──
        if ws_model.concept_bank.size < 10:
            print(f"  >> SKIP SLEEP: concept bank too small ({ws_model.concept_bank.size})")
            continue

        sleep_losses = []
        sleep_sources = {"dream": 0, "real": 0}
        for _ in range(config.ws_sleep_steps):
            if step >= args.max_steps:
                break

            freeze_all(ws_model.encoder, ws_model.decoder)
            unfreeze_all(base_model, ws_model.z_to_context)

            sleep_opt.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = ws_model.sleep_step(
                    args.batch_size, device, arc_tasks=arc_train_tasks)

            source = result.get("sleep_source", "dream")
            sleep_sources[source] = sleep_sources.get(source, 0) + 1

            loss = result["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ws_model.sleep_parameters(), args.grad_clip)
            sleep_opt.step()

            sleep_losses.append(loss.item())

            if step % args.log_every == 0:
                cv = result["metric_cv"]
                if isinstance(cv, torch.Tensor):
                    cv = cv.item()
                kappa = result["avg_kappa"]
                if isinstance(kappa, torch.Tensor):
                    kappa = kappa.item()
                xf_loss = result.get("xform_loss", torch.tensor(0.0))
                if isinstance(xf_loss, torch.Tensor):
                    xf_loss = xf_loss.item()
                xf_acc = result.get("transform_accuracy", torch.tensor(0.0))
                if isinstance(xf_acc, torch.Tensor):
                    xf_acc = xf_acc.item()

                print(f"  [step={step}] SLEEP({source}) loss={loss.item():.4f}, "
                      f"xf_loss={xf_loss:.4f}, xf_acc={xf_acc:.4f}, "
                      f"cv={cv:.4f}, |k|={kappa:.4f}")
                writer.add_scalar("sleep/loss", loss.item(), step)
                writer.add_scalar("sleep/xf_loss", xf_loss, step)
                writer.add_scalar("sleep/xf_acc", xf_acc, step)
                writer.add_scalar("sleep/cv", cv, step)
                writer.add_scalar("sleep/kappa", kappa, step)

            step += 1

        # Cycle summary
        avg_wake = sum(wake_losses) / max(len(wake_losses), 1)
        avg_recon = sum(wake_recon_losses) / max(len(wake_recon_losses), 1)
        avg_vq = sum(wake_vq_losses) / max(len(wake_vq_losses), 1)
        avg_sleep = sum(sleep_losses) / max(len(sleep_losses), 1)
        elapsed = time.time() - t0
        print(f"\n  >> CYCLE {cycle}: wake_loss={avg_wake:.4f} "
              f"(recon={avg_recon:.4f}, vq={avg_vq:.4f}), "
              f"sleep_loss={avg_sleep:.4f}, "
              f"sleep_src=({sleep_sources.get('dream', 0)}d/{sleep_sources.get('real', 0)}r), "
              f"step={step}, elapsed={elapsed:.0f}s\n")
        writer.add_scalar("cycle/wake_loss", avg_wake, cycle)
        writer.add_scalar("cycle/recon_loss", avg_recon, cycle)
        writer.add_scalar("cycle/vq_loss", avg_vq, cycle)
        writer.add_scalar("cycle/sleep_loss", avg_sleep, cycle)
        writer.add_scalar("cycle/sleep_dream_frac",
                          sleep_sources.get("dream", 0) / max(sum(sleep_sources.values()), 1),
                          cycle)
        writer.add_scalar("cycle/codebook_usage",
                          ws_model.encoder.vq.codebook_usage(), cycle)

        # Checkpoint every cycle
        save_checkpoint(ws_model, base_model, wake_opt, sleep_opt, config, step,
                      os.path.join(out_dir, "checkpoints", f"step_{step}.pt"))

        # Eval with Dream-TTT
        if step >= config.ws_wake_only_steps + config.ws_sleep_steps and cycle % args.eval_every_cycles == 0:
            print(f"  >> Dream-TTT V2 eval at cycle {cycle}...")
            cell_acc, xform_acc = evaluate_dream_ttt(
                base_model, ws_model.encoder, ws_model.decoder, ws_model.z_to_context,
                args.data_dir, config, device, n_tasks=args.eval_n_tasks, verbose=False,
            )
            writer.add_scalar("eval/cell_acc", cell_acc, step)
            writer.add_scalar("eval/xform_acc", xform_acc, step)

            if xform_acc > best_eval_xform:
                best_eval_xform = xform_acc
                save_checkpoint(ws_model, base_model, wake_opt, sleep_opt, config, step,
                              os.path.join(out_dir, "checkpoints", "best.pt"),
                              extra={"eval_xform": xform_acc})
                print(f"  >> New best xform: {xform_acc:.4f}")

            # Re-enter training mode
            base_model.train()
            ws_model.train()

    # Final checkpoint
    save_checkpoint(ws_model, base_model, wake_opt, sleep_opt, config, step,
                   os.path.join(out_dir, "checkpoints", "final.pt"))
    writer.close()

    print(f"\n  Training complete. Best eval xform: {best_eval_xform:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Wake-Sleep V2 Training for LiquidARC")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/arc")
    parser.add_argument("--output_dir", type=str, default="output_wake_sleep_v2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=2500)
    parser.add_argument("--eval_every_cycles", type=int, default=2,
                        help="Run Dream-TTT eval every N wake-sleep cycles")
    parser.add_argument("--eval_n_tasks", type=int, default=50,
                        help="Number of ARC eval tasks for Dream-TTT eval")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ode_checkpoint", type=str, default=None,
                        help="Path to pre-trained ODE checkpoint (warm-start)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to Wake-Sleep V2 checkpoint to resume from")
    args = parser.parse_args()

    config = WakeSleepConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mem_frac = float(os.environ.get("CUDA_MEMORY_FRACTION", "0"))
    if mem_frac > 0 and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(mem_frac)
        print(f"CUDA memory fraction capped at {mem_frac:.0%}")

    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Data dir: {args.data_dir}")
    print(f"Max steps: {args.max_steps}")

    train(args, config, device)


if __name__ == "__main__":
    main()
