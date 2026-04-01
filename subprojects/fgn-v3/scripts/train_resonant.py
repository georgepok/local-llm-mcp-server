"""Resonant Geometry — Training script with structural energy.

Supports FluidNet (with/without structural energy) and flat transformer.
Works with ContinuousGridWorld (CW) task.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model_fluid import FluidNetModel
from fgn.flat_model import FlatTransformerModel
from fgn.tasks import get_task


def create_model(config: FGNConfig, device: torch.device):
    """Create model based on config."""
    if config.model_type == "flat":
        return FlatTransformerModel(config).to(device)
    elif config.architecture_version == "fluid":
        return FluidNetModel(config).to(device)
    else:
        return FlatTransformerModel(config).to(device)


def create_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def print_status(model, result, step, tok_per_sec, epoch=None):
    """Print compact status line with resonance metrics."""
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    is_fluid = isinstance(m, FluidNetModel)

    extra = ""
    if is_fluid:
        t_local = result.get("avg_t_local", torch.tensor(0.0)).item()
        t_medium = result.get("avg_t_medium", torch.tensor(0.0)).item()
        t_global = result.get("avg_t_global", torch.tensor(0.0)).item()
        extra = f", t=[{t_local:.2f},{t_medium:.2f},{t_global:.2f}]"

    e_struct = result.get("structural_energy", torch.tensor(0.0))
    if isinstance(e_struct, torch.Tensor):
        e_struct = e_struct.item()

    cv_val = result["metric_cv"]
    if isinstance(cv_val, torch.Tensor):
        cv_val = cv_val.item()

    # Weight norm — key grokking signal (plunge = structural collapse)
    w_norm = sum(p.data.norm().item() ** 2 for p in m.parameters()) ** 0.5

    aux_dist = result.get("aux_dist_loss", torch.tensor(0.0))
    if isinstance(aux_dist, torch.Tensor):
        aux_dist = aux_dist.item()
    aux_str = f", aux_d={aux_dist:.4f}" if aux_dist > 0 else ""

    kf = result.get("kappa_floor_loss", torch.tensor(0.0))
    if isinstance(kf, torch.Tensor):
        kf = kf.item()
    kf_str = f", kf={kf:.4f}" if kf > 0 else ""

    epoch_str = f", ep={epoch}" if epoch is not None else ""
    print(f"  [step={step}] loss={result['loss'].item():.4f}, "
          f"ce={result['ce_loss'].item():.4f}, "
          f"e_struct={e_struct:.4f}, "
          f"cv={cv_val:.4f}, "
          f"|k|={result['avg_kappa'].item():.4f}, "
          f"W={w_norm:.1f}, "
          f"tok/s={tok_per_sec:.0f}{aux_str}{kf_str}{epoch_str}{extra}")


def save_checkpoint(model, optimizer, config, step, path, extra=None):
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


class FixedDataset:
    """Pre-generated dataset that cycles through episodes with shuffling."""

    def __init__(self, path: str, device: torch.device):
        print(f"  Loading fixed dataset: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.input_ids = data["input_ids"]           # [N, seq_len]
        self.labels = data["labels"]                  # [N, seq_len]
        self.context_mask = data["context_mask"]      # [N, seq_len]
        self.room_distances = data["room_distances"]  # [N, R_max, R_max]
        self.room_positions = data["room_token_positions"]  # [N, R_max]
        self.n_rooms = data["n_rooms"]                # [N]
        self.n_episodes = data["n_episodes"]
        self.device = device

        # Shuffle order
        self._perm = torch.randperm(self.n_episodes)
        self._cursor = 0
        self._epoch = 0

        print(f"  Fixed dataset: {self.n_episodes} episodes, "
              f"R_max={data['R_max']}")

    def get_batch(self, batch_size: int):
        """Get next batch, cycling with shuffle at epoch boundaries."""
        indices = []
        for _ in range(batch_size):
            if self._cursor >= self.n_episodes:
                self._cursor = 0
                self._epoch += 1
                self._perm = torch.randperm(self.n_episodes)
            indices.append(self._perm[self._cursor].item())
            self._cursor += 1

        idx = torch.tensor(indices, dtype=torch.long)

        input_ids = self.input_ids[idx].to(self.device)
        labels = self.labels[idx].to(self.device)
        meta = {
            "context_mask": self.context_mask[idx].to(self.device),
            "room_distances": self.room_distances[idx].to(self.device),
            "room_token_positions": self.room_positions[idx].to(self.device),
            "n_rooms": self.n_rooms[idx].to(self.device),
        }
        return input_ids, labels, meta

    @property
    def epoch(self):
        return self._epoch


def train(args, config, device):
    """Training loop."""
    tokenizer = _get_tokenizer()
    task_kwargs = json.loads(args.task_kwargs)

    # Override config from CLI
    config.structural_energy_lambda = args.lambda_struct
    config.aux_distance_max_hops = args.aux_distance_max_hops
    config.aux_distance_weight = args.aux_distance_weight
    config.kappa_floor = args.kappa_floor
    config.kappa_floor_mu = args.kappa_floor_mu

    print(f"\n{'='*70}")
    print(f"Resonant Geometry Training — {config.architecture_version}")
    print(f"{'='*70}")

    model = create_model(config, device)
    is_fluid = isinstance(model, FluidNetModel)

    # Resume from checkpoint (model weights only, fresh optimizer)
    if args.resume_checkpoint:
        print(f"  Resuming from: {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        # Load with strict=False to handle new modules (distance_predictor)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  New params (randomly initialized): {missing}")
        if unexpected:
            print(f"  Ignored checkpoint params: {unexpected}")
        del ckpt

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    if is_fluid:
        print(f"  Architecture: FluidNet (pure geometric diffusion)")
        print(f"  Scales: {config.n_scales}, d_metric: {config.d_metric}")
        print(f"  lambda_struct: {config.structural_energy_lambda}")
    else:
        print(f"  Architecture: flat baseline")

    out_dir = args.output_dir
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, "logs"))

    total_steps = args.max_steps

    # Progressive complexity schedule
    complexity_phases = []
    if args.complexity_schedule:
        for phase_str in args.complexity_schedule.split(","):
            step_s, min_s, max_s = phase_str.strip().split(":")
            complexity_phases.append((int(step_s), int(min_s), int(max_s)))
        complexity_phases.sort(key=lambda x: x[0])
        print(f"  Complexity schedule: {complexity_phases}")
        # Start with first phase
        task_kwargs["n_rooms_min"] = complexity_phases[0][1]
        task_kwargs["n_rooms_max"] = complexity_phases[0][2]

    # Fixed dataset mode (for grokking) vs online generation
    fixed_ds = None
    if args.fixed_dataset:
        fixed_ds = FixedDataset(args.fixed_dataset, device)
    else:
        task = get_task(args.task, tokenizer, seq_len=config.max_seq_len, **task_kwargs)
    current_phase_idx = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args.warmup_steps, total_steps)

    # Keep reference to raw model for aux computations outside torch.compile
    model_raw = model

    if config.use_torch_compile and device.type == "cuda":
        compiled_model = torch.compile(model, mode="default")
    else:
        compiled_model = model

    compiled_model.train()
    t0 = time.time()
    loss = torch.tensor(0.0)

    # Hot-swap signal file: drop a new .pt dataset path into this file to trigger swap
    swap_signal_path = os.path.join(out_dir, "SWAP_DATASET")

    for step in range(total_steps):
        # Check for dataset hot-swap (every 100 steps to minimize I/O)
        if fixed_ds is not None and step % 100 == 0 and os.path.exists(swap_signal_path):
            try:
                with open(swap_signal_path) as f:
                    new_path = f.read().strip()
                if new_path and os.path.exists(new_path):
                    print(f"\n  >>> WORLD SWAP @ step {step}: loading {new_path}\n")
                    fixed_ds = FixedDataset(new_path, device)
                    os.remove(swap_signal_path)
                    # Save checkpoint at swap point
                    save_checkpoint(model, optimizer, config, step,
                                  os.path.join(out_dir, "checkpoints", f"pre_swap_{step}.pt"))
            except Exception as e:
                print(f"  >>> Swap failed: {e}")

        # Check for complexity phase transition
        if complexity_phases:
            new_phase_idx = current_phase_idx
            for i, (phase_step, _, _) in enumerate(complexity_phases):
                if step >= phase_step:
                    new_phase_idx = i
            if new_phase_idx != current_phase_idx:
                current_phase_idx = new_phase_idx
                _, n_min, n_max = complexity_phases[current_phase_idx]
                task_kwargs["n_rooms_min"] = n_min
                task_kwargs["n_rooms_max"] = n_max
                task = get_task(args.task, tokenizer, seq_len=config.max_seq_len, **task_kwargs)
                print(f"\n  >>> Phase transition @ step {step}: rooms {n_min}-{n_max}\n")

        if fixed_ds is not None:
            input_ids, labels, meta = fixed_ds.get_batch(args.batch_size)
        else:
            input_ids, labels, meta = task.generate_batch(args.batch_size, device=device)
        context_mask = meta.get("context_mask")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            if is_fluid:
                result = compiled_model(
                    input_ids, labels=labels,
                    context_mask=context_mask,
                    room_distances=meta.get("room_distances"),
                    room_token_positions=meta.get("room_token_positions"),
                    n_rooms=meta.get("n_rooms"),
                )
            else:
                result = compiled_model(input_ids, labels=labels)
            loss = result["loss"]

        # Auxiliary distance prediction (outside torch.compile to avoid graph breaks)
        if (is_fluid and args.aux_distance_weight > 0 and
                model_raw.distance_predictor is not None and
                meta.get("room_distances") is not None):
            aux_dist_loss = model_raw.distance_predictor(
                result["h_pre_norm"], meta["room_token_positions"],
                meta["room_distances"], meta["n_rooms"])
            result["aux_dist_loss"] = aux_dist_loss
            loss = loss + args.aux_distance_weight * aux_dist_loss

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            scheduler.step()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            dt = time.time() - t0
            tok_s = args.batch_size * config.max_seq_len * (step + 1) / max(dt, 1e-6)
            ep = fixed_ds.epoch if fixed_ds is not None else None
            print_status(model, result, step, tok_s, epoch=ep)

            writer.add_scalar("loss/total", result["loss"].item(), step)
            writer.add_scalar("loss/ce", result["ce_loss"].item(), step)

            e_struct = result.get("structural_energy", torch.tensor(0.0))
            if isinstance(e_struct, torch.Tensor):
                e_struct = e_struct.item()
            writer.add_scalar("loss/structural_energy", e_struct, step)

            cv_val = result["metric_cv"]
            if isinstance(cv_val, torch.Tensor):
                cv_val = cv_val.item()
            writer.add_scalar("metric/cv", cv_val, step)
            writer.add_scalar("metric/kappa", result["avg_kappa"].item(), step)

            aux_d_val = result.get("aux_dist_loss", torch.tensor(0.0))
            if isinstance(aux_d_val, torch.Tensor):
                aux_d_val = aux_d_val.item()
            if aux_d_val > 0:
                writer.add_scalar("loss/aux_distance", aux_d_val, step)

            kf_val = result.get("kappa_floor_loss", torch.tensor(0.0))
            if isinstance(kf_val, torch.Tensor):
                kf_val = kf_val.item()
            if kf_val > 0:
                writer.add_scalar("loss/kappa_floor", kf_val, step)

            if is_fluid:
                writer.add_scalar("timescale/local",
                                  result.get("avg_t_local", torch.tensor(0.0)).item(), step)
                writer.add_scalar("timescale/medium",
                                  result.get("avg_t_medium", torch.tensor(0.0)).item(), step)
                writer.add_scalar("timescale/global",
                                  result.get("avg_t_global", torch.tensor(0.0)).item(), step)

        if step > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, config, step,
                          os.path.join(out_dir, "checkpoints", f"step_{step}.pt"))

    save_checkpoint(model, optimizer, config, total_steps,
                   os.path.join(out_dir, "checkpoints", "final.pt"),
                   extra={"task": args.task,
                          "architecture_version": config.architecture_version,
                          "lambda_struct": args.lambda_struct})
    writer.close()

    print(f"\n  Training complete. Final loss: {loss.item():.4f}")

    if is_fluid:
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        print(f"\n  Final timescales per layer:")
        for i, layer in enumerate(m.layers):
            import torch.nn.functional as F
            t_bias = F.softplus(layer.time_net_linear2.bias)
            print(f"    Layer {i}: t_init=[{','.join(f'{t:.3f}' for t in t_bias.tolist())}]")


def _get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


def main():
    parser = argparse.ArgumentParser(description="Resonant Geometry Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default="CW",
                        help="Task: CW=continuous gridworld")
    parser.add_argument("--output_dir", type=str, default="output_resonant")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--lambda_struct", type=float, default=0.0,
                        help="Structural energy weight (0.0 = disabled)")
    parser.add_argument("--task_kwargs", type=str, default="{}",
                        help="JSON dict of extra kwargs for task constructor")
    parser.add_argument("--complexity_schedule", type=str, default="",
                        help="Progressive complexity: 'step:min:max,...' e.g. '0:5:5,20000:5:15,40000:5:25'")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Path to checkpoint to resume from (loads model weights only)")
    parser.add_argument("--aux_distance_weight", type=float, default=0.0,
                        help="Auxiliary distance prediction loss weight")
    parser.add_argument("--aux_distance_max_hops", type=int, default=0,
                        help="Max hop classes for distance prediction (0=disabled)")
    parser.add_argument("--kappa_floor", type=float, default=0.0,
                        help="Minimum |kappa| target (0=disabled)")
    parser.add_argument("--kappa_floor_mu", type=float, default=0.0,
                        help="Penalty weight for kappa below floor")
    parser.add_argument("--fixed_dataset", type=str, default="",
                        help="Path to pre-generated .pt dataset (enables grokking mode)")
    args = parser.parse_args()

    config = FGNConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mem_frac = float(os.environ.get("CUDA_MEMORY_FRACTION", "0"))
    if mem_frac > 0 and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(mem_frac)
        print(f"CUDA memory fraction capped at {mem_frac:.0%}")

    print(f"Device: {device}")
    print(f"Model: {config.model_type}, arch: {config.architecture_version}")
    print(f"Task: {args.task}, kwargs: {args.task_kwargs}")
    print(f"lambda_struct: {args.lambda_struct}")
    if args.aux_distance_max_hops > 0:
        print(f"Aux distance: max_hops={args.aux_distance_max_hops}, weight={args.aux_distance_weight}")
    if args.kappa_floor > 0:
        print(f"Kappa floor: {args.kappa_floor}, mu={args.kappa_floor_mu}")
    if args.resume_checkpoint:
        print(f"Resume from: {args.resume_checkpoint}")
    if args.fixed_dataset:
        print(f"Fixed dataset: {args.fixed_dataset}")
    print(f"Total steps: {args.max_steps}")

    train(args, config, device)


if __name__ == "__main__":
    main()
