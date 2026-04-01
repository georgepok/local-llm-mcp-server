"""Criticality Training — sustain edge-of-chaos geometric plasticity.

Combines four mechanisms:
1. MasterWorld with topology mutations every K steps
2. MetricMonitor with shrink-and-perturb for dead dimensions
3. Dynamic weight decay conditioned on CE loss velocity
4. Plasticity telemetry (adaptation half-life, metric volatility)
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
from fgn.tasks.master_world import MasterWorld
from fgn.tasks.continuous_gridworld import ContinuousGridWorldTask
from fgn.metric_monitor import MetricMonitor, DynamicWeightDecay


def create_model(config: FGNConfig, device: torch.device):
    if config.model_type == "flat":
        return FlatTransformerModel(config).to(device)
    elif config.architecture_version == "fluid":
        return FluidNetModel(config).to(device)
    else:
        return FlatTransformerModel(config).to(device)


def create_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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


def generate_batch_from_master(master: MasterWorld, task: ContinuousGridWorldTask,
                                batch_size: int, device: torch.device):
    """Generate a training batch using the MasterWorld's current topology.

    Creates episode worlds that share the master's graph structure
    but have fresh agent/object placement.
    """
    pad_id = task.tokenizer.eos_token_id or 0
    all_input_ids = []
    all_labels = []
    all_context_masks = []
    episode_worlds = []
    episode_room_positions = []

    for _ in range(batch_size):
        # Generate episode ensuring supervised tokens survive truncation
        for _retry in range(100):
            world = master.create_episode_world()
            try:
                ep_result = task._try_generate_episode(override_world=world)
            except Exception:
                ep_result = None

            if ep_result is None:
                continue

            episode_text, _actions, n_steps, optimal_cost, step_costs, world = ep_result
            input_ids, labels, context_end_pos, action_spans, room_token_pos = \
                task._tokenize_episode(episode_text)

            # Truncate/pad
            if len(input_ids) > task.seq_len:
                input_ids = input_ids[:task.seq_len]
                labels = labels[:task.seq_len]
            else:
                pad_len = task.seq_len - len(input_ids)
                input_ids += [pad_id] * pad_len
                labels += [-100] * pad_len

            # Check: enough supervised tokens after truncation
            n_supervised = sum(1 for l in labels if l != -100)
            if n_supervised >= 5:
                break
        else:
            # Final fallback: use normal generation (variable room count)
            ep_result = task._generate_valid_episode()
            episode_text, _actions, n_steps, optimal_cost, step_costs, world = ep_result
            input_ids, labels, context_end_pos, action_spans, room_token_pos = \
                task._tokenize_episode(episode_text)
            if len(input_ids) > task.seq_len:
                input_ids = input_ids[:task.seq_len]
                labels = labels[:task.seq_len]
            else:
                pad_len = task.seq_len - len(input_ids)
                input_ids += [pad_id] * pad_len
                labels += [-100] * pad_len

        episode_worlds.append(world)
        episode_room_positions.append(room_token_pos)

        context_mask_row = [False] * task.seq_len
        for i in range(min(context_end_pos, task.seq_len)):
            context_mask_row[i] = True
        all_context_masks.append(context_mask_row)
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    # Build graph-distance tensors
    R_max = max(w.n_rooms for w in episode_worlds)
    room_distances = torch.ones(batch_size, R_max, R_max)
    room_positions = torch.full((batch_size, R_max), -1, dtype=torch.long)
    n_rooms_tensor = torch.zeros(batch_size, dtype=torch.long)

    for b, (world, rtp) in enumerate(zip(episode_worlds, episode_room_positions)):
        R = world.n_rooms
        n_rooms_tensor[b] = R
        sp = world.all_pairs_shortest_paths()
        finite_dists = [d for d in sp.values() if d < float('inf') and d > 0]
        max_dist = max(finite_dists) if finite_dists else 1.0
        for i in range(R):
            for j in range(R):
                d = sp.get((i, j), float('inf'))
                if d < float('inf'):
                    room_distances[b, i, j] = d / max_dist
            if i in rtp and rtp[i] < task.seq_len:
                room_positions[b, i] = rtp[i]

    input_ids_t = torch.tensor(all_input_ids, dtype=torch.long, device=device)
    labels_t = torch.tensor(all_labels, dtype=torch.long, device=device)

    meta = {
        "context_mask": torch.tensor(all_context_masks, dtype=torch.bool, device=device),
        "room_distances": room_distances.to(device),
        "room_token_positions": room_positions.to(device),
        "n_rooms": n_rooms_tensor.to(device),
    }

    return input_ids_t, labels_t, meta


def train(args, config, device):
    """Criticality training loop."""
    tokenizer = _get_tokenizer()
    task_kwargs = json.loads(args.task_kwargs)
    config.structural_energy_lambda = args.lambda_struct

    print(f"\n{'='*70}")
    print(f"Criticality Training — Edge of Chaos")
    print(f"{'='*70}")

    # Create model
    model = create_model(config, device)
    is_fluid = isinstance(model, FluidNetModel)

    if args.resume_checkpoint:
        print(f"  Resuming from: {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  New params: {missing}")
        del ckpt

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Architecture: {'FluidNet' if is_fluid else 'flat'}")

    # Master World
    master = MasterWorld(
        n_rooms=task_kwargs.get("n_rooms_max", 15),
        n_objects=task_kwargs.get("n_objects", 4),
        space_size=task_kwargs.get("space_size", 150.0),
        connect_radius=task_kwargs.get("connect_radius", 40.0),
        locked_door_prob=task_kwargs.get("locked_door_prob", 0.3),
        seed=args.master_seed,
    )
    stats = master.get_topology_stats()
    print(f"  MasterWorld: {stats['n_rooms']} rooms, {stats['n_edges']} edges, "
          f"{stats['n_locked']} locked")
    print(f"  Mutate every {args.mutate_every} steps "
          f"(catastrophic {args.catastrophic_fraction:.0%} of edges)")

    # Task (for tokenization and episode generation)
    task = get_task("CW", tokenizer, seq_len=config.max_seq_len, **task_kwargs)

    # Metric monitor (shrink-and-perturb)
    metric_monitor = MetricMonitor(
        crystal_percentile=args.crystal_percentile,
        perturb_alpha=args.perturb_alpha,
        perturb_sigma=args.perturb_sigma,
        check_every=args.perturb_every,
    )
    print(f"  MetricMonitor: percentile={args.crystal_percentile}, "
          f"check every {args.perturb_every} steps")

    # Dynamic weight decay
    dyn_decay = DynamicWeightDecay(
        base_decay=args.weight_decay,
        min_decay=args.weight_decay * 0.1,
        max_decay=args.weight_decay * 3.0,
    )
    print(f"  DynamicWeightDecay: base={args.weight_decay}, "
          f"range=[{args.weight_decay*0.1:.3f}, {args.weight_decay*3:.3f}]")

    # Output
    out_dir = args.output_dir
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, "logs"))

    total_steps = args.max_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args.warmup_steps, total_steps)

    model_raw = model
    if config.use_torch_compile and device.type == "cuda":
        compiled_model = torch.compile(model, mode="default")
    else:
        compiled_model = model

    compiled_model.train()
    t0 = time.time()
    loss = torch.tensor(0.0)

    # Plasticity telemetry state
    last_mutation_step = -1
    post_mutation_ce_history = []
    pre_mutation_ce = 0.0
    adaptation_half_life = 0.0

    for step in range(total_steps):
        # --- Mechanism 1: Topology mutation ---
        if step > 0 and step % args.mutate_every == 0:
            # Save pre-mutation CE for adaptation tracking
            pre_mutation_ce = loss.item() if isinstance(loss, torch.Tensor) else loss

            result_str = master.catastrophic_mutate(args.catastrophic_fraction)
            last_mutation_step = step
            post_mutation_ce_history = []

            stats = master.get_topology_stats()
            print(f"\n  >>> MUTATION @ step {step}: {result_str}")
            print(f"      Topology: {stats['n_edges']} edges, "
                  f"{stats['n_locked']} locked, "
                  f"avg_dist={stats['avg_edge_dist']:.1f}\n")

            writer.add_scalar("mutation/n_edges", stats["n_edges"], step)
            writer.add_scalar("mutation/n_locked", stats["n_locked"], step)

            # Save pre-mutation checkpoint
            save_checkpoint(model, optimizer, config, step,
                          os.path.join(out_dir, "checkpoints", f"pre_mut_{step}.pt"))

        # --- Mechanism 2: Shrink-and-perturb (skip during warmup) ---
        if is_fluid and step >= args.warmup_steps:
            sp_stats = metric_monitor.check_and_perturb(model_raw, step)
            if sp_stats and sp_stats.get("n_perturbed", 0) > 0:
                print(f"  [step={step}] Shrink-perturb: "
                      f"{sp_stats['n_crystallized']} crystallized, "
                      f"{sp_stats['n_perturbed']} perturbed")
                writer.add_scalar("monitor/n_crystallized",
                                  sp_stats["n_crystallized"], step)
                writer.add_scalar("monitor/n_perturbed",
                                  sp_stats["n_perturbed"], step)

        # --- Generate batch from MasterWorld ---
        input_ids, labels, meta = generate_batch_from_master(
            master, task, args.batch_size, device)
        context_mask = meta.get("context_mask")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            if is_fluid:
                # Only pass room metadata when structural energy is active
                extra_kwargs = {}
                if args.lambda_struct > 0:
                    extra_kwargs["room_distances"] = meta.get("room_distances")
                    extra_kwargs["room_token_positions"] = meta.get("room_token_positions")
                    extra_kwargs["n_rooms"] = meta.get("n_rooms")
                result = compiled_model(
                    input_ids, labels=labels,
                    context_mask=context_mask,
                    **extra_kwargs,
                )
            else:
                result = compiled_model(input_ids, labels=labels)
            loss = result["loss"]

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [step={step}] WARNING: NaN/Inf loss, skipping")
            optimizer.zero_grad()
            scheduler.step()
            continue

        # --- Mechanism 3: Dynamic weight decay ---
        ce_val = result["ce_loss"].item()
        current_decay = dyn_decay.update(ce_val, optimizer)

        # --- Plasticity telemetry: adaptation tracking ---
        if last_mutation_step >= 0 and step > last_mutation_step:
            post_mutation_ce_history.append(ce_val)
            # Compute adaptation half-life: steps until CE recovers halfway
            if len(post_mutation_ce_history) >= 2:
                peak_ce = max(post_mutation_ce_history[:10]) if len(post_mutation_ce_history) >= 10 else max(post_mutation_ce_history)
                target = (peak_ce + pre_mutation_ce) / 2
                for i, ce in enumerate(post_mutation_ce_history):
                    if ce <= target:
                        adaptation_half_life = i + 1
                        break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # --- Mechanism 4: Metric volatility ---
        if is_fluid and step % args.log_every == 0:
            h_last = result.get("h_pre_norm")
            if h_last is not None:
                ctx_vol = model_raw.context_pool(h_last, context_mask)
                metric_monitor.compute_volatility(model_raw, h_last, ctx_vol)

        # --- Logging ---
        if step % args.log_every == 0:
            dt = time.time() - t0
            tok_s = args.batch_size * config.max_seq_len * (step + 1) / max(dt, 1e-6)

            m = model._orig_mod if hasattr(model, '_orig_mod') else model

            cv_val = result["metric_cv"]
            if isinstance(cv_val, torch.Tensor):
                cv_val = cv_val.item()

            kappa = result["avg_kappa"].item()
            w_norm = sum(p.data.norm().item() ** 2 for p in m.parameters()) ** 0.5

            extra = ""
            if is_fluid:
                t_local = result.get("avg_t_local", torch.tensor(0.0)).item()
                t_medium = result.get("avg_t_medium", torch.tensor(0.0)).item()
                t_global = result.get("avg_t_global", torch.tensor(0.0)).item()
                extra = f", t=[{t_local:.2f},{t_medium:.2f},{t_global:.2f}]"

            print(f"  [step={step}] loss={loss.item():.4f}, "
                  f"ce={ce_val:.4f}, "
                  f"cv={cv_val:.4f}, "
                  f"|k|={kappa:.4f}, "
                  f"W={w_norm:.1f}, "
                  f"wd={current_decay:.4f}, "
                  f"Vg={metric_monitor.last_volatility:.4f}, "
                  f"t½={adaptation_half_life:.0f}, "
                  f"tok/s={tok_s:.0f}{extra}")

            writer.add_scalar("loss/total", loss.item(), step)
            writer.add_scalar("loss/ce", ce_val, step)
            writer.add_scalar("metric/cv", cv_val, step)
            writer.add_scalar("metric/kappa", kappa, step)
            writer.add_scalar("metric/weight_norm", w_norm, step)
            writer.add_scalar("decay/weight_decay", current_decay, step)
            writer.add_scalar("plasticity/volatility",
                              metric_monitor.last_volatility, step)
            writer.add_scalar("plasticity/adaptation_half_life",
                              adaptation_half_life, step)
            writer.add_scalar("plasticity/total_perturbations",
                              metric_monitor.total_perturbations, step)

        if step > 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, config, step,
                          os.path.join(out_dir, "checkpoints", f"step_{step}.pt"))

    save_checkpoint(model, optimizer, config, total_steps,
                   os.path.join(out_dir, "checkpoints", "final.pt"),
                   extra={"master_mutations": master.mutation_count})
    writer.close()
    print(f"\n  Training complete. Final loss: {loss.item():.4f}")
    print(f"  Total mutations: {master.mutation_count}")
    print(f"  Total metric perturbations: {metric_monitor.total_perturbations}")


def _get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


def main():
    parser = argparse.ArgumentParser(description="Criticality Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_criticality")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--lambda_struct", type=float, default=0.0)
    parser.add_argument("--task_kwargs", type=str, default="{}",
                        help="JSON dict of CW task kwargs")
    parser.add_argument("--resume_checkpoint", type=str, default="")

    # Criticality-specific
    parser.add_argument("--mutate_every", type=int, default=2000,
                        help="Steps between topology mutations")
    parser.add_argument("--master_seed", type=int, default=42,
                        help="RNG seed for master world")
    parser.add_argument("--crystal_percentile", type=float, default=0.05,
                        help="Bottom percentile of dims to consider crystallized (0.05=5%)")
    parser.add_argument("--catastrophic_fraction", type=float, default=0.2,
                        help="Fraction of edges to mutate simultaneously")
    parser.add_argument("--perturb_alpha", type=float, default=0.9,
                        help="Shrink-and-perturb mixing coefficient")
    parser.add_argument("--perturb_sigma", type=float, default=0.01,
                        help="Noise std for metric perturbation")
    parser.add_argument("--perturb_every", type=int, default=500,
                        help="Steps between metric health checks")

    args = parser.parse_args()
    config = FGNConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mem_frac = float(os.environ.get("CUDA_MEMORY_FRACTION", "0"))
    if mem_frac > 0 and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(mem_frac)
        print(f"CUDA memory fraction capped at {mem_frac:.0%}")

    print(f"Device: {device}")
    print(f"Mutate every: {args.mutate_every} steps")
    print(f"Metric perturb: alpha={args.perturb_alpha}, sigma={args.perturb_sigma}")
    print(f"Total steps: {args.max_steps}")

    train(args, config, device)


if __name__ == "__main__":
    main()
