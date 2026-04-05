#!/usr/bin/env python3
"""Train geometric coupling between LiquidARC and Qwen3-4B.

Training objectives:
  A. NTP improvement: Does LiquidARC prefix reduce Qwen3's perplexity?
  B. State prediction: Does Qwen3's read-back predict LiquidARC's next state?

Only coupling layers (W_inject, W_read) are trained.
Optionally LiquidARC dynamics at 100x slower LR.
Qwen3 is completely frozen.
"""

import argparse
import math
import os
import random
import sys
import time
import yaml
from pathlib import Path

import torch
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel
from liquid_arc.coupling import GeometricCoupling
from liquid_arc.coupled_system import CoupledSystem


def load_arc_model(config: LiquidARCConfig, checkpoint_path: str,
                   device: torch.device) -> LiquidARCModel:
    """Load LiquidARC fluid metric model from checkpoint."""
    model = LiquidARCModel(config).to(device).to(torch.bfloat16)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    # Strip _orig_mod prefix from compiled checkpoints
    cleaned = {}
    for k, v in state_dict.items():
        clean_k = k.replace("_orig_mod.", "")
        cleaned[clean_k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} — {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} — {unexpected[:5]}")

    print(f"  LiquidARC loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    return model


def load_qwen_model(model_path: str, device: torch.device, gradient_checkpointing: bool = True):
    """Load frozen Qwen3-4B from local path."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading Qwen3-4B from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={'': device},
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Qwen3-4B loaded: {n_params/1e9:.2f}B params, d_model={model.config.hidden_size}")
    return model, tokenizer


def generate_event_sequences(tokenizer, n_sequences: int = 100,
                             n_events: int = 8, event_len: int = 128,
                             dataset_name: str = "wikitext",
                             dataset_config: str = "wikitext-2-raw-v1"):
    """Generate sequential event data from text corpus.

    Splits text into chunks that serve as sequential "events" —
    mimicking the Mind's conversation event stream.
    """
    from datasets import load_dataset

    print(f"  Loading {dataset_name}/{dataset_config}...")
    ds = load_dataset(dataset_name, dataset_config, split="train")
    all_text = "\n".join(line for line in ds["text"] if line.strip())
    all_tokens = tokenizer.encode(all_text)
    print(f"  Corpus: {len(all_tokens):,} tokens")

    # Generate event sequences: contiguous chunks
    chunk_len = event_len  # tokens per event
    seq_len = chunk_len * n_events
    max_start = len(all_tokens) - seq_len - 1

    sequences = []
    for _ in range(n_sequences):
        start = random.randint(0, max_start)
        events = []
        for j in range(n_events):
            event_start = start + j * chunk_len
            event_tokens = all_tokens[event_start:event_start + chunk_len]
            event_text = tokenizer.decode(event_tokens, skip_special_tokens=True)
            events.append(event_text)
        sequences.append(events)

    # Also generate eval sequences from validation split
    ds_val = load_dataset(dataset_name, dataset_config, split="validation")
    val_text = "\n".join(line for line in ds_val["text"] if line.strip())
    val_tokens = tokenizer.encode(val_text)

    eval_sequences = []
    max_start_val = len(val_tokens) - seq_len - 1
    for _ in range(min(20, max(1, max_start_val // seq_len))):
        start = random.randint(0, max(0, max_start_val))
        events = []
        for j in range(n_events):
            event_start = start + j * chunk_len
            event_tokens = val_tokens[event_start:event_start + chunk_len]
            event_text = tokenizer.decode(event_tokens, skip_special_tokens=True)
            events.append(event_text)
        eval_sequences.append(events)

    print(f"  Generated {len(sequences)} train sequences, {len(eval_sequences)} eval sequences")
    print(f"  Each: {n_events} events × ~{event_len} tokens")
    return sequences, eval_sequences


def evaluate_perplexity(system: CoupledSystem, eval_sequences: list,
                        device: torch.device) -> dict:
    """Compare perplexity: baseline vs random prefix vs LiquidARC prefix."""
    system.eval()

    baseline_losses = []
    random_losses = []
    coupled_losses = []

    for seq_idx, events in enumerate(eval_sequences[:5]):
        system.reset_state()

        for i, event_text in enumerate(events[:-1]):
            # Accumulate state through events
            h_state = system.observe_event_arc(event_text, device)

        # Test on the LAST event (has temporal context from previous events)
        test_text = events[-1]
        if len(test_text.strip()) < 10:
            continue

        # Baseline: Qwen3 alone
        logits_base, ids_base = system.baseline_forward(test_text, device)
        if ids_base.shape[1] < 2:
            continue
        loss_base = F.cross_entropy(
            logits_base[:, :-1, :].contiguous().view(-1, logits_base.size(-1)),
            ids_base[:, 1:].contiguous().view(-1),
        )
        baseline_losses.append(loss_base.item())

        # Random prefix
        logits_rand, ids_rand = system.random_prefix_forward(test_text, device)
        loss_rand = F.cross_entropy(
            logits_rand[:, :-1, :].contiguous().view(-1, logits_rand.size(-1)),
            ids_rand[:, 1:].contiguous().view(-1),
        )
        random_losses.append(loss_rand.item())

        # Coupled: LiquidARC prefix
        result = system.coupled_forward(h_state, test_text, device)
        loss_coupled = system.compute_ntp_loss(result['logits'], result['input_ids'])
        coupled_losses.append(loss_coupled.item())

    system.train()

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else float('nan')

    baseline_ppl = math.exp(safe_mean(baseline_losses))
    random_ppl = math.exp(safe_mean(random_losses))
    coupled_ppl = math.exp(safe_mean(coupled_losses))

    return {
        'baseline_ppl': baseline_ppl,
        'random_prefix_ppl': random_ppl,
        'coupled_ppl': coupled_ppl,
        'ppl_improvement': (baseline_ppl - coupled_ppl) / baseline_ppl * 100,
        'n_eval': len(baseline_losses),
    }


def get_arc_diagnostics(arc_model: LiquidARCModel, h0: torch.Tensor) -> dict:
    """Get metric CV and tau stats from current state."""
    raw = getattr(arc_model, '_orig_mod', arc_model)
    with torch.no_grad():
        g = raw.dynamics.compute_metric_diag(h0)
        g_mean = g.mean(dim=(0, 1))  # [d]
        cv = g_mean.std() / (g_mean.mean() + 1e-8)

        tau = raw.dynamics.compute_tau(h0)
        tau_avg = tau.mean().item()
        tau_std = tau.std().item()

    return {'metric_cv': cv.item(), 'tau_avg': tau_avg, 'tau_std': tau_std}


def main():
    parser = argparse.ArgumentParser(description="Train geometric coupling")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/geometric_coupling")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # ── Load LiquidARC ──
    print("═══ Loading LiquidARC ═══")
    arc_config = LiquidARCConfig(
        d_model=cfg['d_model'],
        d_metric=cfg['d_metric'],
        d_metric_bottleneck=cfg.get('d_metric_bottleneck', 0),
        metric_rank=cfg.get('metric_rank', 0),
        d_ffn=cfg['d_ffn'],
        n_ode_steps=cfg['n_ode_steps'],
        tau_min=cfg['tau_min'],
        tau_max=cfg.get('tau_max', 3.0),
        t_diffusion_init=cfg.get('t_diffusion_init', 1.0),
        curvature_lambda=cfg.get('curvature_lambda', 0.05),
        tau_var_lambda=cfg.get('tau_var_lambda', 0.001),
        cv_floor_target=cfg.get('cv_floor_target', 3.0),
        cv_ceiling_target=cfg.get('cv_ceiling_target', 8.0),
        cv_floor_lambda=cfg.get('cv_floor_lambda', 0.1),
    )
    arc_model = load_arc_model(arc_config, cfg['arc_checkpoint'], device)
    arc_model.eval()  # Mostly frozen (dynamics at 100x slower LR)

    # ── Load Qwen3-4B ──
    print("\n═══ Loading Qwen3-4B ═══")
    qwen_model, tokenizer = load_qwen_model(
        cfg['qwen_model_path'], device,
        gradient_checkpointing=cfg.get('gradient_checkpointing', True))

    d_qwen = qwen_model.config.hidden_size
    print(f"  Verified d_qwen = {d_qwen}")

    # ── Create coupling ──
    print("\n═══ Creating GeometricCoupling ═══")
    coupling = GeometricCoupling(
        d_arc=cfg['d_model'],
        d_qwen=d_qwen,
        n_virtual_tokens=cfg.get('n_virtual_tokens', 8),
    ).to(device).to(torch.bfloat16)
    print(f"  Coupling: {coupling.param_count()/1e6:.2f}M params")
    print(f"  Virtual tokens: {coupling.n_virtual_tokens}")

    # ── Build coupled system ──
    system = CoupledSystem(
        arc_model, qwen_model, tokenizer, coupling,
        gradient_checkpointing=cfg.get('gradient_checkpointing', True))

    # Memory report
    mem_gb = torch.cuda.memory_allocated(device) / 1e9
    print(f"\n  Total VRAM: {mem_gb:.1f} GB")

    # ── Generate training data ──
    print("\n═══ Generating training data ═══")
    train_seqs, eval_seqs = generate_event_sequences(
        tokenizer,
        n_sequences=200,
        n_events=cfg.get('n_events_per_sequence', 8),
        event_len=cfg.get('event_seq_len', 128),
        dataset_name=cfg.get('text_dataset', 'wikitext'),
        dataset_config=cfg.get('text_dataset_config', 'wikitext-2-raw-v1'),
    )

    # ── Optimizer ──
    print("\n═══ Setting up optimizer ═══")
    param_groups = [
        {'params': coupling.parameters(), 'lr': float(cfg.get('coupling_lr', 3e-4))},
    ]

    # Optionally include LiquidARC dynamics at 100x slower LR
    arc_dynamics_lr = float(cfg.get('arc_dynamics_lr', 0))
    if arc_dynamics_lr > 0:
        raw_arc = getattr(arc_model, '_orig_mod', arc_model)
        for p in raw_arc.dynamics.parameters():
            p.requires_grad_(True)
        param_groups.append({
            'params': list(raw_arc.dynamics.parameters()),
            'lr': arc_dynamics_lr,
        })
        print(f"  LiquidARC dynamics: LR={arc_dynamics_lr} (unfrozen)")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(cfg.get('weight_decay', 0.01)))

    ntp_weight = float(cfg.get('ntp_weight', 1.0))
    state_pred_weight = float(cfg.get('state_pred_weight', 0.1))
    max_steps = int(cfg.get('max_steps', 5000))
    eval_interval = int(cfg.get('eval_interval', 100))
    log_interval = int(cfg.get('log_interval', 10))

    # ── Initial evaluation ──
    print("\n═══ Initial evaluation ═══")
    with torch.no_grad():
        init_eval = evaluate_perplexity(system, eval_seqs, device)
    print(f"  Baseline PPL: {init_eval['baseline_ppl']:.1f}")
    print(f"  Random prefix PPL: {init_eval['random_prefix_ppl']:.1f}")
    print(f"  Coupled PPL: {init_eval['coupled_ppl']:.1f}")

    # ── Training loop ──
    print(f"\n═══ Training ({max_steps} steps) ═══")
    coupling.train()

    step = 0
    ntp_losses = []
    state_losses = []
    total_losses = []
    cv_history = []

    t_start = time.time()

    while step < max_steps:
        # Pick a random event sequence
        events = random.choice(train_seqs)
        system.reset_state()

        for i in range(len(events) - 1):
            if step >= max_steps:
                break

            current_event = events[i]
            next_event = events[i + 1]

            # Skip very short events
            if len(current_event.strip()) < 10 or len(next_event.strip()) < 10:
                continue

            # ── LiquidARC observes current event ──
            with torch.no_grad():
                h_current = system.observe_event_arc(current_event, device)

            # ── Coupled forward: Qwen3 processes NEXT event with LiquidARC prefix ──
            result = system.coupled_forward(h_current, next_event, device)

            # ── NTP loss ──
            ntp_loss = system.compute_ntp_loss(result['logits'], result['input_ids'])

            # ── State prediction loss ──
            with torch.no_grad():
                h_next = system.observe_event_arc(next_event, device)
            state_loss = system.compute_state_pred_loss(result['arc_signal'], h_next)

            # ── Combined loss ──
            loss = ntp_weight * ntp_loss + state_pred_weight * state_loss

            # ── Backward + step ──
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping on coupling params
            torch.nn.utils.clip_grad_norm_(coupling.parameters(), 1.0)

            # NaN scrubbing (pre-existing bfloat16 SDPA issue)
            for pg in optimizer.param_groups:
                for p in pg['params']:
                    if p.grad is not None and p.grad.isnan().any():
                        p.grad.nan_to_num_(nan=0.0)

            optimizer.step()

            # ── Logging ──
            ntp_losses.append(ntp_loss.item())
            state_losses.append(state_loss.item())
            total_losses.append(loss.item())
            step += 1

            if step % log_interval == 0:
                elapsed = time.time() - t_start
                avg_ntp = sum(ntp_losses[-log_interval:]) / min(log_interval, len(ntp_losses))
                avg_state = sum(state_losses[-log_interval:]) / min(log_interval, len(state_losses))
                avg_total = sum(total_losses[-log_interval:]) / min(log_interval, len(total_losses))

                # Get diagnostics
                if system._h_state is not None:
                    diag = get_arc_diagnostics(arc_model, system._h_state)
                    cv_history.append(diag['metric_cv'])
                    cv_str = f"CV={diag['metric_cv']:.2f} τ={diag['tau_avg']:.2f}"
                else:
                    cv_str = "CV=? τ=?"

                steps_per_sec = step / elapsed
                print(f"  step {step:5d} | loss={avg_total:.3f} ntp={avg_ntp:.3f} "
                      f"state={avg_state:.3f} | {cv_str} | "
                      f"{steps_per_sec:.1f} step/s")

            # ── Evaluation ──
            if step % eval_interval == 0:
                with torch.no_grad():
                    eval_result = evaluate_perplexity(system, eval_seqs, device)
                print(f"  ── EVAL step {step}: "
                      f"baseline={eval_result['baseline_ppl']:.1f} "
                      f"random={eval_result['random_prefix_ppl']:.1f} "
                      f"coupled={eval_result['coupled_ppl']:.1f} "
                      f"(Δ={eval_result['ppl_improvement']:+.1f}%)")

                # Save checkpoint
                ckpt_path = os.path.join(args.output_dir, "checkpoints",
                                         f"step_{step}.pt")
                torch.save({
                    'step': step,
                    'coupling_state_dict': coupling.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'eval_result': eval_result,
                    'config': cfg,
                }, ckpt_path)

    # ── Final evaluation ──
    print("\n═══ Final evaluation ═══")
    with torch.no_grad():
        final_eval = evaluate_perplexity(system, eval_seqs, device)
    print(f"  Baseline PPL: {final_eval['baseline_ppl']:.1f}")
    print(f"  Random prefix PPL: {final_eval['random_prefix_ppl']:.1f}")
    print(f"  Coupled PPL: {final_eval['coupled_ppl']:.1f}")
    print(f"  Improvement: {final_eval['ppl_improvement']:+.1f}%")

    # ── Temporal context test ──
    print("\n═══ Temporal context test ═══")
    temporal_test_events = [
        "The research meeting is scheduled for Thursday at 3pm in Conference Room B.",
        "Dr. Chen will present the latest results on geometric neural networks.",
        "The projector in Conference Room B has been reported as broken since Monday.",
        "Several team members have requested video conferencing capability.",
    ]
    test_query = "What preparations should be made for the Thursday presentation?"

    system.reset_state()
    for event in temporal_test_events:
        h_state = system.observe_event_arc(event, device)

    # With prefix
    with torch.no_grad():
        coupled_result = system.coupled_forward(h_state, test_query, device)
        coupled_ids = coupled_result['input_ids']
        coupled_logits = coupled_result['logits']

        # Without prefix
        base_logits, base_ids = system.baseline_forward(test_query, device)

    # Compare top predictions
    coupled_topk = coupled_logits[0, -1].topk(10)
    base_topk = base_logits[0, -1].topk(10)

    print(f"  Context events: {len(temporal_test_events)}")
    print(f"  Query: '{test_query}'")
    print(f"  Baseline top tokens: {[tokenizer.decode([t]) for t in base_topk.indices]}")
    print(f"  Coupled top tokens:  {[tokenizer.decode([t]) for t in coupled_topk.indices]}")

    # ── Knowledge navigation test ──
    print("\n═══ Knowledge navigation test ═══")
    topic_sets = {
        'physics': [
            "The experiment measured gravitational wave amplitude at the LIGO detector.",
            "Quantum entanglement between photon pairs was confirmed at 100km distance.",
            "The Higgs boson mass measurement was refined to 125.35 GeV.",
        ],
        'ecology': [
            "The coral reef bleaching event covered 60% of the Great Barrier Reef.",
            "Monarch butterfly migration patterns shifted 200km northward this decade.",
            "Amazon deforestation reduced primary forest cover by 17% since 2000.",
        ],
        'mathematics': [
            "The Riemann hypothesis implies specific zero distribution of the zeta function.",
            "Grothendieck's algebraic geometry unified number theory and topology.",
            "The Langlands program connects representation theory to number theory.",
        ],
    }

    query = "What is the most important recent development?"
    topic_predictions = {}

    for topic, events in topic_sets.items():
        system.reset_state()
        for event in events:
            h_state = system.observe_event_arc(event, device)

        with torch.no_grad():
            result = system.coupled_forward(h_state, query, device)
            topk = result['logits'][0, -1].topk(10)
            top_tokens = [tokenizer.decode([t]) for t in topk.indices]
            topic_predictions[topic] = top_tokens

    print(f"  Query: '{query}'")
    for topic, tokens in topic_predictions.items():
        print(f"  {topic:12s}: {tokens}")

    # ── Save final checkpoint ──
    final_path = os.path.join(args.output_dir, "final.pt")
    torch.save({
        'step': step,
        'coupling_state_dict': coupling.state_dict(),
        'config': cfg,
        'init_eval': init_eval,
        'final_eval': final_eval,
        'cv_history': cv_history,
        'ntp_losses': ntp_losses,
        'state_losses': state_losses,
    }, final_path)
    print(f"\n  Saved final checkpoint: {final_path}")

    # ── Memory report ──
    mem_gb = torch.cuda.memory_allocated(device) / 1e9
    mem_peak = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  Memory: current={mem_gb:.1f}GB, peak={mem_peak:.1f}GB")

    elapsed = time.time() - t_start
    print(f"\n═══ Training complete: {step} steps in {elapsed/60:.1f} min ═══")


if __name__ == "__main__":
    main()
