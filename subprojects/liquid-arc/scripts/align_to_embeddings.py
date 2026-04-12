#!/usr/bin/env python3
"""Phase 2b: Align ODE state distribution to LLM embedding space.

Instead of backpropping through the full 30B LLM (slow, vanishing gradients),
directly align the ODE output to match the LLM's embedding table distribution.

The embedding table defines the submanifold of R^2688 where Nemotron expects
its inputs to live. The ODE must learn to produce states in this region.

Three alignment losses:
  1. Moment matching: ODE output mean/std matches embedding table mean/std per dim
  2. Nearest-neighbor: each ODE position should be close to SOME real embedding
  3. Projection: ODE output projected through embed table should produce valid logits

No LLM forward pass needed. Just the embedding table (705MB on CPU).
Fast, strong gradient, minimal memory.

Usage:
    python scripts/align_to_embeddings.py \
        --ode_checkpoint PRECIOUS_CHECKPOINTS/distilled_2688_step3000.pt \
        --llm_path /workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
        --output_dir output/embed_aligned_2688 \
        --max_steps 5000
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
from liquid_arc.dynamics import ContinuousDynamics
from liquid_arc.context_pool import ContextPool
from liquid_arc.solver import euler_solve


def load_embedding_stats(llm_path: str, device: str = 'cuda'):
    """Load LLM embedding table and compute distribution statistics.

    Returns the embedding weight matrix and precomputed stats
    for alignment losses.
    """
    import json
    from safetensors.torch import load_file

    index_file = os.path.join(llm_path, 'model.safetensors.index.json')
    with open(index_file) as f:
        index = json.load(f)

    # Find embedding key
    emb_key = None
    for candidate in ['model.embed_tokens.weight', 'backbone.embeddings.weight',
                       'model.embedding.word_embeddings.weight']:
        if candidate in index['weight_map']:
            emb_key = candidate
            break
    if emb_key is None:
        raise ValueError(f"Cannot find embedding key in {llm_path}")

    shard = index['weight_map'][emb_key]
    weights = load_file(os.path.join(llm_path, shard))
    emb_weight = weights[emb_key].float()  # [vocab, d]

    vocab, d = emb_weight.shape
    print(f"  Embedding table: [{vocab}, {d}]")

    # Compute per-dimension statistics
    emb_mean = emb_weight.mean(dim=0)  # [d]
    emb_std = emb_weight.std(dim=0)    # [d]
    emb_norm_mean = emb_weight.norm(dim=-1).mean().item()
    emb_norm_std = emb_weight.norm(dim=-1).std().item()

    print(f"  Per-dim mean: [{emb_mean.min():.4f}, {emb_mean.max():.4f}]")
    print(f"  Per-dim std:  [{emb_std.min():.4f}, {emb_std.max():.4f}]")
    print(f"  Token norm:   {emb_norm_mean:.3f} ± {emb_norm_std:.3f}")

    # Sample subset for nearest-neighbor (full vocab too large)
    n_sample = min(10000, vocab)
    sample_idx = torch.randperm(vocab)[:n_sample]
    emb_sample = emb_weight[sample_idx].to(device)  # [n_sample, d]

    return {
        'weight': emb_weight,        # full table, CPU
        'sample': emb_sample,        # subset, GPU
        'mean': emb_mean.to(device), # [d]
        'std': emb_std.to(device),   # [d]
        'norm_mean': emb_norm_mean,
        'norm_std': emb_norm_std,
        'd': d,
        'vocab': vocab,
    }


def moment_loss(h: torch.Tensor, emb_stats: dict) -> torch.Tensor:
    """Match per-dimension mean and std of ODE output to embedding distribution.

    h: [B, N, d] — ODE output positions
    """
    # Flatten to [B*N, d]
    flat = h.reshape(-1, h.shape[-1])

    h_mean = flat.mean(dim=0)  # [d]
    h_std = flat.std(dim=0)    # [d]

    mean_loss = F.mse_loss(h_mean, emb_stats['mean'])
    std_loss = F.mse_loss(h_std, emb_stats['std'])

    return mean_loss + std_loss


def norm_loss(h: torch.Tensor, target_norm: float) -> torch.Tensor:
    """Match per-token norm to embedding norm.

    h: [B, N, d]
    """
    h_norms = h.norm(dim=-1)  # [B, N]
    return F.mse_loss(h_norms, torch.full_like(h_norms, target_norm))


def nearest_neighbor_loss(h: torch.Tensor, emb_sample: torch.Tensor) -> torch.Tensor:
    """Each ODE position should be close to some real embedding.

    h: [B, N, d]
    emb_sample: [K, d] — sampled embeddings on GPU

    Uses cosine distance to find nearest neighbor.
    """
    flat = h.reshape(-1, h.shape[-1])  # [B*N, d]

    # Normalize for cosine similarity
    h_norm = F.normalize(flat, dim=-1)         # [B*N, d]
    e_norm = F.normalize(emb_sample, dim=-1)   # [K, d]

    # Cosine similarity: [B*N, K]
    sim = h_norm @ e_norm.T

    # Max similarity per ODE position (best match)
    max_sim = sim.max(dim=-1).values  # [B*N]

    # Loss: 1 - cosine similarity (want to maximize similarity)
    return (1.0 - max_sim).mean()


def projection_loss(h: torch.Tensor, emb_weight: torch.Tensor,
                     tokenizer, device: str) -> torch.Tensor:
    """ODE output projected through embedding table should produce valid token logits.

    h: [B, N, d] — ODE output
    emb_weight: [vocab, d] — full embedding table

    Projects ODE output to logits via h @ emb.T, then checks if the
    resulting distribution is peaky (high confidence on real tokens)
    rather than flat (uniform noise).
    """
    flat = h.reshape(-1, h.shape[-1])  # [B*N, d]

    # Project to logits: [B*N, vocab] — this is large, use a subset
    emb_sub = emb_weight[:5000].to(device)  # first 5K tokens
    logits = flat @ emb_sub.T  # [B*N, 5000]

    # Entropy of the softmax distribution — should be LOW (peaky)
    # For random directions, entropy ≈ log(5000) ≈ 8.5
    # For aligned embeddings, entropy should be much lower
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    # Target: entropy well below uniform
    return entropy


def split_parameters(dynamics, context_pool):
    """Split into geometric (slow) and content (fast) parameters."""
    geo_names = ['metric_net', 'tau_net', 't_diffusion', 'alpha_logit']
    geo_params, content_params = [], []
    for name, p in dynamics.named_parameters():
        if any(g in name for g in geo_names):
            geo_params.append(p)
        else:
            content_params.append(p)
    for p in context_pool.parameters():
        geo_params.append(p)
    return geo_params, content_params


def main():
    parser = argparse.ArgumentParser(description="Phase 2b: Embed alignment")
    parser.add_argument("--ode_checkpoint", type=str, required=True)
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/embed_aligned")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--n_events", type=int, default=8)
    parser.add_argument("--event_len", type=int, default=64)
    parser.add_argument("--geo_lr", type=float, default=1e-4)
    parser.add_argument("--content_lr", type=float, default=1e-2)
    parser.add_argument("--moment_weight", type=float, default=1.0)
    parser.add_argument("--norm_weight", type=float, default=1.0)
    parser.add_argument("--nn_weight", type=float, default=0.5)
    parser.add_argument("--proj_weight", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # ── Load ODE ──
    print("═══ Loading ODE ═══")
    ode_ckpt = torch.load(args.ode_checkpoint, map_location=device, weights_only=False)
    ode_config = ode_ckpt['config']
    dynamics = ContinuousDynamics(ode_config).to(device)
    context_pool = ContextPool(ode_config).to(device)

    ode_sd = ode_ckpt.get('model_state_dict', {})
    dynamics.load_state_dict(
        {k.replace('dynamics.', ''): v for k, v in ode_sd.items()
         if k.startswith('dynamics.')}, strict=False)
    context_pool.load_state_dict(
        {k.replace('context_pool.', ''): v for k, v in ode_sd.items()
         if k.startswith('context_pool.')}, strict=False)
    print(f"  d={ode_config.d_model}, dynamics={sum(p.numel() for p in dynamics.parameters())/1e6:.1f}M")

    # ── Load embedding stats ──
    print(f"\n═══ Loading LLM embeddings ═══")
    emb_stats = load_embedding_stats(args.llm_path, device)

    # ── Load tokenizer for generating input events ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Text data ──
    print(f"\n═══ Loading text data ═══")
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text = ' '.join([t for t in ds['text'] if len(t.strip()) > 50])
    all_tokens = tokenizer.encode(text)
    print(f"  {len(all_tokens):,} tokens")

    # Embedding lookup on CPU
    embed_table = emb_stats['weight']  # [vocab, d] on CPU

    # ── Freeze geometry, only train content ──
    geo_params, content_params = split_parameters(dynamics, context_pool)
    for p in geo_params:
        p.requires_grad_(False)
    n_frozen = sum(p.numel() for p in geo_params)
    n_trainable = sum(p.numel() for p in content_params)
    optimizer = torch.optim.AdamW([
        {'params': content_params, 'lr': args.content_lr},
    ], weight_decay=0.01)
    print(f"\n  Geometry FROZEN: {n_frozen/1e6:.1f}M params")
    print(f"  Content trainable: {n_trainable/1e6:.1f}M params at LR={args.content_lr}")

    # ── Training ──
    print(f"\n═══ Training ({args.max_steps} steps) ═══")
    dynamics.train()
    t0 = time.time()
    metrics = {}
    d = ode_config.d_model
    n_steps = ode_config.n_ode_steps

    for step in range(1, args.max_steps + 1):
        # Sample text events, embed through LLM table
        seq_len = args.n_events * args.event_len
        start = random.randint(0, max(0, len(all_tokens) - seq_len - 1))

        event_embeds = []
        for ev_idx in range(args.n_events):
            ev_start = start + ev_idx * args.event_len
            ev_ids = all_tokens[ev_start:ev_start + args.event_len]
            # Lookup embeddings from table (CPU → GPU)
            ev_emb = embed_table[ev_ids].to(device)  # [T, d]
            event_embeds.append(ev_emb)

        h = torch.stack(event_embeds).unsqueeze(0)  # [1, n_events*event_len, d]
        # Take mean per event to get [1, n_events, d]
        h = h.view(1, args.n_events, args.event_len, d).mean(dim=2)  # [1, n_events, d]

        # Run ODE
        context = context_pool(h)
        dynamics.set_context(context, mask=None)
        dynamics.set_n_steps(n_steps)
        h_ode = euler_solve(dynamics, h, t_span=(0.0, 2.0), n_steps=n_steps)

        # ── Alignment losses ──
        loss_moment = moment_loss(h_ode, emb_stats)
        loss_norm = norm_loss(h_ode, emb_stats['norm_mean'])
        loss_nn = nearest_neighbor_loss(h_ode, emb_stats['sample'])
        loss_proj = projection_loss(h_ode, emb_stats['weight'], tokenizer, device)

        loss = (args.moment_weight * loss_moment +
                args.norm_weight * loss_norm +
                args.nn_weight * loss_nn +
                args.proj_weight * loss_proj)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(dynamics.parameters()) + list(context_pool.parameters()), 1.0)
        optimizer.step()

        # ── Metrics ──
        with torch.no_grad():
            metrics['loss'] = metrics.get('loss', 0) + loss.item()
            metrics['moment'] = metrics.get('moment', 0) + loss_moment.item()
            metrics['norm'] = metrics.get('norm', 0) + loss_norm.item()
            metrics['nn'] = metrics.get('nn', 0) + loss_nn.item()
            metrics['proj'] = metrics.get('proj', 0) + loss_proj.item()
            metrics['h_norm'] = metrics.get('h_norm', 0) + h_ode.norm(dim=-1).mean().item()
            metrics['n'] = metrics.get('n', 0) + 1

        if step % args.log_every == 0:
            n = metrics['n']
            elapsed = time.time() - t0
            h_n = metrics['h_norm'] / n
            target_n = emb_stats['norm_mean']
            print(f"  step {step:>5d} | loss={metrics['loss']/n:.4f} "
                  f"moment={metrics['moment']/n:.4f} norm={metrics['norm']/n:.4f} "
                  f"nn={metrics['nn']/n:.4f} proj={metrics['proj']/n:.4f} | "
                  f"h_norm={h_n:.3f}/{target_n:.3f} "
                  f"ratio={h_n/target_n:.2f}x | "
                  f"{step/elapsed:.1f} step/s")
            metrics = {}

        # ── Save ──
        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, "checkpoints", f"step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': {
                    **{f'dynamics.{k}': v for k, v in dynamics.state_dict().items()},
                    **{f'context_pool.{k}': v for k, v in context_pool.state_dict().items()},
                },
                'config': ode_config,
                'phase': 'embed_aligned',
            }, ckpt_path)
            print(f"  → Saved {ckpt_path}")

    print(f"\nDone. Total time: {(time.time()-t0)/60:.1f} minutes")


if __name__ == '__main__':
    main()
