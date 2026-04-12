#!/usr/bin/env python3
"""Phase 2: NTP alignment — train ODE to produce LLM-compatible distributions.

After Phase 1 (ARC distillation), the ODE has routing STRUCTURE but ARC CONTENT.
This script trains the ODE through NTP loss with the frozen LLM, teaching it
what distributions the LLM can actually use.

The ODE processes text events → accumulates state → state becomes prefix tokens
→ LLM generates with prefix → NTP loss → gradient flows back through frozen LLM
to ODE → ODE learns text-compatible distributions.

Usage:
    python scripts/train_ntp_alignment.py \
        --ode_checkpoint PRECIOUS_CHECKPOINTS/distilled_2688_step3000.pt \
        --llm_path /workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
        --output_dir output/ntp_aligned_2688 \
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


def split_parameters(dynamics, context_pool):
    """Split into geometric (slow LR) and content (fast LR) parameters."""
    geo_names = ['metric_net', 'tau_net', 't_diffusion', 'alpha_logit']
    geo_params = []
    content_params = []
    for name, p in dynamics.named_parameters():
        if any(g in name for g in geo_names):
            geo_params.append(p)
        else:
            content_params.append(p)
    for p in context_pool.parameters():
        geo_params.append(p)
    return geo_params, content_params


def main():
    parser = argparse.ArgumentParser(description="Phase 2: NTP alignment")
    parser.add_argument("--ode_checkpoint", type=str, required=True,
                        help="Phase 1 distilled ODE checkpoint")
    parser.add_argument("--llm_path", type=str, required=True,
                        help="Path to frozen LLM (Nemotron FP8)")
    parser.add_argument("--output_dir", type=str, default="output/ntp_aligned")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--n_events", type=int, default=8,
                        help="Events per training sequence")
    parser.add_argument("--event_len", type=int, default=128,
                        help="Tokens per event")
    parser.add_argument("--geo_lr", type=float, default=1e-5,
                        help="LR for geometric params (slow)")
    parser.add_argument("--content_lr", type=float, default=1e-3,
                        help="LR for content params (fast)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # ── Load ODE from Phase 1 checkpoint ──
    print("═══ Loading ODE (Phase 1 distilled) ═══")
    ode_ckpt = torch.load(args.ode_checkpoint, map_location=device, weights_only=False)
    ode_config = ode_ckpt.get('config')
    if ode_config is None:
        raise ValueError("ODE checkpoint must contain config")

    dynamics = ContinuousDynamics(ode_config).to(device)
    context_pool = ContextPool(ode_config).to(device)

    ode_sd = ode_ckpt.get('model_state_dict', {})
    # Extract dynamics and context_pool weights
    dyn_sd = {}
    for k, v in ode_sd.items():
        if k.startswith('dynamics.'):
            dyn_sd[k.replace('dynamics.', '')] = v
        elif k.startswith('context_pool.'):
            dyn_sd[k] = v  # keep prefix for holder loading

    dynamics.load_state_dict(
        {k: v for k, v in ode_sd.items() if k.startswith('dynamics.')},
        strict=False)

    # Load context_pool separately
    cp_sd = {k.replace('context_pool.', ''): v
             for k, v in ode_sd.items() if k.startswith('context_pool.')}
    context_pool.load_state_dict(cp_sd, strict=False)

    d = ode_config.d_model
    n_steps = ode_config.n_ode_steps
    print(f"  d={d}, n_steps={n_steps}")
    print(f"  Dynamics: {sum(p.numel() for p in dynamics.parameters())/1e6:.2f}M")

    # ── Load frozen LLM ──
    print(f"\n═══ Loading LLM (frozen) ═══")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_path, device_map='auto', trust_remote_code=True,
        offload_folder='/tmp/offload',
    )
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    # Enable gradient checkpointing for memory
    try:
        llm.gradient_checkpointing_enable()
    except Exception:
        pass

    embed_fn = llm.get_input_embeddings()
    llm_d = embed_fn.weight.shape[1]
    assert llm_d == d, f"LLM d_model={llm_d} != ODE d_model={d}"
    print(f"  LLM: d={llm_d}, vocab={embed_fn.weight.shape[0]}")
    print(f"  Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # ── Training data: WikiText ──
    print(f"\n═══ Loading training data ═══")
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text = ' '.join([t for t in ds['text'] if len(t.strip()) > 50])
    tokens = tokenizer.encode(text)
    print(f"  {len(tokens):,} tokens from WikiText-2")

    val_ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    val_text = ' '.join([t for t in val_ds['text'] if len(t.strip()) > 50])
    val_tokens = tokenizer.encode(val_text)
    print(f"  {len(val_tokens):,} validation tokens")

    # ── Optimizer ──
    geo_params, content_params = split_parameters(dynamics, context_pool)
    optimizer = torch.optim.AdamW([
        {'params': geo_params, 'lr': args.geo_lr},
        {'params': content_params, 'lr': args.content_lr},
    ], weight_decay=0.01)
    print(f"\n  Geo LR: {args.geo_lr} ({sum(p.numel() for p in geo_params)/1e6:.1f}M)")
    print(f"  Content LR: {args.content_lr} ({sum(p.numel() for p in content_params)/1e6:.1f}M)")
    print(f"  Ratio: {args.content_lr/args.geo_lr:.0f}x")

    # ── Training loop ──
    print(f"\n═══ Training ({args.max_steps} steps) ═══")
    dynamics.train()
    t0 = time.time()
    losses = []

    for step in range(1, args.max_steps + 1):
        # Sample a sequence of events
        seq_len = args.n_events * args.event_len
        max_start = len(tokens) - seq_len - args.event_len - 1
        start = random.randint(0, max(0, max_start))

        # Process events through ODE, accumulating state
        h = None
        for ev_idx in range(args.n_events):
            ev_start = start + ev_idx * args.event_len
            ev_tokens = tokens[ev_start:ev_start + args.event_len]
            ev_ids = torch.tensor([ev_tokens], device=device)

            # Embed through LLM's frozen embedding table (cast to float32 for ODE)
            with torch.no_grad():
                ev_embeds = embed_fn(ev_ids).float()  # [1, T, d]

            # Initialize or update ODE state
            if h is None:
                h = ev_embeds  # first event initializes state
            else:
                # Append new event embeddings to existing state
                h = torch.cat([h[:, -((args.n_events - 1) * args.event_len):, :],
                                ev_embeds], dim=1)

        # Run ODE integration on accumulated state
        N = h.shape[1]
        context = context_pool(h)
        dynamics.set_context(context, mask=None)
        dynamics.set_n_steps(n_steps)
        h_ode = euler_solve(dynamics, h, t_span=(0.0, 2.0), n_steps=n_steps)

        # Use ODE output as prefix for the NEXT chunk of text
        next_start = start + seq_len
        next_tokens = tokens[next_start:next_start + args.event_len]
        if len(next_tokens) < 10:
            continue
        next_ids = torch.tensor([next_tokens], device=device)

        with torch.no_grad():
            next_embeds = embed_fn(next_ids)  # [1, T_next, d]

        # Mean-pool ODE state to get a compact prefix (not all N positions)
        # Use last few positions as they carry most recent context
        n_prefix = min(8, N)
        prefix = h_ode[:, -n_prefix:, :]  # [1, n_prefix, d]

        # Concatenate: [prefix, next_text_embeds] — match dtype
        combined = torch.cat([prefix.to(next_embeds.dtype), next_embeds], dim=1)

        # Forward through frozen LLM
        outputs = llm(inputs_embeds=combined, use_cache=False)
        logits = outputs.logits

        # NTP loss on text positions only (skip prefix positions)
        text_logits = logits[:, n_prefix:-1, :]  # predict next token
        text_targets = next_ids[:, 1:]  # shifted targets

        loss = F.cross_entropy(
            text_logits.contiguous().view(-1, text_logits.size(-1)),
            text_targets.contiguous().view(-1),
        )

        # Backward (gradient flows through frozen LLM to ODE)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(dynamics.parameters()) + list(context_pool.parameters()), 1.0)
        optimizer.step()

        losses.append(loss.item())

        # ── Logging ──
        if step % args.log_every == 0:
            avg_loss = sum(losses[-args.log_every:]) / min(args.log_every, len(losses))
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t0

            # Check ODE state distribution
            with torch.no_grad():
                h_norm = h_ode.norm(dim=-1).mean().item()
                h_std = h_ode.std(dim=-1).mean().item()
                # Compare to LLM embedding distribution
                emb_norm = next_embeds.norm(dim=-1).mean().item()

            print(f"  step {step:>5d} | loss={avg_loss:.3f} ppl={ppl:.1f} | "
                  f"h_norm={h_norm:.1f} h_std={h_std:.3f} "
                  f"emb_norm={emb_norm:.1f} ratio={h_norm/max(emb_norm,0.01):.1f}x | "
                  f"{step/elapsed:.1f} step/s")

        # ── Eval ──
        if step % args.eval_every == 0:
            dynamics.eval()
            eval_losses = []
            with torch.no_grad():
                for _ in range(10):
                    v_start = random.randint(0, max(0, len(val_tokens) - seq_len - args.event_len - 1))
                    v_h = None
                    for ev_idx in range(args.n_events):
                        ev_s = v_start + ev_idx * args.event_len
                        ev_ids = torch.tensor([val_tokens[ev_s:ev_s + args.event_len]], device=device)
                        ev_emb = embed_fn(ev_ids).float()
                        if v_h is None:
                            v_h = ev_emb
                        else:
                            v_h = torch.cat([v_h[:, -((args.n_events-1)*args.event_len):, :],
                                              ev_emb], dim=1)

                    ctx = context_pool(v_h)
                    dynamics.set_context(ctx, mask=None)
                    dynamics.set_n_steps(n_steps)
                    v_ode = euler_solve(dynamics, v_h, t_span=(0.0, 2.0), n_steps=n_steps)

                    v_next_s = v_start + seq_len
                    v_next = val_tokens[v_next_s:v_next_s + args.event_len]
                    if len(v_next) < 10:
                        continue
                    v_next_ids = torch.tensor([v_next], device=device)
                    v_next_emb = embed_fn(v_next_ids)

                    v_prefix = v_ode[:, -min(8, v_h.shape[1]):, :]
                    v_combined = torch.cat([v_prefix.to(v_next_emb.dtype), v_next_emb], dim=1)
                    v_out = llm(inputs_embeds=v_combined, use_cache=False)
                    v_logits = v_out.logits[:, v_prefix.shape[1]:-1, :]
                    v_loss = F.cross_entropy(
                        v_logits.contiguous().view(-1, v_logits.size(-1)),
                        v_next_ids[:, 1:].contiguous().view(-1),
                    )
                    eval_losses.append(v_loss.item())

            if eval_losses:
                eval_ppl = math.exp(min(sum(eval_losses)/len(eval_losses), 20))
                # Baseline: LLM without prefix
                base_losses = []
                with torch.no_grad():
                    for _ in range(10):
                        b_start = random.randint(0, max(0, len(val_tokens) - args.event_len - 1))
                        b_ids = torch.tensor([val_tokens[b_start:b_start + args.event_len]], device=device)
                        b_out = llm(input_ids=b_ids, use_cache=False)
                        b_loss = F.cross_entropy(
                            b_out.logits[:, :-1, :].contiguous().view(-1, b_out.logits.size(-1)),
                            b_ids[:, 1:].contiguous().view(-1),
                        )
                        base_losses.append(b_loss.item())
                base_ppl = math.exp(min(sum(base_losses)/len(base_losses), 20))
                improvement = (base_ppl - eval_ppl) / base_ppl * 100

                print(f"  ── EVAL step {step}: "
                      f"prefix_ppl={eval_ppl:.1f} baseline_ppl={base_ppl:.1f} "
                      f"improvement={improvement:+.1f}%")
            dynamics.train()

        # ── Save ──
        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, "checkpoints", f"step_{step}.pt")
            # Save as compatible with Mind loading
            torch.save({
                'step': step,
                'model_state_dict': {
                    **{f'dynamics.{k}': v for k, v in dynamics.state_dict().items()},
                    **{f'context_pool.{k}': v for k, v in context_pool.state_dict().items()},
                },
                'config': ode_config,
                'phase': 'ntp_aligned',
                'train_loss': sum(losses[-args.save_every:]) / min(args.save_every, len(losses)),
            }, ckpt_path)
            print(f"  → Saved {ckpt_path}")

    print(f"\nPhase 2 complete. Total time: {(time.time()-t0)/60:.1f} minutes")


if __name__ == '__main__':
    main()
