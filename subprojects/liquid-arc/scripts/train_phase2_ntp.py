#!/usr/bin/env python3
"""Phase 2: Event-level NTP + ARC mix — train LiquidARC at the right granularity.

The ODE operates on EVENT positions, not TOKEN positions. Each ODE position
is a mean-pooled TextEmbedding of a text chunk (sentence/paragraph).

Training:
  1. Split text into N event chunks
  2. Embed each chunk → mean pool → [1, N, d] event representations
  3. Run through ODE (16 steps)
  4. Predict the NEXT event from the ODE state
  5. Mix with ARC tasks to maintain geometric routing

This matches how the Mind actually operates: 64 event positions, each
representing a conversation event, processed through ODE dynamics.

Usage:
    python scripts/train_phase2_ntp.py \
        --ode_checkpoint PRECIOUS_CHECKPOINTS/distilled_2688_step3000.pt \
        --llm_path /workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
        --data_dir /workspace/fgn-v3/data/arc-repo/data \
        --output_dir output/phase2_events \
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

from liquid_arc.model import LiquidARCModel
from liquid_arc.solver import euler_solve
from liquid_arc.tasks.text_task import TextEmbedding, TextHead


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Event-level NTP + ARC")
    parser.add_argument("--ode_checkpoint", type=str, default=None,
                        help="Start from existing LiquidARC checkpoint (ARC-pretrained)")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config for from-scratch training (mutually exclusive with --ode_checkpoint)")
    parser.add_argument("--llm_path", type=str, default="gpt2",
                        help="Path to LLM for tokenizer (or HF model name, default gpt2)")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/phase2_events")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--n_events", type=int, default=8,
                        help="Number of event positions per sequence")
    parser.add_argument("--event_tokens", type=int, default=64,
                        help="Tokens per event chunk (mean-pooled to one position)")
    parser.add_argument("--arc_mix", type=float, default=0.3)
    parser.add_argument("--geo_lr", type=float, default=1e-4)
    parser.add_argument("--content_lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    args = parser.parse_args()

    if not args.ode_checkpoint and not args.config:
        parser.error("Must provide either --ode_checkpoint or --config")
    if args.ode_checkpoint and args.config:
        parser.error("--ode_checkpoint and --config are mutually exclusive")

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # ── Load model ──
    print("═══ Loading LiquidARC ═══")
    if args.ode_checkpoint:
        ode_ckpt = torch.load(args.ode_checkpoint, map_location=device, weights_only=False)
        ode_config = ode_ckpt['config']
        print(f"  loaded config from checkpoint: {args.ode_checkpoint}")
    else:
        from liquid_arc.config import LiquidARCConfig
        ode_config = LiquidARCConfig.from_yaml(args.config)
        ode_ckpt = None
        print(f"  from-scratch: loaded config from {args.config}")

    d = ode_config.d_model
    ode_config.tau_freeze_steps = 0
    ode_config.tau_min = 0.1
    ode_config.tau_max = 3.0

    arc_model = LiquidARCModel(ode_config).to(device)
    if ode_ckpt is not None:
        arc_model.load_state_dict(ode_ckpt.get('model_state_dict', {}), strict=False)
        print("  loaded weights from checkpoint")
    else:
        print("  fresh random initialization (no checkpoint)")
    arc_model.dynamics.freeze_tau = False
    print(f"  d={d}, freeze_tau={arc_model.dynamics.freeze_tau}")
    print(f"  t_diffusion={F.softplus(arc_model.dynamics.t_diffusion).item():.2f}")

    # ── TextEmbedding + TextHead ──
    print(f"\n═══ TextEmbedding ═══")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    text_embed = TextEmbedding(
        vocab_size=vocab_size, d_model=d,
        max_seq_len=args.event_tokens, dropout=0.1,
    ).to(device)
    text_head = TextHead(d_model=d, vocab_size=vocab_size).to(device)
    print(f"  Vocab={vocab_size}, event_tokens={args.event_tokens}")

    # ── Data ──
    print(f"\n═══ Data ═══")
    from liquid_arc.tasks.procedural import ProceduralARCTask
    arc_task = ProceduralARCTask(seq_len=ode_config.max_seq_len, augment=True)
    eval_task = None
    if args.data_dir:
        try:
            from fgn.tasks.arc import ARCTask
            eval_task = ARCTask(seq_len=ode_config.max_seq_len, data_dir=args.data_dir,
                                split='eval', augment=False)
            print(f"  ARC: procedural + real eval")
        except Exception:
            pass
    if eval_task is None:
        eval_task = ProceduralARCTask(seq_len=ode_config.max_seq_len, augment=False)
        eval_task._seed_counter = 999999

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text_data = ' '.join([t for t in ds['text'] if len(t.strip()) > 50])
    all_tokens = tokenizer.encode(text_data)
    print(f"  NTP: {len(all_tokens):,} tokens, {args.n_events} events × {args.event_tokens} tokens")

    # ── Optimizer ──
    geo_names = ['metric_net', 'tau_net', 't_diffusion', 'alpha_logit', 'context_pool']
    geo_params, content_params = [], []
    for name, p in arc_model.named_parameters():
        if any(g in name for g in geo_names):
            geo_params.append(p)
        else:
            content_params.append(p)
    for p in text_embed.parameters():
        content_params.append(p)
    for p in text_head.parameters():
        content_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': geo_params, 'lr': args.geo_lr},
        {'params': content_params, 'lr': args.content_lr},
    ], weight_decay=0.01)
    print(f"\n  Geo: {sum(p.numel() for p in geo_params)/1e6:.1f}M at LR={args.geo_lr}")
    print(f"  Content: {sum(p.numel() for p in content_params)/1e6:.1f}M at LR={args.content_lr}")

    # ── Training ──
    print(f"\n═══ Training ({args.max_steps} steps, {args.arc_mix:.0%} ARC) ═══")
    arc_model.train()
    text_embed.train()
    text_head.train()
    t0 = time.time()
    metrics = {}
    n_steps = ode_config.n_ode_steps

    for step in range(1, args.max_steps + 1):
        use_arc = random.random() < args.arc_mix

        if use_arc:
            # ═══ ARC path ═══
            try:
                _, _, meta = arc_task.generate_batch(batch_size=4, device=device)
                result = arc_model(
                    colors=meta['colors'], xs=meta['xs'], ys=meta['ys'],
                    roles=meta['roles'], sep_mask=meta['sep_mask'],
                    sep_types=meta['sep_types'], target_mask=meta['target_mask'],
                    target_labels=meta.get('target_labels'),
                    grid_ids=meta.get('grid_ids'),
                )
                loss = result['ce_loss'] + result.get('curv_loss', 0) + result.get('tau_var_loss', 0)
                task_type = 'arc'
                metrics['arc_xform'] = metrics.get('arc_xform', 0) + result.get('transform_accuracy', 0)
            except Exception:
                continue
        else:
            # ═══ Event-level NTP ═══
            # Split text into N+1 event chunks, process N through ODE, predict N+1
            total_tokens = (args.n_events + 1) * args.event_tokens
            start = random.randint(0, max(0, len(all_tokens) - total_tokens - 1))

            # Embed each event chunk and mean-pool to one position
            event_positions = []
            for ev_idx in range(args.n_events + 1):
                ev_start = start + ev_idx * args.event_tokens
                ev_ids = torch.tensor(
                    [all_tokens[ev_start:ev_start + args.event_tokens]], device=device)
                ev_emb = text_embed(ev_ids)  # [1, T, d]
                ev_pooled = ev_emb.mean(dim=1)  # [1, d]
                event_positions.append(ev_pooled)

            # Stack into [1, N+1, d]
            all_events = torch.stack(event_positions, dim=1)  # [1, N+1, d]
            input_events = all_events[:, :args.n_events, :]  # [1, N, d]
            target_event = all_events[:, -1:, :]  # [1, 1, d]

            # ODE on the N input events
            context = arc_model.context_pool(input_events)
            arc_model.dynamics.set_context(context, mask=None)
            arc_model.dynamics.set_n_steps(n_steps)

            # Pre-ODE diagnostics needed by auxiliary losses (see
            # LiquidARCModel.forward in model.py:490-530 for the canonical
            # path). Required here because this script bypasses .forward()
            # and calls euler_solve() directly.
            g_init_t = arc_model.dynamics.compute_metric_diag(input_events)
            tau_init_t = arc_model.dynamics.compute_tau(input_events)
            # compute_metric_diag may return (D, L) when metric_rank > 0
            g_init_diag = g_init_t[0] if isinstance(g_init_t, tuple) else g_init_t

            h_ode = euler_solve(arc_model.dynamics, input_events,
                                t_span=(0.0, 2.0), n_steps=n_steps)
            # euler_solve may return (h, efficiency) — normalize
            if isinstance(h_ode, tuple):
                h_ode = h_ode[0]

            # Predict next event: use mean-pooled ODE state → TextHead → match target
            # The ODE state at the last position should predict the next event
            h_last = h_ode[:, -1:, :]  # [1, 1, d]

            # Project through TextHead to get token logits for the target event
            # Expand to match target tokens for NTP loss
            target_start = start + args.n_events * args.event_tokens
            target_ids = torch.tensor(
                [all_tokens[target_start:target_start + args.event_tokens]], device=device)
            target_emb = text_embed(target_ids)  # [1, T_target, d]

            # Concatenate ODE summary with target tokens for autoregressive prediction
            # [h_last, target_emb[:, :-1]] → predict target_emb[:, 1:]'s token IDs
            combined = torch.cat([h_last, target_emb[:, :-1, :]], dim=1)  # [1, 1+T-1, d]
            logits = text_head(combined)  # [1, T, vocab]

            # NTP loss on target tokens (skip the ODE summary position)
            ntp_loss = F.cross_entropy(
                logits[:, 1:, :].contiguous().view(-1, vocab_size),
                target_ids[:, 1:].contiguous().view(-1),
            )

            # Auxiliary geometry-shaping losses — mirror LiquidARCModel.forward.
            # Without these the event-level NTP path is task-loss-only, which
            # produces flat geometry (confirmed empirically in baseline run).
            aux_loss = torch.zeros((), device=device)
            crit_ratio_log = 0.0
            if getattr(ode_config, 'criticality_loss_enabled', False):
                from liquid_arc.sustained_criticality import compute_criticality_loss
                _crit_loss, _crit_diag = compute_criticality_loss(
                    h_ode, g_init_diag, tau_init_t, arc_model.dynamics.t_diffusion,
                    target_ratio=ode_config.criticality_target_ratio,
                    d_sq_target=ode_config.criticality_D_sq_target,
                )
                aux_loss = aux_loss + ode_config.criticality_loss_lambda * _crit_loss
                crit_ratio_log = float(_crit_diag.get('ratio', 0.0))
            if getattr(ode_config, 'tau_quality_loss_enabled', False):
                from liquid_arc.sustained_criticality import compute_tau_quality_loss
                tql_mean_target = ode_config.tau_mean_target
                if tql_mean_target <= 0:
                    tql_mean_target = 2.0 / max(1, n_steps) * 16.0  # auto
                _tq_loss = compute_tau_quality_loss(
                    tau_init_t,
                    mean_target=tql_mean_target,
                    log_spread_target=ode_config.tau_log_spread_target,
                )
                aux_loss = aux_loss + ode_config.tau_quality_lambda * _tq_loss
            if getattr(ode_config, 'curvature_diversity_loss_enabled', False):
                from liquid_arc.sustained_criticality import compute_curvature_diversity_loss
                _cd_loss = compute_curvature_diversity_loss(
                    g_init_diag,
                    cv_floor=ode_config.curvature_cv_floor,
                    cv_ceiling=ode_config.curvature_cv_ceiling,
                )
                aux_loss = aux_loss + ode_config.curvature_diversity_lambda * _cd_loss

            loss = ntp_loss + aux_loss
            # Stash a compact diagnostic so the logger can surface it
            metrics["_last_crit_ratio"] = crit_ratio_log
            metrics["_last_aux_loss"] = float(aux_loss.detach().item()) if aux_loss.requires_grad else 0.0
            task_type = 'ntp'

        optimizer.zero_grad()
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  step {step}: NaN/Inf loss detected ({task_type}), skipping")
            continue
        loss.backward()
        # NaN scrubbing — bfloat16 SDPA can produce NaN gradients
        all_params = (list(arc_model.parameters()) +
                      list(text_embed.parameters()) + list(text_head.parameters()))
        for p in all_params:
            if p.grad is not None and p.grad.isnan().any():
                p.grad.nan_to_num_(nan=0.0)
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        # ── Metrics ──
        with torch.no_grad():
            metrics[f'{task_type}_loss'] = metrics.get(f'{task_type}_loss', 0) + loss.item()
            metrics[f'{task_type}_n'] = metrics.get(f'{task_type}_n', 0) + 1

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            arc_n = max(metrics.get('arc_n', 0), 1)
            ntp_n = max(metrics.get('ntp_n', 0), 1)
            arc_loss = metrics.get('arc_loss', 0) / arc_n
            ntp_loss = metrics.get('ntp_loss', 0) / ntp_n
            ntp_ppl = math.exp(min(ntp_loss, 20)) if ntp_loss > 0 else 0
            avg_xform = metrics.get('arc_xform', 0) / arc_n * 100

            with torch.no_grad():
                t_diff = F.softplus(arc_model.dynamics.t_diffusion).item()
                alpha = torch.sigmoid(arc_model.dynamics.alpha_logit).item()
                if task_type == 'ntp' and 'h_ode' in dir():
                    g = arc_model.dynamics.compute_metric_diag(h_ode.detach())
                    cv = (g.std() / (g.mean() + 1e-8)).item()
                    tau = arc_model.dynamics.compute_tau(h_ode.detach())
                    tau_str = f"tau={tau.mean():.2f}±{tau.std():.2f}"
                else:
                    cv = result.get('metric_cv', 0) if task_type == 'arc' else 0
                    tau_str = "tau=arc"

            crit_ratio = metrics.get("_last_crit_ratio", 0.0)
            aux_extra = metrics.get("_last_aux_loss", 0.0)
            aux_str = (f" aux={aux_extra:.3f} D²/4τ={crit_ratio:.1f}"
                        if aux_extra != 0.0 else "")
            print(f"  step {step:>5d} | arc={arc_loss:.3f} xform={avg_xform:.1f}% | "
                  f"ntp={ntp_loss:.3f} ppl={ntp_ppl:.0f} | "
                  f"CV={cv:.2f} {tau_str} t={t_diff:.1f} α={alpha:.2f}{aux_str} | "
                  f"{step/elapsed:.1f} step/s")
            metrics = {}

        # ── Eval ──
        if step % args.eval_every == 0:
            arc_model.eval()
            eval_m = {'xform': 0, 'n': 0}
            with torch.no_grad():
                for _ in range(20):
                    try:
                        _, _, emeta = eval_task.generate_batch(batch_size=4, device=device)
                        er = arc_model(
                            colors=emeta['colors'], xs=emeta['xs'], ys=emeta['ys'],
                            roles=emeta['roles'], sep_mask=emeta['sep_mask'],
                            sep_types=emeta['sep_types'],
                            target_mask=emeta['target_mask'],
                            target_labels=emeta.get('target_labels'),
                            grid_ids=emeta.get('grid_ids'),
                        )
                        eval_m['xform'] += er.get('transform_accuracy', 0)
                        eval_m['n'] += 1
                    except Exception:
                        pass
            ne = max(eval_m['n'], 1)
            print(f"  ── EVAL step {step}: arc_xform={eval_m['xform']/ne*100:.1f}%")
            arc_model.train()

        # ── Save ──
        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, "checkpoints", f"step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': {
                    **{f'dynamics.{k}': v for k, v in arc_model.dynamics.state_dict().items()},
                    **{f'context_pool.{k}': v for k, v in arc_model.context_pool.state_dict().items()},
                },
                'text_embed_state': text_embed.state_dict(),
                'text_head_state': text_head.state_dict(),
                'config': ode_config,
                'vocab_size': vocab_size,
                'phase': 'phase2_events',
            }, ckpt_path)
            print(f"  → Saved {ckpt_path}")

    print(f"\nDone. Total time: {(time.time()-t0)/60:.1f} minutes")


if __name__ == '__main__':
    main()
