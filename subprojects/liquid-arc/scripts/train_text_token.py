"""Token-level NTP pretraining for LiquidARC — semantic-first from scratch.

Hypothesis under test: the event-level pooling in train_phase2_ntp.py
collapses 64 tokens into 1 mean-pooled position, leaving structural_tau
only 8 positions to differentiate and discarding within-event signal
that could drive geometry. A token-level path with seq_len=512 gives
structural_tau 512 positions and preserves local token structure.

Architecture:
  TextEmbedding(ids) -> h0 [B, T, d]
  context = ContextPool(h0)
  h_final = euler_solve(dynamics, h0, t_span=(0, T_ode))
  logits = TextHead(h_final) [B, T, vocab]
  loss = CE(logits[:-1], ids[1:])   # next-token prediction
       + λ_crit * criticality_loss(h_final, g_init, tau_init)
       + λ_τq  * tau_quality_loss(tau_init)

All three LiquidARC aux losses are applied directly (we bypass
LiquidARCModel.forward because it is ARC-coupled, but we replicate
its loss assembly from model.py:490-530).

Run:
  python -u scripts/train_text_token.py \
      --config configs/liquid_arc_text_first.yaml \
      --max_steps 3000 --seq_len 512 \
      --output_dir output_text_token
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/liquid-arc")
sys.path.insert(0, "/home/pokazge/liquid-arc")

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel
from liquid_arc.solver import euler_solve
from liquid_arc.tasks.text_task import TextEmbedding, TextHead


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="YAML config (e.g. configs/liquid_arc_text_first.yaml)")
    p.add_argument("--llm_path", default="gpt2",
                   help="HF tokenizer name or path; default gpt2")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--seq_len", type=int, default=512,
                   help="Token-level sequence length (structural_tau needs "
                        "max_seq_len >= this in config)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--geo_lr", type=float, default=1e-4)
    p.add_argument("--content_lr", type=float, default=1e-3)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--code_mix_ratio", type=float, default=0.0,
                   help="Probability of sampling a batch from the Python code "
                        "corpus. 0 = text only; 0.5 = 50/50 with text.")
    p.add_argument("--md_mix_ratio", type=float, default=0.0,
                   help="Probability of sampling a batch from the markdown "
                        "corpus. Combined with code_mix_ratio for 3-domain.")
    p.add_argument("--decorr_lambda", type=float, default=0.0,
                   help="Capacity-preservation loss: Barlow-Twins-style "
                        "feature decorrelation of post-ODE hidden state. "
                        "0 = off; 1e-3 is a reasonable starting value. "
                        "Tests whether explicit capacity preservation "
                        "replicates multi-domain OOD gains.")
    p.add_argument("--code_cache", default="python_code",
                   help="Cache name for the code corpus")
    p.add_argument("--dataset", choices=["wikitext-2", "wikitext-103"],
                   default="wikitext-2",
                   help="Training data source")
    p.add_argument("--criticality_target", type=float, default=None,
                   help="Override config's criticality_target_ratio (sweep knob)")
    p.add_argument("--tag", default="",
                   help="Extra tag appended to log banner for identification")
    p.add_argument("--causal_mask", action="store_true",
                   help="Apply lower-triangular mask to the heat kernel. "
                        "Makes NTP task well-posed — position t only attends "
                        "to 0..t (no leakage from future). Required for "
                        "meaningful causal LM training on a bidirectional ODE.")
    p.add_argument("--training_mode",
                   choices=["causal", "mlm", "distill"], default="causal",
                   help="causal = NTP; mlm = masked-LM; "
                        "distill = KD from a pretrained teacher (matches "
                        "teacher's next-token logits; student MUST use "
                        "causal mask to avoid leaking future the teacher "
                        "cannot see)")
    p.add_argument("--mlm_mask_prob", type=float, default=0.15,
                   help="Fraction of positions masked in mlm mode")
    p.add_argument("--teacher_model", default="gpt2",
                   help="HF model name for distillation teacher (frozen)")
    p.add_argument("--distill_temp", type=float, default=2.0,
                   help="Softmax temperature for KD logits")
    p.add_argument("--distill_hard_mix", type=float, default=0.0,
                   help="Blend coefficient: total = (1-α)*KD + α*hard_CE. "
                        "0 = pure KD (default)")
    p.add_argument("--use_teacher_embed", action="store_true",
                   help="Use teacher's token+positional embeddings as h0 "
                        "and teacher's LM head as the student head, "
                        "both FROZEN. Removes the ~76M embedding+head "
                        "gradient sink so all learning capacity goes to "
                        "the LiquidARC ODE. Distill mode only.")
    p.add_argument("--n_phases", type=int, default=1,
                   help="Un-tie ODE weights: use N distinct "
                        "ContinuousDynamics modules in sequence instead of "
                        "weight-tied iteration. n_ode_steps must be "
                        "divisible by N. 1=current behavior.")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint .pt to resume from. Loads "
                        "arc_model, text_embed, text_head, and (if present) "
                        "phased_dynamics. Starts optimizer fresh.")
    p.add_argument("--microcircuit_M", type=int, default=0,
                   help="Phase 1a: compress input to M microcircuit slots "
                        "before running dynamics. 0 = disabled (token-level). "
                        "Typical: 32 slots for T=512 tokens.")
    p.add_argument("--chunked_M", type=int, default=0,
                   help="Phase 1a.2: split T tokens into M contiguous chunks; "
                        "run dynamics independently on each chunk plus sparse "
                        "inter-chunk summary routing. 0 = disabled. Typical: "
                        "8 chunks for T=256 (32 tokens/chunk). Mutually "
                        "exclusive with --microcircuit_M.")
    p.add_argument("--chunked_no_routing", action="store_true",
                   help="Pure locality ablation: disable inter-chunk summary "
                        "routing. Tests whether long-range information is "
                        "load-bearing for NTP at this scale.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    # ── Config ──
    config = LiquidARCConfig.from_yaml(args.config)
    config.tau_freeze_steps = 0
    if args.criticality_target is not None:
        config.criticality_target_ratio = args.criticality_target
        print(f"  [override] criticality_target_ratio = {args.criticality_target}")
    if args.tag:
        print(f"  [tag] {args.tag}")
    assert config.max_seq_len >= args.seq_len, (
        f"config.max_seq_len={config.max_seq_len} < seq_len={args.seq_len}; "
        f"structural_tau param would be undersized")
    d = config.d_model
    n_steps = config.n_ode_steps
    print(f"═══ LiquidARC token-level text-first pretraining ═══")
    print(f"  d_model={d}  d_metric={config.d_metric}  "
          f"d_ffn={config.d_ffn}  rank={config.metric_rank}")
    print(f"  seq_len={args.seq_len}  n_ode_steps={n_steps}")
    print(f"  structural_tau_enabled={config.structural_tau_enabled}  "
          f"criticality={config.criticality_loss_enabled}  "
          f"tau_quality={config.tau_quality_loss_enabled}")

    # ── Model ──
    arc_model = LiquidARCModel(config).to(device)
    arc_model.dynamics.freeze_tau = False
    print(f"  params: {sum(p.numel() for p in arc_model.parameters())/1e6:.1f}M")

    # Phase 1a microcircuit wrapper: compress to M slots before dynamics.
    # Mutually exclusive with phased dynamics for now.
    assert not (args.microcircuit_M > 0 and args.chunked_M > 0), (
        "--microcircuit_M and --chunked_M are mutually exclusive")
    microcircuit = None
    if args.microcircuit_M > 0:
        from liquid_arc.microcircuit import MicroCircuitWrapper
        microcircuit = MicroCircuitWrapper(
            d=config.d_model, M=args.microcircuit_M,
            dynamics=arc_model.dynamics,
            n_ode_steps=config.n_ode_steps,
        ).to(device)
        mc_params = sum(p.numel() for p in microcircuit.parameters()
                        if p is not arc_model.dynamics)
        print(f"  MicroCircuit wrapper: M={args.microcircuit_M} slots, "
              f"+{mc_params/1e6:.1f}M params (compress/expand cross-attn)")
    elif args.chunked_M > 0:
        from liquid_arc.microcircuit import ChunkedMicroCircuitWrapper
        assert args.seq_len % args.chunked_M == 0, (
            f"seq_len={args.seq_len} not divisible by chunked_M="
            f"{args.chunked_M}")
        routing_on = not args.chunked_no_routing
        microcircuit = ChunkedMicroCircuitWrapper(
            d=config.d_model, M=args.chunked_M,
            dynamics=arc_model.dynamics,
            n_ode_steps=config.n_ode_steps,
            inter_chunk_routing=routing_on,
        ).to(device)
        mc_params = sum(
            p.numel() for n, p in microcircuit.named_parameters()
            if not n.startswith("dynamics."))
        L = args.seq_len // args.chunked_M
        print(f"  ChunkedMicroCircuit: M={args.chunked_M} chunks × "
              f"L={L} tokens, routing={'on' if routing_on else 'OFF (pure locality)'}, "
              f"+{mc_params/1e6:.2f}M params")

    # Un-tied dynamics phases: build (n_phases - 1) additional
    # ContinuousDynamics modules. The first phase reuses arc_model.dynamics.
    # All phases share context_pool (on arc_model) and are driven by the
    # same step_index / n_steps plumbing.
    phased_dynamics = None
    if args.n_phases > 1:
        from liquid_arc.dynamics import ContinuousDynamics
        assert config.n_ode_steps % args.n_phases == 0, (
            f"n_ode_steps={config.n_ode_steps} not divisible by "
            f"n_phases={args.n_phases}")
        # ModuleList so PyTorch registers the params for optim/state_dict
        phased_dynamics = torch.nn.ModuleList([arc_model.dynamics])
        for _ in range(args.n_phases - 1):
            extra = ContinuousDynamics(config).to(device)
            extra.freeze_tau = False
            phased_dynamics.append(extra)
        extra_params = sum(
            p.numel() for p in phased_dynamics[1:].parameters())
        print(f"  n_phases={args.n_phases}: +"
              f"{extra_params/1e6:.1f}M additional ODE params "
              f"({config.n_ode_steps // args.n_phases} steps per phase, "
              f"total ODE params: "
              f"{sum(p.numel() for p in phased_dynamics.parameters())/1e6:.1f}M)")

    # ── Tokenizer + text embed/head ──
    from transformers import AutoTokenizer
    # Distillation mode uses the teacher's tokenizer so vocabularies line up
    tok_path = (args.teacher_model if args.training_mode == "distill"
                else args.llm_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab = len(tokenizer)
    text_embed = TextEmbedding(vocab_size=vocab, d_model=d,
                                max_seq_len=args.seq_len, dropout=0.1).to(device)
    text_head = TextHead(d_model=d, vocab_size=vocab).to(device)
    print(f"  vocab={vocab}  text params: "
          f"{(sum(p.numel() for p in text_embed.parameters())+sum(p.numel() for p in text_head.parameters()))/1e6:.1f}M")

    # ── Data ──
    # Cache tokenized output across sweep runs to avoid re-encoding
    # 100M+ tokens on CPU for each run.
    import pickle
    cache_dir = "/workspace/liquid-arc/cache"
    os.makedirs(cache_dir, exist_ok=True)
    tok_name = args.llm_path.replace("/", "_")
    cache_path = os.path.join(cache_dir, f"{args.dataset}_{tok_name}_tokens.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            all_tokens = pickle.load(f)
        ds_label = ("WikiText-103" if args.dataset == "wikitext-103"
                     else "WikiText-2")
        print(f"  {ds_label} (cached): {len(all_tokens):,} tokens from {cache_path}")
    else:
        from datasets import load_dataset
        if args.dataset == "wikitext-103":
            ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
            ds_label = "WikiText-103"
        else:
            ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            ds_label = "WikiText-2"
        joined = ' '.join([t for t in ds['text'] if len(t.strip()) > 50])
        print(f"  tokenizing {ds_label} ({len(joined)/1e6:.0f} MB text)...",
              flush=True)
        t_tok = time.time()
        all_tokens = tokenizer.encode(joined)
        print(f"  {ds_label}: {len(all_tokens):,} tokens "
              f"(tokenize took {time.time()-t_tok:.1f}s)", flush=True)
        with open(cache_path, "wb") as f:
            pickle.dump(all_tokens, f)
        print(f"  cached tokens to {cache_path}", flush=True)

    # ── Markdown corpus for 3-domain training ──
    md_tokens = None
    if args.md_mix_ratio > 0:
        md_cache_path = os.path.join(
            cache_dir, f"markdown_{tok_name}_tokens.pkl")
        if os.path.exists(md_cache_path):
            with open(md_cache_path, "rb") as f:
                md_tokens = pickle.load(f)
            print(f"  Markdown (cached): {len(md_tokens):,} tokens "
                  f"from {md_cache_path}")
        else:
            import glob
            roots = ["/workspace/liquid-arc",
                     "/workspace/fgn-v3",
                     "/workspace/lewm-integration"]
            pieces = []
            for r in roots:
                for p in sorted(glob.glob(os.path.join(r, "**/*.md"),
                                            recursive=True)):
                    try:
                        with open(p, 'r', encoding='utf-8',
                                  errors='ignore') as f:
                            pieces.append(f.read())
                    except Exception:
                        pass
            joined = "\n\n".join(pieces)
            print(f"  tokenizing markdown ({len(joined)/1e6:.1f} MB "
                  f"from {len(pieces)} files)...", flush=True)
            t_tok = time.time()
            md_tokens = tokenizer.encode(joined)
            print(f"  Markdown: {len(md_tokens):,} tokens "
                  f"(tokenize took {time.time()-t_tok:.1f}s)", flush=True)
            with open(md_cache_path, "wb") as f:
                pickle.dump(md_tokens, f)
        print(f"  md_mix_ratio={args.md_mix_ratio}")

    # ── Code corpus for mixed-domain training ──
    code_tokens = None
    if args.code_mix_ratio > 0:
        code_cache_path = os.path.join(
            cache_dir, f"{args.code_cache}_{tok_name}_tokens.pkl")
        if os.path.exists(code_cache_path):
            with open(code_cache_path, "rb") as f:
                code_tokens = pickle.load(f)
            print(f"  Python code (cached): {len(code_tokens):,} tokens "
                  f"from {code_cache_path}")
        else:
            # Walk Python sources bind-mounted into the container
            import glob
            roots = ["/workspace/liquid-arc/liquid_arc",
                     "/workspace/liquid-arc/scripts",
                     "/workspace/fgn-v3/fgn"]
            pieces = []
            for r in roots:
                for p in sorted(
                    glob.glob(os.path.join(r, "**/*.py"), recursive=True)
                    + glob.glob(os.path.join(r, "*.py"))):
                    try:
                        with open(p, 'r', encoding='utf-8',
                                  errors='ignore') as f:
                            pieces.append(f.read())
                    except Exception:
                        pass
            joined = "\n\n".join(pieces)
            print(f"  tokenizing Python code ({len(joined)/1e6:.1f} MB "
                  f"from {len(pieces)} files)...", flush=True)
            t_tok = time.time()
            code_tokens = tokenizer.encode(joined)
            print(f"  Python code: {len(code_tokens):,} tokens "
                  f"(tokenize took {time.time()-t_tok:.1f}s)", flush=True)
            with open(code_cache_path, "wb") as f:
                pickle.dump(code_tokens, f)
            print(f"  cached code tokens to {code_cache_path}", flush=True)
        print(f"  code_mix_ratio={args.code_mix_ratio}: "
              f"{int(100*args.code_mix_ratio)}% of batches from code")

    # ── Optimizer (only include trainable params — skip frozen embedding/head) ──
    geo_names = ['metric_net', 'tau_net', 't_diffusion', 'alpha_logit',
                 'context_pool', 'structural_tau', 'W_q', 'W_k']
    geo_params, content_params = [], []
    seen_ids = set()
    def _add(name: str, param):
        if not param.requires_grad or id(param) in seen_ids:
            return
        seen_ids.add(id(param))
        (geo_params if any(g in name for g in geo_names)
         else content_params).append(param)
    for name, param in arc_model.named_parameters():
        _add(name, param)
    # Add un-tied phase modules (phased_dynamics[0] is arc_model.dynamics,
    # its params already added)
    if phased_dynamics is not None:
        for i, dyn in enumerate(phased_dynamics):
            if i == 0:
                continue
            for name, param in dyn.named_parameters():
                _add(f"phase{i}.{name}", param)
    for param in text_embed.parameters():
        _add("text_embed", param)
    for param in text_head.parameters():
        _add("text_head", param)
    # Microcircuit wrapper params (compress/expand cross-attn + slot queries).
    # The inner dynamics module is already counted via arc_model.
    if microcircuit is not None:
        for name, param in microcircuit.named_parameters():
            if not name.startswith("dynamics."):
                _add(f"microcircuit.{name}", param)

    optimizer = torch.optim.AdamW([
        {'params': geo_params, 'lr': args.geo_lr},
        {'params': content_params, 'lr': args.content_lr},
    ], weight_decay=0.01)
    print(f"  geo: {sum(p.numel() for p in geo_params)/1e6:.2f}M @ "
          f"{args.geo_lr}; content: "
          f"{sum(p.numel() for p in content_params)/1e6:.1f}M @ {args.content_lr}")

    # ── Resume from checkpoint (optional) ──
    start_step = 0
    if args.resume:
        print(f"\n  resuming from {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing_m, unexpected_m = arc_model.load_state_dict(
            ckpt['model_state_dict'], strict=False)
        if missing_m:
            print(f"    arc_model missing keys: {len(missing_m)} "
                  f"(first: {missing_m[:3]})")
        if unexpected_m:
            print(f"    arc_model unexpected keys: {len(unexpected_m)} "
                  f"(first: {unexpected_m[:3]})")
        text_embed.load_state_dict(ckpt['text_embed_state_dict'], strict=False)
        text_head.load_state_dict(ckpt['text_head_state_dict'], strict=False)
        # Load phased_dynamics if saved (newer format)
        if phased_dynamics is not None and 'phased_dynamics_state_dict' in ckpt:
            phased_dynamics.load_state_dict(
                ckpt['phased_dynamics_state_dict'], strict=False)
            print(f"    phased_dynamics restored")
        elif phased_dynamics is not None and args.n_phases > 1:
            print(f"    WARNING: phased_dynamics not in checkpoint; "
                  f"phase 0 was restored via arc_model, phases 1-"
                  f"{args.n_phases-1} remain at fresh init")
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print(f"    optimizer state restored")
            except Exception as e:
                print(f"    optimizer state NOT restored: {e}")
        start_step = int(ckpt.get('step', 0))
        print(f"    start_step set to {start_step}")

    # ── Training ──
    print(f"\n═══ Training ({args.max_steps} steps, seq={args.seq_len}) ═══")
    arc_model.train()
    text_embed.train()
    text_head.train()
    t0 = time.time()

    from liquid_arc.sustained_criticality import (
        compute_criticality_loss, compute_tau_quality_loss,
        compute_curvature_diversity_loss,
    )

    # Pre-build the causal mask once (same for every step).
    # Convention: True means BLOCKED. Upper-triangular (j>i) is masked out.
    # Distillation mode FORCES the mask — teacher is causal, student must be too.
    causal_mask = None
    if args.causal_mask or args.training_mode == "distill":
        causal_mask = torch.triu(
            torch.ones(args.seq_len, args.seq_len, dtype=torch.bool,
                       device=device),
            diagonal=1,
        )
        print(f"  causal mask active: blocks {causal_mask.sum().item()} "
              f"future-position pairs per forward")

    # Teacher model for distillation (frozen)
    teacher = None
    if args.training_mode == "distill":
        from transformers import AutoModelForCausalLM
        print(f"  loading teacher {args.teacher_model}...")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model).to(device).eval()
        for pp in teacher.parameters():
            pp.requires_grad = False
        t_vocab = teacher.config.vocab_size
        assert t_vocab == vocab, (
            f"teacher vocab={t_vocab} != student vocab={vocab}")
        print(f"  teacher loaded: "
              f"{sum(p.numel() for p in teacher.parameters())/1e6:.0f}M params, "
              f"temp={args.distill_temp}, hard_mix={args.distill_hard_mix}")

        # Optional: use teacher's frozen embeddings as student's h0 source.
        # Removes the ~38M embedding gradient sink so the LiquidARC ODE
        # gets undiluted learning signal. Head stays TRAINABLE because
        # teacher's lm_head expects teacher-transformer-processed hidden
        # states, not LiquidARC ODE output — they're in different spaces.
        if args.use_teacher_embed:
            t_wte = teacher.transformer.wte.weight.data  # [vocab, d]
            t_wpe = teacher.transformer.wpe.weight.data  # [max_pos, d]
            assert t_wte.shape[1] == d, (
                f"teacher d={t_wte.shape[1]} != student d={d}")
            text_embed.token_embed.weight.data.copy_(t_wte)
            text_embed.pos_embed.weight.data.copy_(t_wpe[:args.seq_len])
            for pp in text_embed.parameters():
                pp.requires_grad = False
            # Head initialized from teacher but left TRAINABLE.
            text_head.proj.weight.data.copy_(teacher.lm_head.weight.data)
            for pp in text_head.parameters():
                pp.requires_grad = True
            frozen_params = sum(p.numel() for p in text_embed.parameters())
            trainable_head = sum(p.numel() for p in text_head.parameters())
            print(f"  FROZE teacher embed: "
                  f"{frozen_params/1e6:.1f}M params frozen (vocab lookup). "
                  f"Head init'd from teacher, {trainable_head/1e6:.1f}M "
                  f"trainable (student learns its own hidden-to-logit map)")

    # MLM mode uses the EOS token as the mask token id (GPT-2 has no
    # dedicated [MASK]; EOS is rare enough in WikiText-103 not to
    # interfere. No vocabulary change, no resize of the embedding table.)
    mask_token_id = tokenizer.eos_token_id

    for step in range(start_step + 1, args.max_steps + 1):
        # ── Sample a batch of windows ──
        batch_inputs, batch_targets = [], []
        if args.training_mode == "mlm":
            # MLM: input == window; target == window; 15% positions masked
            # in input, loss computed only at those positions.
            for _ in range(args.batch_size):
                start = random.randint(
                    0, max(0, len(all_tokens) - args.seq_len - 1))
                w = all_tokens[start:start + args.seq_len]
                batch_inputs.append(w)
                batch_targets.append(w)
            input_ids = torch.tensor(batch_inputs, device=device)     # [B, T]
            target_ids = torch.tensor(batch_targets, device=device)   # [B, T]
            # Bernoulli mask: which positions get masked
            mlm_mask = torch.rand(input_ids.shape, device=device) < args.mlm_mask_prob
            # Apply BERT-style: 80% [MASK], 10% random, 10% keep
            rnd = torch.rand(input_ids.shape, device=device)
            replace_mask = mlm_mask & (rnd < 0.80)
            replace_random = mlm_mask & (rnd >= 0.80) & (rnd < 0.90)
            # (10% keep the original token — no op)
            input_ids = input_ids.clone()
            input_ids[replace_mask] = mask_token_id
            rnd_tokens = torch.randint(0, vocab, input_ids.shape, device=device)
            input_ids[replace_random] = rnd_tokens[replace_random]
        else:
            # Choose corpus per-batch. Roll one random float and route:
            #   r < code_ratio          → code
            #   code_ratio..sum_ratios  → markdown
            #   sum_ratios..1.0         → text
            # Zero or missing corpora fall through to text.
            r = random.random()
            code_ratio = args.code_mix_ratio if code_tokens is not None else 0.0
            md_ratio = args.md_mix_ratio if md_tokens is not None else 0.0
            if r < code_ratio:
                src_tokens = code_tokens
            elif r < code_ratio + md_ratio:
                src_tokens = md_tokens
            else:
                src_tokens = all_tokens
            for _ in range(args.batch_size):
                start = random.randint(
                    0, max(0, len(src_tokens) - args.seq_len - 2))
                w = src_tokens[start:start + args.seq_len + 1]
                batch_inputs.append(w[:-1])
                batch_targets.append(w[1:])
            input_ids = torch.tensor(batch_inputs, device=device)     # [B, T]
            target_ids = torch.tensor(batch_targets, device=device)   # [B, T]
            mlm_mask = None  # loss over all positions

        # ── Forward ──
        h0 = text_embed(input_ids)                        # [1, T, d]
        context = arc_model.context_pool(h0)              # uses pool over tokens
        arc_model.dynamics.set_context(context, mask=causal_mask)
        arc_model.dynamics.set_n_steps(n_steps)

        # Pre-ODE diagnostics for aux losses. In attention routing mode the
        # metric is not meaningful — aux losses that depend on g are skipped.
        routing_is_attention = (getattr(config, 'routing_mode', 'metric')
                                  == 'attention')
        tau_init_t = arc_model.dynamics.compute_tau(h0)
        if routing_is_attention:
            g_init_t = None
            g_init = None
        else:
            g_init_t = arc_model.dynamics.compute_metric_diag(h0)
            g_init = g_init_t[0] if isinstance(g_init_t, tuple) else g_init_t

        if microcircuit is not None:
            # Phase 1a microcircuit path: compress → dynamics on M slots → expand
            h_ode = microcircuit(h0, context, mask=causal_mask,
                                   euler_solve_fn=euler_solve)
        elif phased_dynamics is None:
            # Weight-tied single-dynamics path (original behavior)
            h_ode = euler_solve(arc_model.dynamics, h0,
                                 t_span=(0.0, 2.0), n_steps=n_steps)
            if isinstance(h_ode, tuple):
                h_ode = h_ode[0]
        else:
            # Phased un-tied ODE: iterate distinct dynamics modules per phase.
            # Each phase gets its own slice of the 16 Euler steps.
            steps_per_phase = n_steps // args.n_phases
            T_integration = 2.0
            dt = T_integration / n_steps
            h_ode = h0
            for phase_idx, dyn in enumerate(phased_dynamics):
                dyn.set_context(context, mask=causal_mask)
                dyn.set_n_steps(n_steps)
                for step_in_phase in range(steps_per_phase):
                    global_step = phase_idx * steps_per_phase + step_in_phase
                    # Keep step_index consistent with weight-tied interpretation
                    # so downstream (step_embed, tau_step_embed) still fire
                    if hasattr(dyn, '_current_step_index_buf'):
                        dyn._current_step_index_buf.fill_(global_step)
                    t_ode = torch.tensor(global_step * dt, device=device,
                                          dtype=h0.dtype)
                    dh_dt = dyn(t_ode, h_ode)
                    h_ode = h_ode + dt * dh_dt

        # Next-token (or masked-token, or distilled) prediction.
        logits = text_head(h_ode)                         # [B, T, vocab]
        if args.training_mode == "mlm" and mlm_mask is not None:
            logits_masked = logits[mlm_mask]             # [n_masked, vocab]
            targets_masked = target_ids[mlm_mask]        # [n_masked]
            ntp_loss = F.cross_entropy(logits_masked, targets_masked)
        elif args.training_mode == "distill" and teacher is not None:
            # Teacher logits at same positions (teacher is causal; student
            # uses causal mask, so alignment is 1:1 per position).
            with torch.no_grad():
                t_logits = teacher(input_ids).logits     # [B, T, vocab]
            T_soft = args.distill_temp
            log_student = F.log_softmax(logits / T_soft, dim=-1)
            teacher_probs = F.softmax(t_logits / T_soft, dim=-1)
            kd_loss = F.kl_div(
                log_student.reshape(-1, vocab),
                teacher_probs.reshape(-1, vocab),
                reduction='batchmean',
            ) * (T_soft * T_soft)
            if args.distill_hard_mix > 0:
                hard_ce = F.cross_entropy(
                    logits.reshape(-1, vocab), target_ids.reshape(-1))
                ntp_loss = (1.0 - args.distill_hard_mix) * kd_loss + \
                           args.distill_hard_mix * hard_ce
            else:
                ntp_loss = kd_loss
        else:
            ntp_loss = F.cross_entropy(
                logits.view(-1, vocab), target_ids.view(-1))

        # ── Aux losses (mirroring LiquidARCModel.forward at model.py:490-530) ──
        # Skipped in attention routing mode — no meaningful g/metric exists.
        aux_loss = torch.zeros((), device=device)
        crit_ratio_val = 0.0
        if not routing_is_attention and config.criticality_loss_enabled:
            _c_loss, _c_diag = compute_criticality_loss(
                h_ode, g_init, tau_init_t, arc_model.dynamics.t_diffusion,
                target_ratio=config.criticality_target_ratio,
                d_sq_target=config.criticality_D_sq_target,
            )
            aux_loss = aux_loss + config.criticality_loss_lambda * _c_loss
            crit_ratio_val = float(_c_diag.get('ratio', 0.0))
        if config.tau_quality_loss_enabled:
            # tau_quality operates on TauNet output — valid in both routing modes.
            mean_target = config.tau_mean_target
            if mean_target <= 0:
                mean_target = 2.0 / max(1, n_steps) * 16.0
            _tq = compute_tau_quality_loss(
                tau_init_t, mean_target=mean_target,
                log_spread_target=config.tau_log_spread_target,
            )
            aux_loss = aux_loss + config.tau_quality_lambda * _tq
        if not routing_is_attention and config.curvature_diversity_loss_enabled:
            _cd = compute_curvature_diversity_loss(
                g_init, cv_floor=config.curvature_cv_floor,
                cv_ceiling=config.curvature_cv_ceiling,
            )
            aux_loss = aux_loss + config.curvature_diversity_lambda * _cd

        # CV floor/ceiling hinge — metric-dependent, skipped in attention mode.
        # Capacity-preservation regularizer (Barlow-Twins style):
        # penalize off-diagonal of normalized hidden-state covariance so
        # features remain decorrelated → effective rank stays high.
        if args.decorr_lambda > 0:
            H = h_ode.reshape(-1, h_ode.shape[-1])        # [B*T, d]
            H = H - H.mean(dim=0, keepdim=True)
            H = H / (H.std(dim=0, keepdim=True) + 1e-6)
            C = (H.transpose(0, 1) @ H) / H.shape[0]       # [d, d]
            d_ = C.shape[0]
            off_diag = C - torch.diag(torch.diagonal(C))
            decorr_loss = (off_diag ** 2).sum() / (d_ * (d_ - 1))
            aux_loss = aux_loss + args.decorr_lambda * decorr_loss

        if not routing_is_attention and config.cv_floor_lambda > 0:
            cv_val = g_init.std() / (g_init.mean() + 1e-8)
            deficit = torch.clamp(config.cv_floor_target - cv_val, min=0.0)
            ceiling = getattr(config, 'cv_ceiling_target', 0.0)
            excess = (torch.clamp(cv_val - ceiling, min=0.0)
                      if ceiling > 0 else torch.zeros((), device=device))
            aux_loss = aux_loss + config.cv_floor_lambda * (
                deficit ** 2 + excess ** 2)

        loss = ntp_loss + aux_loss

        # ── Backward ──
        optimizer.zero_grad()
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  step {step}: NaN/Inf, skipping")
            continue
        loss.backward()
        for pp in list(arc_model.parameters()) + list(text_embed.parameters()) + list(text_head.parameters()):
            if pp.grad is not None:
                torch.nan_to_num_(pp.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(
            list(arc_model.parameters()) + list(text_embed.parameters()) +
            list(text_head.parameters()), 1.0)
        optimizer.step()

        # ── Logging ──
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            if g_init is not None:
                cv = (g_init.std() / (g_init.mean() + 1e-8)).item()
            else:
                cv = 0.0  # metric undefined in attention mode
            tau_flat = tau_init_t.squeeze(-1)[0]
            tau_mean = tau_flat.mean().item()
            tau_std = tau_flat.std().item()
            log_tau_std = torch.log(tau_flat + 1e-8).std().item()
            s_tau_std = 0.0
            s_tau_grad_norm = 0.0
            if hasattr(arc_model.dynamics, "structural_tau"):
                s = arc_model.dynamics.structural_tau
                with torch.no_grad():
                    s_tau_std = torch.sigmoid(s).std().item()
                if s.grad is not None:
                    s_tau_grad_norm = s.grad.norm().item()
            t_diff = F.softplus(arc_model.dynamics.t_diffusion).item()
            ppl = math.exp(min(ntp_loss.item(), 20))
            # MLM-specific diagnostic: accuracy at masked positions
            extra = ""
            if args.training_mode == "mlm" and mlm_mask is not None:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    n_mask = int(mlm_mask.sum().item())
                    n_correct = int(((preds == target_ids) & mlm_mask).sum().item())
                    mlm_acc = n_correct / max(1, n_mask)
                extra = f" mlm_acc={mlm_acc:.3f} n_mask={n_mask}"
            elif args.training_mode == "distill" and teacher is not None:
                # Student-teacher argmax agreement rate
                with torch.no_grad():
                    t_logits_diag = teacher(input_ids).logits
                    s_preds = logits.argmax(dim=-1)
                    t_preds = t_logits_diag.argmax(dim=-1)
                    agree = float((s_preds == t_preds).float().mean().item())
                    hard_acc = float((s_preds == target_ids).float().mean().item())
                extra = f" agree_w_teacher={agree:.3f} hard_acc={hard_acc:.3f}"
                # Coupled-mode gate diagnostic: if the gate collapses toward
                # 0 or 1 and has low std, one mechanism dominates; if it
                # varies per-position, both are being used.
                if getattr(config, 'routing_mode', '') == 'coupled':
                    dyn0 = (phased_dynamics[0]
                            if phased_dynamics is not None
                            else arc_model.dynamics)
                    if hasattr(dyn0, '_last_gate_mean'):
                        gm = float(dyn0._last_gate_mean.item())
                        gs = float(dyn0._last_gate_std.item())
                        extra += f" gate_μ={gm:.3f} gate_σ={gs:.3f}"
            print(f"  step {step:>5d} | loss={ntp_loss.item():.3f} ppl={ppl:.0f}"
                  f"{extra} | CV={cv:.3f} D²/4τ={crit_ratio_val:.1f} "
                  f"| tau={tau_mean:.2f}±{tau_std:.2f} log_τσ={log_tau_std:.3f} "
                  f"| s_τσ={s_tau_std:.4f} s_τ∇={s_tau_grad_norm:.2e} "
                  f"| t={t_diff:.2f} aux={aux_loss.item():.3f} "
                  f"| {step/elapsed:.1f} step/s", flush=True)

    # ── Save checkpoint ──
    ckpt_path = os.path.join(args.output_dir, "final.pt")
    payload = {
        'step': args.max_steps,
        'model_state_dict': arc_model.state_dict(),
        'text_embed_state_dict': text_embed.state_dict(),
        'text_head_state_dict': text_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    if phased_dynamics is not None and args.n_phases > 1:
        payload['phased_dynamics_state_dict'] = phased_dynamics.state_dict()
        payload['n_phases'] = args.n_phases
    torch.save(payload, ckpt_path)
    print(f"\nDone. Saved {ckpt_path}")


if __name__ == "__main__":
    main()
