"""Domain-shift eval: same checkpoint evaluated on several text distributions.

Tests whether architecture-level wins on WikiText-103 val transfer to other
domains, or whether they're a form of in-domain regularization.

Distributions evaluated:
  - wikipedia  — WikiText-103 validation (in-distribution baseline)
  - code       — concatenated Python source from liquid-arc/ repo (OOD)
  - narrative  — project-gutenberg text sample via HF datasets if available,
                 else falls back to wikitext-2 (related-domain)
  - scientific — arxiv abstracts via HF datasets if available, else skipped

For each distribution: samples n_windows at the trained seq_len, measures
CE loss, ppl, hard_acc.

Supports chunked microcircuit variants via the same --chunked_M /
--chunked_no_routing flags as train_text_token.py.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import random
import glob
import json

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/liquid-arc")
sys.path.insert(0, "/home/pokazge/liquid-arc")

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel
from liquid_arc.solver import euler_solve
from liquid_arc.tasks.text_task import TextEmbedding, TextHead


def load_checkpoint(ckpt_path: str, device: str, vocab_size: int):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: LiquidARCConfig = ckpt['config']
    cfg.tau_freeze_steps = 0

    arc = LiquidARCModel(cfg).to(device).eval()
    arc.load_state_dict(ckpt['model_state_dict'], strict=False)

    txt_embed = TextEmbedding(vocab_size=vocab_size, d_model=cfg.d_model,
                              max_seq_len=cfg.max_seq_len, dropout=0.0).to(device).eval()
    txt_head = TextHead(d_model=cfg.d_model, vocab_size=vocab_size).to(device).eval()
    txt_embed.load_state_dict(ckpt['text_embed_state_dict'])
    txt_head.load_state_dict(ckpt['text_head_state_dict'])

    return cfg, arc, txt_embed, txt_head


def load_corpus(domain: str, tok):
    """Load tokenized text for a domain. Returns list of token IDs."""
    if domain == "wikipedia":
        from datasets import load_dataset
        val = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
        text = ' '.join([t for t in val['text'] if len(t.strip()) > 50])
    elif domain == "code":
        # Python source from liquid-arc/ — guaranteed available, clearly OOD
        roots = [
            "/workspace/liquid-arc/liquid_arc",
            "/workspace/liquid-arc/scripts",
            "/workspace/fgn-v3/fgn",
        ]
        pieces = []
        for r in roots:
            for p in sorted(glob.glob(os.path.join(r, "*.py")) +
                             glob.glob(os.path.join(r, "**/*.py"),
                                        recursive=True)):
                try:
                    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                        pieces.append(f.read())
                except Exception:
                    pass
        text = "\n\n".join(pieces)
    elif domain == "narrative":
        # WikiText-2 val — same domain family but different slice from WT-103
        from datasets import load_dataset
        val = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
        text = ' '.join([t for t in val['text'] if len(t.strip()) > 50])
    elif domain == "yaml":
        # YAML configs — 4th OOD domain not seen in any training
        roots = ["/workspace/liquid-arc",
                 "/workspace/fgn-v3",
                 "/workspace/lewm-integration"]
        pieces = []
        for r in roots:
            for p in sorted(
                glob.glob(os.path.join(r, "**/*.yaml"), recursive=True)
                + glob.glob(os.path.join(r, "**/*.yml"), recursive=True)):
                try:
                    with open(p, 'r', encoding='utf-8',
                              errors='ignore') as f:
                        pieces.append(f.read())
                except Exception:
                    pass
        text = "\n\n".join(pieces)
    elif domain == "markdown":
        # Project markdown docs — third domain not seen in training
        # (mixed50 training was WikiText + Python .py files)
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
        text = "\n\n".join(pieces)
    elif domain == "dialogue":
        # Maximally-different OOD: turn-based conversational text.
        # Tries several HF sources; if all fail, skips with a note.
        text = None
        for ds_spec in [
            ('Anthropic/hh-rlhf', 'test', 'chosen'),
            ('tatsu-lab/alpaca', 'train', 'output'),
            ('yahma/alpaca-cleaned', 'train', 'output'),
        ]:
            try:
                from datasets import load_dataset
                name, split, field = ds_spec
                ds = load_dataset(name, split=split, streaming=False)
                n = min(500, len(ds))
                pieces = []
                for i in range(n):
                    v = ds[i].get(field, '')
                    if isinstance(v, str):
                        pieces.append(v)
                text = '\n\n'.join(pieces)
                print(f"  [dialogue] loaded {name} ({n} samples)")
                break
            except Exception as e:
                print(f"  [dialogue] {ds_spec[0]} failed: "
                      f"{str(e)[:80]}")
        if text is None:
            print(f"  [dialogue] no source available, skipping")
            return None
    elif domain == "scientific":
        try:
            from datasets import load_dataset
            ds = load_dataset('scientific_papers', 'arxiv',
                               split='validation', streaming=False,
                               trust_remote_code=True)
            text = ' '.join([ds[i]['abstract'] for i in range(50)])
        except Exception as e:
            print(f"  [scientific] unavailable ({e}), skipping")
            return None
    else:
        raise ValueError(f"unknown domain: {domain}")

    ids = tok.encode(text)
    return ids


@torch.no_grad()
def embed_absolute(txt_embed, input_ids):
    B, T = input_ids.shape
    positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
    h = txt_embed.token_embed(input_ids) + txt_embed.pos_embed(positions)
    return txt_embed.dropout(txt_embed.norm(h))


@torch.no_grad()
def run_forward(arc, cfg, h0, microcircuit=None, causal_mask=None):
    context = arc.context_pool(h0)
    if microcircuit is not None:
        return microcircuit(h0, context, mask=causal_mask,
                             euler_solve_fn=euler_solve)
    arc.dynamics.set_context(context, mask=causal_mask)
    arc.dynamics.set_n_steps(cfg.n_ode_steps)
    h = euler_solve(arc.dynamics, h0, t_span=(0.0, 2.0),
                     n_steps=cfg.n_ode_steps)
    return h[0] if isinstance(h, tuple) else h


@torch.no_grad()
def evaluate(arc, cfg, txt_embed, txt_head, ids: list, device: str,
             seq_len: int, n_windows: int,
             microcircuit=None, causal: bool = True, seed: int = 0):
    n = len(ids)
    if n < seq_len + 2:
        return None

    causal_mask = None
    if causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    random.seed(seed)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    for _ in range(n_windows):
        start = random.randint(0, n - seq_len - 2)
        w = ids[start:start + seq_len + 1]
        inp = torch.tensor([w[:-1]], device=device)
        tgt = torch.tensor([w[1:]], device=device)

        h0 = embed_absolute(txt_embed, inp)
        h = run_forward(arc, cfg, h0, microcircuit=microcircuit,
                         causal_mask=causal_mask)
        logits = txt_head(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                tgt.view(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += tgt.numel()
        total_correct += (logits.argmax(-1) == tgt).sum().item()

    avg = total_loss / total_tokens
    return {
        "loss": avg,
        "ppl": math.exp(min(avg, 20)),
        "hard_acc": total_correct / total_tokens,
        "n_tokens": total_tokens,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--n_windows", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunked_M", type=int, default=0)
    p.add_argument("--chunked_no_routing", action="store_true")
    p.add_argument("--teacher_tokenizer", default="gpt2-medium")
    p.add_argument("--domains", default="wikipedia,code,narrative",
                   help="comma-separated: wikipedia,code,narrative,"
                        "dialogue,scientific")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    device = args.device
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher_tokenizer,
                                         trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vocab = len(tok)

    cfg, arc, txt_embed, txt_head = load_checkpoint(
        args.checkpoint, device, vocab_size=vocab)

    microcircuit = None
    if args.chunked_M > 0:
        from liquid_arc.microcircuit import ChunkedMicroCircuitWrapper
        microcircuit = ChunkedMicroCircuitWrapper(
            d=cfg.d_model, M=args.chunked_M, dynamics=arc.dynamics,
            n_ode_steps=cfg.n_ode_steps,
            inter_chunk_routing=(not args.chunked_no_routing),
        ).to(device).eval()

    print(f"═══ Domain-shift eval ═══")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  d_model={cfg.d_model}  max_seq_len_trained={cfg.max_seq_len}")
    print(f"  chunked_M={args.chunked_M}  no_routing={args.chunked_no_routing}")
    print(f"  seq_len={args.seq_len}  n_windows={args.n_windows}")
    print()
    print(f"  {'domain':>12}  {'n_tokens':>10}  {'loss':>8}  {'ppl':>10}  {'hard_acc':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}")

    results = {}
    for domain in args.domains.split(","):
        domain = domain.strip()
        try:
            ids = load_corpus(domain, tok)
        except Exception as e:
            print(f"  {domain:>12}  LOAD FAILED: {e}")
            continue
        if ids is None:
            continue
        r = evaluate(arc, cfg, txt_embed, txt_head, ids, device,
                      seq_len=args.seq_len, n_windows=args.n_windows,
                      microcircuit=microcircuit)
        if r is None:
            print(f"  {domain:>12}  TOO SHORT (n={len(ids)} < {args.seq_len+2})")
            continue
        print(f"  {domain:>12}  {r['n_tokens']:>10,}  {r['loss']:>8.3f}  "
              f"{r['ppl']:>10.2f}  {r['hard_acc']:>10.4f}")
        results[domain] = r

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({
                "checkpoint": args.checkpoint,
                "chunked_M": args.chunked_M,
                "chunked_no_routing": args.chunked_no_routing,
                "seq_len": args.seq_len,
                "results": results,
            }, f, indent=2)
        print(f"\n  saved → {args.json_out}")


if __name__ == "__main__":
    main()
