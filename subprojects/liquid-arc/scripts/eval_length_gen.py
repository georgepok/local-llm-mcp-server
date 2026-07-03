"""Length-generalization eval: measure PPL / hard_acc across seq_lens.

Tests whether a locality-based substrate (chunked, no-routing) is invariant
to eval sequence length. A dense baseline's PPL typically improves with T
(more context); a chunked model should be roughly flat across T because
each chunk is a local window independent of absolute T.

Supports the same checkpoints saved by train_text_token.py:
  - arc_model (with optional structural_tau, routing_mode)
  - text_embed / text_head
  - phased_dynamics (if n_phases > 1; not used here)

For chunked variants:
  - Rebuilds ChunkedMicroCircuitWrapper using --chunked_M / --chunked_no_routing
  - Uses cyclic per-chunk positional embeddings (position = i % L within chunk)
    so pos_embed never sees indices > max_seq_len it was trained on.

Evaluates on WikiText-103 validation. Samples n_windows at each seq_len.
"""

from __future__ import annotations

import argparse
import math
import sys
import random

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

    # Rebuild text_embed/head at the config-trained max_seq_len (typically 512).
    txt_embed = TextEmbedding(vocab_size=vocab_size, d_model=cfg.d_model,
                              max_seq_len=cfg.max_seq_len, dropout=0.0).to(device).eval()
    txt_head = TextHead(d_model=cfg.d_model, vocab_size=vocab_size).to(device).eval()
    txt_embed.load_state_dict(ckpt['text_embed_state_dict'])
    txt_head.load_state_dict(ckpt['text_head_state_dict'])

    return cfg, arc, txt_embed, txt_head


@torch.no_grad()
def embed_cyclic(txt_embed: TextEmbedding, input_ids: torch.Tensor,
                  cycle_L: int) -> torch.Tensor:
    """Embed with positions = i % cycle_L per token.

    If cycle_L == 0 (or equals T), falls back to standard positional embedding.
    Used for chunked models so each chunk sees the same local position sequence.
    """
    B, T = input_ids.shape
    if cycle_L <= 0 or cycle_L >= T:
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
    else:
        local_pos = torch.arange(T, device=input_ids.device) % cycle_L
        positions = local_pos.unsqueeze(0).expand(B, -1)
    h = txt_embed.token_embed(input_ids) + txt_embed.pos_embed(positions)
    return txt_embed.dropout(txt_embed.norm(h))


@torch.no_grad()
def run_forward(arc, cfg, h0: torch.Tensor, microcircuit=None,
                causal_mask: torch.Tensor = None) -> torch.Tensor:
    """Single forward: context_pool → ODE dynamics → h_out."""
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
def evaluate_at_seq_len(arc, cfg, txt_embed, txt_head, tok, device,
                        seq_len: int, n_windows: int,
                        microcircuit=None, chunked_L: int = 0,
                        causal: bool = True):
    """Run eval at a specific seq_len. Returns loss, ppl, hard_acc."""
    from datasets import load_dataset
    val = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    text = ' '.join([t for t in val['text'] if len(t.strip()) > 50])
    ids = tok.encode(text)
    n = len(ids)

    if seq_len + 2 > n:
        return None

    causal_mask = None
    if causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    random.seed(0)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    for _ in range(n_windows):
        start = random.randint(0, n - seq_len - 2)
        w = ids[start:start + seq_len + 1]
        inp = torch.tensor([w[:-1]], device=device)
        tgt = torch.tensor([w[1:]], device=device)

        h0 = embed_cyclic(txt_embed, inp, chunked_L)

        h = run_forward(arc, cfg, h0, microcircuit=microcircuit,
                         causal_mask=causal_mask)
        logits = txt_head(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                tgt.view(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += tgt.numel()

        preds = logits.argmax(dim=-1)
        total_correct += (preds == tgt).sum().item()

    avg_loss = total_loss / total_tokens
    return {
        "seq_len": seq_len,
        "loss": avg_loss,
        "ppl": math.exp(min(avg_loss, 20)),
        "hard_acc": total_correct / total_tokens,
        "n_tokens": total_tokens,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seq_lens", default="128,256,384,512",
                   help="Comma-separated seq_lens to evaluate")
    p.add_argument("--n_windows", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunked_M", type=int, default=0,
                   help="If >0, rebuild ChunkedMicroCircuitWrapper with this M")
    p.add_argument("--chunked_no_routing", action="store_true")
    p.add_argument("--positional", choices=["absolute", "cyclic"],
                   default="absolute",
                   help="absolute: positions 0..T-1 (matches training); "
                        "cyclic: positions i%%L per chunk (tests locality)")
    p.add_argument("--teacher_tokenizer", default="gpt2-medium")
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

    seq_lens = [int(s) for s in args.seq_lens.split(",")]

    print(f"═══ Length-generalization eval ═══")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  d_model={cfg.d_model}  max_seq_len_trained={cfg.max_seq_len}")
    print(f"  chunked_M={args.chunked_M}  no_routing={args.chunked_no_routing}")
    print(f"  n_windows={args.n_windows}")
    print()
    print(f"  {'T':>6}  {'L/chunk':>8}  {'loss':>8}  {'ppl':>8}  {'hard_acc':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

    for T in seq_lens:
        L = T // args.chunked_M if args.chunked_M > 0 else T
        if args.chunked_M > 0 and T % args.chunked_M != 0:
            print(f"  skip T={T}: not divisible by M={args.chunked_M}")
            continue
        # cycle=0 → absolute positions; cycle=L → per-chunk cyclic.
        pos_cycle = L if (args.positional == "cyclic" and microcircuit is not None) else 0
        r = evaluate_at_seq_len(
            arc, cfg, txt_embed, txt_head, tok, device,
            seq_len=T, n_windows=args.n_windows,
            microcircuit=microcircuit, chunked_L=pos_cycle,
        )
        if r is None:
            print(f"  T={T}: skipped (val text too short)")
            continue
        print(f"  {T:>6}  {L:>8}  {r['loss']:>8.3f}  {r['ppl']:>8.2f}  "
              f"{r['hard_acc']:>10.4f}")


if __name__ == "__main__":
    main()
