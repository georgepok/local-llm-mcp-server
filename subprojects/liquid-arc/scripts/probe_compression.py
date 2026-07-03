"""Compression probe — measure effective rank of hidden states.

Tests the hypothesis that OOD-transfer quality correlates with representation
compression: models with lower effective rank (more compressed hidden state
distributions) generalize better.

For each checkpoint:
  1. Sample N windows from multiple domains (wiki, code, dialogue)
  2. Forward pass: collect h_embedding (pre-dynamics) and h_post_ode (post).
  3. Concatenate tokens → [N*T, d_model] matrix.
  4. SVD; report:
       - entropy-based effective rank   exp(H_norm) where H_norm = -Σ p_i log p_i
         and p_i = σ_i² / Σ σ_j² (Roy & Vetterli 2007)
       - stable rank                    (Σ σ²) / σ_max²
       - participation ratio            (Σ σ²)² / (Σ σ⁴)

Predictions:
  - More-OOD-competent models should have LOWER eff_rank(h_post_ode).
  - Compression over ODE: eff_rank(h_post_ode) < eff_rank(h_embedding) within
    a single model.
  - Cross-domain compression: eff_rank on a training-seen domain should be
    lower (more stable) than eff_rank on an unseen OOD domain (hidden state
    less structured for inputs outside the learned distribution).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, "/workspace/liquid-arc")
sys.path.insert(0, "/home/pokazge/liquid-arc")

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel
from liquid_arc.solver import euler_solve
from liquid_arc.tasks.text_task import TextEmbedding, TextHead

# Reuse domain loaders and forward logic from the eval script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_domain_shift import load_corpus, embed_absolute, run_forward


def load_checkpoint(ckpt_path: str, device: str, vocab_size: int):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: LiquidARCConfig = ckpt['config']
    cfg.tau_freeze_steps = 0
    arc = LiquidARCModel(cfg).to(device).eval()
    arc.load_state_dict(ckpt['model_state_dict'], strict=False)
    txt_embed = TextEmbedding(
        vocab_size=vocab_size, d_model=cfg.d_model,
        max_seq_len=cfg.max_seq_len, dropout=0.0).to(device).eval()
    txt_head = TextHead(d_model=cfg.d_model,
                         vocab_size=vocab_size).to(device).eval()
    txt_embed.load_state_dict(ckpt['text_embed_state_dict'])
    txt_head.load_state_dict(ckpt['text_head_state_dict'])
    return cfg, arc, txt_embed, txt_head


@torch.no_grad()
def compute_ranks(H: torch.Tensor) -> dict:
    """H: [N, d] — return effective/stable/participation rank measures."""
    # Center before SVD (rank of variation, not of mean)
    H = H - H.mean(dim=0, keepdim=True)
    # SVD on CPU — container's cusolver stub is broken, and the matrix
    # is small enough ([N*T, d] ≈ [5120, 768]) that CPU SVD is fast.
    s = torch.linalg.svdvals(H.float().cpu())           # [min(N,d)]
    s2 = s ** 2
    total = s2.sum()
    if total.item() == 0.0:
        return {"eff_rank": 0.0, "stable_rank": 0.0, "pr": 0.0, "d": H.shape[1]}
    # Entropy-based effective rank (Roy & Vetterli)
    p = s2 / total
    H_ent = -(p * (p + 1e-30).log()).sum()
    eff_rank = float(H_ent.exp().item())
    # Stable rank: ||A||_F² / ||A||_2²
    stable_rank = float((total / (s2.max() + 1e-30)).item())
    # Participation ratio (inverse form matching intrinsic-dim convention)
    pr = float(((total ** 2) / (s2 ** 2).sum()).item())
    return {
        "eff_rank": eff_rank,
        "stable_rank": stable_rank,
        "pr": pr,
        "d": int(H.shape[1]),
        "n_samples": int(H.shape[0]),
        "sigma_max": float(s[0].item()),
        "sigma_min_nonzero": float(s[s > 1e-8][-1].item()) if (s > 1e-8).any() else 0.0,
    }


@torch.no_grad()
def probe(arc, cfg, txt_embed, tok, ids: list, device: str,
          seq_len: int, n_windows: int, microcircuit=None,
          seed: int = 0):
    """Collect h_embedding and h_post_ode across n_windows, compute ranks."""
    import random as _r
    _r.seed(seed)
    H_emb_chunks = []
    H_ode_chunks = []
    n = len(ids)

    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )

    for _ in range(n_windows):
        start = _r.randint(0, n - seq_len - 2)
        w = ids[start:start + seq_len + 1]
        inp = torch.tensor([w[:-1]], device=device)
        h0 = embed_absolute(txt_embed, inp)              # [1, T, d]
        h_ode = run_forward(arc, cfg, h0,
                             microcircuit=microcircuit,
                             causal_mask=causal_mask)     # [1, T, d]
        H_emb_chunks.append(h0.squeeze(0))
        H_ode_chunks.append(h_ode.squeeze(0))
    H_emb = torch.cat(H_emb_chunks, dim=0)               # [N*T, d]
    H_ode = torch.cat(H_ode_chunks, dim=0)
    return {
        "embedding": compute_ranks(H_emb),
        "post_ode":  compute_ranks(H_ode),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", required=True,
                   help="Comma-sep paths to checkpoints to probe")
    p.add_argument("--labels", default=None,
                   help="Comma-sep human labels for each ckpt (matches order)")
    p.add_argument("--chunked_M", default="0",
                   help="Comma-sep chunked_M per checkpoint (0 = dense)")
    p.add_argument("--chunked_no_routing", action="store_true")
    p.add_argument("--domains", default="wikipedia,code,dialogue")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--n_windows", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--teacher_tokenizer", default="gpt2-medium")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher_tokenizer,
                                         trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vocab = len(tok)

    ckpts = args.checkpoints.split(",")
    chunked_Ms = [int(x) for x in args.chunked_M.split(",")]
    if len(chunked_Ms) == 1:
        chunked_Ms *= len(ckpts)
    labels = args.labels.split(",") if args.labels else ckpts
    domains = args.domains.split(",")

    # Pre-load corpora once
    corpora = {}
    for d in domains:
        try:
            corpora[d] = load_corpus(d, tok)
        except Exception as e:
            print(f"  skip {d}: {e}")

    print(f"═══ Compression probe ═══")
    print(f"  seq_len={args.seq_len}  n_windows={args.n_windows}  "
          f"tokens_per_domain≈{args.seq_len * args.n_windows}")
    print(f"  domains: {list(corpora.keys())}")
    print()

    # Format: for each ckpt × domain, print embedding + post_ode ranks
    header = f"  {'label':>22}  {'domain':>10}  {'layer':>10}  " \
             f"{'eff_rank':>10}  {'stable':>10}  {'pr':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for ckpt, M, label in zip(ckpts, chunked_Ms, labels):
        cfg, arc, txt_embed, _ = load_checkpoint(ckpt, args.device, vocab)
        microcircuit = None
        if M > 0:
            from liquid_arc.microcircuit import ChunkedMicroCircuitWrapper
            microcircuit = ChunkedMicroCircuitWrapper(
                d=cfg.d_model, M=M, dynamics=arc.dynamics,
                n_ode_steps=cfg.n_ode_steps,
                inter_chunk_routing=(not args.chunked_no_routing),
            ).to(args.device).eval()

        for d, ids in corpora.items():
            if ids is None or len(ids) < args.seq_len + 2:
                continue
            r = probe(arc, cfg, txt_embed, tok, ids, args.device,
                       seq_len=args.seq_len, n_windows=args.n_windows,
                       microcircuit=microcircuit)
            for layer in ("embedding", "post_ode"):
                s = r[layer]
                print(f"  {label:>22}  {d:>10}  {layer:>10}  "
                      f"{s['eff_rank']:>10.2f}  {s['stable_rank']:>10.2f}  "
                      f"{s['pr']:>10.2f}")
            # Compression ratio within this model×domain
            ratio = r['post_ode']['eff_rank'] / max(1e-9,
                                                     r['embedding']['eff_rank'])
            print(f"  {'':>22}  {d:>10}  {'compression':>10}  "
                  f"{ratio:>10.3f}  (post/emb eff_rank)")


if __name__ == "__main__":
    main()
