"""Evaluate semantics capabilities of the token-level text-first LiquidARC.

Checkpoint: output_text_token/final.pt
  - 573K param LiquidARCModel (d=512) trained 3000 steps on WikiText-2 NTP
  - Phase transition at step 300, CV=7.6 at end, log_τσ=0.58

Probes:
  1. Held-out PPL on WikiText-2 validation
  2. CV response to input class (repeated / random / natural / syntactic)
  3. structural_τ per-position pattern (autocorrelation + rank profile)
  4. MetricNet response to semantic vs random token pairs
  5. Attention row for a fixed sentence

Each probe is isolated; any can be skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/liquid-arc")
sys.path.insert(0, "/home/pokazge/liquid-arc")

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel
from liquid_arc.solver import euler_solve
from liquid_arc.tasks.text_task import TextEmbedding, TextHead


def load_everything(ckpt_path: str, device: str):
    """Rebuild model + embedding + head from a saved checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: LiquidARCConfig = ckpt['config']
    cfg.tau_freeze_steps = 0

    arc = LiquidARCModel(cfg).to(device).eval()
    arc.load_state_dict(ckpt['model_state_dict'], strict=False)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vocab = len(tok)

    txt_embed = TextEmbedding(vocab_size=vocab, d_model=cfg.d_model,
                               max_seq_len=512, dropout=0.0).to(device).eval()
    txt_head = TextHead(d_model=cfg.d_model, vocab_size=vocab).to(device).eval()
    txt_embed.load_state_dict(ckpt['text_embed_state_dict'])
    txt_head.load_state_dict(ckpt['text_head_state_dict'])

    return cfg, arc, txt_embed, txt_head, tok


@torch.no_grad()
def run_ode(arc, h0: torch.Tensor, cfg: LiquidARCConfig):
    """One forward pass through context_pool + ODE."""
    ctx = arc.context_pool(h0)
    arc.dynamics.set_context(ctx, mask=None)
    arc.dynamics.set_n_steps(cfg.n_ode_steps)
    h = euler_solve(arc.dynamics, h0, t_span=(0.0, 2.0),
                     n_steps=cfg.n_ode_steps)
    return h[0] if isinstance(h, tuple) else h


# ────────────────────────────────────────────────────────────
# Probe 1: held-out perplexity
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def probe_heldout_ppl(arc, txt_embed, txt_head, tok, cfg, device,
                       seq_len: int = 512, n_windows: int = 20) -> Dict:
    from datasets import load_dataset
    val = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    text = ' '.join([t for t in val['text'] if len(t.strip()) > 50])
    ids = tok.encode(text)
    n = len(ids)
    total_loss = 0.0
    total_tokens = 0
    import random
    random.seed(0)
    for _ in range(n_windows):
        start = random.randint(0, max(0, n - seq_len - 2))
        w = ids[start:start + seq_len + 1]
        inp = torch.tensor([w[:-1]], device=device)
        tgt = torch.tensor([w[1:]], device=device)
        h0 = txt_embed(inp)
        h = run_ode(arc, h0, cfg)
        logits = txt_head(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                tgt.view(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += tgt.numel()
    avg = total_loss / max(1, total_tokens)
    return {"val_loss": avg, "val_ppl": math.exp(min(avg, 20)),
            "n_tokens": total_tokens}


# ────────────────────────────────────────────────────────────
# Probe 2: CV response to input class
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def probe_cv_response(arc, txt_embed, tok, cfg, device,
                      seq_len: int = 512) -> Dict:
    """Is metric CV content-sensitive? Compare four input types."""
    inputs = {}
    # A: all the same token (should give low CV — uniform input)
    tid = tok.encode("the")[0]
    inputs["repeated_token"] = torch.full((1, seq_len), tid,
                                            dtype=torch.long, device=device)
    # B: uniform random tokens (should give moderate CV)
    inputs["random_tokens"] = torch.randint(
        0, len(tok), (1, seq_len), device=device)
    # C: natural text
    from datasets import load_dataset
    val = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    natural_ids = tok.encode(' '.join([t for t in val['text']
                                         if len(t.strip()) > 50])[:2500])[:seq_len]
    # pad if short
    if len(natural_ids) < seq_len:
        natural_ids = natural_ids + [tok.eos_token_id] * (seq_len - len(natural_ids))
    inputs["natural_text"] = torch.tensor([natural_ids], device=device)
    # D: repeated alternation "the . the . the ." — syntactic rhythm
    rhythm_ids = (tok.encode("the")[0:1] + tok.encode(".")[0:1]) * (seq_len // 2)
    inputs["alternating"] = torch.tensor([rhythm_ids[:seq_len]],
                                            device=device)

    out = {}
    for name, ids in inputs.items():
        h0 = txt_embed(ids)
        g_init_t = arc.dynamics.compute_metric_diag(h0)
        g = g_init_t[0] if isinstance(g_init_t, tuple) else g_init_t
        tau = arc.dynamics.compute_tau(h0).squeeze(-1)[0]
        out[name] = {
            "cv": (g.std() / (g.mean() + 1e-8)).item(),
            "g_mean": g.mean().item(),
            "g_std": g.std().item(),
            "tau_mean": tau.mean().item(),
            "tau_std": tau.std().item(),
            "log_tau_std": torch.log(tau + 1e-8).std().item(),
        }
    return out


# ────────────────────────────────────────────────────────────
# Probe 3: structural_τ pattern
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def probe_structural_tau(arc) -> Dict:
    if not hasattr(arc.dynamics, "structural_tau"):
        return {"skipped": "no structural_tau"}
    s = arc.dynamics.structural_tau.detach().cpu()
    s_sig = torch.sigmoid(s)
    # Autocorrelation at lag 1 (smoothness)
    s_centered = s_sig - s_sig.mean()
    ac_lag1 = (s_centered[:-1] * s_centered[1:]).mean().item() / \
              (s_centered.var().item() + 1e-8)
    # Rank profile: sort and look at spread
    sorted_vals, _ = s_sig.sort()
    quartiles = [
        float(sorted_vals[int(0.05 * len(sorted_vals))]),
        float(sorted_vals[int(0.25 * len(sorted_vals))]),
        float(sorted_vals[int(0.50 * len(sorted_vals))]),
        float(sorted_vals[int(0.75 * len(sorted_vals))]),
        float(sorted_vals[int(0.95 * len(sorted_vals))]),
    ]
    # Position 0-32 mean (front of sequence) vs 480-512 (end)
    front_mean = s_sig[:32].mean().item()
    mid_mean = s_sig[240:272].mean().item()
    back_mean = s_sig[-32:].mean().item()
    return {
        "n_positions": len(s),
        "sigmoid_mean": s_sig.mean().item(),
        "sigmoid_std": s_sig.std().item(),
        "sigmoid_min": s_sig.min().item(),
        "sigmoid_max": s_sig.max().item(),
        "lag1_autocorr": ac_lag1,
        "quartiles_5_25_50_75_95": quartiles,
        "front_vs_back_mean": {
            "front_32": front_mean, "middle_32": mid_mean, "back_32": back_mean,
        },
    }


# ────────────────────────────────────────────────────────────
# Probe 4: semantic vs random pair distances
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def probe_semantic_pairs(arc, txt_embed, tok, cfg, device) -> Dict:
    """Feed a fixed sentence + same with random-word insertions; compare pair distances."""
    # Fixed short context, then two continuations: coherent vs random
    coherent = "The capital of France is Paris. The capital of Germany is"
    random_cont = "The capital of France is Paris. Xylophone refrigerator tangent because"

    results = {}
    for name, text in [("coherent", coherent), ("random_insert", random_cont)]:
        ids = tok.encode(text)
        padded = ids + [tok.eos_token_id] * (512 - len(ids))
        inp = torch.tensor([padded[:512]], device=device)
        h0 = txt_embed(inp)
        real_n = min(len(ids), 512)
        # Compute pairwise D² over the real (non-pad) positions using the
        # learned metric. Use h0 (pre-ODE) to match heat-kernel routing.
        g_t = arc.dynamics.compute_metric_diag(h0)
        g = (g_t[0] if isinstance(g_t, tuple) else g_t)[0, :real_n, :]
        h_normed = arc.dynamics.norm_geo(h0)[0, :real_n, :]
        sqrt_g = torch.sqrt(g)
        k = h_normed * sqrt_g
        d2 = (k.unsqueeze(0) - k.unsqueeze(1)).pow(2).sum(-1)  # [n, n]
        # Token pairs for near (adjacent), far-same-sentence, across-sentence
        # Sentence 1: positions 0..9 (approx "The capital of France is Paris.")
        # Sentence 2: 10..end
        near = d2[0, 1].item()  # "The" → "capital"
        far_same = d2[0, 5].item()  # "The" → "Paris"
        across = d2[0, real_n - 1].item() if real_n > 10 else 0.0
        results[name] = {
            "d2_near_adj": near,
            "d2_far_same_sent": far_same,
            "d2_across_full_span": across,
            "d2_median_all_pairs": d2.masked_fill(
                torch.eye(real_n, dtype=torch.bool, device=device),
                float('inf')).flatten().median().item(),
        }
    return results


# ────────────────────────────────────────────────────────────
# Probe 5: attention row for a fixed sentence
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def probe_attention_row(arc, txt_embed, tok, cfg, device) -> Dict:
    """For 'The cat sat on the mat .' (7 tokens), compute heat-kernel attention
    for position 0 ('The') and position 1 ('cat'). Report the top-5 attended
    positions each."""
    sentence = "The cat sat on the mat."
    ids = tok.encode(sentence)
    n = len(ids)
    padded = ids + [tok.eos_token_id] * (512 - n)
    inp = torch.tensor([padded[:512]], device=device)
    h0 = txt_embed(inp)
    g_t = arc.dynamics.compute_metric_diag(h0)
    g = (g_t[0] if isinstance(g_t, tuple) else g_t)[0, :n, :]
    # Pairwise D² over the sentence tokens only
    h_normed = arc.dynamics.norm_geo(h0)[0, :n, :]
    sqrt_g = torch.sqrt(g)
    k = h_normed * sqrt_g
    d2 = (k.unsqueeze(0) - k.unsqueeze(1)).pow(2).sum(-1)
    t_diff = F.softplus(arc.dynamics.t_diffusion).item()
    attn = F.softmax(-d2 / (4.0 * t_diff), dim=-1)
    out = {
        "sentence": sentence,
        "tokens": [tok.decode([i]) for i in ids],
        "t_diffusion": t_diff,
        "attn_row_0_top5": [
            {"token_idx": int(idx.item()),
             "token": tok.decode([ids[int(idx.item())]]),
             "prob": float(prob.item())}
            for prob, idx in zip(*attn[0].topk(min(5, n)))
        ],
        "attn_row_1_top5": [
            {"token_idx": int(idx.item()),
             "token": tok.decode([ids[int(idx.item())]]),
             "prob": float(prob.item())}
            for prob, idx in zip(*attn[1].topk(min(5, n)))
        ],
        "attn_entropy_per_pos": [
            -float((a * (a + 1e-12).log()).sum().item()) for a in attn
        ],
    }
    return out


# ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="output_text_token/final.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="output_text_token/semantics_eval.json")
    args = p.parse_args()

    print(f"═══ Loading {args.ckpt} ═══")
    cfg, arc, emb, head, tok = load_everything(args.ckpt, args.device)
    print(f"  d={cfg.d_model}, seq_len={cfg.max_seq_len}, "
          f"structural_tau_enabled={cfg.structural_tau_enabled}")

    results = {}
    print("\n--- Probe 1: held-out PPL ---")
    results["heldout_ppl"] = probe_heldout_ppl(arc, emb, head, tok, cfg, args.device)
    print(json.dumps(results["heldout_ppl"], indent=2))

    print("\n--- Probe 2: CV response to input class ---")
    results["cv_response"] = probe_cv_response(arc, emb, tok, cfg, args.device)
    print(json.dumps(results["cv_response"], indent=2))

    print("\n--- Probe 3: structural_tau pattern ---")
    results["structural_tau"] = probe_structural_tau(arc)
    print(json.dumps(results["structural_tau"], indent=2))

    print("\n--- Probe 4: semantic vs random pair distances ---")
    results["semantic_pairs"] = probe_semantic_pairs(arc, emb, tok, cfg, args.device)
    print(json.dumps(results["semantic_pairs"], indent=2))

    print("\n--- Probe 5: attention row ---")
    results["attention_row"] = probe_attention_row(arc, emb, tok, cfg, args.device)
    print(json.dumps(results["attention_row"], indent=2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
