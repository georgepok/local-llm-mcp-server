"""Generate text continuations from the distilled LiquidARC.

Evaluates semantic coherence by greedy continuation on prompts, side-by-side
with the GPT-2 teacher. Student is causal-masked (matches training).
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/workspace/liquid-arc")
sys.path.insert(0, "/home/pokazge/liquid-arc")

import torch
import torch.nn.functional as F

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel
from liquid_arc.solver import euler_solve
from liquid_arc.tasks.text_task import TextEmbedding, TextHead


PROMPTS = [
    "The capital of France is",
    "In the beginning God created the",
    "The quick brown fox jumps over",
    "Machine learning is a subfield of",
    "The Earth revolves around the",
    "Neural networks consist of layers of",
]


@torch.no_grad()
def student_continue(prompt_ids, arc, text_embed, text_head, cfg, device,
                     n_new=12, phased_dynamics=None,
                     rep_penalty: float = 1.0,
                     top_k: int = 1, temp: float = 1.0):
    """Greedy (top_k=1) or sampled continuation with optional repetition penalty.

    rep_penalty > 1 divides logits of previously-seen tokens by this factor
    (GPT-2-style). top_k=1 + rep_penalty=1.0 reproduces greedy behavior.
    """
    ids = list(prompt_ids)
    for _ in range(n_new):
        inp = torch.tensor([ids], device=device)
        T = inp.shape[1]
        cm = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device),
                         diagonal=1)
        h0 = text_embed(inp)
        ctx = arc.context_pool(h0)

        if phased_dynamics is None:
            arc.dynamics.set_context(ctx, mask=cm)
            arc.dynamics.set_n_steps(cfg.n_ode_steps)
            h = euler_solve(arc.dynamics, h0, t_span=(0.0, 2.0),
                             n_steps=cfg.n_ode_steps)
            if isinstance(h, tuple):
                h = h[0]
        else:
            n_phases = len(phased_dynamics)
            steps_per_phase = cfg.n_ode_steps // n_phases
            dt = 2.0 / cfg.n_ode_steps
            h = h0
            for phase_idx, dyn in enumerate(phased_dynamics):
                dyn.set_context(ctx, mask=cm)
                dyn.set_n_steps(cfg.n_ode_steps)
                for s_in in range(steps_per_phase):
                    gstep = phase_idx * steps_per_phase + s_in
                    if hasattr(dyn, '_current_step_index_buf'):
                        dyn._current_step_index_buf.fill_(gstep)
                    t_ode = torch.tensor(gstep * dt, device=device,
                                          dtype=h.dtype)
                    h = h + dt * dyn(t_ode, h)

        logits = text_head(h)[0, -1, :]  # [vocab]
        logits = logits / max(temp, 1e-6)
        if rep_penalty != 1.0:
            for prev in set(ids):
                logits[prev] = (logits[prev] / rep_penalty
                                 if logits[prev] > 0
                                 else logits[prev] * rep_penalty)
        if top_k <= 1:
            next_id = int(logits.argmax().item())
        else:
            topv, topi = torch.topk(logits, top_k)
            probs = F.softmax(topv, dim=-1)
            pick = torch.multinomial(probs, 1).item()
            next_id = int(topi[pick].item())
        ids.append(next_id)
    return ids


@torch.no_grad()
def teacher_continue(prompt_ids, teacher, device, n_new=12):
    ids = torch.tensor([prompt_ids], device=device)
    for _ in range(n_new):
        out = teacher(ids)
        next_id = int(out.logits[0, -1, :].argmax().item())
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return ids[0].tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--teacher", default="gpt2")
    p.add_argument("--n_new", type=int, default=12)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rep_penalty", type=float, default=1.5,
                   help="Divide logits of already-seen tokens by this factor")
    p.add_argument("--top_k", type=int, default=1,
                   help=">1 samples from top-k (temperature-weighted)")
    p.add_argument("--temp", type=float, default=1.0)
    args = p.parse_args()

    device = args.device
    print(f"Loading {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg: LiquidARCConfig = ckpt['config']
    cfg.tau_freeze_steps = 0

    # Rebuild phased dynamics if present in checkpoint
    phased_dynamics = None
    n_phases = ckpt.get('n_phases', 1)
    if n_phases > 1 and 'phased_dynamics_state_dict' in ckpt:
        from liquid_arc.dynamics import ContinuousDynamics
        print(f"  detected phased training: n_phases={n_phases}")

    arc = LiquidARCModel(cfg).to(device).eval()
    arc.load_state_dict(ckpt['model_state_dict'], strict=False)

    # Rebuild phased dynamics list if present
    if n_phases > 1 and 'phased_dynamics_state_dict' in ckpt:
        from liquid_arc.dynamics import ContinuousDynamics
        phased_dynamics = torch.nn.ModuleList([arc.dynamics])
        for _ in range(n_phases - 1):
            dyn = ContinuousDynamics(cfg).to(device).eval()
            phased_dynamics.append(dyn)
        phased_dynamics.load_state_dict(
            ckpt['phased_dynamics_state_dict'], strict=False)
        for dyn in phased_dynamics:
            dyn.eval()
        print(f"  loaded {n_phases} phased dynamics modules")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vocab = len(tok)
    text_embed = TextEmbedding(
        vocab_size=vocab, d_model=cfg.d_model, max_seq_len=512, dropout=0.0
    ).to(device).eval()
    text_head = TextHead(d_model=cfg.d_model, vocab_size=vocab).to(device).eval()
    text_embed.load_state_dict(ckpt['text_embed_state_dict'])
    text_head.load_state_dict(ckpt['text_head_state_dict'])

    print(f"Loading teacher {args.teacher}...")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher).to(device).eval()

    print("\n═══ Greedy continuations — student vs teacher ═══\n")
    for prompt in PROMPTS:
        print(f"PROMPT: {prompt!r}")
        ids = tok.encode(prompt)
        s_ids = student_continue(ids, arc, text_embed, text_head, cfg, device,
                                   n_new=args.n_new,
                                   phased_dynamics=phased_dynamics,
                                   rep_penalty=args.rep_penalty,
                                   top_k=args.top_k, temp=args.temp)
        t_ids = teacher_continue(ids, teacher, device, n_new=args.n_new)
        s_text = tok.decode(s_ids)
        t_text = tok.decode(t_ids)
        print(f"  STUDENT: {s_text!r}")
        print(f"  TEACHER: {t_text!r}")
        print()


if __name__ == "__main__":
    main()
