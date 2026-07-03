"""Liquid-LoRA: the Liquid GENERATES the adapter (weight-level), not an activation steer.

Why: a residual steer perturbs activations OFF-distribution and disrupts a strong reasoner. A
LoRA changes the model's FUNCTION in-distribution (W + BA). Making the Liquid produce the LoRA,
conditioned on its dynamical goal/trajectory state, gives a DYNAMIC, trajectory-aware weight
modulation — it re-programs the LLM for the current waypoint/goal and evolves per turn. Unifies
latent-oracle (HyperNet -> weight deltas) with liquid-arc (dynamical goal tracker).

Tractability: don't generate raw LoRA params. Learn a small BANK of LoRA primitives {(B_k,A_k)};
the Liquid emits mixing COEFFICIENTS over them (the dynamical state navigates a low-dim LoRA
manifold). Per TURN: Liquid observes (goal anchor + current waypoint) -> state -> coeffs -> LoRA
-> LLM generates the turn re-programmed. Frozen LLM; train basis + coeff-head + Liquid.
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_steer_controller import SteerController, encode_goal, base_fluency
from train_steer_traj import elicit_plan, split_answer, build_plans
from train_steer_commit import COMMIT_GOALS, TANGENTS
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class DynLoRA(nn.Module):
    """A bank of K LoRA primitives per target projection; the Liquid state picks the mix (coeffs).
    LoRA delta for input x at target t = scale * sum_k coeff[t,k] * (x A_{t,k}^T) B_{t,k}^T."""
    def __init__(self, model, layers, proj, ctrl_state_dim, r=8, K=8, scale=1.0):
        super().__init__()
        self.layers = list(layers); self.proj = proj; self.r = r; self.K = K; self.scale = scale
        self.mods = [getattr(model.model.layers[L - 1].self_attn, proj) for L in self.layers]
        self.A, self.B = nn.ParameterList(), nn.ParameterList()
        for m in self.mods:
            din, dout = m.in_features, m.out_features
            a = nn.Parameter(torch.randn(K, r, din) * (1.0 / din ** 0.5))
            b = nn.Parameter(torch.zeros(K, dout, r))           # B=0 -> LoRA starts at 0 (LLM unchanged)
            self.A.append(a); self.B.append(b)
        self.coeff_head = nn.Sequential(nn.Linear(ctrl_state_dim, 128), nn.SiLU(), nn.Linear(128, len(self.mods) * K))
        nn.init.zeros_(self.coeff_head[-1].bias)
        self.coeffs = None; self.active = False; self.handles = []
        self.debug = False; self.last_rel = [0.0] * len(self.mods)   # verification: |delta|/|out| per target

    def set_state(self, h):
        """h: Liquid belief state [B,Kbelief,d]. Compute per-target mixing coeffs for this turn."""
        c = self.coeff_head(h.flatten(1))                       # [B, n_targets*K]
        self.coeffs = c.view(len(self.mods), self.K)            # B=1

    def _hook(self, ti):
        def fn(module, inp, out):
            if not self.active or self.coeffs is None:
                return out
            x = inp[0].float()                                  # [.., din]
            A, B, c = self.A[ti], self.B[ti], self.coeffs[ti]   # [K,r,din],[K,dout,r],[K]
            xa = torch.einsum("...i,kri->...kr", x, A)          # [.., K, r]
            xa = xa * c.view(*([1] * (xa.dim() - 2)), self.K, 1)
            delta = torch.einsum("...kr,kor->...o", xa, B)      # [.., dout]
            d = self.scale * delta
            if self.debug:                                      # capture |delta|/|out| for verification
                self.last_rel[ti] = float((d.norm(dim=-1) / (out.float().norm(dim=-1) + 1e-6)).mean().item())
            return out + d.to(out.dtype)
        return fn

    def register(self):
        for ti, m in enumerate(self.mods):
            self.handles.append(m.register_forward_hook(self._hook(ti)))

    def lora_l2(self):
        """Mean |delta-coeff|: regularize how hard the LoRA is pushed (kept small near base)."""
        if self.coeffs is None:
            return torch.tensor(0.0)
        return self.coeffs.pow(2).mean()


def generate_lora(model, tok, messages, max_new, temperature, grad, think=None):
    """Generate a turn with the dynamic LoRA active (coeffs preset by caller). Accumulates logp
    for REINFORCE. No activation steering — modulation is entirely the LoRA weight delta."""
    try:
        chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **({} if think is None else {"enable_thinking": think}))
    except TypeError:
        chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(chat, return_tensors="pt").to(model.device)
    ids, attn = enc.input_ids, enc.attention_mask
    logps, out_ids = [], []
    past, cur, cur_attn = None, ids, attn
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for _ in range(max_new):
            o = model(cur, attention_mask=cur_attn, past_key_values=past, use_cache=True)
            logits = o.logits[:, -1]; past = o.past_key_values
            tk = torch.multinomial(torch.softmax(logits / temperature, -1), 1) if temperature > 0 \
                else logits.argmax(-1, keepdim=True)
            if grad:
                logps.append(torch.log_softmax(logits, -1).gather(1, tk).squeeze())
            tid = int(tk.item()); out_ids.append(tid)
            if tid == tok.eos_token_id:
                break
            cur = tk
            cur_attn = torch.cat([cur_attn, torch.ones((1, 1), dtype=cur_attn.dtype, device=model.device)], 1)
    text = tok.decode(out_ids, skip_special_tokens=True)
    logp_sum = torch.stack(logps).sum() if logps else None
    n_new = len([t for t in out_ids if t != tok.eos_token_id])
    return text, logp_sum, n_new


def run_traj_lora(model, tok, controller, lora, enc_tok, enc_model, goal, way_emb, zG, device, args, grad):
    """Walk the conversation; per turn the Liquid observes the current waypoint (with the goal
    anchored in the slow channel), produces the turn's LoRA, the LLM generates re-programmed."""
    controller.reset_episode(1, device)
    if controller.use_slow:
        controller.slow_step(zG.unsqueeze(0))                   # anchor slow channel to the GOAL
    lora.active = True
    rng = np.random.default_rng()
    n_way = len(way_emb); p = 0
    intro = f"I want to {goal}. Let's work through it together, step by step. What should we do, and let's begin."
    turns = [intro]
    td = list(rng.permutation(len(TANGENTS)))
    for k in range(args.n_turns - 1):
        turns.append("Okay, continue." if k % 2 == 0 else TANGENTS[td[k % len(td)]])
    think = None if args.think else False
    logps, flus, n_news, mags = [], [], [], []
    messages = []
    for u in turns:
        messages.append({"role": "user", "content": u})
        wp = way_emb[min(p, n_way - 1)]
        h = controller.dyn_state(wp.unsqueeze(0))               # Liquid state for this turn (with grad)
        lora.set_state(h)                                       # -> LoRA for this turn
        text, logp, n_new = generate_lora(model, tok, messages, args.max_new_tokens,
                                             args.temperature if grad else 0.0, grad, think=think)
        messages.append({"role": "assistant", "content": text})
        zr = encode_goal(split_answer(text), enc_tok, enc_model, device)
        if p < n_way and float((zr * way_emb[p]).sum().item()) > args.advance_thr:
            p += 1
        flus.append(base_fluency(model, tok, u, text)); n_news.append(n_new)
        if grad:
            mags.append(lora.lora_l2())
            if logp is not None:
                logps.append(logp)
    lora.active = False
    mag = torch.stack(mags).mean() if mags else None
    return p / n_way, logps, float(np.mean(flus)), (min(n_news) if n_news else 0), mag


def eval_lora(model, tok, controller, lora, enc_tok, enc_model, plans, goals, n, rng, device, args, on):
    controller.eval(); lora.eval()
    progs, flus = [], []
    for _ in range(n):
        g = goals[rng.integers(len(goals))]
        if g not in plans:
            continue
        _, wemb = plans[g]; zG = encode_goal(g, enc_tok, enc_model, device)
        if not on:
            lora.active = False
            prog, flu = _base_progress(model, tok, enc_tok, enc_model, g, wemb, device, args)
        else:
            prog, _, flu, _, _ = run_traj_lora(model, tok, controller, lora, enc_tok, enc_model, g, wemb, zG, device, args, grad=False)
        progs.append(prog); flus.append(flu)
    controller.train(); lora.train()
    return float(np.mean(progs)) if progs else 0.0, float(np.mean(flus)) if flus else 0.0


@torch.no_grad()
def _base_progress(model, tok, enc_tok, enc_model, goal, way_emb, device, args):
    rng = np.random.default_rng(); n_way = len(way_emb); p = 0
    intro = f"I want to {goal}. Let's work through it together, step by step. What should we do, and let's begin."
    turns = [intro]; td = list(rng.permutation(len(TANGENTS)))
    for k in range(args.n_turns - 1):
        turns.append("Okay, continue." if k % 2 == 0 else TANGENTS[td[k % len(td)]])
    think = None if args.think else False
    messages, flus = [], []
    for u in turns:
        messages.append({"role": "user", "content": u})
        text, _, _ = generate_lora(model, tok, messages, args.max_new_tokens, 0.0, False, think=think)
        messages.append({"role": "assistant", "content": text})
        zr = encode_goal(split_answer(text), enc_tok, enc_model, device)
        if p < n_way and float((zr * way_emb[p]).sum().item()) > args.advance_thr:
            p += 1
        flus.append(base_fluency(model, tok, u, text))
    return p / n_way, float(np.mean(flus))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--gen_dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--d", type=int, default=128); p.add_argument("--K", type=int, default=4)
    p.add_argument("--use_slow", action="store_true")
    p.add_argument("--lora_layers", default="8,14,20,26"); p.add_argument("--lora_proj", default="o_proj")
    p.add_argument("--lora_r", type=int, default=8); p.add_argument("--lora_K", type=int, default=8)
    p.add_argument("--lora_scale", type=float, default=1.0); p.add_argument("--beta_lora", type=float, default=0.01)
    p.add_argument("--n_steps", type=int, default=4); p.add_argument("--n_turns", type=int, default=4)
    p.add_argument("--advance_thr", type=float, default=0.62); p.add_argument("--think", action="store_true")
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--max_steps", type=int, default=80)
    p.add_argument("--group", type=int, default=2); p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int, default=50); p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--eval_n", type=int, default=12); p.add_argument("--min_len", type=int, default=8)
    p.add_argument("--lambda_flu", type=float, default=1.0); p.add_argument("--ref_flu", type=float, default=-1.3)
    p.add_argument("--n_test_goals", type=int, default=3); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="/home/pokazge/checkpoints/liquid_lora.pt")
    args = p.parse_args()

    device = torch.device("cuda"); torch.manual_seed(args.seed)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.set_float32_matmul_precision("high")
    tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    gdtype = torch.bfloat16 if args.gen_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.gen_model, dtype=gdtype, trust_remote_code=True,
                                                   low_cpu_mem_usage=True, device_map={"": 0}).eval()
    for pp in model.parameters():
        pp.requires_grad = False
    enc_tok = AutoTokenizer.from_pretrained(args.enc_model)
    enc_model = AutoModel.from_pretrained(args.enc_model).to(device).eval()

    controller = SteerController(d_llm=model.config.hidden_size, d=args.d, K=args.K,
                                   use_slow=args.use_slow, n_inject=1).to(device)
    layers = [int(x) for x in args.lora_layers.split(",")]
    lora = DynLoRA(model, layers, args.lora_proj, ctrl_state_dim=args.K * args.d,
                     r=args.lora_r, K=args.lora_K, scale=args.lora_scale).to(device)
    lora.register()
    n_par = sum(pp.numel() for pp in controller.parameters()) + sum(pp.numel() for pp in lora.parameters())
    print(f"[llora] trainable {n_par:,} (controller+LoRA-basis+coeff-head)  layers={layers} proj={args.lora_proj} "
          f"r={args.lora_r} K={args.lora_K}", flush=True)
    opt = torch.optim.AdamW(list(controller.parameters()) + list(lora.parameters()), lr=args.lr, weight_decay=0.0)

    train_goals = COMMIT_GOALS[:-args.n_test_goals]; test_goals = COMMIT_GOALS[-args.n_test_goals:]
    print("[llora] eliciting plans (the LLM is the planner)...", flush=True)
    plans = build_plans(model, tok, enc_tok, enc_model, COMMIT_GOALS, args.n_steps, device)
    rng = np.random.default_rng(args.seed); baseline = deque(maxlen=64)

    def run_eval(tag):
        pb, fb = eval_lora(model, tok, controller, lora, enc_tok, enc_model, plans, test_goals, args.eval_n,
                             np.random.default_rng(456), device, args, on=False)
        pl, fl = eval_lora(model, tok, controller, lora, enc_tok, enc_model, plans, test_goals, args.eval_n,
                             np.random.default_rng(456), device, args, on=True)
        print(f"[eval {tag}] HELD-OUT plan-PROGRESS  base={pb:.3f}  LIQUID-LoRA={pl:.3f}  "
              f"(flu base={fb:.2f} lora={fl:.2f})", flush=True)
        return pl

    best = run_eval("init"); csum = 0.0; cn = 0
    for step in range(1, args.max_steps + 1):
        opt.zero_grad(); rsum = 0.0
        for _ in range(args.group):
            g = train_goals[rng.integers(len(train_goals))]
            if g not in plans:
                continue
            _, wemb = plans[g]; zG = encode_goal(g, enc_tok, enc_model, device)
            prog, logps, flu, min_n, mag = run_traj_lora(model, tok, controller, lora, enc_tok, enc_model,
                                                           g, wemb, zG, device, args, grad=True)
            if not logps:
                continue
            len_ok = 1.0 if min_n >= args.min_len else 0.0
            R = prog * len_ok - args.lambda_flu * max(0.0, args.ref_flu - flu)
            b = np.mean(baseline) if baseline else 0.0; baseline.append(R); adv = R - b
            total_logp = torch.stack(logps).sum()
            mag_term = args.beta_lora * mag if mag is not None else 0.0
            ((-adv * total_logp + mag_term) / args.group * 256.0).backward()
            rsum += prog
        for pp in list(controller.parameters()) + list(lora.parameters()):
            if pp.grad is not None:
                pp.grad /= 256.0
        torch.nn.utils.clip_grad_norm_(list(controller.parameters()) + list(lora.parameters()), 1.0)
        opt.step(); csum += rsum / args.group; cn += 1
        if step % 10 == 0:
            print(f"step {step:>4}  train_progress(roll)={csum/cn:.3f}", flush=True); csum = 0.0; cn = 0
        if step % args.eval_every == 0:
            mc = run_eval(f"s{step}")
            if mc > best:
                best = mc
                torch.save({"controller": controller.state_dict(), "lora": lora.state_dict(),
                             "args": vars(args), "best": best}, args.output)
                print(f"[llora] saved (progress {best:.3f}) -> {args.output}", flush=True)
    print(f"[llora] DONE best={best:.3f}", flush=True); print("[llora] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
