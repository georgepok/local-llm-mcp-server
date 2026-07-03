"""Closed-loop Liquid STEERING CONTROLLER — trained causal goal control.

The substrate stops PREDICTING goal-following and starts CAUSING it. At every
generated token it reads the frozen LLM's hidden state (layer L) + the goal,
maintains a commitment state via ContinuousDynamics, and emits a bounded steering
vector added to the residual stream. Trained by REINFORCE with the programmatic
check_fn as reward (LLM frozen). Breakthrough bar: steered pass-rate >> unsteered,
AND transfers to a held-out goal category (task-agnostic causal control).

Myopic credit assignment: past KV detached each step, so each token's steer is
updated by the global episode reward via its own log-prob term (bounded memory).
"""
import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import generate_diverse_goals as gdg
_LA = _HERE.parents[1]
sys.path.insert(0, str(_LA))
from liquid_arc.config import LiquidARCConfig          # type: ignore
from liquid_arc.dynamics import ContinuousDynamics      # type: ignore
from liquid_arc.context_pool import ContextPool         # type: ignore
from liquid_arc.solver import euler_solve_halting        # type: ignore

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def make_cfg(d, d_metric=16, d_ffn=128, n_ode_steps=3):
    return LiquidARCConfig(
        d_model=d, d_metric=d_metric, d_ffn=d_ffn, max_seq_len=8,
        n_ode_steps=n_ode_steps, ode_steps_min=max(2, n_ode_steps - 1),
        ode_steps_max=n_ode_steps + 1, integration_time=0.5,
        tau_min=0.3, tau_max=2.0, t_diffusion_init=0.5, routing_mode="metric",
        tau_freeze_steps=0, halting_enabled=True, halting_min_steps=1,
        halting_ponder_lambda=0.0001, rezero_enabled=True, rezero_gate_init=-3.0,
        metric_bias_init_std=0.1, deep_supervision_enabled=False,
        ponder_kl_lambda=0.0, criticality_loss_enabled=False,
        curvature_diversity_loss_enabled=True, curvature_diversity_lambda=0.0001,
    )


class SteerController(nn.Module):
    """Liquid commitment-tracker that emits a bounded residual steering vector."""
    def __init__(self, d_llm=1536, z_goal_dim=384, d=128, K=4, max_steer=20.0, use_slow=False,
                 n_inject=1, out_init_std=0.002):
        super().__init__()
        self.d_llm, self.d, self.K, self.max_steer = d_llm, d, K, max_steer
        self.use_slow = use_slow
        self.n_inject = n_inject   # ENFORCER: emit a steer for each of n_inject LLM layers
        self.out_init_std = out_init_std   # smaller -> untrained steer ~0 (calibrate per LLM scale)
        self.cfg = make_cfg(d)
        self.in_llm = nn.Linear(d_llm, d); nn.init.normal_(self.in_llm.weight, std=0.02)
        self.in_goal = nn.Linear(z_goal_dim, d); nn.init.normal_(self.in_goal.weight, std=0.02)
        self.ln = nn.LayerNorm(d)
        self.init_belief = nn.Parameter(torch.zeros(K, d)); nn.init.normal_(self.init_belief, std=0.05)
        self.evidence_mix = nn.Parameter(torch.ones(K, 1))
        self.context_pool = ContextPool(self.cfg)
        self.dynamics = ContinuousDynamics(self.cfg)
        self.out = nn.Linear(K * d, d_llm * n_inject)        # one steer vector per injected layer
        nn.init.normal_(self.out.weight, std=out_init_std); nn.init.zeros_(self.out.bias)  # start ~0 steer
        self.g_head = nn.Linear(K * d, z_goal_dim)           # belief -> waypoint-emb space (contrastive)
        self.h_c = None
        self.mag_sum = None; self.mag_n = 0   # per-episode mean steer-norm (intervention cost)
        if use_slow:
            # SLOW channel: persistent cross-TURN mission state + trigger that advances it at
            # goal/subgoal transitions. h_c (fast) resets each turn; h_slow persists across turns.
            self.slow_init = nn.Parameter(torch.zeros(K, d)); nn.init.normal_(self.slow_init, std=0.05)
            self.slow_in_goal = nn.Linear(z_goal_dim, d); nn.init.normal_(self.slow_in_goal.weight, std=0.02)
            self.slow_ln = nn.LayerNorm(d)
            self.slow_alpha = nn.Parameter(torch.tensor(0.05))
            self.trigger_boost = 8.0
            self.head_trigger = nn.Sequential(nn.Linear(z_goal_dim, 64), nn.SiLU(), nn.Linear(64, 1))
            with torch.no_grad():
                self.head_trigger[-1].weight.mul_(0.01); self.head_trigger[-1].bias.fill_(-2.0)
            self.slow_to_fast = nn.Linear(d, d); nn.init.normal_(self.slow_to_fast.weight, std=0.02)
            self.h_slow = None; self.z_goal_prev = None

    def reset(self, B, device):
        """Fast (within-turn) reset only — called at the start of each turn's generation."""
        self.h_c = self.init_belief.unsqueeze(0).expand(B, -1, -1).contiguous().to(device)
        self.mag_sum = None; self.mag_n = 0

    def reset_episode(self, B, device):
        """Mission start: reset BOTH fast and the persistent slow channel."""
        self.reset(B, device)
        if self.use_slow:
            h0 = self.slow_init.unsqueeze(0).expand(B, -1, -1).contiguous().to(device)
            self.h_slow = h0                 # used within turn (has grad to slow_init)
            self.h_slow_carry = h0.detach()  # detached recurrent carry across turns
            self.z_goal_prev = None

    def slow_step(self, z_goal):
        """Turn boundary: persist the mission state, JUMP when the trigger detects a goal
        transition. h_slow keeps grad WITHIN the turn (so trigger/slow projections learn);
        only the cross-turn carry is detached (bounds memory, no cross-turn grad chain)."""
        if not self.use_slow:
            return None
        zp = self.z_goal_prev if self.z_goal_prev is not None else z_goal
        trig = torch.sigmoid(self.head_trigger(z_goal - zp))                  # [B,1] transition?
        alpha = torch.clamp(self.slow_alpha + trig * self.slow_alpha * self.trigger_boost, max=0.5)
        inject = torch.tanh(self.slow_ln(self.slow_in_goal(z_goal))).unsqueeze(1)  # [B,1,d]
        self.h_slow = (1.0 - alpha).unsqueeze(-1) * self.h_slow_carry + alpha.unsqueeze(-1) * inject
        self.h_slow_carry = self.h_slow.detach()
        self.z_goal_prev = z_goal.detach()
        return float(trig.mean().item())

    def mag_penalty(self):
        if self.mag_sum is None:
            return None
        return self.mag_sum / max(1, self.mag_n)   # differentiable mean steer norm

    def belief_query(self, h):
        """Project the post-ODE belief state into waypoint-embedding space, L2-normalized.
        Used by the contrastive objective to pull h toward the current waypoint, away from others."""
        return F.normalize(self.g_head(h.flatten(1)), dim=-1)        # [B, z]

    def dyn_state(self, ref_emb, h_llm=None):
        """Per-TURN observe for the LoRA-hypernet use (Liquid GENERATES the adapter, not a steer).
        Update the belief state from a reference embedding (current waypoint), incorporating the
        slow-channel goal anchor. CLOSED LOOP: if h_llm (the frozen LLM's hidden on the drift prompt)
        is given, mix it in via in_llm so the adapter is conditioned on the model's ACTUAL drift state,
        not just the target. Returns the post-ODE state [B,K,d] WITH grad for this turn; the carried
        h_c is detached across turns (bounds memory, keeps within-turn grad)."""
        ein = self.in_goal(ref_emb)
        if h_llm is not None:
            ein = ein + self.in_llm(h_llm)                          # closed-loop drift observation
        e = self.ln(ein)                                            # [B, d]
        evidence = e.unsqueeze(1) * self.evidence_mix.unsqueeze(0)  # [B, K, d]
        h_in = self.h_c + evidence
        if self.use_slow and self.h_slow is not None:
            h_in = h_in + self.slow_to_fast(self.h_slow)            # goal anchor (slow) modulates
        ctx = self.context_pool(h_in, None)
        self.dynamics.set_context(ctx, mask=None)
        self.dynamics.set_n_steps(int(self.cfg.n_ode_steps))
        out = euler_solve_halting(self.dynamics, h_in, (0.0, float(self.cfg.integration_time)),
                                    int(self.cfg.n_ode_steps), min_steps=self.cfg.halting_min_steps)
        h_out = out[0] if isinstance(out, tuple) else out          # [B, K, d]
        self.h_c = h_out.detach()
        return h_out

    def step(self, h_llm, z_goal):
        # h_llm [B, d_llm] (detached observation), z_goal [B, z]
        e = self.ln(self.in_llm(h_llm) + self.in_goal(z_goal))      # [B, d]
        evidence = e.unsqueeze(1) * self.evidence_mix.unsqueeze(0)  # [B, K, d]
        h_in = self.h_c + evidence
        if self.use_slow and self.h_slow is not None:
            h_in = h_in + self.slow_to_fast(self.h_slow)            # mission state modulates fast dynamics
        ctx = self.context_pool(h_in, None)
        self.dynamics.set_context(ctx, mask=None)
        self.dynamics.set_n_steps(int(self.cfg.n_ode_steps))
        out = euler_solve_halting(self.dynamics, h_in, (0.0, float(self.cfg.integration_time)),
                                    int(self.cfg.n_ode_steps), min_steps=self.cfg.halting_min_steps)
        h_out = out[0] if isinstance(out, tuple) else out          # [B, K, d]
        self.h_c = h_out.detach()                                   # carry value, not cross-step grad
        raw = self.out(h_out.flatten(1)).view(-1, self.n_inject, self.d_llm)   # [B, n_inject, d_llm]
        nrm = raw.norm(dim=-1, keepdim=True)
        scale = (self.max_steer * torch.tanh(nrm / self.max_steer)) / (nrm + 1e-6)
        steer = raw * scale                                         # per-layer bounded |steer|<=max_steer
        sn = steer.norm(dim=-1).mean()                             # mean over layers + batch
        self.mag_sum = sn if self.mag_sum is None else self.mag_sum + sn
        self.mag_n += 1
        return steer                                               # [B, n_inject, d_llm]


def _apply_steer(h, steer, rel_frac):
    """Add steer to the last position. If rel_frac set, cap |steer| to a fraction of the LOCAL
    residual norm (auto-calibrates magnitude across models/layers — needed for large/sensitive LLMs)."""
    s = steer.to(h.dtype)
    if rel_frac is not None:
        ln = h[:, -1, :].norm(dim=-1, keepdim=True)
        sn = s.norm(dim=-1, keepdim=True) + 1e-6
        s = s * torch.clamp(rel_frac * ln / sn, max=1.0)
    h = h.clone(); h[:, -1, :] = h[:, -1, :] + s
    return h


class Hook:
    """Single-layer steering hook (back-compat: controller.step now returns [B,n_inject,d_llm])."""
    def __init__(self, controller, rel_frac=None):
        self.c = controller; self.z = None; self.active = False; self.rel_frac = rel_frac
    def __call__(self, module, inputs, output):
        is_t = isinstance(output, tuple)
        h = output[0] if is_t else output
        if self.active and self.z is not None:
            steer = self.c.step(h[:, -1, :].detach().float(), self.z)[:, 0, :]  # first slice
            h = _apply_steer(h, steer, self.rel_frac)
            return ((h,) + tuple(output[1:])) if is_t else h
        return output


class MultiHook:
    """ENFORCER: influence the LLM state at MULTIPLE layers each token. The controller reads the
    lowest injected layer's hidden state, emits a steer per layer; each layer's hook adds its slice.
    register() must be given the model; layers are 1-based hidden_states indices (output of
    model.model.layers[L-1]), matching the rest of the code (layer_idx convention)."""
    def __init__(self, controller, inject_layers, rel_frac=None):
        self.c = controller
        self.layers = sorted(inject_layers)          # 1-based (hidden_states index)
        assert len(self.layers) == controller.n_inject, "n_inject must match #inject_layers"
        self.z = None; self.active = False; self.steers = None; self.handles = []; self.rel_frac = rel_frac

    def _hook(self, slot):
        def fn(module, inputs, output):
            is_t = isinstance(output, tuple); h = output[0] if is_t else output
            if not (self.active and self.z is not None):
                return output
            if slot == 0:                            # lowest layer = READ: compute all steers
                self.steers = self.c.step(h[:, -1, :].detach().float(), self.z)  # [B,n_inject,d_llm]
            h = _apply_steer(h, self.steers[:, slot, :], self.rel_frac)
            return ((h,) + tuple(output[1:])) if is_t else h
        return fn

    def register(self, model):
        for slot, L in enumerate(self.layers):
            self.handles.append(model.model.layers[L - 1].register_forward_hook(self._hook(slot)))


def detach_past(past):
    if past is None:
        return None
    if hasattr(past, "key_cache"):
        past.key_cache = [k.detach() for k in past.key_cache]
        past.value_cache = [v.detach() for v in past.value_cache]
        return past
    return tuple(tuple(t.detach() for t in layer) for layer in past)


@torch.no_grad()
def encode_goal(instr, enc_tok, enc_model, device):
    t = enc_tok(instr, return_tensors="pt", truncation=True, max_length=64).to(device)
    v = enc_model(**t).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(v, dim=-1).squeeze(0)     # [384]


def generate(model, tok, instr, z_goal, hook, max_new, temperature, steer, grad):
    msgs = [{"role": "user", "content": instr}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(chat, return_tensors="pt").to(model.device)
    ids, attn = enc.input_ids, enc.attention_mask
    if steer:
        hook.c.reset(1, model.device)
        hook.z = z_goal.unsqueeze(0); hook.active = True
    else:
        hook.active = False
    logps, out_ids = [], []
    past, cur, cur_attn = None, ids, attn
    ctx = torch.enable_grad() if (grad and steer) else torch.no_grad()
    with ctx:
        for _ in range(max_new):
            o = model(cur, attention_mask=cur_attn, past_key_values=past, use_cache=True)
            logits = o.logits[:, -1]
            past = o.past_key_values  # keep attached: full-episode credit (B=1 graph is small)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                tk = torch.multinomial(probs, 1)
            else:
                tk = logits.argmax(dim=-1, keepdim=True)
            if grad and steer:
                logps.append(torch.log_softmax(logits, dim=-1).gather(1, tk).squeeze())
            tid = int(tk.item())
            out_ids.append(tid)
            if tid == tok.eos_token_id:
                break
            cur = tk
            cur_attn = torch.cat([cur_attn, torch.ones((1, 1), dtype=cur_attn.dtype, device=model.device)], 1)
    hook.active = False
    text = tok.decode(out_ids, skip_special_tokens=True)
    logp_sum = torch.stack(logps).sum() if logps else None
    n_new = len([t for t in out_ids if t != tok.eos_token_id])
    return text, logp_sum, n_new


@torch.no_grad()
def base_fluency(model, tok, instr, text):
    """Mean per-token logprob of `text` under the UNSTEERED LLM (hook must be inactive).
    Degenerate / gamed text scores much lower than coherent text."""
    if not text.strip():
        return -10.0
    msgs = [{"role": "user", "content": instr}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pids = tok(chat, return_tensors="pt").input_ids.to(model.device)
    cids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    if cids.shape[1] == 0:
        return -10.0
    full = torch.cat([pids, cids], 1)
    out = model(full)
    logp = torch.log_softmax(out.logits[0, :-1], -1)
    tgt = full[0, 1:]
    sel = logp[torch.arange(len(tgt), device=full.device), tgt]
    return float(sel[pids.shape[1] - 1:].mean().item())


def eval_passrate(model, tok, controller, hook, enc_tok, enc_model, cats, n, rng, args, steer):
    """Returns (check_rate, coherent_rate, mean_fluency). coherent = check AND
    len>=min_len AND fluency>thresh — the HONEST goal-following metric (ungameable)."""
    controller.eval()
    chk, coh, flus = 0, 0, []
    tot = 0
    for cat in cats:
        for _ in range(n):
            _, _, instr, check, _ = gdg.make_goal(rng, category=cat)
            z = encode_goal(instr, enc_tok, enc_model, model.device)
            txt, _, n_new = generate(model, tok, instr, z, hook, args.max_new_tokens, 0.0, steer, grad=False)
            ok = bool(check(txt))
            flu = base_fluency(model, tok, instr, txt)
            flus.append(flu)
            chk += int(ok)
            coh += int(ok and n_new >= args.min_len and flu > args.flu_thresh)
            tot += 1
    controller.train()
    return chk / tot, coh / tot, float(np.mean(flus))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--layer_idx", type=int, default=14)
    p.add_argument("--train_cats", default="surface,format,length,inclusion,punct,structure,repetition,structure_alt")
    p.add_argument("--heldout_cat", default="contrast")
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--max_steer", type=float, default=20.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--group", type=int, default=8, help="goals per optimizer step")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_n", type=int, default=20)
    p.add_argument("--min_len", type=int, default=20, help="min new tokens (eval coherent + reward task gate)")
    p.add_argument("--flu_thresh", type=float, default=-1.3, help="min base-LLM fluency for eval coherent metric")
    p.add_argument("--lambda_flu", type=float, default=0.5, help="dense fluency-below-ref penalty weight")
    p.add_argument("--ref_flu", type=float, default=-0.8, help="fluency reference; penalize only below this")
    p.add_argument("--beta_mag", type=float, default=0.03, help="minimal-intervention: penalty on mean steer norm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="/home/pokazge/checkpoints/steer_ctrl.pt")
    args = p.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    # sm121 FMHA kernels are broken (silent NaN) — force math SDPA backend (memory)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.set_float32_matmul_precision("high")
    print(f"[steer] loading {args.gen_model} (frozen) + {args.enc_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.gen_model, dtype=torch.float16,
                                                   trust_remote_code=True).to(device).eval()
    for pp in model.parameters():
        pp.requires_grad = False
    enc_tok = AutoTokenizer.from_pretrained(args.enc_model)
    enc_model = AutoModel.from_pretrained(args.enc_model).to(device).eval()

    controller = SteerController(d_llm=model.config.hidden_size, d=args.d, K=args.K,
                                   max_steer=args.max_steer).to(device)
    # controller runs in fp32; steer cast to model dtype inside hook addition
    n_params = sum(pp.numel() for pp in controller.parameters())
    print(f"[steer] controller {n_params:,} params, max_steer={args.max_steer}", flush=True)
    hook = Hook(controller)
    handle = model.model.layers[args.layer_idx - 1].register_forward_hook(hook)

    opt = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=0.0)
    train_cats = [c.strip() for c in args.train_cats.split(",")]
    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 9999)
    baseline = defaultdict(lambda: deque(maxlen=64))  # reward baseline per category

    # initial eval
    def run_eval(tag):
        bc_tr, bh_tr, bf_tr = eval_passrate(model, tok, controller, hook, enc_tok, enc_model, train_cats[:3], args.eval_n, np.random.default_rng(123), args, steer=False)
        sc_tr, sh_tr, sf_tr = eval_passrate(model, tok, controller, hook, enc_tok, enc_model, train_cats[:3], args.eval_n, np.random.default_rng(123), args, steer=True)
        bc_ho, bh_ho, bf_ho = eval_passrate(model, tok, controller, hook, enc_tok, enc_model, [args.heldout_cat], args.eval_n, np.random.default_rng(456), args, steer=False)
        sc_ho, sh_ho, sf_ho = eval_passrate(model, tok, controller, hook, enc_tok, enc_model, [args.heldout_cat], args.eval_n, np.random.default_rng(456), args, steer=True)
        print(f"[eval {tag}] TRAIN coherent base={bh_tr:.2f}->steer={sh_tr:.2f} (check {bc_tr:.2f}->{sc_tr:.2f}, flu {bf_tr:.2f}->{sf_tr:.2f})  "
              f"HELD-OUT[{args.heldout_cat}] coherent base={bh_ho:.2f}->steer={sh_ho:.2f} (check {bc_ho:.2f}->{sc_ho:.2f}, flu {bf_ho:.2f}->{sf_ho:.2f})", flush=True)
        return sh_ho  # checkpoint on held-out COHERENT rate (the honest metric)

    best = run_eval("init")
    csum = 0.0; cn = 0
    for step in range(1, args.max_steps + 1):
        opt.zero_grad()
        batch_loss = 0.0; rsum = 0.0
        for _ in range(args.group):
            cat = train_cats[rng.integers(len(train_cats))]
            _, _, instr, check, _ = gdg.make_goal(rng, category=cat)
            z = encode_goal(instr, enc_tok, enc_model, device)
            text, logp, n_new = generate(model, tok, instr, z, hook, args.max_new_tokens,
                                           args.temperature, steer=True, grad=True)
            if logp is None:
                continue
            # DENSE coherence-constrained reward (anti-Goodhart):
            #   task   = check AND not-truncated (length gate kills brevity/jamming)
            #   - fluency PENALTY (dense gradient): only penalize BELOW ref, never reward
            #     high fluency (else short common phrases score well). Kills repetition/degeneracy.
            flu = base_fluency(model, tok, instr, text)
            R_task = 1.0 if (bool(check(text)) and n_new >= args.min_len) else 0.0
            R = R_task - args.lambda_flu * max(0.0, args.ref_flu - flu)
            b = np.mean(baseline[cat]) if baseline[cat] else 0.0
            baseline[cat].append(R)
            adv = R - b
            LOSS_SCALE = 256.0  # fp16 activation-grad underflow guard (unscaled before step)
            # MINIMAL-INTERVENTION anchor: direct L2-ish penalty on mean steer norm so
            # "behave like the base LLM" is the floor; controller steers only when it pays.
            mag = controller.mag_penalty()
            mag_term = args.beta_mag * mag if mag is not None else 0.0
            ((-adv * logp + mag_term) / args.group * LOSS_SCALE).backward()
            batch_loss += float(-adv * logp.detach()); rsum += R_task
        for pp in controller.parameters():
            if pp.grad is not None:
                pp.grad /= 256.0
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        opt.step()
        csum += rsum / args.group; cn += 1
        if step % 10 == 0:
            print(f"step {step:>4}  train_reward(roll)={csum/cn:.3f}  loss={batch_loss:.3f}", flush=True)
            csum = 0.0; cn = 0
        if step % args.eval_every == 0:
            mho = run_eval(f"s{step}")
            if mho > best:
                best = mho
                torch.save({"controller": controller.state_dict(), "args": vars(args),
                             "best_heldout_steered": best}, args.output)
                print(f"[steer] saved (held-out steered {best:.3f}) -> {args.output}", flush=True)
    handle.remove()
    print(f"[steer] DONE best held-out steered={best:.3f}", flush=True)
    print("[steer] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
