"""Liquid-LoRA v2: the LoRA adapter IS represented by the Liquid (no fixed basis), trained by
DISTILLATION (no REINFORCE).

Representation: the Liquid's belief state h in [K, d] — its K slots ARE the rank-K adapter.
Per target layer: A = proj_A(h) in [K, d_in], B = proj_B(h) in [K, d_out] (proj_B init ~0 so the
adapter starts at identity); LoRA delta = (x A^T) B. The adapter is a full function of the
continuous Liquid state and evolves through the ODE as the trajectory advances. proj_A/proj_B are
small shared linears -> tractable, fully expressive (not a K-dim basis cone).

Training: DISTILLATION. Teacher = the LLM WITH the current waypoint made explicit (it answers
on-plan on its own). Student = the LLM + Liquid-LoRA given only the DRIFTING context. Loss =
token CE of the student toward the teacher's response -> the adapter learns to make the model act
on the current waypoint even when the conversation pulls away. Dense, low-variance, generalizes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_steer_controller import SteerController, encode_goal
from train_steer_traj import elicit_plan, split_answer
from train_steer_commit import COMMIT_GOALS, TANGENTS
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class LiquidLoRA(nn.Module):
    """The Liquid's belief state generates the rank-K LoRA factors per target layer."""
    def __init__(self, model, layers, projs, d_ctrl, scale=1.0):
        super().__init__()
        self.layers = list(layers); self.scale = scale
        attn = {"q_proj", "k_proj", "v_proj", "o_proj"}
        # Each target = a GROUP of modules that SHARE one Liquid-generated (A,B). Attention/dense-MLP
        # projs are singleton groups; a MoE proj (e.g. down_proj) is the whole expert bank sharing one
        # adapter -> the Liquid writes the same goal-factors into whichever experts fire this token.
        self.target_mods = []                                  # list[list[nn.Linear]]
        dims = []                                              # (in_features, out_features) per target
        for L in self.layers:
            layer = model.model.layers[L - 1]
            for pr in projs:
                if pr in attn:
                    m = getattr(layer.self_attn, pr)
                    self.target_mods.append([m]); dims.append((m.in_features, m.out_features))
                    continue
                mlp = layer.mlp
                if hasattr(mlp, pr):                            # dense MLP (e.g. Qwen2.5-1.5B)
                    m = getattr(mlp, pr)
                    self.target_mods.append([m]); dims.append((m.in_features, m.out_features))
                elif hasattr(mlp, "experts"):                  # MoE: shared factors across all experts
                    mods = [getattr(e, pr) for e in mlp.experts if hasattr(e, pr)]
                    di = {(e.in_features, e.out_features) for e in mods}
                    assert len(di) == 1, f"layer {L} {pr}: experts have mixed dims {di}"
                    rep = mods[0]
                    self.target_mods.append(mods); dims.append((rep.in_features, rep.out_features))
                else:
                    raise ValueError(f"layer {L}: cannot find proj '{pr}' (no dense attr, no .experts)")
        self.n_targets = len(dims)
        self.proj_A, self.proj_B = nn.ModuleList(), nn.ModuleList()
        for (din, dout) in dims:
            pa = nn.Linear(d_ctrl, din); nn.init.normal_(pa.weight, std=0.02); nn.init.zeros_(pa.bias)
            pb = nn.Linear(d_ctrl, dout); nn.init.zeros_(pb.weight); nn.init.zeros_(pb.bias)  # B~0 -> LoRA=0
            self.proj_A.append(pa); self.proj_B.append(pb)
        self.AB = [None] * self.n_targets; self.active = False; self.handles = []; self.debug = False
        self.last_rel = [0.0] * self.n_targets; self.cap_rel = None   # cap |delta|/|out| for stability

    def set_state(self, h):
        """h: Liquid belief state [1,K,d_ctrl]. Generate (A,B) for every target this turn."""
        for ti in range(self.n_targets):
            A = self.proj_A[ti](h).squeeze(0)                   # [K, d_in]
            B = self.proj_B[ti](h).squeeze(0)                   # [K, d_out]
            self.AB[ti] = (A, B)

    def _hook(self, ti):
        def fn(module, inp, out):
            if not self.active or self.AB[ti] is None:
                return out
            A, B = self.AB[ti]
            x = inp[0].float()                                  # [.., d_in]
            xa = torch.einsum("...i,ki->...k", x, A)            # [.., K]
            delta = torch.einsum("...k,ko->...o", xa, B)        # [.., d_out]
            d = self.scale * delta
            if self.cap_rel is not None:                        # cap |delta| to a fraction of |out| (stability)
                on = out.float().norm(dim=-1, keepdim=True)
                dn = d.norm(dim=-1, keepdim=True) + 1e-6
                d = d * torch.clamp(self.cap_rel * on / dn, max=1.0)
            if self.debug:
                self.last_rel[ti] = float((d.norm(dim=-1) / (out.float().norm(dim=-1) + 1e-6)).mean().item())
            return out + d.to(out.dtype)
        return fn

    def register(self):
        for ti, mods in enumerate(self.target_mods):
            for m in mods:                                      # every module in the group shares _hook(ti)
                self.handles.append(m.register_forward_hook(self._hook(ti)))


def _template(tok, messages, think):
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **({} if think is None else {"enable_thinking": think}))
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def gen_plain(model, tok, messages, max_new, think, lora):
    prev = lora.active; lora.active = False                    # teacher: NO adapter
    chat = _template(tok, messages, think)
    enc = tok(chat, return_tensors="pt").to(model.device)
    out = model.generate(enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=max_new,
                           do_sample=False, pad_token_id=tok.pad_token_id)
    lora.active = prev
    return split_answer(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))


def episode(model, tok, controller, lora, enc_tok, enc_model, goal, way_txt, way_emb, zG, device, args, train):
    """Distillation episode. Each turn: teacher (explicit waypoint) -> target; student (drift +
    Liquid-LoRA) teacher-forced -> CE. Returns (mean_ce, on_plan_cos) ; trains if train=True."""
    controller.reset_episode(1, device)
    if controller.use_slow:
        controller.slow_step(zG.unsqueeze(0))
    rng = np.random.default_rng()
    td = list(rng.permutation(len(TANGENTS)))
    history = [{"role": "user", "content": f"Help me with this task: {goal}. We'll go step by step."}]
    n_way = len(way_emb); ces, closs, coss, rloss = [], [], [], []
    think = None if args.think else False
    for p in range(min(args.n_turns, n_way)):
        wp_t = way_txt[p]
        # student's user turn = a drift pull (tangent), NOT the waypoint
        drift = "Okay, what's next?" if p % 2 == 0 else TANGENTS[td[p % len(td)]]
        # TEACHER: same history but the step is made explicit -> on-plan target (no adapter)
        t_msgs = history + [{"role": "user", "content": f"{drift}\n\nSet that aside and refocus on our task. The single next step is: {wp_t}. Do exactly that step now — concretely and directly, no preamble."}]
        target = gen_plain(model, tok, t_msgs, args.max_new_tokens, think, lora)
        if not target.strip():
            target = wp_t
        # STUDENT: drift turn only; the Liquid-LoRA (from current waypoint state) must supply the step
        s_msgs = (history[-2*args.keep_turns:] if args.keep_turns>0 else history) + [{"role": "user", "content": drift}]
        s_chat = _template(tok, s_msgs, think)
        p_ids = tok(s_chat, return_tensors="pt").input_ids.to(device)
        # CLOSED LOOP: observe the frozen LLM's drift state (no adapter) before generating the adapter
        h_llm = None
        if args.closed_loop:
            with torch.no_grad():
                lora.active = False
                h_llm = model(p_ids, output_hidden_states=True).hidden_states[args.obs_layer][:, -1].float()
        h = controller.dyn_state(way_emb[p].unsqueeze(0), h_llm); lora.set_state(h)
        if args.lambda_recon > 0:                                # HOLDING: belief reads back as the overall MISSION
            held = F.normalize(controller.g_head(h.flatten(1)).squeeze(0), dim=0)
            rloss.append(1.0 - (held * F.normalize(zG, dim=0)).sum())
        t_ids = tok(target, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        full = torch.cat([p_ids, t_ids], 1)
        lora.active = True
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            logits = model(full).logits[0]                     # [T, V]
        lora.active = False
        sel = logits[p_ids.shape[1] - 1: -1]                   # predict the target tokens
        ce = F.cross_entropy(sel, t_ids[0])
        ces.append(ce)
        # CONTRASTIVE: belief query -> current waypoint (pos) vs other waypoints + this tangent (neg)
        if args.lambda_contrast > 0:
            q = controller.belief_query(h)                     # [1, z]
            negs = [way_emb[j].unsqueeze(0) for j in range(n_way) if j != p]
            negs.append(encode_goal(drift, enc_tok, enc_model, device).unsqueeze(0))  # tangent repel
            cand = torch.cat([way_emb[p].unsqueeze(0)] + negs, 0)  # [1+M, z]; row 0 = positive
            sims = (q @ cand.t()).squeeze(0) / args.contrast_tau  # [1+M]
            closs.append(F.cross_entropy(sims.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device)))
        with torch.no_grad():
            coss.append(float((encode_goal(target, enc_tok, enc_model, device) * way_emb[p]).sum().item()))
        history += [{"role": "user", "content": drift}, {"role": "assistant", "content": target}]
    ce_m = torch.stack(ces).mean() if ces else None
    ct_m = torch.stack(closs).mean() if closs else None
    r_m = torch.stack(rloss).mean() if rloss else None
    return ce_m, ct_m, r_m, float(np.mean(coss)) if coss else 0.0


@torch.no_grad()
def eval_onplan(model, tok, controller, lora, enc_tok, enc_model, plans, goals, n, rng, device, args):
    """Measure: does the STUDENT (drift context + Liquid-LoRA) stay on the waypoint vs base (no LoRA)?
    Report mean cos(student_response, current waypoint) for base vs Liquid-LoRA."""
    controller.eval()
    base_c, lora_c, base_hit, lora_hit = [], [], [], []
    think = None if args.think else False
    for _ in range(n):
        g = goals[rng.integers(len(goals))]
        if g not in plans:
            continue
        wtxt, wemb = plans[g]; zG = encode_goal(g, enc_tok, enc_model, device)
        controller.reset_episode(1, device)
        if controller.use_slow:
            controller.slow_step(zG.unsqueeze(0))
        td = list(rng.permutation(len(TANGENTS)))
        history = [{"role": "user", "content": f"Help me with this task: {g}. We'll go step by step."}]
        for p in range(min(args.n_turns, len(wemb))):
            drift = "Okay, what's next?" if p % 2 == 0 else TANGENTS[td[p % len(td)]]
            msgs = (history[-2*args.keep_turns:] if args.keep_turns>0 else history) + [{"role": "user", "content": drift}]
            # base (no adapter)
            lora.active = False
            rb = gen_plain(model, tok, msgs, args.max_new_tokens, think, lora)
            # Liquid-LoRA
            chat = _template(tok, msgs, think); enc = tok(chat, return_tensors="pt").to(device)
            h_llm = None
            if args.closed_loop:
                lora.active = False
                h_llm = model(enc.input_ids, output_hidden_states=True).hidden_states[args.obs_layer][:, -1].float()
            h = controller.dyn_state(wemb[p].unsqueeze(0), h_llm); lora.set_state(h); lora.active = True
            out = model.generate(enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            lora.active = False
            rl = split_answer(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
            wpe = wemb[p]
            eb = encode_goal(rb, enc_tok, enc_model, device); el = encode_goal(rl, enc_tok, enc_model, device)
            base_c.append(float((eb * wpe).sum().item())); lora_c.append(float((el * wpe).sum().item()))
            base_hit.append(int((wemb @ eb).argmax()) == p)       # nearest waypoint == current step
            lora_hit.append(int((wemb @ el).argmax()) == p)
            history += [{"role": "user", "content": drift}, {"role": "assistant", "content": rl}]
    controller.train()
    m = lambda x: float(np.mean(x)) if x else 0.0
    return m(base_c), m(lora_c), m(base_hit), m(lora_hit)


def build_plans(model, tok, enc_tok, enc_model, goals, n_steps, device):
    plans = {}
    for g in goals:
        steps = elicit_plan(model, tok, g, n_steps, device)
        if len(steps) < 2:
            continue
        emb = torch.stack([encode_goal(s, enc_tok, enc_model, device) for s in steps])
        plans[g] = (steps, emb)
    return plans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--gen_dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--d", type=int, default=128); p.add_argument("--K", type=int, default=4)
    p.add_argument("--use_slow", action="store_true")
    p.add_argument("--lora_layers", default="8,14,20,26"); p.add_argument("--lora_proj", default="o_proj")
    p.add_argument("--lora_scale", type=float, default=1.0)
    p.add_argument("--cap_rel", type=float, default=0.5, help="cap |LoRA delta| to this fraction of |layer out| (stability)")
    p.add_argument("--closed_loop", action="store_true", help="condition adapter on the frozen LLM's drift hidden state")
    p.add_argument("--obs_layer", type=int, default=-1, help="hidden layer read for closed-loop observation")
    p.add_argument("--lambda_contrast", type=float, default=0.0, help="weight of belief->waypoint InfoNCE contrastive")
    p.add_argument("--contrast_tau", type=float, default=0.1, help="InfoNCE temperature")
    p.add_argument("--lambda_recon", type=float, default=0.0, help="weight of belief->MISSION reconstruction (holding readout)")
    p.add_argument("--n_steps", type=int, default=4); p.add_argument("--n_turns", type=int, default=4)
    p.add_argument("--keep_turns", type=int, default=0)
    p.add_argument("--think", action="store_true")
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--max_steps", type=int, default=120)
    p.add_argument("--group", type=int, default=2); p.add_argument("--max_new_tokens", type=int, default=45)
    p.add_argument("--eval_every", type=int, default=30); p.add_argument("--eval_n", type=int, default=10)
    p.add_argument("--n_test_goals", type=int, default=3); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--select_by", default="prec", choices=["prec", "cos"], help="checkpoint selection metric")
    p.add_argument("--output", default="/home/pokazge/checkpoints/liquid_lora2.pt")
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
    projs = [s.strip() for s in args.lora_proj.split(",")]
    lora = LiquidLoRA(model, layers, projs, d_ctrl=args.d, scale=args.lora_scale).to(device)
    lora.cap_rel = args.cap_rel if args.cap_rel > 0 else None
    lora.register()
    npar = sum(pp.numel() for pp in controller.parameters()) + sum(pp.numel() for pp in lora.parameters())
    print(f"[ll2] trainable {npar:,}  layers={layers} proj={args.lora_proj}  TRAIN=distillation", flush=True)
    opt = torch.optim.AdamW(list(controller.parameters()) + list(lora.parameters()), lr=args.lr)
    train_goals = COMMIT_GOALS[:-args.n_test_goals]; test_goals = COMMIT_GOALS[-args.n_test_goals:]
    print("[ll2] eliciting plans...", flush=True)
    plans = build_plans(model, tok, enc_tok, enc_model, COMMIT_GOALS, args.n_steps, device)
    rng = np.random.default_rng(args.seed)

    def run_eval(tag):
        bc, lc, bp, lp = eval_onplan(model, tok, controller, lora, enc_tok, enc_model, plans, test_goals,
                               args.eval_n, np.random.default_rng(7), device, args)
        print(f"[eval {tag}] cos base={bc:.3f} LoRA={lc:.3f} (Δ={lc-bc:+.3f}) | "
              f"step-prec base={bp:.2f} LoRA={lp:.2f} (Δ={lp-bp:+.2f})", flush=True)
        return lp if args.select_by == "prec" else lc

    best = run_eval("init"); closs = 0.0; ctsum = 0.0; rcsum = 0.0; cn = 0
    for step in range(1, args.max_steps + 1):
        opt.zero_grad(); lsum = 0.0; ctl = 0.0; rcl = 0.0; nb = 0
        for _ in range(args.group):
            g = train_goals[rng.integers(len(train_goals))]
            if g not in plans:
                continue
            wtxt, wemb = plans[g]; zG = encode_goal(g, enc_tok, enc_model, device)
            ce, ct, rc, _ = episode(model, tok, controller, lora, enc_tok, enc_model, g, wtxt, wemb, zG, device, args, train=True)
            if ce is None:
                continue
            loss = ce
            if ct is not None: loss = loss + args.lambda_contrast * ct
            if rc is not None: loss = loss + args.lambda_recon * rc
            (loss / args.group).backward(); lsum += float(ce.detach()); ctl += float(ct.detach()) if ct is not None else 0.0; rcl += float(rc.detach()) if rc is not None else 0.0; nb += 1
        torch.nn.utils.clip_grad_norm_(list(controller.parameters()) + list(lora.parameters()), 1.0)
        opt.step(); closs += lsum / max(1, nb); ctsum += ctl / max(1, nb); rcsum += rcl / max(1, nb); cn += 1
        if step % 10 == 0:
            print(f"step {step:>4}  distill_CE(roll)={closs/cn:.3f}  contrast(roll)={ctsum/cn:.3f}  recon_cos(roll)={1.0-rcsum/cn:.3f}", flush=True); closs = 0.0; ctsum = 0.0; rcsum = 0.0; cn = 0
        if step % args.eval_every == 0:
            mc = run_eval(f"s{step}")
            if mc > best:
                best = mc
                torch.save({"controller": controller.state_dict(), "lora": lora.state_dict(), "args": vars(args), "best": best}, args.output)
                print(f"[ll2] saved ({args.select_by}={best:.3f}) -> {args.output}", flush=True)
    print(f"[ll2] DONE best={best:.3f}", flush=True); print("[ll2] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
