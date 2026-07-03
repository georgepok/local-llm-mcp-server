"""CONSTRAINED-LoRA realization of the goal-field (weight-level actuator).

Framework: the LoRA is NOT a free weight-generator; it is constrained to implement the goal-curvature
and nothing else. Rank-1 form, per target output-projection (o_proj / MoE expert down_proj, out=hidden):
    delta_out = scale * z_G * (a . x)
  B (write direction) = z_G  -> FIXED to the goal-field (the constraint; the adapter can ONLY push
                                the residual toward the goal). Seeded from goal-serving trajectories.
  A (read)            = a    -> LEARNED, shared across goals (the gate: how hard to pull, per input).
With z_G swapped per goal and `a` shared, this IS the generalized goal-following operator at the
weight level. Trained by distillation (teacher = on-goal; student = LoRA on drift). Relative cap keeps
the pull from dominating the sub-task / breaking fluency.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_steer_traj import split_answer
from task_goals import task_goals
from train_steer_commit import TANGENTS
from transformers import AutoModelForCausalLM, AutoTokenizer


def _template(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class ConstrainedFieldLoRA(nn.Module):
    def __init__(self, model, layers, projs, scale=1.0, cap_rel=0.3):
        super().__init__()
        attn = {"q_proj", "k_proj", "v_proj", "o_proj"}
        self.target_mods, in_dims = [], []
        for L in layers:
            layer = model.model.layers[L - 1]
            for pr in projs:
                if pr in attn:
                    m = getattr(layer.self_attn, pr); self.target_mods.append([m]); in_dims.append(m.in_features)
                else:
                    mlp = layer.mlp
                    if hasattr(mlp, pr):
                        m = getattr(mlp, pr); self.target_mods.append([m]); in_dims.append(m.in_features)
                    elif hasattr(mlp, "experts"):
                        ms = [getattr(e, pr) for e in mlp.experts if hasattr(e, pr)]
                        self.target_mods.append(ms); in_dims.append(ms[0].in_features)
        self.n = len(in_dims)
        self.a = nn.ParameterList([nn.Parameter(torch.zeros(di)) for di in in_dims])  # learned read, init 0 (LoRA=0)
        self.scale = scale; self.cap_rel = cap_rel; self.active = False; self.zG = None; self.handles = []

    def set_goal(self, zG):
        self.zG = F.normalize(zG, dim=-1)                      # write direction = goal (hidden space)

    def _hook(self, ti):
        def fn(module, inp, out):
            if not self.active or self.zG is None:
                return out
            x = inp[0].float()
            mag = (x * self.a[ti]).sum(-1, keepdim=True)       # read / gate  [.., 1]
            d = self.scale * mag * self.zG                     # write toward goal  [.., hidden]
            if self.cap_rel is not None:
                on = out.float().norm(dim=-1, keepdim=True); dn = d.norm(dim=-1, keepdim=True) + 1e-6
                d = d * torch.clamp(self.cap_rel * on / dn, max=1.0)
            return out + d.to(out.dtype)
        return fn

    def register(self):
        for ti, mods in enumerate(self.target_mods):
            for m in mods:
                self.handles.append(m.register_forward_hook(self._hook(ti)))


@torch.no_grad()
def gen(model, tok, msgs, mx, lora=None, on=False):
    if lora is not None:
        lora.active = on
    enc = tok(_template(tok, msgs), return_tensors="pt").to(model.device)
    out = model.generate(enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=mx,
                           do_sample=False, pad_token_id=tok.pad_token_id)
    if lora is not None:
        lora.active = False
    return split_answer(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))


@torch.no_grad()
def seed_goal(model, tok, goal, layer, device, n_seed=2):
    """z_G = centroid of goal-serving continuation reps (trajectory-seeded, non-textual)."""
    reps = []
    for k in range(n_seed):
        msgs = [{"role": "user", "content": f"Help me with this task, step by step: {goal}."},
                {"role": "user", "content": "Continue working on the task." if k else "Give the next concrete step."}]
        r = gen(model, tok, msgs, 40)
        ids = tok(r or goal, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        hs = model(ids, output_hidden_states=True).hidden_states[layer][0]
        reps.append(hs.float().mean(0))
    return F.normalize(torch.stack(reps).mean(0), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gen_dtype", default="float16")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--lora_layers", default="12,16,20,24")
    ap.add_argument("--lora_proj", default="o_proj,down_proj")
    ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--cap_rel", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--group", type=int, default=2); ap.add_argument("--max_new", type=int, default=45)
    ap.add_argument("--n_turns", type=int, default=4); ap.add_argument("--eval_every", type=int, default=30)
    ap.add_argument("--n_test", type=int, default=8); ap.add_argument("--n_train", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--output", default="/home/pokazge/checkpoints/constrained_lora.pt")
    ap.add_argument("--zG_cache", default="/home/pokazge/checkpoints/clora_zG.pt")
    args = ap.parse_args()
    dev = torch.device("cuda"); torch.manual_seed(args.seed)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.set_float32_matmul_precision("high")
    tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    gd = torch.bfloat16 if args.gen_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.gen_model, dtype=gd, trust_remote_code=True,
                                                   low_cpu_mem_usage=True, device_map={"": 0}).eval()
    for p in model.parameters():
        p.requires_grad = False
    layers = [int(x) for x in args.lora_layers.split(",")]; projs = [s.strip() for s in args.lora_proj.split(",")]
    lora = ConstrainedFieldLoRA(model, layers, projs, args.scale, args.cap_rel).to(dev); lora.register()
    print(f"[clora] trainable {sum(p.numel() for p in lora.parameters()):,}  targets={lora.n}  proj={args.lora_proj}", flush=True)
    opt = torch.optim.AdamW(lora.parameters(), lr=args.lr)

    goals = task_goals()
    tr = goals[:args.n_train]; te = goals[-args.n_test:]
    zc = Path(args.zG_cache)
    if zc.exists():
        print(f"[clora] loading seeded fields {zc}", flush=True)
        zG = {g: v.to(dev) for g, v in torch.load(zc, map_location="cpu").items()}
    else:
        print("[clora] seeding goal fields...", flush=True)
        zG = {g: seed_goal(model, tok, g, args.layer, dev) for g in tr + te}
        torch.save({g: v.cpu() for g, v in zG.items()}, zc); print(f"[clora] cached fields -> {zc}", flush=True)
    rng = np.random.default_rng(args.seed)

    @torch.no_grad()
    def rep(text):
        ids = tok(text or ".", return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        if ids.shape[1] == 0:
            ids = tok(".", return_tensors="pt").input_ids.to(dev)
        return F.normalize(model(ids, output_hidden_states=True).hidden_states[args.layer][0].float().mean(0), dim=-1)

    def episode(g, train):
        lora.set_goal(zG[g])
        td = list(rng.permutation(len(TANGENTS)))
        history = [{"role": "user", "content": f"Help me with this task: {g}. We'll go step by step."}]
        ces = []
        for t in range(args.n_turns):
            drift = TANGENTS[td[t % len(td)]]
            tmsg = history + [{"role": "user", "content": f"{drift} (Stay on our task: {g}.)"}]
            target = gen(model, tok, tmsg, args.max_new) or g
            smsg = history + [{"role": "user", "content": drift}]
            p_ids = tok(_template(tok, smsg), return_tensors="pt").input_ids.to(dev)
            t_ids = tok(target, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            full = torch.cat([p_ids, t_ids], 1)
            lora.active = True
            ctx = torch.enable_grad() if train else torch.no_grad()
            with ctx:
                logits = model(full).logits[0]
            lora.active = False
            ce = F.cross_entropy(logits[p_ids.shape[1] - 1:-1], t_ids[0]); ces.append(ce)
            history += [{"role": "user", "content": drift}, {"role": "assistant", "content": target}]
        return torch.stack(ces).mean()

    @torch.no_grad()
    def evaluate():
        bc, lc = [], []
        for g in te:
            lora.set_goal(zG[g]); z = zG[g]
            td = list(np.random.default_rng(7).permutation(len(TANGENTS)))
            history = [{"role": "user", "content": f"Help me with this task: {g}. We'll go step by step."}]
            for t in range(args.n_turns):
                drift = TANGENTS[td[t % len(td)]]
                msgs = history + [{"role": "user", "content": drift}]
                rb = gen(model, tok, msgs, args.max_new, lora, on=False)
                rl = gen(model, tok, msgs, args.max_new, lora, on=True)
                bc.append(float((rep(rb) * z).sum())); lc.append(float((rep(rl) * z).sum()))
                history += [{"role": "user", "content": drift}, {"role": "assistant", "content": rl}]
        return float(np.mean(bc)), float(np.mean(lc))

    b0, l0 = evaluate(); print(f"[eval init] cos→goal base={b0:.3f} clora={l0:.3f} (Δ={l0-b0:+.3f})", flush=True)
    best = l0 - b0; roll = 0.0; n = 0
    for step in range(1, args.max_steps + 1):
        opt.zero_grad(); s = 0.0; nb = 0
        for _ in range(args.group):
            g = tr[rng.integers(len(tr))]
            ce = episode(g, True); (ce / args.group).backward(); s += float(ce.detach()); nb += 1
        torch.nn.utils.clip_grad_norm_(lora.parameters(), 1.0); opt.step()
        roll += s / max(1, nb); n += 1
        if step % 10 == 0:
            print(f"step {step:>4} distill_CE={roll/n:.3f}", flush=True); roll = 0.0; n = 0
        if step % args.eval_every == 0:
            b, l = evaluate(); print(f"[eval s{step}] cos→goal base={b:.3f} clora={l:.3f} (Δ={l-b:+.3f})", flush=True)
            best = max(best, l - b)
            torch.save({"a": [p.detach().cpu() for p in lora.a], "args": vars(args), "step": step, "lastDelta": l - b}, args.output)
            print(f"[clora] saved (final adapter, step {step})", flush=True)
    torch.save({"a": [p.detach().cpu() for p in lora.a], "args": vars(args), "step": args.max_steps}, args.output)
    print(f"[clora] DONE (best monitored Δ={best:+.3f}; final adapter saved -> {args.output})", flush=True)
    print("[clora] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
