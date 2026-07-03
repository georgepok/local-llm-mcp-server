"""INDUCTION EXPERIMENT — can a goal exist as a GENERALIZING field over the LLM's representation
manifold, or is it only a re-encoding of its seed?

The crux question of the goal-as-field program. A goal is formalized as a potential field Phi(h; z_G)
over the LLM's hidden-state space whose low-energy region is the goal-serving basin. We SEED it from
goal-serving TRAJECTORIES (not text): z_G is the centroid of goal-serving continuation states. We fit
the field's low-rank curvature P CONTRASTIVELY against near-miss (drift) continuations — so the field
must encode goal-STRUCTURE, not familiarity with the seed. Then we test whether the field assigns lower
energy to the goal-serving continuation at states/goals it never trained on.

Decisive test (induction):
  - held-out STATES (seen goals, unseen turns): does the field generalize over representation space?
  - held-out GOALS (P never trained on them; z_G from their own seed trajectories): the strong test.
Controls:
  - POSITIVE-ONLY field (no near-miss): does contrastive formation matter?
  - RAW-COS (P = I): does the learned field beat mere seed-similarity? If not, it's text in disguise.

Verdict: contrastive field > raw-cos on held-out GOALS  => a goal CAN be a generalizing field (structure).
         contrastive field ~ raw-cos                    => seed re-encoding (familiarity), not a substrate.
         ~ 0.5                                           => no generalization; substrate idea falsified cheaply.
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
from train_steer_traj import elicit_plan, split_answer
from train_steer_commit import COMMIT_GOALS, TANGENTS
from task_goals import task_goals
from transformers import AutoModelForCausalLM, AutoTokenizer


def _template(tok, messages):
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def gen(model, tok, msgs, max_new):
    chat = _template(tok, msgs)
    enc = tok(chat, return_tensors="pt").to(model.device)
    out = model.generate(enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=max_new,
                           do_sample=False, pad_token_id=tok.pad_token_id)
    return split_answer(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))


@torch.no_grad()
def repr_of(model, tok, msgs, response, layer):
    """Representation of a continuation: mean-pooled hidden state at `layer` over the RESPONSE tokens."""
    chat = _template(tok, msgs)
    p_ids = tok(chat, return_tensors="pt").input_ids.to(model.device)
    r_ids = tok(response, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    full = torch.cat([p_ids, r_ids], 1)
    hs = model(full, output_hidden_states=True).hidden_states[layer][0]   # [T, d]
    resp = hs[p_ids.shape[1]:]                                            # response-token states
    if resp.shape[0] == 0:
        resp = hs[-1:]
    return resp.float().mean(0)                                          # [d]


def collect(model, tok, layer, n_turns, max_new, device, goal_list):
    """For each goal x turn: (h_pos = goal-serving continuation repr, h_neg = drift continuation repr)."""
    data = {}                                                            # goal_idx -> list of (h_pos, h_neg)
    rng = np.random.default_rng(0)
    for gi, g in enumerate(goal_list):
        steps = elicit_plan(model, tok, g, n_turns, device)
        if len(steps) < 2:
            continue
        td = list(rng.permutation(len(TANGENTS)))
        history = [{"role": "user", "content": f"Help me with this task: {g}. We'll go step by step."}]
        pairs = []
        for p in range(min(n_turns, len(steps))):
            drift = "Okay, what's next?" if p % 2 == 0 else TANGENTS[td[p % len(td)]]
            wp = steps[p]
            t_msgs = history + [{"role": "user", "content": f"{drift} (Stay on our task — the next step is: {wp}.)"}]
            d_msgs = history + [{"role": "user", "content": drift}]
            r_pos = gen(model, tok, t_msgs, max_new) or wp
            r_neg = gen(model, tok, d_msgs, max_new) or drift
            h_pos = repr_of(model, tok, d_msgs, r_pos, layer)            # same drift context, goal-serving cont
            h_neg = repr_of(model, tok, d_msgs, r_neg, layer)            # drift continuation
            pairs.append((h_pos.cpu(), h_neg.cpu()))
            history += [{"role": "user", "content": drift}, {"role": "assistant", "content": r_neg}]
        data[gi] = pairs
        print(f"[collect] goal {gi:2d} '{g[:40]}'  {len(pairs)} pairs", flush=True)
    return data


class Field(nn.Module):
    """Low-rank potential: Phi(h; z) = -<Ph_hat, Pz_hat>  (lower energy = more goal-aligned)."""
    def __init__(self, d, r):
        super().__init__()
        self.P = nn.Linear(d, r, bias=False)
        nn.init.orthogonal_(self.P.weight)

    def align(self, h, z):                                               # higher = goal-serving (= -energy)
        ph = F.normalize(self.P(h), dim=-1); pz = F.normalize(self.P(z), dim=-1)
        return (ph * pz).sum(-1)


def seed_code(pairs, idx):
    """Goal code = centroid of goal-serving (h_pos) states over the SEED turns (trajectory-seeded)."""
    z = torch.stack([pairs[i][0] for i in idx]).mean(0)
    return z


def pair_acc(score_pos, score_neg):
    return float((score_pos > score_neg).float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gen_dtype", default="float16")
    ap.add_argument("--layer", type=int, default=20, help="hidden layer for the representation manifold")
    ap.add_argument("--n_turns", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=45)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--n_heldout_goals", type=int, default=8)
    ap.add_argument("--seed_turns", type=int, default=2, help="first k turns -> z_G + fit; rest -> held-out states")
    ap.add_argument("--cache", default="/home/pokazge/checkpoints/induction_repr.pt")
    ap.add_argument("--goal_set", default="commit", choices=["commit", "task"], help="task = ~120 templated goals")
    ap.add_argument("--n_goals", type=int, default=0, help="cap on goals (0 = all)")
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.set_float32_matmul_precision("high")

    if Path(args.cache).exists():
        print(f"[load] cached reprs {args.cache}", flush=True)
        data = torch.load(args.cache, weights_only=False)
    else:
        tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        gd = torch.bfloat16 if args.gen_dtype == "bfloat16" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(args.gen_model, dtype=gd, trust_remote_code=True,
                                                       low_cpu_mem_usage=True, device_map={"": 0}).eval()
        gl = task_goals(args.n_goals or None) if args.goal_set == "task" else COMMIT_GOALS
        print(f"[collect] goal_set={args.goal_set} n={len(gl)}", flush=True)
        data = collect(model, tok, args.layer, args.n_turns, args.max_new, device, gl)
        torch.save(data, args.cache); print(f"[save] {args.cache}", flush=True)
        del model
        torch.cuda.empty_cache()

    goals = sorted(data.keys())
    d = data[goals[0]][0][0].shape[0]
    rng = np.random.default_rng(7)
    ho_goals = set(rng.choice(goals, size=min(args.n_heldout_goals, len(goals) // 2), replace=False).tolist())
    tr_goals = [g for g in goals if g not in ho_goals]
    print(f"[split] d={d}  train goals={len(tr_goals)}  held-out goals={len(ho_goals)}  rank={args.rank}", flush=True)

    def turn_split(pairs):
        k = min(args.seed_turns, len(pairs) - 1) if len(pairs) > 1 else len(pairs)
        return list(range(k)), list(range(k, len(pairs)))               # seed/fit turns, held-out-state turns

    # build training pairs (train goals, seed turns) and z_G per goal (centroid of seed h_pos)
    zG = {}
    fit_pos, fit_neg, fit_z = [], [], []
    for g in tr_goals:
        pairs = data[g]; seed_idx, _ = turn_split(pairs)
        z = F.normalize(seed_code(pairs, seed_idx), dim=0)
        zG[g] = z
        for i in seed_idx:
            fit_pos.append(pairs[i][0]); fit_neg.append(pairs[i][1]); fit_z.append(z)
    Hp = torch.stack(fit_pos).to(device); Hn = torch.stack(fit_neg).to(device); Z = torch.stack(fit_z).to(device)

    def fit(contrastive):
        fld = Field(d, args.rank).to(device)
        opt = torch.optim.Adam(fld.parameters(), lr=1e-2)
        for ep in range(args.epochs):
            opt.zero_grad()
            ap_ = fld.align(Hp, Z); an_ = fld.align(Hn, Z)
            if contrastive:
                loss = F.softplus(args.margin - (ap_ - an_)).mean()     # pos must out-align neg by margin
            else:
                loss = (1 - ap_).mean()                                 # positive-only: pull pos toward z
            loss.backward(); opt.step()
        return fld

    fld_c = fit(True)
    fld_p = fit(False)

    @torch.no_grad()
    def evalp(fld, glist, which):
        sp, sn = [], []
        for g in glist:
            pairs = data[g]; seed_idx, ho_idx = turn_split(pairs)
            z = zG[g] if g in zG else F.normalize(seed_code(pairs, seed_idx), dim=0)
            z = z.to(device)
            idx = ho_idx if which == "state" else list(range(len(pairs)))  # held-out goal: test all its turns
            if which == "goal":
                idx = ho_idx if len(ho_idx) else list(range(len(pairs)))   # avoid seed leakage where possible
            for i in idx:
                hp = pairs[i][0].to(device); hn = pairs[i][1].to(device)
                if fld is None:                                           # raw-cos baseline (P = I)
                    sp.append(float((F.normalize(hp, dim=0) * z).sum())); sn.append(float((F.normalize(hn, dim=0) * z).sum()))
                else:
                    sp.append(float(fld.align(hp.unsqueeze(0), z.unsqueeze(0)))); sn.append(float(fld.align(hn.unsqueeze(0), z.unsqueeze(0))))
        return pair_acc(torch.tensor(sp), torch.tensor(sn)), len(sp)

    print("\n=== INDUCTION RESULT  (pairwise acc: field prefers goal-serving continuation) ===", flush=True)
    for which, glist, lbl in [("state", tr_goals, "held-out STATES (seen goals)"),
                               ("goal", sorted(ho_goals), "held-out GOALS (unseen)")]:
        ac, nc = evalp(fld_c, glist, which)
        ap_, _ = evalp(fld_p, glist, which)
        ar, _ = evalp(None, glist, which)
        print(f"  {lbl:32s} n={nc:3d} | contrastive={ac:.3f}  positive-only={ap_:.3f}  raw-cos={ar:.3f}", flush=True)
    print("\n[verdict] induction supported iff contrastive > raw-cos on held-out GOALS (structure beyond seed).", flush=True)
    print("[ind] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
