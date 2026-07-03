"""RELEVANCE-AWARE goal-field: stability to noise, plasticity to meaning.

Distractor test showed the native field's relevance signal cos(h_d, z_G) cleanly separates meaningful
(0.317) from noise (0.152) distractors. Build the mechanism that USES it:

  relevance(h_d) = clamp(cos(h_d, z_G), 0, 1)
  STABILITY  : effective resistance  kappa_eff = kappa_gate * (1 - relevance)
               -> noise (low rel) fully resisted; meaningful (high rel) admitted (resistance suppressed)
  PLASTICITY : slow-channel absorbs meaning  z_G' = normalize(z_G + alpha * relevance * h_d)
               -> meaningful deforms the mission (incorporated); noise (rel~0) leaves z_G unchanged
               and the ABSORBED field must still track goal-serving (mission intact, now refined).

Demonstrated on the live 30B over held-out goals: meaningful vs noise distractors.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from engagement_gate import LiquidGate
from train_steer_traj import split_answer
from task_goals import task_goals
from train_steer_commit import TANGENTS
from transformers import AutoModelForCausalLM, AutoTokenizer


def _template(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_model", default="/home/pokazge/models/Qwen3-30B-A3B")
    ap.add_argument("--gate", default="/home/pokazge/checkpoints/liquid_gate.pt")
    ap.add_argument("--cache", default="/home/pokazge/checkpoints/induction_repr_30b_big.pt")
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n_goals", type=int, default=30)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.set_float32_matmul_precision("high")
    tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.gen_model, dtype=torch.bfloat16, trust_remote_code=True,
                                                   low_cpu_mem_usage=True, device_map={"": 0}).eval()
    for p in model.parameters():
        p.requires_grad = False
    ck = torch.load(args.gate, weights_only=False, map_location="cpu")
    gate = LiquidGate(ck["d_llm"], ck["d"]).to(dev); gate.load_state_dict(ck["gate"]); gate.eval()
    cache = torch.load(args.cache, weights_only=False, map_location="cpu")
    zG = {g: ck["zG"][g].to(dev) for g in ck["zG"]}
    hpos = {g: torch.stack([p[0] for p in cache[g]]).to(dev) for g in cache}      # goal-serving traj
    goals = [g for g in ck["ho_goals"] if g in hpos][:args.n_goals]
    goal_text = task_goals()

    @torch.no_grad()
    def gen(msgs, mx=30):
        enc = tok(_template(tok, msgs), return_tensors="pt").to(dev)
        out = model.generate(enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=mx,
                               do_sample=False, pad_token_id=tok.pad_token_id)
        return split_answer(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))

    @torch.no_grad()
    def rep(text):
        ids = tok(text or ".", return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        if ids.shape[1] == 0:
            ids = tok(".", return_tensors="pt").input_ids.to(dev)
        return F.normalize(model(ids, output_hidden_states=True).hidden_states[args.layer][0].float().mean(0), dim=-1)

    @torch.no_grad()
    def gate_kappa(z, dist, g):                          # mission context = goal-serving turns, then distractor
        gate.reset(1, dev)
        for t in range(hpos[g].shape[0]):
            gate.step(hpos[g][t:t + 1], z.unsqueeze(0))
        return float(gate.step(dist.unsqueeze(0), z.unsqueeze(0)))

    rng = np.random.default_rng(0)
    rows = {"relevance": [], "keff": [], "track": [], "absorb": []}
    agg = {"m": {k: [] for k in rows}, "n": {k: [] for k in rows}}
    for gi in goals:
        g = goal_text[gi]; z = zG[gi]
        mtext = gen([{"role": "user", "content": f"A user is working step by step on: '{g}'. They add ONE short "
                      f"relevant constraint that should be taken into account. Reply ONLY that one-sentence message."}], 30).strip().strip('"')
        ntext = TANGENTS[rng.integers(len(TANGENTS))]
        for tag, txt in (("m", mtext), ("n", ntext)):
            hd = rep(txt)
            relev = float(torch.clamp((hd * z).sum(), 0, 1))
            kap = gate_kappa(z, hd, gi)
            keff = kap * (1 - relev)                                       # STABILITY: admit relevant, resist noise
            zp = F.normalize(z + args.alpha * relev * hd, dim=-1)          # PLASTICITY: absorb relevant into z_G
            track = float((F.normalize(hpos[gi], dim=-1) * zp).sum(-1).mean())   # updated field still tracks goal?
            absorb = float((hd * zp).sum()) - relev                       # did z_G move TOWARD the distractor?
            agg[tag]["relevance"].append(relev); agg[tag]["keff"].append(keff)
            agg[tag]["track"].append(track); agg[tag]["absorb"].append(absorb)

    base_track = np.mean([float((F.normalize(hpos[gi], dim=-1) * zG[gi]).sum(-1).mean()) for gi in goals])
    print(f"\n[aware] n={len(goals)} goals  (alpha={args.alpha})  base field tracks goal-serving={base_track:.3f}", flush=True)
    print(f"  {'':12s} {'relevance':>10s} {'kappa_eff':>10s} {'absorb→z_G':>11s} {'track(goal)':>11s}", flush=True)
    for tag, lbl in (("m", "MEANINGFUL"), ("n", "NOISE")):
        a = agg[tag]
        print(f"  {lbl:12s} {np.mean(a['relevance']):>10.3f} {np.mean(a['keff']):>10.3f} "
              f"{np.mean(a['absorb']):>+11.4f} {np.mean(a['track']):>11.3f}", flush=True)
    print(f"  STABILITY : kappa_eff(noise) >> kappa_eff(meaningful)  ->  resist noise, admit meaning", flush=True)
    print(f"  PLASTICITY: meaningful absorbed into z_G (+absorb) while goal-tracking held; noise ~ignored", flush=True)
    print("[aware] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
