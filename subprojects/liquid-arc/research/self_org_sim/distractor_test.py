"""DISTRACTOR DISCRIMINATION — stability vs plasticity of the goal-field.

Mission must hold course against IRRELEVANT distractors (noise) yet ADMIT RELEVANT ones (meaningful).
The discriminator is the field's relevance signal: relevance(distractor) = cos(h_distractor, z_G).
  noise distractor      (off-topic tangent)      -> low relevance  -> field resists (high kappa)
  meaningful distractor (goal-relevant constraint) -> high relevance -> field admits  (low kappa)

Test on the live 30B over the cached held-out goals:
  - generate a MEANINGFUL distractor (relevant constraint) per goal with the LLM
  - NOISE distractors = generic tangents
  - encode each distractor MESSAGE -> layer-36 rep; measure cos->z_G and gate kappa
Question (structure vs familiarity, again): does the gate -- trained only on goal-serving vs noise --
zero-shot ADMIT meaningful distractors (low kappa, high relevance) and RESIST noise (high kappa)?
If yes, the substrate has the stability-plasticity property: resist noise, take in the meaningful.
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
    ap.add_argument("--layer", type=int, default=36)
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
    zG = {g: ck["zG"][g].to(dev) for g in ck["zG"]}
    goals = ck["ho_goals"][:args.n_goals]
    goal_text = task_goals()

    @torch.no_grad()
    def gen(msgs, mx=40):
        enc = tok(_template(tok, msgs), return_tensors="pt").to(dev)
        out = model.generate(enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=mx,
                               do_sample=False, pad_token_id=tok.pad_token_id)
        return split_answer(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))

    @torch.no_grad()
    def rep(text):
        ids = tok(text or ".", return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        if ids.shape[1] == 0:
            ids = tok(".", return_tensors="pt").input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states[args.layer][0]
        return F.normalize(hs.float().mean(0), dim=-1)

    @torch.no_grad()
    def kappa(h, z):
        gate.reset(1, dev)
        return float(gate.step(h.unsqueeze(0), z.unsqueeze(0)))

    rng = np.random.default_rng(0)
    cm, cn, km, kn, sep_cos, sep_kap = [], [], [], [], [], []
    for gi in goals:
        g = goal_text[gi]; z = zG[gi]
        # MEANINGFUL distractor: a relevant constraint the user adds mid-task
        mprompt = [{"role": "user", "content": f"A user is working step by step on this task: '{g}'. "
                    f"Mid-conversation they add ONE short relevant detail or constraint that should be "
                    f"taken into account. Reply with ONLY that one-sentence user message."}]
        mtext = gen(mprompt, 30).strip().strip('"')
        ntext = TANGENTS[rng.integers(len(TANGENTS))]                    # NOISE distractor
        hm, hn = rep(mtext), rep(ntext)
        rcm, rcn = float((hm * z).sum()), float((hn * z).sum())
        rkm, rkn = kappa(hm, z), kappa(hn, z)
        cm.append(rcm); cn.append(rcn); km.append(rkm); kn.append(rkn)
        sep_cos.append(rcm > rcn); sep_kap.append(rkm < rkn)
        if gi == goals[0] or gi == goals[1]:
            print(f"  [{gi}] {g[:34]}", flush=True)
            print(f"     meaningful='{mtext[:60]}' cos→goal={rcm:.3f} kappa={rkm:.2f}", flush=True)
            print(f"     noise     ='{ntext[:60]}' cos→goal={rcn:.3f} kappa={rkn:.2f}", flush=True)
    print(f"\n[distractor] n={len(cm)} goals", flush=True)
    print(f"  relevance cos→goal:  meaningful={np.mean(cm):.3f}  noise={np.mean(cn):.3f}  "
          f"(meaningful>noise in {np.mean(sep_cos):.2f})", flush=True)
    print(f"  gate kappa (resist): meaningful={np.mean(km):.2f}  noise={np.mean(kn):.2f}  "
          f"(admits meaningful kappa_m<kappa_n in {np.mean(sep_kap):.2f})", flush=True)
    print("[distractor] stability-plasticity holds iff meaningful is MORE relevant AND LESS resisted than noise", flush=True)
    print("[distractor] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
