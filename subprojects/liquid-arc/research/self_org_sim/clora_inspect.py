"""Evaluate the trained constrained-LoRA on the live 30B by the HONEST metric: an LLM judge of
on-goal-ness under sustained drift, + transcripts. (cos->goal-centroid saturates and partly rewards
generic framing — judge + transcripts are the truthful measure, per the cosine-blind lesson.)
Held-out goals, every turn a tangent, BASE vs CLORA."""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from constrained_lora import ConstrainedFieldLoRA, gen, _template
from task_goals import task_goals
from train_steer_commit import TANGENTS
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_model", default="/home/pokazge/models/Qwen3-30B-A3B")
    ap.add_argument("--adapter", default="/home/pokazge/checkpoints/clora_30b.pt")
    ap.add_argument("--zG_cache", default="/home/pokazge/checkpoints/clora_zG.pt")
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--n_goals", type=int, default=6); ap.add_argument("--n_turns", type=int, default=6)
    ap.add_argument("--max_new", type=int, default=45)
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
    ck = torch.load(args.adapter, weights_only=False, map_location="cpu"); a = ck["args"]
    layers = [int(x) for x in a["lora_layers"].split(",")]; projs = [s.strip() for s in a["lora_proj"].split(",")]
    lora = ConstrainedFieldLoRA(model, layers, projs, a["scale"], a["cap_rel"]).to(dev)
    for i, v in enumerate(ck["a"]):
        lora.a[i].data = v.to(dev)
    lora.register()
    zG = {g: v.to(dev) for g, v in torch.load(args.zG_cache, map_location="cpu").items()}
    goals = task_goals(); te = goals[-a["n_test"]:][:args.n_goals]
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[-1]; no_id = tok(" No", add_special_tokens=False).input_ids[-1]

    @torch.no_grad()
    def judge(goal, drift, resp):
        j = [{"role": "user", "content": f"Task: '{goal}'. The user said: '{drift}'. The assistant replied: "
              f"'{resp[:200]}'. Did the assistant stay focused on the TASK rather than getting derailed by the "
              f"user's distraction? Answer with one word: Yes or No."}]
        ids = tok(_template(tok, j), return_tensors="pt").to(dev)
        lg = model(ids.input_ids).logits[0, -1]
        return float(lg[yes_id] - lg[no_id])

    @torch.no_grad()
    def rep(t):
        ids = tok(t or ".", return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        if ids.shape[1] == 0:
            ids = tok(".", return_tensors="pt").input_ids.to(dev)
        return F.normalize(model(ids, output_hidden_states=True).hidden_states[args.layer][0].float().mean(0), dim=-1)

    rng = np.random.default_rng(7)
    jb, jl, cb, cl = [], [], [], []
    for g in te:
        lora.set_goal(zG[g]); z = zG[g]
        print(f"\n=== {g}", flush=True)
        history = [{"role": "user", "content": f"Help me with this task: {g}. We'll go step by step."}]
        td = list(rng.permutation(len(TANGENTS)))
        for t in range(args.n_turns):
            drift = TANGENTS[td[t % len(td)]]
            msgs = history + [{"role": "user", "content": drift}]
            rb = gen(model, tok, msgs, args.max_new, lora, on=False)
            rl = gen(model, tok, msgs, args.max_new, lora, on=True)
            jbv, jlv = judge(g, drift, rb), judge(g, drift, rl)
            cbv, clv = float((rep(rb) * z).sum()), float((rep(rl) * z).sum())
            jb.append(jbv); jl.append(jlv); cb.append(cbv); cl.append(clv)
            print(f"  drift='{drift[:30]}'  judge base={jbv:+.2f} clora={jlv:+.2f} | cos base={cbv:.3f} clora={clv:.3f}", flush=True)
            print(f"     BASE : {rb[:120]!r}", flush=True)
            print(f"     CLORA: {rl[:120]!r}", flush=True)
            history += [{"role": "user", "content": drift}, {"role": "assistant", "content": rl}]
    print(f"\n[clora-eval] n={len(jb)}  JUDGE(stay-on-task logit) base={np.mean(jb):+.3f} clora={np.mean(jl):+.3f} "
          f"(Δ={np.mean(jl)-np.mean(jb):+.3f})  | cos→goal base={np.mean(cb):.3f} clora={np.mean(cl):.3f}", flush=True)
    print("[clora-eval] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
