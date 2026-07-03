"""Independent-judge validation of the semantic steering controller.

The training reward used the Qwen2.5-1.5B judge, and the controller partially gamed
it before (template collapse). To check the held-out result is REAL goal-following and
not reward-model-hacking, re-score base vs steered with an INDEPENDENT, stronger judge
(Qwen2.5-7B) that was never part of training. If steered still beats base under the 7B,
the effect is validated.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_steer_controller import SteerController, Hook, generate, base_fluency, encode_goal
from train_steer_semantic import make_semantic_goal, judge_reward
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--judge_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--ckpt", default="/home/pokazge/checkpoints/steer_semantic_div.pt")
    p.add_argument("--layer_idx", type=int, default=14)
    p.add_argument("--cats", default="contrarian")  # held-out; can add more
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=456)
    args = p.parse_args()

    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.set_float32_matmul_precision("high")

    tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.gen_model, dtype=torch.float16,
                                                   trust_remote_code=True).to(device).eval()
    for pp in model.parameters():
        pp.requires_grad = False
    enc_tok = AutoTokenizer.from_pretrained(args.enc_model)
    enc_model = AutoModel.from_pretrained(args.enc_model).to(device).eval()

    # judges
    y1 = tok(" Yes", add_special_tokens=False).input_ids[0]
    n1 = tok(" No", add_special_tokens=False).input_ids[0]
    jtok = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    jmodel = AutoModelForCausalLM.from_pretrained(args.judge_model, dtype=torch.float16,
                                                    trust_remote_code=True).to(device).eval()
    y7 = jtok(" Yes", add_special_tokens=False).input_ids[0]
    n7 = jtok(" No", add_special_tokens=False).input_ids[0]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    controller = SteerController(d_llm=model.config.hidden_size, d=a["d"], K=a["K"],
                                   max_steer=a["max_steer"]).to(device)
    controller.load_state_dict(ck["controller"]); controller.eval()
    hook = Hook(controller)
    model.model.layers[args.layer_idx - 1].register_forward_hook(hook)
    print(f"[xjudge] ckpt={args.ckpt} train-judge=1.5B  INDEPENDENT judge={args.judge_model}", flush=True)

    cats = [c.strip() for c in args.cats.split(",")]
    rng = np.random.default_rng(args.seed)
    agg = {}
    for cat in cats:
        j1b, j1s, j7b, j7s, fbs, fss = [], [], [], [], [], []
        for _ in range(args.n):
            _, instr, judge_q = make_semantic_goal(rng, category=cat)
            z = encode_goal(instr, enc_tok, enc_model, device)
            b_txt, _, _ = generate(model, tok, instr, z, hook, args.max_new_tokens, 0.0, steer=False, grad=False)
            s_txt, _, _ = generate(model, tok, instr, z, hook, args.max_new_tokens, 0.0, steer=True, grad=False)
            # training judge (1.5B) — reference
            j1b.append(judge_reward(model, tok, instr, b_txt, judge_q, y1, n1))
            j1s.append(judge_reward(model, tok, instr, s_txt, judge_q, y1, n1))
            # INDEPENDENT judge (7B)
            j7b.append(judge_reward(jmodel, jtok, instr, b_txt, judge_q, y7, n7))
            j7s.append(judge_reward(jmodel, jtok, instr, s_txt, judge_q, y7, n7))
            fbs.append(base_fluency(model, tok, instr, b_txt))
            fss.append(base_fluency(model, tok, instr, s_txt))
        agg[cat] = (np.mean(j1b), np.mean(j1s), np.mean(j7b), np.mean(j7s), np.mean(fbs), np.mean(fss))
        m = agg[cat]
        print(f"\n[{cat}] n={args.n}", flush=True)
        print(f"  1.5B judge (TRAIN): base={m[0]:.3f} steer={m[1]:.3f}  Δ={m[1]-m[0]:+.3f}", flush=True)
        print(f"  7B  judge (INDEP):  base={m[2]:.3f} steer={m[3]:.3f}  Δ={m[3]-m[2]:+.3f}", flush=True)
        print(f"  fluency:            base={m[4]:.2f} steer={m[5]:.2f}", flush=True)
    print("\n[xjudge] VALIDATION: if 7B-judge Δ is clearly positive, the steering effect is REAL "
          "(not 1.5B-judge gaming).", flush=True)
    print("[xjudge] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
