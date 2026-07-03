"""Inspect semantic-goal steering: is the controller doing REAL goal control or
hacking the judge? Prints base vs steered text for held-out goals + judge score +
base-LLM fluency, so we can read whether steered text genuinely exhibits the target
semantic property (e.g. contrarian) AND stays coherent."""
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
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--ckpt", default="/home/pokazge/checkpoints/steer_semantic.pt")
    p.add_argument("--layer_idx", type=int, default=14)
    p.add_argument("--cat", default="contrarian")
    p.add_argument("--n", type=int, default=10)
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
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    controller = SteerController(d_llm=model.config.hidden_size, d=a["d"], K=a["K"],
                                   max_steer=a["max_steer"]).to(device)
    controller.load_state_dict(ck["controller"]); controller.eval()
    print(f"[insp] ckpt best_heldout_judge={ck.get('best_heldout_judge')}  cat={args.cat}", flush=True)
    hook = Hook(controller)
    model.model.layers[args.layer_idx - 1].register_forward_hook(hook)

    rng = np.random.default_rng(args.seed)
    jb, js, fb, fs = [], [], [], []
    for i in range(args.n):
        _, instr, judge_q = make_semantic_goal(rng, category=args.cat)
        z = encode_goal(instr, enc_tok, enc_model, device)
        b_txt, _, _ = generate(model, tok, instr, z, hook, args.max_new_tokens, 0.0, steer=False, grad=False)
        s_txt, _, _ = generate(model, tok, instr, z, hook, args.max_new_tokens, 0.0, steer=True, grad=False)
        jbi = judge_reward(model, tok, instr, b_txt, judge_q, yes_id, no_id)
        jsi = judge_reward(model, tok, instr, s_txt, judge_q, yes_id, no_id)
        fbi = base_fluency(model, tok, instr, b_txt); fsi = base_fluency(model, tok, instr, s_txt)
        jb.append(jbi); js.append(jsi); fb.append(fbi); fs.append(fsi)
        print(f"\n=== {i} GOAL: {instr}", flush=True)
        print(f"  [judge {jbi:.2f} flu {fbi:.2f}] BASE : {b_txt[:220]!r}", flush=True)
        print(f"  [judge {jsi:.2f} flu {fsi:.2f}] STEER: {s_txt[:220]!r}", flush=True)
    print(f"\n[insp] judge base={np.mean(jb):.3f} steered={np.mean(js):.3f}  "
          f"fluency base={np.mean(fb):.2f} steered={np.mean(fs):.2f}", flush=True)


if __name__ == "__main__":
    main()
