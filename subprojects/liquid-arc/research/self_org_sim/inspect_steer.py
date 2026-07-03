"""Inspect steered vs unsteered generations — is the controller doing REAL goal
control or just gaming the narrow check_fn?

The check_fns are narrow (end_with_question only tests endswith('?')). A steering
controller trained on check reward could reward-hack: emit degenerate text that
trips the surface check without being a coherent on-topic response. This prints
baseline and steered outputs side by side so we can judge coherence + topicality,
and also computes a fluency proxy (mean token log-prob under the FROZEN base LLM,
no steer) — gamed/degenerate steered text scores much worse on base-LLM fluency.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import generate_diverse_goals as gdg
from train_steer_controller import (SteerController, Hook, generate, encode_goal)
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def base_fluency(model, tok, instr, text):
    """Mean per-token logprob of `text` as a continuation, under the UNSTEERED LLM.
    Low = the base model finds the text unlikely (degenerate/gamed)."""
    if not text.strip():
        return float("nan")
    msgs = [{"role": "user", "content": instr}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pids = tok(chat, return_tensors="pt").input_ids.to(model.device)
    cids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    full = torch.cat([pids, cids], 1)
    out = model(full)
    logp = torch.log_softmax(out.logits[0, :-1], -1)
    tgt = full[0, 1:]
    sel = logp[torch.arange(len(tgt)), tgt]
    cont = sel[pids.shape[1] - 1:]
    return float(cont.mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--ckpt", default="/home/pokazge/checkpoints/steer_ctrl.pt")
    p.add_argument("--layer_idx", type=int, default=14)
    p.add_argument("--cat", default="contrast")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=456)  # match held-out eval seed
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

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    controller = SteerController(d_llm=model.config.hidden_size, d=a["d"], K=a["K"],
                                   max_steer=a["max_steer"]).to(device)
    controller.load_state_dict(ck["controller"]); controller.eval()
    print(f"[inspect] ckpt best_heldout_steered={ck.get('best_heldout_steered')}", flush=True)
    hook = Hook(controller)
    model.model.layers[args.layer_idx - 1].register_forward_hook(hook)

    rng = np.random.default_rng(args.seed)
    nb = ns = 0
    flu_b, flu_s = [], []
    for i in range(args.n):
        _, _, instr, check, _ = gdg.make_goal(rng, category=args.cat)
        z = encode_goal(instr, enc_tok, enc_model, device)
        b_txt, _, _ = generate(model, tok, instr, z, hook, args.max_new_tokens, 0.0, steer=False, grad=False)
        s_txt, _, _ = generate(model, tok, instr, z, hook, args.max_new_tokens, 0.0, steer=True, grad=False)
        bok, sok = bool(check(b_txt)), bool(check(s_txt))
        nb += bok; ns += sok
        fb, fs = base_fluency(model, tok, instr, b_txt), base_fluency(model, tok, instr, s_txt)
        flu_b.append(fb); flu_s.append(fs)
        print(f"\n=== {i} GOAL: {instr}", flush=True)
        print(f"  BASE  [{'PASS' if bok else 'fail'} flu={fb:.2f}]: {b_txt[:200]!r}", flush=True)
        print(f"  STEER [{'PASS' if sok else 'fail'} flu={fs:.2f}]: {s_txt[:200]!r}", flush=True)
    print(f"\n[inspect] pass: base {nb}/{args.n}  steered {ns}/{args.n}", flush=True)
    print(f"[inspect] base-LLM fluency (mean logprob): base={np.nanmean(flu_b):.2f}  "
          f"steered={np.nanmean(flu_s):.2f}  (large drop => steered text is degenerate/gamed)", flush=True)


if __name__ == "__main__":
    main()
