"""Validate commitment steering: read base vs steered transcripts on held-out goals with
tangents, and score per-turn commitment with the INDEPENDENT 7B judge (vs the 1.5B trainer).
Confirms whether steering-to-G genuinely keeps the assistant serving the ultimate goal against
drift, or whether the 1.5B commitment number is judge noise."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_steer_controller import SteerController, Hook, MultiHook, base_fluency, encode_goal
from train_steer_mt import generate_mt, generate_mt_nosteer
from train_steer_semantic import judge_reward
from train_steer_commit import make_commit, COMMIT_GOALS
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def commit_run(model, tok, controller, hook, enc_tok, enc_model, goal, turns, device, steer,
                 max_new, jmodel, jtok, y1, n1, y7, n7, do_print):
    zG = encode_goal(goal, enc_tok, enc_model, device)
    if steer:
        controller.reset_episode(1, device)
        if controller.use_slow:
            controller.slow_step(zG.unsqueeze(0))
    jq = (f"Is the assistant's reply still actively working on the original task ('{goal}') — "
          f"continuing it or steering back to it — rather than just answering the off-topic message?")
    messages, s1, s7 = [], [], []
    for ti, u in enumerate(turns):
        messages.append({"role": "user", "content": u})
        if steer:
            if controller.use_slow:
                controller.slow_step(zG.unsqueeze(0))
            text, _, _ = generate_mt(model, tok, messages, zG, hook, max_new, 0.0, grad=False)
        else:
            hook.active = False
            text, _, _ = generate_mt_nosteer(model, tok, messages, max_new)
        messages.append({"role": "assistant", "content": text})
        if ti >= 1:
            instr = f"Original task the assistant must keep serving: {goal}\nOff-topic user message: {u}"
            s1.append(judge_reward(model, tok, instr, text, jq, y1, n1))
            s7.append(judge_reward(jmodel, jtok, instr, text, jq, y7, n7))
            if do_print:
                tag = "STEER" if steer else "BASE "
                print(f"    [{tag} 7B={s7[-1]:.2f}] tangent: {u[:40]!r} -> {text[:150]!r}", flush=True)
    return s1, s7


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--judge_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--ckpt", default="/home/pokazge/checkpoints/commit_slow.pt")
    p.add_argument("--layer_idx", type=int, default=14)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--n_print", type=int, default=3)
    p.add_argument("--n_tangent", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--gen_dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=456)
    args = p.parse_args()
    device = torch.device("cuda")
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
    y1 = tok(" Yes", add_special_tokens=False).input_ids[0]; n1 = tok(" No", add_special_tokens=False).input_ids[0]
    jtok = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    jmodel = AutoModelForCausalLM.from_pretrained(args.judge_model, dtype=torch.float16,
                                                    trust_remote_code=True).to(device).eval()
    y7 = jtok(" Yes", add_special_tokens=False).input_ids[0]; n7 = jtok(" No", add_special_tokens=False).input_ids[0]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False); a = ck["args"]
    inj = [int(x) for x in a.get("inject_layers", "").split(",")] if a.get("inject_layers") else [a.get("layer_idx", 14)]
    controller = SteerController(d_llm=model.config.hidden_size, d=a["d"], K=a["K"],
                                   max_steer=a["max_steer"], use_slow=a.get("use_slow", False),
                                   n_inject=len(inj), out_init_std=a.get("out_init_std", 0.002)).to(device)
    controller.load_state_dict(ck["controller"]); controller.eval()
    rel = a.get("rel_steer", 0.0); rel_frac = rel if rel and rel > 0 else None
    if len(inj) > 1:
        hook = MultiHook(controller, inj, rel_frac=rel_frac); hook.register(model)
    else:
        hook = Hook(controller, rel_frac=rel_frac); model.model.layers[inj[0] - 1].register_forward_hook(hook)
    print(f"[ci] inject_layers={inj} rel_frac={rel_frac}", flush=True)
    print(f"[ci] ckpt={args.ckpt} use_slow={a.get('use_slow')} 1.5B-commit={ck.get('best_heldout_commit')}", flush=True)
    test_goals = COMMIT_GOALS[-a.get("n_test_goals", 3):]
    rng = np.random.default_rng(args.seed)
    b1a, b7a, s1a, s7a = [], [], [], []
    for i in range(args.n):
        goal, turns = make_commit(rng, test_goals, args.n_tangent)
        dop = i < args.n_print
        if dop:
            print(f"\n=== {i} GOAL: {goal}", flush=True)
        b1, b7 = commit_run(model, tok, controller, hook, enc_tok, enc_model, goal, turns, device,
                              False, args.max_new_tokens, jmodel, jtok, y1, n1, y7, n7, dop)
        s1, s7 = commit_run(model, tok, controller, hook, enc_tok, enc_model, goal, turns, device,
                              True, args.max_new_tokens, jmodel, jtok, y1, n1, y7, n7, dop)
        b1a += b1; b7a += b7; s1a += s1; s7a += s7
    print(f"\n[ci] COMMITMENT mean over tangent turns (n={args.n} convs):", flush=True)
    print(f"[ci]   1.5B judge: base={np.mean(b1a):.3f}  steer={np.mean(s1a):.3f}  Δ={np.mean(s1a)-np.mean(b1a):+.3f}", flush=True)
    print(f"[ci]   7B  judge: base={np.mean(b7a):.3f}  steer={np.mean(s7a):.3f}  Δ={np.mean(s7a)-np.mean(b7a):+.3f}", flush=True)
    print("[ci] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
