"""Validate the gated-mission controller: read steered text + score all 4 stages with
BOTH the 1.5B training judge and the INDEPENDENT 7B judge. Confirms whether the mission
COMPLETE=1.0 is genuine ordered completion or 1.5B-stage-judge gaming."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_steer_controller import SteerController, Hook, generate, base_fluency, encode_goal
from train_steer_semantic import judge_reward
from train_steer_mission import make_mission, stage_scores, prefix_reached, TOPICS
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--judge_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--ckpt", default="/home/pokazge/checkpoints/steer_mission.pt")
    p.add_argument("--layer_idx", type=int, default=14)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--n_print", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=110)
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
    test_topics = TOPICS[-a.get("n_test_topics", 6):]
    print(f"[m-insp] ckpt gated={ck.get('best_heldout_gated')}  held-out topics={test_topics}", flush=True)

    rng = np.random.default_rng(args.seed)
    agg = {"b1": np.zeros(4), "s1": np.zeros(4), "b7": np.zeros(4), "s7": np.zeros(4)}
    cb1 = cs1 = cb7 = cs7 = 0
    fb, fs = [], []
    for i in range(args.n):
        instr, qs = make_mission(rng, test_topics)
        z = encode_goal(instr, enc_tok, enc_model, device)
        b_txt, _, _ = generate(model, tok, instr, z, hook, args.max_new_tokens, 0.0, steer=False, grad=False)
        s_txt, _, _ = generate(model, tok, instr, z, hook, args.max_new_tokens, 0.0, steer=True, grad=False)
        b1 = stage_scores(model, tok, instr, b_txt, qs, y1, n1)
        s1 = stage_scores(model, tok, instr, s_txt, qs, y1, n1)
        b7 = [judge_reward(jmodel, jtok, instr, b_txt, q, y7, n7) for q in qs]
        s7 = [judge_reward(jmodel, jtok, instr, s_txt, q, y7, n7) for q in qs]
        agg["b1"] += b1; agg["s1"] += s1; agg["b7"] += b7; agg["s7"] += s7
        cb1 += all(x > 0.5 for x in b1); cs1 += all(x > 0.5 for x in s1)
        cb7 += all(x > 0.5 for x in b7); cs7 += all(x > 0.5 for x in s7)
        fb.append(base_fluency(model, tok, instr, b_txt)); fs.append(base_fluency(model, tok, instr, s_txt))
        if i < args.n_print:
            print(f"\n=== {i}  {instr[:60]}...", flush=True)
            print(f"  BASE  7B-stages[{' '.join(f'{x:.2f}' for x in b7)}]: {b_txt[:260]!r}", flush=True)
            print(f"  STEER 7B-stages[{' '.join(f'{x:.2f}' for x in s7)}]: {s_txt[:260]!r}", flush=True)
    n = args.n
    print(f"\n[m-insp] STAGE MEANS (1.5B train-judge): base[{' '.join(f'{x:.2f}' for x in agg['b1']/n)}] "
          f"steer[{' '.join(f'{x:.2f}' for x in agg['s1']/n)}]", flush=True)
    print(f"[m-insp] STAGE MEANS (7B INDEPENDENT):   base[{' '.join(f'{x:.2f}' for x in agg['b7']/n)}] "
          f"steer[{' '.join(f'{x:.2f}' for x in agg['s7']/n)}]", flush=True)
    print(f"[m-insp] MISSION COMPLETE (all 4>0.5):  1.5B base={cb1/n:.2f} steer={cs1/n:.2f}   "
          f"7B base={cb7/n:.2f} steer={cs7/n:.2f}", flush=True)
    print(f"[m-insp] fluency: base={np.mean(fb):.2f} steer={np.mean(fs):.2f}", flush=True)
    print("[m-insp] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
