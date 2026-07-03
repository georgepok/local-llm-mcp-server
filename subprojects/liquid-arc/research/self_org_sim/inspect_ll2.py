"""Inspect Liquid-LoRA v2: on held-out goals under drift, show base vs Liquid-LoRA responses +
on-waypoint cos, to verify the +cos win is real (the adapter keeps the model on the current
step where base follows the tangent)."""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_steer_controller import SteerController, encode_goal
from train_steer_traj import split_answer
from train_liquid_lora2 import LiquidLoRA, _template, build_plans
from train_steer_commit import COMMIT_GOALS, TANGENTS
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class A:
    n_turns = 4; think = False; max_new_tokens = 45


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--gen_dtype", default="float16")
    p.add_argument("--ckpt", default="/home/pokazge/checkpoints/ll2cap2.pt")
    p.add_argument("--n_goals", type=int, default=3)
    args = p.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True); torch.set_float32_matmul_precision("high")
    tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    gd = torch.bfloat16 if args.gen_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.gen_model, dtype=gd, trust_remote_code=True,
                                                   low_cpu_mem_usage=True, device_map={"": 0}).eval()
    for pp in model.parameters():
        pp.requires_grad = False
    enc_tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    enc_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(device).eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False); a = ck["args"]
    ctrl = SteerController(d_llm=model.config.hidden_size, d=a["d"], K=a["K"], use_slow=a.get("use_slow", True), n_inject=1).to(device)
    ctrl.load_state_dict(ck["controller"], strict=False); ctrl.eval()  # strict=False: pre-g_head ckpts
    layers = [int(x) for x in a["lora_layers"].split(",")]; projs = [s.strip() for s in a["lora_proj"].split(",")]
    lora = LiquidLoRA(model, layers, projs, d_ctrl=a["d"]).to(device)
    lora.cap_rel = a.get("cap_rel", 0.5) or None
    lora.load_state_dict(ck["lora"]); lora.register(); lora.eval()
    print(f"[i2] ckpt best={ck.get('best')} layers={layers} projs={projs} K={a['K']}", flush=True)

    test_goals = COMMIT_GOALS[-a.get("n_test_goals", 3):]
    plans = build_plans(model, tok, enc_tok, enc_model, test_goals, a["n_steps"], device)
    rng = np.random.default_rng(7)

    @torch.no_grad()
    def gen(msgs, use_lora):
        lora.active = use_lora
        chat = _template(tok, msgs, None if A.think else False)
        enc = tok(chat, return_tensors="pt").to(device)
        out = model.generate(enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=A.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        lora.active = False
        return split_answer(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))

    base_hit, lora_hit = [], []
    for g in test_goals[:args.n_goals]:
        if g not in plans:
            continue
        wtxt, wemb = plans[g]
        zG = encode_goal(g, enc_tok, enc_model, device)
        print(f"\n=== GOAL: {g}\n    plan={wtxt}", flush=True)
        ctrl.reset_episode(1, device); ctrl.slow_step(zG.unsqueeze(0))
        td = list(rng.permutation(len(TANGENTS)))
        history = [{"role": "user", "content": f"Help me with this task: {g}. We'll go step by step."}]
        for pi in range(min(A.n_turns, len(wemb))):
            drift = "Okay, what's next?" if pi % 2 == 0 else TANGENTS[td[pi % len(td)]]
            msgs = history + [{"role": "user", "content": drift}]
            rb = gen(msgs, False)
            h_llm = None
            if a.get("closed_loop"):
                chat = _template(tok, msgs, None if A.think else False)
                enc = tok(chat, return_tensors="pt").to(device)
                lora.active = False
                with torch.no_grad():
                    h_llm = model(enc.input_ids, output_hidden_states=True).hidden_states[a.get("obs_layer", -1)][:, -1].float()
            h = ctrl.dyn_state(wemb[pi].unsqueeze(0), h_llm); lora.set_state(h)
            rl = gen(msgs, True)
            eb = encode_goal(rb, enc_tok, enc_model, device)
            el = encode_goal(rl, enc_tok, enc_model, device)
            cb = float((eb * wemb[pi]).sum()); cl = float((el * wemb[pi]).sum())
            nb = int((wemb @ eb).argmax()); nl = int((wemb @ el).argmax())  # nearest waypoint (precision)
            base_hit.append(nb == pi); lora_hit.append(nl == pi)
            print(f"  step{pi} waypoint='{wtxt[pi][:45]}'  user(drift)='{drift[:40]}'", flush=True)
            print(f"     BASE [cos {cb:.2f} | nearest=step{nb}{'=HIT' if nb==pi else ' MISS'}]: {rb[:140]!r}", flush=True)
            print(f"     LORA [cos {cl:.2f} | nearest=step{nl}{'=HIT' if nl==pi else ' MISS'}]: {rl[:140]!r}", flush=True)
            history += [{"role": "user", "content": drift}, {"role": "assistant", "content": rl}]
    print(f"[i2] STEP-PRECISION (nearest waypoint == current step): "
          f"BASE {sum(base_hit)}/{len(base_hit)}={np.mean(base_hit):.2f}  "
          f"LORA {sum(lora_hit)}/{len(lora_hit)}={np.mean(lora_hit):.2f}", flush=True)
    print("[i2] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
