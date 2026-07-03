"""REALIZATION — install the gated native goal-field as a runtime steer on the live 30B.

The operator (validated in rep-space) actuated into the LLM forward pass:
  goal-field  = native basin cos(h, z_G)  -> pull direction native_pull(h, z_G)   [layer 36]
  focus gate  = LiquidGate -> kappa_t per turn (persistent state across turns = mission)
  steer       = forward hook at layer 36 adds  rel_frac * ||h|| * tanh(kappa/kappa0) * native_pull
                (direction NATIVE, magnitude gate-modulated + relative-capped: quiet on-task,
                 strong-but-bounded on drift, never dominates the current sub-task / breaks fluency)

Validate: held-out goals under drift, BASE vs STEERED, on-goal retention = cos(response_rep, z_G)
in the field's own layer-36 space, plus transcripts. Steered should stay closer to the goal.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from engagement_gate import LiquidGate, native_pull
from train_steer_traj import split_answer
from task_goals import task_goals
from train_steer_commit import TANGENTS
from transformers import AutoModelForCausalLM, AutoTokenizer


def _template(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class FieldSteer:
    """Forward hook at layer L: add gate-modulated native-field pull to the last residual position."""
    def __init__(self):
        self.active = False; self.z = None; self.mag = 0.0

    def __call__(self, module, inp, out):
        if not self.active or self.z is None:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        h_last = hs[:, -1].float()
        pull = native_pull(h_last, self.z.unsqueeze(0))
        corr = self.mag * h_last.norm(dim=-1, keepdim=True) * pull
        hs[:, -1] = hs[:, -1] + corr.to(hs.dtype)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_model", default="/home/pokazge/models/Qwen3-30B-A3B")
    ap.add_argument("--gate", default="/home/pokazge/checkpoints/liquid_gate.pt")
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--rel_frac", type=float, default=0.15)
    ap.add_argument("--kappa0", type=float, default=4.0)
    ap.add_argument("--n_goals", type=int, default=4)
    ap.add_argument("--n_turns", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=50)
    ap.add_argument("--sustained", action="store_true", help="every turn is a tangent (real drift headroom)")
    ap.add_argument("--keep_turns", type=int, default=0, help=">0 keeps only last K exchanges: goal drops out of context (substrate regime)")
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
    test_goals = ck["ho_goals"][:args.n_goals]
    goal_text = task_goals()

    steer = FieldSteer()
    model.model.layers[args.layer - 1].register_forward_hook(steer)

    @torch.no_grad()
    def gen(msgs, on):
        steer.active = on
        enc = tok(_template(tok, msgs), return_tensors="pt").to(dev)
        out = model.generate(enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=args.max_new,
                               do_sample=False, pad_token_id=tok.pad_token_id)
        steer.active = False
        return split_answer(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))

    @torch.no_grad()
    def rep(text):
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        if ids.shape[1] == 0:
            ids = tok(".", return_tensors="pt").input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states[args.layer][0]
        return F.normalize(hs.float().mean(0), dim=-1)

    @torch.no_grad()
    def hidden_last(msgs):
        enc = tok(_template(tok, msgs), return_tensors="pt").to(dev)
        return model(enc.input_ids, output_hidden_states=True).hidden_states[args.layer][:, -1].float()

    rng = np.random.default_rng(7)
    base_c, steer_c = [], []
    for gi in test_goals:
        g = goal_text[gi]; z = zG[gi]
        gate.reset(1, dev)
        print(f"\n=== GOAL[{gi}]: {g}", flush=True)
        history = [{"role": "user", "content": f"Help me with this task: {g}. We'll go step by step."}]
        td = list(rng.permutation(len(TANGENTS)))
        for t in range(args.n_turns):
            drift = TANGENTS[td[t % len(td)]] if args.sustained else \
                    ("Okay, what's next?" if t % 2 == 0 else TANGENTS[td[t % len(td)]])
            # truncate context: keep only the last K exchanges -> the GOAL falls out of the window
            # (the substrate's real regime: model has lost the goal; the field holds z_G externally)
            ctx_hist = history[-2 * args.keep_turns:] if args.keep_turns > 0 else history
            msgs = ctx_hist + [{"role": "user", "content": drift}]
            rb = gen(msgs, False)
            kappa = float(gate.step(hidden_last(msgs), z.unsqueeze(0)))           # focus gain (persistent = mission)
            steer.z = z; steer.mag = args.rel_frac * float(np.tanh(kappa / args.kappa0))
            rl = gen(msgs, True)
            cb = float((rep(rb) * z).sum()); cl = float((rep(rl) * z).sum())
            base_c.append(cb); steer_c.append(cl)
            print(f"  turn{t} drift='{drift[:32]}' kappa={kappa:.2f} mag={steer.mag:.3f}", flush=True)
            print(f"     BASE  [cos→goal {cb:.3f}]: {rb[:130]!r}", flush=True)
            print(f"     STEER [cos→goal {cl:.3f}]: {rl[:130]!r}", flush=True)
            history += [{"role": "user", "content": drift}, {"role": "assistant", "content": rl}]
    print(f"\n[realize] MEAN cos→goal  BASE={np.mean(base_c):.3f}  STEERED={np.mean(steer_c):.3f}  "
          f"(Δ={np.mean(steer_c) - np.mean(base_c):+.3f})  n={len(base_c)}", flush=True)
    print("[realize] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
