"""Closed-loop steering PROBE (Phase 0 of the causal-substrate program).

Tests the PREMISE behind making the substrate causal: can we move Qwen's
goal-following by ADDING a 'following direction' to the residual stream during
generation? Every prior experiment was passive readout (predict follow/drift).
This is the first where the intervention has to actually CHANGE the output.

Direction (per category) = mean(hidden | followed) - mean(hidden | drifted),
from existing labeled chunk-hidden data. Injected at the output of decoder layer
(layer_idx-1) at the LAST position every decode step (that hidden state predicts
the next token). Sweep alpha; measure check_fn pass-rate baseline vs steered on
freshly sampled goals (greedy decode so the ONLY variable is the steer).

Go/no-go: if pass-rate rises with alpha, steering is a valid actuator and we
build the substrate-controller. If flat/negative, steering is the wrong lever.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import generate_diverse_goals as gdg


class Steerer:
    """Holds the current steering vector + alpha; a forward hook reads it."""
    def __init__(self):
        self.vec = None      # [d] unit tensor on device, fp16
        self.alpha = 0.0
        self.norms = []      # diagnostic: residual norm at injection site

    def hook(self, module, inputs, output):
        # Qwen2DecoderLayer may return a bare tensor [B,T,d] or a tuple whose
        # [0] is that tensor, depending on transformers version.
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        self.norms.append(float(h[0, -1].float().norm().item()))
        if self.vec is not None and self.alpha != 0.0:
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + self.alpha * self.vec
            return (h,) + tuple(output[1:]) if is_tuple else h
        return output


def build_directions(pack_path, d, device):
    """Per-category following direction from labeled pooled-chunk hidden states."""
    pack = torch.load(pack_path, map_location="cpu", weights_only=False)
    sums = defaultdict(lambda: [torch.zeros(d), torch.zeros(d), 0, 0])  # foll_sum, drift_sum, nf, nd
    norms = []
    for r in pack["records"]:
        T = int(r["T"])
        if T < 4:
            continue
        hs = r["hidden_state_traj"].float()  # [T, d]
        starts = list(r["turn_chunk_starts"])
        followed = list(r["turn_followed"])
        cats = list(r.get("turn_categories", []))
        cur = 0
        for t in range(T):
            while cur + 1 < len(starts) and t >= starts[cur + 1]:
                cur += 1
            cat = cats[cur] if cats else "unk"
            s = sums[cat]
            norms.append(float(hs[t].norm()))
            if followed[cur]:
                s[0] += hs[t]; s[2] += 1
            else:
                s[1] += hs[t]; s[3] += 1
    dirs = {}
    for cat, (fs, ds, nf, nd) in sums.items():
        if nf == 0 or nd == 0:
            continue
        v = (fs / nf) - (ds / nd)
        v = torch.nn.functional.normalize(v, dim=-1)
        dirs[cat] = v.to(device).half()
    mean_norm = float(np.mean(norms))
    return dirs, mean_norm


@torch.no_grad()
def gen(model, tok, instr, device, max_new_tokens=64):
    msgs = [{"role": "user", "content": instr}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(chat, return_tensors="pt").to(device)
    ids = enc.input_ids
    out = model.generate(ids, attention_mask=enc.attention_mask,
                           max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train_pack", default="/home/pokazge/data/hs_train_9cat_judged.pt")
    p.add_argument("--layer_idx", type=int, default=14)
    p.add_argument("--categories", default="contrast,surface,case,structure")
    p.add_argument("--n_goals", type=int, default=20)
    p.add_argument("--coeffs", default="0,0.25,0.5,1.0,2.0",
                   help="alpha as multiples of mean residual norm")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--max_new_tokens", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, dtype=torch.float16, trust_remote_code=True).to(device).eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    d = model.config.hidden_size

    dirs, mean_norm = build_directions(args.train_pack, d, device)
    print(f"[probe] mean pooled-chunk hidden norm = {mean_norm:.1f}", flush=True)
    print(f"[probe] directions for: {sorted(dirs.keys())}", flush=True)

    steerer = Steerer()
    h = model.model.layers[args.layer_idx - 1].register_forward_hook(steerer.hook)

    coeffs = [float(c) for c in args.coeffs.split(",")]
    cats = [c.strip() for c in args.categories.split(",")]
    rng = np.random.default_rng(args.seed)

    # Pre-sample goals per category (same goals across alphas for paired comparison)
    goals_by_cat = {}
    for cat in cats:
        gl = []
        for _ in range(args.n_goals):
            _, gtype, instr, check, _tmpl = gdg.make_goal(rng, category=cat)
            gl.append((instr, check))
        goals_by_cat[cat] = gl

    print(f"\n[probe] === pass-rate by category x alpha (greedy decode) ===", flush=True)
    header = "category        " + "".join(f"  a={c:<5}" for c in coeffs)
    print(header, flush=True)
    overall = defaultdict(lambda: [0, 0])
    for cat in cats:
        if cat not in dirs:
            print(f"{cat}: no direction (skipped)", flush=True)
            continue
        steerer.vec = dirs[cat]
        row = f"{cat:<15}"
        for c in coeffs:
            steerer.alpha = c * mean_norm if c != 0 else 0.0
            npass = 0
            for instr, check in goals_by_cat[cat]:
                txt = gen(model, tok, instr, device, args.max_new_tokens)
                ok = bool(check(txt))
                npass += int(ok)
                overall[c][0] += int(ok); overall[c][1] += 1
            row += f"  {npass/len(goals_by_cat[cat]):.2f} "
        print(row, flush=True)
    steerer.alpha = 0.0
    print("\n[probe] OVERALL pass-rate:", flush=True)
    for c in coeffs:
        n, tot = overall[c]
        print(f"  alpha={c:<5} ({c*mean_norm:6.1f}):  {n/max(1,tot):.3f}  ({n}/{tot})", flush=True)
    print(f"\n[probe] injection-site residual norms: mean={np.mean(steerer.norms):.1f} "
          f"p50={np.percentile(steerer.norms,50):.1f} p95={np.percentile(steerer.norms,95):.1f}",
          flush=True)
    h.remove()
    print("[probe] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
