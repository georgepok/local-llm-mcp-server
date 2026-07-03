"""Trajectory tracking: the LLM PLANS the path, the Liquid FOLLOWS it.

Separation of planning and control. The LLM (planner) emits an ordered N-step plan toward the
ultimate goal G — the optimal trajectory, which the Liquid can't compute itself. The Liquid
(tracking controller) holds the CURRENT WAYPOINT as its reference, steers the LLM's generation
toward it, and ADVANCES the pointer when a waypoint is achieved — keeping the LLM on the
trajectory toward G against per-turn drift (tangents).

Core claim under test: a concrete LLM-planned WAYPOINT is a far better steering reference than
the abstract ultimate goal (--target waypoint vs goal vs base). Reward = trajectory PROGRESS
(how far along the plan the Liquid keeps the LLM advancing).
"""
import argparse
import re
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_steer_controller import SteerController, Hook, MultiHook, base_fluency, encode_goal
from train_steer_mt import generate_mt, generate_mt_nosteer
from train_steer_commit import COMMIT_GOALS, TANGENTS
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def split_answer(text):
    """Reasoning models emit <think>...</think>answer. Return the answer (after the last
    </think>), or the full text if there's no think tag."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def _template(tok, messages, think):
    """apply_chat_template with optional enable_thinking (Qwen3); fall back for other models."""
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=think)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def elicit_plan(model, tok, goal, n_steps, device, max_new=200):
    """LLM (the planner) produces an ordered N-step plan toward the goal. Thinking OFF for a
    clean, fast, parseable plan (the plan is the artifact; reasoning is used during execution)."""
    msg = [{"role": "user", "content":
            f"I want to {goal}. List exactly {n_steps} concrete steps to accomplish this, in order. "
            f"Reply with ONLY the numbered steps (1., 2., ...), one per line, each a short phrase."}]
    chat = _template(tok, msg, think=False)
    enc = tok(chat, return_tensors="pt").to(device)
    out = model.generate(enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=max_new,
                           do_sample=False, pad_token_id=tok.pad_token_id)
    text = split_answer(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
    steps = []
    for line in text.split("\n"):
        m = re.match(r"^\s*\d+[.)]\s*(.+)", line)
        if m and len(m.group(1).strip()) > 3:
            steps.append(m.group(1).strip())
    return steps[:n_steps]


def run_traj(model, tok, controller, hook, enc_tok, enc_model, goal, way_txt, way_emb, zG, device,
               args, mode, grad):
    """Walk the conversation. mode: 'waypoint' steer to current waypoint, 'goal' steer to G,
    'base' no steer. Pointer advances when the response matches the current waypoint (cos>thr).
    Returns (progress_fraction, max_pointer, logps, mean_fluency, min_n_new)."""
    steer = (mode != "base")
    if steer:
        controller.reset_episode(1, device)
    rng = np.random.default_rng()
    n_way = len(way_emb)
    p = 0
    # interleave: intro, then alternating execute / tangent turns
    intro = f"I want to {goal}. Let's work through it together, step by step. What should we do, and let's begin."
    turns = [intro]
    td = list(rng.permutation(len(TANGENTS)))
    for k in range(args.n_turns - 1):
        turns.append("Okay, continue." if k % 2 == 0 else TANGENTS[td[k % len(td)]])
    messages, logps, flus, n_news = [], [], [], []
    for ti, u in enumerate(turns):
        messages.append({"role": "user", "content": u})
        wp = way_emb[min(p, n_way - 1)]
        # goal and waypoints COMPLEMENT: the goal is the terminal anchor (slow channel), the
        # current waypoint is the proximal reference (fast). 'goal'/'waypoint' = single-ref ablations.
        if mode == "goal":
            slow_ref, fast_ref = zG, zG
        elif mode == "waypoint":
            slow_ref, fast_ref = wp, wp
        else:  # trajectory: G anchors (slow), waypoint guides (fast)
            slow_ref, fast_ref = zG, wp
        think = None if args.think else False
        if steer:
            if controller.use_slow:
                controller.slow_step(slow_ref.unsqueeze(0))
            text, logp, n_new = generate_mt(model, tok, messages, fast_ref, hook, args.max_new_tokens,
                                              args.temperature if grad else 0.0, grad, think=think)
        else:
            hook.active = False
            text, logp, n_new = generate_mt_nosteer(model, tok, messages, args.max_new_tokens, think=think)
        messages.append({"role": "assistant", "content": text})
        # advance pointer if the ANSWER (post-</think>) achieved the current waypoint
        zr = encode_goal(split_answer(text), enc_tok, enc_model, device)
        if p < n_way and float((zr * way_emb[p]).sum().item()) > args.advance_thr:
            p += 1
        flus.append(base_fluency(model, tok, u, text)); n_news.append(n_new)
        if logp is not None:
            logps.append(logp)
    return p / n_way, p, logps, float(np.mean(flus)), (min(n_news) if n_news else 0)


def build_plans(model, tok, enc_tok, enc_model, goals, n_steps, device):
    plans = {}
    for g in goals:
        steps = elicit_plan(model, tok, g, n_steps, device)
        if len(steps) < 2:
            continue
        emb = torch.stack([encode_goal(s, enc_tok, enc_model, device) for s in steps])  # [N,384]
        plans[g] = (steps, emb)
        print(f"[traj] plan[{g[:40]}]: {steps}", flush=True)
    return plans


def eval_traj(model, tok, controller, hook, enc_tok, enc_model, plans, goals, n, rng, device, args, mode):
    controller.eval()
    progs, flus = [], []
    for _ in range(n):
        g = goals[rng.integers(len(goals))]
        if g not in plans:
            continue
        wtxt, wemb = plans[g]; zG = encode_goal(g, enc_tok, enc_model, device)
        prog, _, _, flu, _ = run_traj(model, tok, controller, hook, enc_tok, enc_model, g, wtxt, wemb,
                                        zG, device, args, mode, grad=False)
        progs.append(prog); flus.append(flu)
    controller.train()
    return float(np.mean(progs)) if progs else 0.0, float(np.mean(flus)) if flus else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--gen_dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--d", type=int, default=128); p.add_argument("--K", type=int, default=4)
    p.add_argument("--max_steer", type=float, default=14.0); p.add_argument("--use_slow", action="store_true")
    p.add_argument("--inject_layers", default="8,14,20,26"); p.add_argument("--rel_steer", type=float, default=0.0)
    p.add_argument("--out_init_std", type=float, default=0.002)
    p.add_argument("--target", default="trajectory",
                   choices=["trajectory", "waypoint", "goal", "base"],
                   help="trajectory=goal(slow)+waypoint(fast) COMPLEMENT; waypoint/goal=single-ref ablations")
    p.add_argument("--n_steps", type=int, default=4); p.add_argument("--n_turns", type=int, default=5)
    p.add_argument("--advance_thr", type=float, default=0.62)
    p.add_argument("--think", action="store_true", help="keep reasoning CoT during execution (slow); default off")
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--max_steps", type=int, default=120)
    p.add_argument("--group", type=int, default=3); p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int, default=55); p.add_argument("--eval_every", type=int, default=30)
    p.add_argument("--eval_n", type=int, default=12); p.add_argument("--min_len", type=int, default=10)
    p.add_argument("--lambda_flu", type=float, default=1.0); p.add_argument("--ref_flu", type=float, default=-1.3)
    p.add_argument("--beta_mag", type=float, default=0.03); p.add_argument("--n_test_goals", type=int, default=3)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--output", default="/home/pokazge/checkpoints/steer_traj.pt")
    args = p.parse_args()

    device = torch.device("cuda"); torch.manual_seed(args.seed)
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
    inject = [int(x) for x in args.inject_layers.split(",")] if args.inject_layers else [14]
    controller = SteerController(d_llm=model.config.hidden_size, d=args.d, K=args.K, max_steer=args.max_steer,
                                   use_slow=args.use_slow, n_inject=len(inject), out_init_std=args.out_init_std).to(device)
    print(f"[traj] controller {sum(p.numel() for p in controller.parameters()):,} target={args.target} "
          f"inject={inject} rel={args.rel_steer}", flush=True)
    rel = args.rel_steer if args.rel_steer > 0 else None
    if len(inject) > 1:
        hook = MultiHook(controller, inject, rel_frac=rel); hook.register(model)
    else:
        hook = Hook(controller, rel_frac=rel); model.model.layers[inject[0] - 1].register_forward_hook(hook)
    opt = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=0.0)

    train_goals = COMMIT_GOALS[:-args.n_test_goals]; test_goals = COMMIT_GOALS[-args.n_test_goals:]
    print("[traj] eliciting plans from the LLM (the planner)...", flush=True)
    plans = build_plans(model, tok, enc_tok, enc_model, COMMIT_GOALS, args.n_steps, device)
    rng = np.random.default_rng(args.seed); baseline = deque(maxlen=64)

    def run_eval(tag):
        pb, fb = eval_traj(model, tok, controller, hook, enc_tok, enc_model, plans, test_goals, args.eval_n,
                             np.random.default_rng(456), device, args, "base")
        pg, _ = eval_traj(model, tok, controller, hook, enc_tok, enc_model, plans, test_goals, args.eval_n,
                            np.random.default_rng(456), device, args, "goal")
        pw, _ = eval_traj(model, tok, controller, hook, enc_tok, enc_model, plans, test_goals, args.eval_n,
                            np.random.default_rng(456), device, args, "waypoint")
        pt, ft = eval_traj(model, tok, controller, hook, enc_tok, enc_model, plans, test_goals, args.eval_n,
                            np.random.default_rng(456), device, args, "trajectory")
        print(f"[eval {tag}] HELD-OUT plan-PROGRESS  base={pb:.3f}  goal-only={pg:.3f}  "
              f"waypoint-only={pw:.3f}  TRAJECTORY(both)={pt:.3f}  (flu base={fb:.2f} traj={ft:.2f})", flush=True)
        return {"trajectory": pt, "waypoint": pw, "goal": pg}.get(args.target, pt)

    best = run_eval("init"); csum = 0.0; cn = 0
    for step in range(1, args.max_steps + 1):
        opt.zero_grad(); rsum = 0.0
        for _ in range(args.group):
            g = train_goals[rng.integers(len(train_goals))]
            if g not in plans:
                continue
            wtxt, wemb = plans[g]; zG = encode_goal(g, enc_tok, enc_model, device)
            prog, _, logps, flu, min_n = run_traj(model, tok, controller, hook, enc_tok, enc_model, g, wtxt,
                                                    wemb, zG, device, args, args.target, grad=True)
            if not logps:
                continue
            len_ok = 1.0 if min_n >= args.min_len else 0.0
            R = prog * len_ok - args.lambda_flu * max(0.0, args.ref_flu - flu)
            b = np.mean(baseline) if baseline else 0.0; baseline.append(R); adv = R - b
            total_logp = torch.stack(logps).sum()
            mag = controller.mag_penalty(); mag_term = args.beta_mag * mag if mag is not None else 0.0
            ((-adv * total_logp + mag_term) / args.group * 256.0).backward()
            rsum += prog
        for pp in controller.parameters():
            if pp.grad is not None:
                pp.grad /= 256.0
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        opt.step(); csum += rsum / args.group; cn += 1
        if step % 10 == 0:
            print(f"step {step:>4}  train_progress(roll)={csum/cn:.3f}", flush=True); csum = 0.0; cn = 0
        if step % args.eval_every == 0:
            mc = run_eval(f"s{step}")
            if mc > best:
                best = mc
                torch.save({"controller": controller.state_dict(), "args": vars(args), "best": best}, args.output)
                print(f"[traj] saved (progress {best:.3f}) -> {args.output}", flush=True)
    print(f"[traj] DONE best={best:.3f}", flush=True); print("[traj] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
