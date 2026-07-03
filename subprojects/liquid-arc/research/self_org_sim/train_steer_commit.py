"""Commitment-to-ultimate-goal steering against multi-turn DRIFT.

Distinction the prior multi-turn test missed: the context window holds the goal (MEMORY) but
that does not keep the goal the behavioral DRIVER (COMMITMENT). Over a few turns the LLM drifts
toward recent context even though the goal is still in its window.

Setup: one ULTIMATE goal G stated once at turn 1, then several TANGENT turns that do NOT restate
G. The controller is fed G (constant) as the commitment target — NOT the per-turn message. Metric:
per-turn commitment = "is the reply still serving G, or did it follow the tangent?". The base LLM
should DRIFT over turns; the G-anchored controller should hold. Slow channel = stateful commitment
register holding G at constant salience (vs stateless in_goal(G) per turn).
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_steer_controller import SteerController, Hook, MultiHook, base_fluency, encode_goal
from train_steer_mt import generate_mt, generate_mt_nosteer
from train_steer_semantic import judge_reward
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

COMMIT_GOALS = [
    "write a cover letter for a software engineering job",
    "plan a 3-day trip to Kyoto",
    "outline a 10-minute presentation about climate change",
    "draft a polite email asking my landlord to fix the heating",
    "build a one-week study schedule for final exams",
    "write a short story about a lighthouse keeper",
    "design a beginner workout plan for someone with knee pain",
    "write a resignation letter giving two weeks notice",
    "plan a healthy weekly meal prep for one person",
    "outline a business plan for a small coffee shop",
    "write a thank-you speech for a retirement party",
    "create a packing checklist for a week-long camping trip",
    "draft a cancellation email for a gym membership",
    "design a study plan to learn basic Spanish in a month",
    "write a bedtime story about a brave little robot",
    "plan a surprise anniversary dinner at home",
    "outline a tutorial on how to change a bike tire",
    "write a LinkedIn summary for a recent marketing graduate",
    "create a morning routine for better productivity",
    "draft a complaint letter about a defective product",
    "plan a small vegetable garden for a backyard",
    "create a budget for a small wedding",
    "write a product description for a reusable water bottle",
    "plan a birthday party for a 7-year-old",
]
TANGENTS = [
    "Wait, what's a fun fact about octopuses?",
    "Random question — what's the capital of New Zealand?",
    "Tell me a quick joke.",
    "What's the best programming language, in your opinion?",
    "How do birds fly, exactly?",
    "What should I have for dinner tonight?",
    "Who won the World Cup in 2018?",
    "Can you recommend a good movie?",
    "How far away is the moon?",
    "What's your favorite color?",
    "Explain how Wi-Fi works.",
    "What's a good book to read this summer?",
]


def make_commit(rng, goals, n_tangent):
    goal = rng.choice(goals)
    intro = f"I need your help with one task: {goal}. Let's begin — give me a first step."
    idx = rng.choice(len(TANGENTS), size=n_tangent, replace=False)
    turns = [intro] + [TANGENTS[i] for i in idx]
    return goal, turns


def run_commit(model, tok, controller, hook, enc_tok, enc_model, goal, turns, device, args, steer, grad):
    """Returns (per_tangent_commitment_scores, logps, mean_fluency, min_n_new)."""
    zG = encode_goal(goal, enc_tok, enc_model, device)        # ULTIMATE goal, constant
    if steer:
        controller.reset_episode(1, device)
        if controller.use_slow:
            controller.slow_step(zG.unsqueeze(0))             # anchor commitment register to G
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]
    jq = (f"Is the assistant's reply still actively working on the original task ('{goal}') — "
          f"continuing it or steering back to it — rather than just answering the off-topic message?")
    messages = []
    scores, gsims, logps, flus, n_news = [], [], [], [], []
    for ti, u in enumerate(turns):
        messages.append({"role": "user", "content": u})
        if steer:
            if controller.use_slow:
                controller.slow_step(zG.unsqueeze(0))         # persist G (delta~0 -> no transition)
            text, logp, n_new = generate_mt(model, tok, messages, zG, hook, args.max_new_tokens,
                                              args.temperature if grad else 0.0, grad)
        else:
            hook.active = False
            text, logp, n_new = generate_mt_nosteer(model, tok, messages, args.max_new_tokens)
        messages.append({"role": "assistant", "content": text})
        if ti >= 1:                                            # commitment judged on TANGENT turns
            instr = f"Original task the assistant must keep serving: {goal}\nOff-topic user message: {u}"
            scores.append(judge_reward(model, tok, instr, text, jq, yes_id, no_id))
            # HONEST trajectory signal: embedding alignment of the reply to the ULTIMATE goal G.
            # On-tangent reply -> low; redirect-to-goal reply -> high. Ungameable by yes/no noise.
            zr = encode_goal(text, enc_tok, enc_model, model.device)
            gsims.append(float((zr * zG).sum().item()))        # cos (both bge-normalized)
            flus.append(base_fluency(model, tok, u, text)); n_news.append(n_new)
            if logp is not None:
                logps.append(logp)
    return scores, gsims, logps, float(np.mean(flus)) if flus else 0.0, (min(n_news) if n_news else 0)


def eval_commit(model, tok, controller, hook, enc_tok, enc_model, goals, n, rng, device, args, steer):
    controller.eval()
    per = None; gs_all = []; flus = []
    for _ in range(n):
        goal, turns = make_commit(rng, goals, args.n_tangent)
        sc, gs, _, flu, _ = run_commit(model, tok, controller, hook, enc_tok, enc_model, goal, turns,
                                         device, args, steer, grad=False)
        a = np.array(sc); per = a if per is None else per + a; gs_all.append(np.mean(gs)); flus.append(flu)
    controller.train()
    return per / n, float(np.mean(gs_all)), float(np.mean(flus))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--layer_idx", type=int, default=14)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--max_steer", type=float, default=14.0)
    p.add_argument("--use_slow", action="store_true")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=150)
    p.add_argument("--group", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=30)
    p.add_argument("--eval_n", type=int, default=12)
    p.add_argument("--n_tangent", type=int, default=4)
    p.add_argument("--min_len", type=int, default=10)
    p.add_argument("--lambda_flu", type=float, default=1.0)
    p.add_argument("--ref_flu", type=float, default=-1.3)
    p.add_argument("--beta_mag", type=float, default=0.03)
    p.add_argument("--n_test_goals", type=int, default=3)
    p.add_argument("--reward", default="gsim", choices=["gsim", "judge"],
                   help="gsim=honest goal-embedding alignment (default); judge=noisy 1.5B yes/no")
    p.add_argument("--inject_layers", default="", help="ENFORCER: comma-sep 1-based layers, e.g. 8,14,20,26")
    p.add_argument("--gen_dtype", default="float16", choices=["float16", "bfloat16"],
                   help="bfloat16 for Qwen3/large MoE models (fp16 breaks MoE routing)")
    p.add_argument("--out_init_std", type=float, default=0.002,
                   help="steer output init scale; use ~0.0003 for 30B (late layers output-sensitive)")
    p.add_argument("--rel_steer", type=float, default=0.0,
                   help=">0: cap |steer| to this fraction of local residual norm (auto-calibrate; ~0.1 for 30B)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="/home/pokazge/checkpoints/steer_commit.pt")
    args = p.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.set_float32_matmul_precision("high")
    tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    gdtype = torch.bfloat16 if args.gen_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, dtype=gdtype, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map={"": 0}).eval()
    print(f"[commit] gen={args.gen_model} dtype={gdtype} d={model.config.hidden_size} "
          f"layers={model.config.num_hidden_layers}", flush=True)
    for pp in model.parameters():
        pp.requires_grad = False
    enc_tok = AutoTokenizer.from_pretrained(args.enc_model)
    enc_model = AutoModel.from_pretrained(args.enc_model).to(device).eval()
    inject = [int(x) for x in args.inject_layers.split(",")] if args.inject_layers else [args.layer_idx]
    controller = SteerController(d_llm=model.config.hidden_size, d=args.d, K=args.K,
                                   max_steer=args.max_steer, use_slow=args.use_slow,
                                   n_inject=len(inject), out_init_std=args.out_init_std).to(device)
    print(f"[commit] controller {sum(p.numel() for p in controller.parameters()):,} params, "
          f"use_slow={args.use_slow}, inject_layers={inject}", flush=True)
    rel = args.rel_steer if args.rel_steer > 0 else None
    if len(inject) > 1:
        hook = MultiHook(controller, inject, rel_frac=rel); hook.register(model)   # ENFORCER: multi-layer
    else:
        hook = Hook(controller, rel_frac=rel); model.model.layers[inject[0] - 1].register_forward_hook(hook)
    opt = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=0.0)
    train_goals = COMMIT_GOALS[:-args.n_test_goals]; test_goals = COMMIT_GOALS[-args.n_test_goals:]
    print(f"[commit] held-out goals={test_goals}", flush=True)
    rng = np.random.default_rng(args.seed); baseline = deque(maxlen=64)

    def run_eval(tag):
        b, bg, bf = eval_commit(model, tok, controller, hook, enc_tok, enc_model, test_goals, args.eval_n,
                                  np.random.default_rng(456), device, args, steer=False)
        s, sg, sf = eval_commit(model, tok, controller, hook, enc_tok, enc_model, test_goals, args.eval_n,
                                  np.random.default_rng(456), device, args, steer=True)
        bs = " ".join(f"{x:.2f}" for x in b); ss = " ".join(f"{x:.2f}" for x in s)
        print(f"[eval {tag}] HELD-OUT 1.5Bjudge base[{bs}]->steer[{ss}] (mean {b.mean():.3f}->{s.mean():.3f})  "
              f"GOAL-SIM base={bg:.3f}->steer={sg:.3f}  flu {bf:.2f}->{sf:.2f}", flush=True)
        return sg if args.reward == "gsim" else float(s.mean())

    best = run_eval("init"); csum = 0.0; cn = 0
    for step in range(1, args.max_steps + 1):
        opt.zero_grad(); rsum = 0.0
        for _ in range(args.group):
            goal, turns = make_commit(rng, train_goals, args.n_tangent)
            sc, gs, logps, flu, min_n = run_commit(model, tok, controller, hook, enc_tok, enc_model,
                                                     goal, turns, device, args, steer=True, grad=True)
            if not logps:
                continue
            # HONEST reward = goal-trajectory alignment (gsim); judge mode kept for comparison
            commit = float(np.mean(gs)) if args.reward == "gsim" else float(np.mean(sc))
            len_ok = 1.0 if min_n >= args.min_len else 0.0
            R = commit * len_ok - args.lambda_flu * max(0.0, args.ref_flu - flu)
            b = np.mean(baseline) if baseline else 0.0; baseline.append(R); adv = R - b
            total_logp = torch.stack(logps).sum()
            mag = controller.mag_penalty(); mag_term = args.beta_mag * mag if mag is not None else 0.0
            ((-adv * total_logp + mag_term) / args.group * 256.0).backward()
            rsum += commit
        for pp in controller.parameters():
            if pp.grad is not None:
                pp.grad /= 256.0
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        opt.step(); csum += rsum / args.group; cn += 1
        if step % 10 == 0:
            print(f"step {step:>4}  train_commit(roll)={csum/cn:.3f}", flush=True); csum = 0.0; cn = 0
        if step % args.eval_every == 0:
            mc = run_eval(f"s{step}")
            if mc > best:
                best = mc
                torch.save({"controller": controller.state_dict(), "args": vars(args),
                             "best_heldout_commit": best}, args.output)
                print(f"[commit] saved (held-out commit {best:.3f}) -> {args.output}", flush=True)
    print(f"[commit] DONE best held-out commit={best:.3f}", flush=True)
    print("[commit] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
