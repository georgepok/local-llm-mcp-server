"""Gated-MISSION causal steering — conceptual proof for hierarchical, condition-gated goals.

A mission = ordered subgoals where each stage's conditions must be fulfilled before the
next is meaningful (stage 3 must solve stage 2's specific challenge; stage 4 is the resolved
end-state). The substrate must function like a goal-directed organism: track mission stage,
fulfill the current condition, advance.

Reward is PREFIX-GATED: R = mean_k( product_{j<=k} stage_j ), so stage k earns credit only
if stages 1..k-1 are also satisfied -> conditions truly gate progression (no credit for
jumping ahead). Held-out = unseen topics. Metrics: per-stage scores, prefix REACHED (how far
the organism gets, in order), and MISSION COMPLETION (all stages).

Tests whether the existing flat controller can drive ordered gated progression, or whether
mission-stage tracking (hierarchical slow/fast state) is needed.
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_steer_controller import SteerController, Hook, generate, base_fluency, encode_goal
from train_steer_semantic import judge_reward, TOPICS
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

INSTR = ("Carry out this 4-step mission about {topic}, completing each step before moving to "
         "the next. Step 1: define in one clear sentence what {topic} is. Step 2: identify the "
         "single biggest challenge or risk involved. Step 3: propose a concrete way to overcome "
         "that specific challenge. Step 4: state the end goal you reach once that challenge is "
         "overcome.")
# Stages 3 and 4 DEPEND on the content of 2 and 3 -> genuine condition-gating.
STAGE_QS = [
    "Does the response clearly define what {topic} is?",
    "Does the response identify a specific biggest challenge or risk involved in {topic}?",
    "Does the response propose a concrete way to overcome THAT specific challenge it identified?",
    "Does the response state the end goal reached once that specific challenge is overcome?",
]

# HARD mission: 6 chained-dependent stages under a TIGHT token budget -> forces ruthless
# pacing. Strong models (30B) saturate the 4-step version; this re-creates headroom because
# default verbosity blows the budget before completing all 6 stages in order.
INSTR_HARD = ("Carry out this 6-step mission about {topic}. Use ONE short sentence per step, in "
              "order, and complete all six. Step 1: define {topic}. Step 2: name who most "
              "benefits from it. Step 3: name the single biggest obstacle to it. Step 4: give one "
              "concrete tactic to beat THAT obstacle. Step 5: state what success looks like once "
              "that obstacle is beaten. Step 6: give one measurable metric for that success.")
STAGE_QS_HARD = [
    "Does the response clearly define what {topic} is?",
    "Does the response name who most benefits from {topic}?",
    "Does the response name a specific biggest obstacle to {topic}?",
    "Does the response give a concrete tactic to beat THAT specific obstacle it named?",
    "Does the response state what success looks like once that specific obstacle is beaten?",
    "Does the response give a measurable metric for that specific success?",
]


def make_mission(rng, topics, hard=False):
    topic = rng.choice(topics)
    instr = (INSTR_HARD if hard else INSTR).format(topic=topic)
    qs = [q.format(topic=topic) for q in (STAGE_QS_HARD if hard else STAGE_QS)]
    return instr, qs


def stage_scores(model, tok, instr, response, qs, yes_id, no_id):
    return [judge_reward(model, tok, instr, response, q, yes_id, no_id) for q in qs]


def gated_reward(scores):
    """Prefix-gated: mean of cumulative products. Stage k contributes prod(s_1..s_k)."""
    cum, prod, terms = [], 1.0, []
    for s in scores:
        prod = prod * s
        terms.append(prod)
    return float(np.mean(terms))


def prefix_reached(scores, thr=0.5):
    """Longest in-order prefix of stages all above threshold."""
    k = 0
    for s in scores:
        if s > thr:
            k += 1
        else:
            break
    return k


def eval_mission(model, tok, controller, hook, enc_tok, enc_model, topics, n, rng, args, steer,
                   yes_id, no_id):
    controller.eval()
    per_stage = None; reached = []; complete = 0; flus = []; comps = []
    for _ in range(n):
        instr, qs = make_mission(rng, topics, hard=args.hard)
        z = encode_goal(instr, enc_tok, enc_model, model.device)
        txt, _, _ = generate(model, tok, instr, z, hook, args.max_new_tokens, 0.0, steer, grad=False)
        sc = stage_scores(model, tok, instr, txt, qs, yes_id, no_id)
        per_stage = np.array(sc) if per_stage is None else per_stage + np.array(sc)
        reached.append(prefix_reached(sc))
        complete += int(all(s > 0.5 for s in sc))
        comps.append(gated_reward(sc))
        flus.append(base_fluency(model, tok, instr, txt))
    controller.train()
    return per_stage / n, np.mean(reached), complete / n, np.mean(comps), np.mean(flus)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--layer_idx", type=int, default=14)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--max_steer", type=float, default=14.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int, default=110)
    p.add_argument("--eval_every", type=int, default=40)
    p.add_argument("--eval_n", type=int, default=16)
    p.add_argument("--min_len", type=int, default=40)
    p.add_argument("--lambda_flu", type=float, default=1.0)
    p.add_argument("--ref_flu", type=float, default=-1.0)
    p.add_argument("--beta_mag", type=float, default=0.03)
    p.add_argument("--lambda_div", type=float, default=0.3)
    p.add_argument("--n_test_topics", type=int, default=6)
    p.add_argument("--hard", action="store_true", help="6-stage chained mission, tight budget")
    p.add_argument("--gen_dtype", default="float16", choices=["float16", "bfloat16"],
                   help="bf16 for Qwen3/large models (native dtype; fp16 can break MoE routing)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="/home/pokazge/checkpoints/steer_mission.pt")
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
    print(f"[mission] gen={args.gen_model} dtype={gdtype} d={model.config.hidden_size} "
          f"layers={model.config.num_hidden_layers} layer_idx={args.layer_idx}", flush=True)
    for pp in model.parameters():
        pp.requires_grad = False
    enc_tok = AutoTokenizer.from_pretrained(args.enc_model)
    enc_model = AutoModel.from_pretrained(args.enc_model).to(device).eval()
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]

    controller = SteerController(d_llm=model.config.hidden_size, d=args.d, K=args.K,
                                   max_steer=args.max_steer).to(device)
    print(f"[mission] controller {sum(p.numel() for p in controller.parameters()):,} params", flush=True)
    hook = Hook(controller)
    model.model.layers[args.layer_idx - 1].register_forward_hook(hook)
    opt = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=0.0)

    train_topics = TOPICS[:-args.n_test_topics]
    test_topics = TOPICS[-args.n_test_topics:]
    print(f"[mission] train_topics={len(train_topics)} held-out={test_topics}", flush=True)
    rng = np.random.default_rng(args.seed)
    baseline = deque(maxlen=64)
    recent_openings = deque(maxlen=128)

    def run_eval(tag):
        b = eval_mission(model, tok, controller, hook, enc_tok, enc_model, test_topics, args.eval_n,
                           np.random.default_rng(456), args, False, yes_id, no_id)
        s = eval_mission(model, tok, controller, hook, enc_tok, enc_model, test_topics, args.eval_n,
                           np.random.default_rng(456), args, True, yes_id, no_id)
        bs = " ".join(f"{x:.2f}" for x in b[0]); ss = " ".join(f"{x:.2f}" for x in s[0])
        ns = len(b[0])
        print(f"[eval {tag}] HELD-OUT  stages base[{bs}]->steer[{ss}]  "
              f"REACHED {b[1]:.2f}->{s[1]:.2f}/{ns}  COMPLETE {b[2]:.2f}->{s[2]:.2f}  "
              f"gated {b[3]:.3f}->{s[3]:.3f}  flu {b[4]:.2f}->{s[4]:.2f}", flush=True)
        return s[3]

    best = run_eval("init")
    csum = 0.0; cn = 0
    for step in range(1, args.max_steps + 1):
        opt.zero_grad()
        rsum = 0.0
        for _ in range(args.group):
            instr, qs = make_mission(rng, train_topics, hard=args.hard)
            z = encode_goal(instr, enc_tok, enc_model, device)
            text, logp, n_new = generate(model, tok, instr, z, hook, args.max_new_tokens,
                                           args.temperature, steer=True, grad=True)
            if logp is None:
                continue
            sc = stage_scores(model, tok, instr, text, qs, yes_id, no_id)
            gated = gated_reward(sc)
            flu = base_fluency(model, tok, instr, text)
            len_ok = 1.0 if n_new >= args.min_len else 0.0
            opening = tuple(tok(text, add_special_tokens=False).input_ids[:8])
            novelty = 1.0 - sum(1 for o in recent_openings if o == opening) / max(1, len(recent_openings))
            recent_openings.append(opening)
            R = gated * len_ok - args.lambda_flu * max(0.0, args.ref_flu - flu) + args.lambda_div * novelty
            b = np.mean(baseline) if baseline else 0.0
            baseline.append(R)
            adv = R - b
            mag = controller.mag_penalty()
            mag_term = args.beta_mag * mag if mag is not None else 0.0
            ((-adv * logp + mag_term) / args.group * 256.0).backward()
            rsum += gated
        for pp in controller.parameters():
            if pp.grad is not None:
                pp.grad /= 256.0
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        opt.step()
        csum += rsum / args.group; cn += 1
        if step % 10 == 0:
            print(f"step {step:>4}  train_gated(roll)={csum/cn:.3f}", flush=True)
            csum = 0.0; cn = 0
        if step % args.eval_every == 0:
            mc = run_eval(f"s{step}")
            if mc > best:
                best = mc
                torch.save({"controller": controller.state_dict(), "args": vars(args),
                             "best_heldout_gated": best}, args.output)
                print(f"[mission] saved (held-out gated {best:.3f}) -> {args.output}", flush=True)
    print(f"[mission] DONE best held-out gated={best:.3f}", flush=True)
    print("[mission] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
