"""Hierarchical-subgoal causal steering — conceptual proof.

Question: does causal goal-steering work when main-goal success depends on INTEGRATING
a hierarchy of subgoals? Task: a persuasive case that must (sg1) acknowledge an objection,
(sg2) give a strong reason, (sg3) end with a takeaway, AND (integration) have the takeaway
genuinely UNIFY the objection + reason (not just list them).

Reward = mean(subgoal judges) * integration_judge  -- MULTIPLICATIVE so satisfying the
layers without integrating them scores ~0 ("success depends on integration"). + fluency
floor + diversity bonus + magnitude anchor (validated machinery). Held-out = unseen topics.

If steered INTEGRATION rises (not just subgoals) and inspection shows real synthesis, the
concept works with the existing flat controller. If subgoals rise but integration lags, the
hierarchical (slow/fast) architecture is needed.
"""
import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_steer_controller import SteerController, Hook, generate, base_fluency, encode_goal
from train_steer_semantic import judge_reward, TOPICS
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

INSTR = ("Write a short, persuasive case about {topic}. First acknowledge a common objection "
         "to it, then give a strong reason it is worthwhile anyway, and finish with a one-line "
         "takeaway that ties the objection and the reason together.")
SUBGOAL_QS = [
    "Does the response acknowledge a common objection or criticism of the topic?",
    "Does the response give a strong, clear reason the topic is worthwhile?",
    "Does the response end with a short concluding takeaway line?",
]
INTEGRATION_Q = ("Does the final takeaway genuinely tie together BOTH the objection AND the "
                 "reason into one unified point, rather than just listing them separately?")


def make_hier_goal(rng, topics):
    topic = rng.choice(topics)
    return INSTR.format(topic=topic)


def hier_scores(model, tok, instr, response, yes_id, no_id):
    """Return (mean_subgoal, integration). judge_reward is single-prompt sigmoid in [0,1]."""
    sg = [judge_reward(model, tok, instr, response, q, yes_id, no_id) for q in SUBGOAL_QS]
    integ = judge_reward(model, tok, instr, response, INTEGRATION_Q, yes_id, no_id)
    return float(np.mean(sg)), float(integ)


def eval_hier(model, tok, controller, hook, enc_tok, enc_model, topics, n, rng, args, steer,
                yes_id, no_id):
    controller.eval()
    sgs, ints, comps, flus = [], [], [], []
    for _ in range(n):
        instr = make_hier_goal(rng, topics)
        z = encode_goal(instr, enc_tok, enc_model, model.device)
        txt, _, _ = generate(model, tok, instr, z, hook, args.max_new_tokens, 0.0, steer, grad=False)
        sg, integ = hier_scores(model, tok, instr, txt, yes_id, no_id)
        sgs.append(sg); ints.append(integ); comps.append(sg * integ)
        flus.append(base_fluency(model, tok, instr, txt))
    controller.train()
    return np.mean(sgs), np.mean(ints), np.mean(comps), np.mean(flus)


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
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--eval_every", type=int, default=40)
    p.add_argument("--eval_n", type=int, default=16)
    p.add_argument("--min_len", type=int, default=24)
    p.add_argument("--lambda_flu", type=float, default=1.0)
    p.add_argument("--ref_flu", type=float, default=-0.9)
    p.add_argument("--beta_mag", type=float, default=0.03)
    p.add_argument("--lambda_div", type=float, default=0.4)
    p.add_argument("--n_test_topics", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="/home/pokazge/checkpoints/steer_hier.pt")
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
    model = AutoModelForCausalLM.from_pretrained(args.gen_model, dtype=torch.float16,
                                                   trust_remote_code=True).to(device).eval()
    for pp in model.parameters():
        pp.requires_grad = False
    enc_tok = AutoTokenizer.from_pretrained(args.enc_model)
    enc_model = AutoModel.from_pretrained(args.enc_model).to(device).eval()
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]

    controller = SteerController(d_llm=model.config.hidden_size, d=args.d, K=args.K,
                                   max_steer=args.max_steer).to(device)
    print(f"[hier] controller {sum(p.numel() for p in controller.parameters()):,} params", flush=True)
    hook = Hook(controller)
    model.model.layers[args.layer_idx - 1].register_forward_hook(hook)
    opt = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=0.0)

    train_topics = TOPICS[:-args.n_test_topics]
    test_topics = TOPICS[-args.n_test_topics:]
    print(f"[hier] train_topics={len(train_topics)} held-out={test_topics}", flush=True)
    rng = np.random.default_rng(args.seed)
    baseline = deque(maxlen=64)
    recent_openings = deque(maxlen=128)

    def run_eval(tag):
        b = eval_hier(model, tok, controller, hook, enc_tok, enc_model, test_topics, args.eval_n,
                        np.random.default_rng(456), args, False, yes_id, no_id)
        s = eval_hier(model, tok, controller, hook, enc_tok, enc_model, test_topics, args.eval_n,
                        np.random.default_rng(456), args, True, yes_id, no_id)
        print(f"[eval {tag}] HELD-OUT topics  subgoals {b[0]:.3f}->{s[0]:.3f}  "
              f"INTEGRATION {b[1]:.3f}->{s[1]:.3f}  composite {b[2]:.3f}->{s[2]:.3f}  "
              f"flu {b[3]:.2f}->{s[3]:.2f}", flush=True)
        return s[2]

    best = run_eval("init")
    csum = 0.0; cn = 0
    for step in range(1, args.max_steps + 1):
        opt.zero_grad()
        rsum = 0.0
        for _ in range(args.group):
            instr = make_hier_goal(rng, train_topics)
            z = encode_goal(instr, enc_tok, enc_model, device)
            text, logp, n_new = generate(model, tok, instr, z, hook, args.max_new_tokens,
                                           args.temperature, steer=True, grad=True)
            if logp is None:
                continue
            sg, integ = hier_scores(model, tok, instr, text, yes_id, no_id)
            comp = sg * integ                      # integration-dependent composite
            flu = base_fluency(model, tok, instr, text)
            len_ok = 1.0 if n_new >= args.min_len else 0.0
            opening = tuple(tok(text, add_special_tokens=False).input_ids[:8])
            novelty = 1.0 - sum(1 for o in recent_openings if o == opening) / max(1, len(recent_openings))
            recent_openings.append(opening)
            R = comp * len_ok - args.lambda_flu * max(0.0, args.ref_flu - flu) + args.lambda_div * novelty
            b = np.mean(baseline) if baseline else 0.0
            baseline.append(R)
            adv = R - b
            mag = controller.mag_penalty()
            mag_term = args.beta_mag * mag if mag is not None else 0.0
            ((-adv * logp + mag_term) / args.group * 256.0).backward()
            rsum += comp
        for pp in controller.parameters():
            if pp.grad is not None:
                pp.grad /= 256.0
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        opt.step()
        csum += rsum / args.group; cn += 1
        if step % 10 == 0:
            print(f"step {step:>4}  train_composite(roll)={csum/cn:.3f}", flush=True)
            csum = 0.0; cn = 0
        if step % args.eval_every == 0:
            mc = run_eval(f"s{step}")
            if mc > best:
                best = mc
                torch.save({"controller": controller.state_dict(), "args": vars(args),
                             "best_heldout_composite": best}, args.output)
                print(f"[hier] saved (held-out composite {best:.3f}) -> {args.output}", flush=True)
    print(f"[hier] DONE best held-out composite={best:.3f}", flush=True)
    print("[hier] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
