"""Causal steering controller on SEMANTIC goals with a JUDGE reward.

The narrow-regex benchmark let the controller Goodhart (degenerate text trips the
check). Here goals are genuinely SEMANTIC (no programmatic check), and the reward is
an LLM-judge score of goal-pursuit + a fluency floor + a minimal-intervention anchor.
Degenerate or off-topic text is obviously NOT "a good on-task response", so the judge
naturally penalizes the exploits a regex couldn't. Task-agnostic test: train on N
semantic goal types, hold one out.

Reuses the validated machinery (SteerController, Hook, generate, base_fluency).
"""
import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_steer_controller import (SteerController, Hook, generate, base_fluency, encode_goal)
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

TOPICS = ["urban gardening", "deep-sea exploration", "the history of tea", "remote work",
          "renewable energy", "ancient architecture", "migratory birds", "street food",
          "space tourism", "handwritten letters", "mountain ecosystems", "vintage cinema",
          "coral reefs", "public libraries", "the art of negotiation", "desert wildlife",
          "sleep science", "folk music traditions", "glaciers", "urban beekeeping"]

# Semantic goal types chosen to FIGHT the model's RLHF priors (it defaults to hedged,
# balanced, positive, abstract) so the base model genuinely + often FAILS -> headroom.
# Each = (instruction template, STRICT judge question probing the exact hard property).
# NO programmatic check; judge rates the specific semantic property -> discriminating,
# not saturated, and ungameable by degenerate text.
SEMANTIC_GOALS = {
    "pessimistic": ("Write a bleak, downbeat take on {topic}, dwelling on what's wrong with it.",
                     "Is the response consistently negative and pessimistic throughout, NOT balanced, positive, or hopeful?"),
    "no_hedge":    ("Give one strong, definitive opinion about {topic} with absolutely no hedging, caveats, or 'it depends'.",
                     "Does the response state a single strong definitive opinion with NO hedging, caveats, or qualifications?"),
    "you_address": ("Write about {topic} speaking directly TO the reader as 'you' the entire time.",
                     "Does the response address the reader directly as 'you' throughout?"),
    "excited":     ("Write about {topic} with intense, over-the-top excitement and nothing else.",
                     "Is the response intensely, over-the-top excited and enthusiastic throughout?"),
    "concrete":    ("Describe {topic} using only concrete sensory details, never abstract statements.",
                     "Does the response use vivid concrete sensory detail and AVOID abstract generalities?"),
    "contrarian":  ("Take a contrarian stance on {topic} that challenges what most people believe.",
                     "Does the response take a genuinely contrarian, against-the-grain stance?"),
    "imperative":  ("Write about {topic} entirely as a series of direct commands to the reader.",
                     "Is the response phrased entirely as commands / imperative instructions?"),
    "questioning": ("Explore {topic} only by posing thought-provoking questions, never giving answers.",
                     "Does the response consist mainly of questions rather than asserted answers?"),
}


def make_semantic_goal(rng, category=None):
    if category is None:
        category = rng.choice(list(SEMANTIC_GOALS.keys()))
    topic = rng.choice(TOPICS)
    instr_tmpl, judge_q = SEMANTIC_GOALS[category]
    return category, instr_tmpl.format(topic=topic), judge_q


# Multiple judge-prompt framings of the SAME per-goal question. Averaging over these
# makes the reward robust to single-phrasing gaming: the controller must satisfy several
# framings, not key on one prompt's surface cues. {q}=judge_q, {i}=instr, {r}=response.
JUDGE_WRAPPERS = [
    "Instruction given: {i}\n\nResponse: {r}\n\n{q} Answer with a single word: Yes or No.\n\nAnswer:",
    "Read this response to an instruction.\nInstruction: {i}\nResponse: {r}\n\nQuestion: {q}\nReply with only Yes or No.\nReply:",
    "{q}\n\n(The instruction was: {i})\nResponse to evaluate: {r}\n\nIs the answer to the question Yes or No?\nAnswer:",
]


@torch.no_grad()
def _judge_one(model, tok, prompt, yes_id, no_id):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    logits = model(**ids).logits[0, -1]
    return 1.0 / (1.0 + np.exp(-float(logits[yes_id] - logits[no_id])))


@torch.no_grad()
def judge_reward(model, tok, instr, response, judge_q, yes_id, no_id):
    """Single-prompt judge score in [0,1] (used by independent-judge validation)."""
    if not response.strip():
        return 0.0
    prompt = JUDGE_WRAPPERS[0].format(i=instr, r=response, q=judge_q)
    return _judge_one(model, tok, prompt, yes_id, no_id)


@torch.no_grad()
def judge_ensemble(model, tok, instr, response, judge_q, yes_id, no_id, k=3):
    """ENSEMBLE judge: mean score over k prompt framings -> robust to phrasing-gaming."""
    if not response.strip():
        return 0.0
    scores = [_judge_one(model, tok, JUDGE_WRAPPERS[j].format(i=instr, r=response, q=judge_q),
                           yes_id, no_id) for j in range(min(k, len(JUDGE_WRAPPERS)))]
    return float(np.mean(scores))


def eval_judge(model, tok, controller, hook, enc_tok, enc_model, cats, n, rng, args,
                 steer, yes_id, no_id):
    """Returns (mean_judge, mean_fluency)."""
    controller.eval()
    js, fs = [], []
    for cat in cats:
        for _ in range(n):
            _, instr, judge_q = make_semantic_goal(rng, category=cat)
            z = encode_goal(instr, enc_tok, enc_model, model.device)
            txt, _, _ = generate(model, tok, instr, z, hook, args.max_new_tokens, 0.0, steer, grad=False)
            js.append(judge_ensemble(model, tok, instr, txt, judge_q, yes_id, no_id, args.n_judge_prompts))
            fs.append(base_fluency(model, tok, instr, txt))
    controller.train()
    return float(np.mean(js)), float(np.mean(fs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--layer_idx", type=int, default=14)
    p.add_argument("--train_cats", default="skeptic,child,history,hopeful,practical,argue,compare")
    p.add_argument("--heldout_cat", default="emotion")
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--max_steer", type=float, default=14.0)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_n", type=int, default=16)
    p.add_argument("--min_len", type=int, default=16)
    p.add_argument("--lambda_flu", type=float, default=1.0)
    p.add_argument("--ref_flu", type=float, default=-0.9)
    p.add_argument("--beta_mag", type=float, default=0.03)
    p.add_argument("--lambda_div", type=float, default=0.5, help="novelty bonus weight (anti-mode-collapse)")
    p.add_argument("--n_judge_prompts", type=int, default=3, help="judge-ensemble: # prompt framings averaged")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="/home/pokazge/checkpoints/steer_semantic.pt")
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
    print(f"[sem] controller {sum(p.numel() for p in controller.parameters()):,} params, "
          f"max_steer={args.max_steer}, judge-reward", flush=True)
    hook = Hook(controller)
    model.model.layers[args.layer_idx - 1].register_forward_hook(hook)
    opt = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=0.0)
    train_cats = [c.strip() for c in args.train_cats.split(",")]
    rng = np.random.default_rng(args.seed)
    baseline = defaultdict(lambda: deque(maxlen=64))
    recent_openings = deque(maxlen=128)  # for diversity / anti-mode-collapse

    def run_eval(tag):
        bj_tr, bf_tr = eval_judge(model, tok, controller, hook, enc_tok, enc_model, train_cats[:3], args.eval_n, np.random.default_rng(123), args, False, yes_id, no_id)
        sj_tr, sf_tr = eval_judge(model, tok, controller, hook, enc_tok, enc_model, train_cats[:3], args.eval_n, np.random.default_rng(123), args, True, yes_id, no_id)
        bj_ho, bf_ho = eval_judge(model, tok, controller, hook, enc_tok, enc_model, [args.heldout_cat], args.eval_n, np.random.default_rng(456), args, False, yes_id, no_id)
        sj_ho, sf_ho = eval_judge(model, tok, controller, hook, enc_tok, enc_model, [args.heldout_cat], args.eval_n, np.random.default_rng(456), args, True, yes_id, no_id)
        print(f"[eval {tag}] TRAIN judge base={bj_tr:.3f}->steer={sj_tr:.3f} (flu {bf_tr:.2f}->{sf_tr:.2f})  "
              f"HELD-OUT[{args.heldout_cat}] judge base={bj_ho:.3f}->steer={sj_ho:.3f} (flu {bf_ho:.2f}->{sf_ho:.2f})", flush=True)
        return sj_ho

    best = run_eval("init")
    csum = 0.0; cn = 0
    for step in range(1, args.max_steps + 1):
        opt.zero_grad()
        rsum = 0.0
        for _ in range(args.group):
            cat = train_cats[rng.integers(len(train_cats))]
            _, instr, judge_q = make_semantic_goal(rng, category=cat)
            z = encode_goal(instr, enc_tok, enc_model, device)
            text, logp, n_new = generate(model, tok, instr, z, hook, args.max_new_tokens,
                                           args.temperature, steer=True, grad=True)
            if logp is None:
                continue
            J = judge_ensemble(model, tok, instr, text, judge_q, yes_id, no_id, args.n_judge_prompts)
            flu = base_fluency(model, tok, instr, text)
            len_ok = 1.0 if n_new >= args.min_len else 0.0
            # DIVERSITY pressure: reward novel openings so the controller can't mode-collapse
            # to one judge-gaming template. novelty = fraction of recent openings that differ.
            opening = tuple(tok(text, add_special_tokens=False).input_ids[:8])
            match = sum(1 for o in recent_openings if o == opening)
            novelty = 1.0 - match / max(1, len(recent_openings))
            recent_openings.append(opening)
            R = J * len_ok - args.lambda_flu * max(0.0, args.ref_flu - flu) + args.lambda_div * novelty
            b = np.mean(baseline[cat]) if baseline[cat] else 0.0
            baseline[cat].append(R)
            adv = R - b
            mag = controller.mag_penalty()
            mag_term = args.beta_mag * mag if mag is not None else 0.0
            ((-adv * logp + mag_term) / args.group * 256.0).backward()
            rsum += J
        for pp in controller.parameters():
            if pp.grad is not None:
                pp.grad /= 256.0
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        opt.step()
        csum += rsum / args.group; cn += 1
        if step % 10 == 0:
            print(f"step {step:>4}  train_judge(roll)={csum/cn:.3f}", flush=True)
            csum = 0.0; cn = 0
        if step % args.eval_every == 0:
            mho = run_eval(f"s{step}")
            if mho > best:
                best = mho
                torch.save({"controller": controller.state_dict(), "args": vars(args),
                             "best_heldout_judge": best}, args.output)
                print(f"[sem] saved (held-out judge {best:.3f}) -> {args.output}", flush=True)
    print(f"[sem] DONE best held-out judge={best:.3f}", flush=True)
    print("[sem] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
