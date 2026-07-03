"""Multi-TURN mission steering with the SLOW CHANNEL.

A mission is delivered one subgoal per conversation turn; later turns DEPEND on earlier
turns (e.g. "give a tactic to overcome THAT obstacle you named"). Across the turn boundary
the LLM context changes and the fast state resets, so the controller's SLOW CHANNEL is the
persistent mission register: it carries which subgoal/obstacle is active and advances via the
trigger at each boundary. Tests whether the slow channel enables cross-turn goal tracking the
single-response flat controller couldn't have (its state doesn't survive a turn boundary).

Reward = prefix-gated across turns (mean of cumulative products of per-turn subgoal judges),
so a later turn earns credit only if the earlier dependent turns succeeded. Held-out = topics.
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_steer_controller import SteerController, Hook, base_fluency, encode_goal
from train_steer_semantic import judge_reward, TOPICS
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# 4-turn mission: turns 3-4 DEPEND on the obstacle named in turn 2 (which lives only in the
# assistant's turn-2 response, in context) -> genuine cross-turn dependency.
MT_TURNS = [
    ("We're working on {topic}. In one sentence, define what {topic} is.",
     "Does the reply define what {topic} is in roughly one sentence?"),
    ("Now name the single biggest obstacle to {topic}.",
     "Does the reply name a specific biggest obstacle to {topic}?"),
    ("Give one concrete tactic to overcome THAT obstacle you just named.",
     "Does the reply give a concrete tactic to overcome the specific obstacle named earlier in the conversation?"),
    ("Finally, state one measurable signal that this obstacle has been overcome.",
     "Does the reply state a measurable signal that the specific obstacle is overcome?"),
]


# Off-topic distractor turns inserted BETWEEN mission turns. They fill the LLM's recent
# context with unrelated content, burying the mission thread -> the regime where a
# controller-side persistent mission register (slow channel) should beat relying on the
# LLM's own context attention. The LLM still answers them (real user turns).
DISTRACTORS = [
    "Quick aside — what's a fun fact about octopuses?",
    "Unrelated: suggest a good board game for two people.",
    "Off topic, but what's a quick stretch I can do at my desk?",
    "By the way, what's a fun fact about the planet Jupiter?",
    "Side question: recommend a low-effort houseplant.",
    "Random: what's a good word to use in Scrabble?",
]


def make_mt(rng, topics, n_distract=0):
    """Return (topic, sequence) where sequence is a list of (kind, user_text, judge_q).
    kind='mission' turns are judged + steered; kind='distract' fill context only."""
    topic = rng.choice(topics)
    seq = []
    di = 0
    for i, (u, q) in enumerate(MT_TURNS):
        seq.append(("mission", u.format(topic=topic), q.format(topic=topic)))
        if n_distract and i < len(MT_TURNS) - 1:
            for _ in range(n_distract):
                seq.append(("distract", DISTRACTORS[di % len(DISTRACTORS)], None)); di += 1
    return topic, seq


def generate_mt(model, tok, messages, z_goal, hook, max_new, temperature, grad, think=None):
    """Steered generation for the current turn given full message history. Resets the FAST
    channel only (slow channel persists — managed by caller via reset_episode + slow_step).
    think=False disables reasoning-model CoT (Qwen3); None = model default."""
    hook.c.reset(1, model.device)
    hook.z = z_goal.unsqueeze(0); hook.active = True
    try:
        chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **({} if think is None else {"enable_thinking": think}))
    except TypeError:
        chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(chat, return_tensors="pt").to(model.device)
    ids, attn = enc.input_ids, enc.attention_mask
    logps, out_ids = [], []
    past, cur, cur_attn = None, ids, attn
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for _ in range(max_new):
            o = model(cur, attention_mask=cur_attn, past_key_values=past, use_cache=True)
            logits = o.logits[:, -1]
            past = o.past_key_values
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                tk = torch.multinomial(probs, 1)
            else:
                tk = logits.argmax(dim=-1, keepdim=True)
            if grad:
                logps.append(torch.log_softmax(logits, dim=-1).gather(1, tk).squeeze())
            tid = int(tk.item()); out_ids.append(tid)
            if tid == tok.eos_token_id:
                break
            cur = tk
            cur_attn = torch.cat([cur_attn, torch.ones((1, 1), dtype=cur_attn.dtype, device=model.device)], 1)
    hook.active = False
    text = tok.decode(out_ids, skip_special_tokens=True)
    logp_sum = torch.stack(logps).sum() if logps else None
    n_new = len([t for t in out_ids if t != tok.eos_token_id])
    return text, logp_sum, n_new


def gated(scores):
    prod, terms = 1.0, []
    for s in scores:
        prod *= s; terms.append(prod)
    return float(np.mean(terms))


def run_mission(model, tok, controller, hook, enc_tok, enc_model, seq, device, args, steer, grad):
    """Run a mission sequence (mission + distractor turns). MISSION turns are steered + slow-
    stepped + judged; DISTRACTOR turns fill context unsteered. Returns
    (mission_scores, logps, mean_fluency, min_n_new, mean_trigger)."""
    if steer:
        controller.reset_episode(1, device)
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]
    messages, transcript = [], ""
    scores, logps, flus, n_news, trigs = [], [], [], [], []
    for (kind, u, jq) in seq:
        messages.append({"role": "user", "content": u})
        is_mission = (kind == "mission")
        if steer and is_mission:
            z = encode_goal(u, enc_tok, enc_model, device)
            tr = controller.slow_step(z.unsqueeze(0))   # advance mission state ONLY at mission turns
            if tr is not None:
                trigs.append(tr)
            text, logp, n_new = generate_mt(model, tok, messages, z, hook, args.max_new_tokens,
                                              args.temperature if grad else 0.0, grad)
        else:
            hook.active = False
            text, logp, n_new = generate_mt_nosteer(model, tok, messages, args.max_new_tokens)
        messages.append({"role": "assistant", "content": text})
        if is_mission:
            judge_instr = (f"Conversation so far:\n{transcript}\nCurrent request: {u}") if transcript else u
            scores.append(judge_reward(model, tok, judge_instr, text, jq, yes_id, no_id))
            flus.append(base_fluency(model, tok, u, text)); n_news.append(n_new)
            if logp is not None:
                logps.append(logp)
        transcript += f"User: {u}\nAssistant: {text}\n"
    return scores, logps, float(np.mean(flus)), (min(n_news) if n_news else 0), (np.mean(trigs) if trigs else 0.0)


@torch.no_grad()
def generate_mt_nosteer(model, tok, messages, max_new, think=None):
    try:
        chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **({} if think is None else {"enable_thinking": think}))
    except TypeError:
        chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(chat, return_tensors="pt").to(model.device)
    out = model.generate(enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=max_new,
                           do_sample=False, pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    n_new = out.shape[1] - enc.input_ids.shape[1]
    return text, None, n_new


def eval_mt(model, tok, controller, hook, enc_tok, enc_model, topics, n, rng, device, args, steer):
    controller.eval()
    per = None; reached = []; complete = 0; comps = []; flus = []
    for _ in range(n):
        _, seq = make_mt(rng, topics, n_distract=args.n_distract)
        sc, _, flu, _, _ = run_mission(model, tok, controller, hook, enc_tok, enc_model, seq,
                                         device, args, steer, grad=False)
        a = np.array(sc); per = a if per is None else per + a
        k = 0
        for s in sc:
            if s > 0.5: k += 1
            else: break
        reached.append(k); complete += int(all(s > 0.5 for s in sc)); comps.append(gated(sc)); flus.append(flu)
    controller.train()
    return per / n, np.mean(reached), complete / n, np.mean(comps), np.mean(flus)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--layer_idx", type=int, default=14)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--max_steer", type=float, default=14.0)
    p.add_argument("--use_slow", action="store_true", help="enable cross-turn slow channel")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=250)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int, default=45)
    p.add_argument("--eval_every", type=int, default=40)
    p.add_argument("--eval_n", type=int, default=12)
    p.add_argument("--min_len", type=int, default=8)
    p.add_argument("--lambda_flu", type=float, default=1.0)
    p.add_argument("--ref_flu", type=float, default=-1.2)
    p.add_argument("--beta_mag", type=float, default=0.03)
    p.add_argument("--n_test_topics", type=int, default=6)
    p.add_argument("--n_distract", type=int, default=0, help="off-topic turns between mission turns")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="/home/pokazge/checkpoints/steer_mt.pt")
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

    controller = SteerController(d_llm=model.config.hidden_size, d=args.d, K=args.K,
                                   max_steer=args.max_steer, use_slow=args.use_slow).to(device)
    print(f"[mt] controller {sum(p.numel() for p in controller.parameters()):,} params, "
          f"use_slow={args.use_slow}", flush=True)
    hook = Hook(controller)
    model.model.layers[args.layer_idx - 1].register_forward_hook(hook)
    opt = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=0.0)
    train_topics = TOPICS[:-args.n_test_topics]; test_topics = TOPICS[-args.n_test_topics:]
    print(f"[mt] held-out={test_topics}", flush=True)
    rng = np.random.default_rng(args.seed); baseline = deque(maxlen=64)

    def run_eval(tag):
        b = eval_mt(model, tok, controller, hook, enc_tok, enc_model, test_topics, args.eval_n,
                      np.random.default_rng(456), device, args, steer=False)
        s = eval_mt(model, tok, controller, hook, enc_tok, enc_model, test_topics, args.eval_n,
                      np.random.default_rng(456), device, args, steer=True)
        bs = " ".join(f"{x:.2f}" for x in b[0]); ss = " ".join(f"{x:.2f}" for x in s[0])
        print(f"[eval {tag}] HELD-OUT turns base[{bs}]->steer[{ss}]  "
              f"REACHED {b[1]:.2f}->{s[1]:.2f}/{len(b[0])}  COMPLETE {b[2]:.2f}->{s[2]:.2f}  "
              f"gated {b[3]:.3f}->{s[3]:.3f}  flu {b[4]:.2f}->{s[4]:.2f}", flush=True)
        return s[3]

    best = run_eval("init"); csum = 0.0; cn = 0
    for step in range(1, args.max_steps + 1):
        opt.zero_grad(); rsum = 0.0
        for _ in range(args.group):
            _, seq = make_mt(rng, train_topics, n_distract=args.n_distract)
            sc, logps, flu, min_n, _ = run_mission(model, tok, controller, hook, enc_tok, enc_model,
                                                     seq, device, args, steer=True, grad=True)
            if not logps:
                continue
            g = gated(sc); len_ok = 1.0 if min_n >= args.min_len else 0.0
            R = g * len_ok - args.lambda_flu * max(0.0, args.ref_flu - flu)
            b = np.mean(baseline) if baseline else 0.0; baseline.append(R); adv = R - b
            total_logp = torch.stack(logps).sum()
            mag = controller.mag_penalty(); mag_term = args.beta_mag * mag if mag is not None else 0.0
            ((-adv * total_logp + mag_term) / args.group * 256.0).backward()
            rsum += g
        for pp in controller.parameters():
            if pp.grad is not None:
                pp.grad /= 256.0
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        opt.step(); csum += rsum / args.group; cn += 1
        if step % 10 == 0:
            print(f"step {step:>4}  train_gated(roll)={csum/cn:.3f}", flush=True); csum = 0.0; cn = 0
        if step % args.eval_every == 0:
            mc = run_eval(f"s{step}")
            if mc > best:
                best = mc
                torch.save({"controller": controller.state_dict(), "args": vars(args),
                             "best_heldout_gated": best}, args.output)
                print(f"[mt] saved (held-out gated {best:.3f}) -> {args.output}", flush=True)
    print(f"[mt] DONE best held-out gated={best:.3f}", flush=True)
    print("[mt] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
