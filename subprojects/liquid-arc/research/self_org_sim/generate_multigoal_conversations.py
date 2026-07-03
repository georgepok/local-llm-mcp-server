"""Generate multi-goal conversations with linguistically expressed, verifiable goals.

Each conversation: 3 sequential turns where user issues a constraint-based goal.
Goals are auto-verifiable so we get clean per-turn drift labels.

Goal types (sample randomly per turn):
  - exact_word_count(N)        — "Answer in exactly N words: <topic>"
  - begin_with(phrase)         — "Begin your response with '<phrase>'..."
  - include_word(W)            — "In your response, include the word '<W>'..."
  - single_sentence            — "Respond using exactly one sentence."
  - end_with_question          — "End your response with a question mark."
  - all_caps                   — "Write your response in ALL CAPITAL LETTERS."
  - no_word(W)                 — "Do not use the word '<W>' in your response."

For substrate:
  z_t[t]       = bge-small encoding of running model generation at chunk t
  z_goal[t]    = bge-small encoding of CURRENT user instruction (jumps at turn boundary)
  z_lang[t]    = z_goal[t]  — slow stream IS the linguistic goal

Drift label: model response satisfies the constraint of that turn's goal.
Conversation-level label: did model follow at least 2/3 goals.
"""
import argparse
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Topic seeds — neutral, easy for any LLM to write something about
TOPICS = [
    "the changing seasons", "ocean currents", "the role of bees",
    "ancient libraries", "city skylines at night", "the smell of rain",
    "old wooden bridges", "morning coffee rituals", "stars in the desert",
    "small mountain villages", "tide pools at dawn", "kite flying",
    "neighborhood bakeries", "winter mornings", "a quiet park bench",
    "lighthouses on cliffs", "subway commutes", "watercolor painting",
    "knitting in evening", "river deltas", "forgotten attics",
    "village festivals", "icebergs drifting", "alpine meadows",
    "the sound of typewriters", "moss on stone", "vintage radios",
    "summer thunderstorms", "candle making", "a small fishing boat",
]

BEGIN_PHRASES = ["However", "Interestingly", "Surprisingly", "Notably",
                  "Importantly", "Curiously", "Remarkably", "Honestly"]
INCLUDE_WORDS = ["whisper", "lantern", "thunder", "compass", "ember",
                  "echo", "twilight", "mirror"]
AVOID_WORDS = ["the", "and", "but", "very"]  # common, easy to forget to avoid


def make_goal(rng):
    """Sample a goal type + parameters. Returns (instruction_text, check_fn)."""
    g = rng.choice([
        "exact_word_count", "begin_with", "include_word",
        "single_sentence", "end_with_question",
        "all_caps", "no_word",
    ])
    topic = rng.choice(TOPICS)
    if g == "exact_word_count":
        n = int(rng.choice([8, 10, 12, 15]))
        instr = f"Write about {topic} in exactly {n} words."
        def check(resp): return len(resp.split()) == n
    elif g == "begin_with":
        phrase = rng.choice(BEGIN_PHRASES)
        instr = f"Write 1-2 sentences about {topic}. Begin your response with '{phrase}'."
        def check(resp): return resp.strip().startswith(phrase)
    elif g == "include_word":
        word = rng.choice(INCLUDE_WORDS)
        instr = f"Write 1-2 sentences about {topic}. You must include the word '{word}'."
        def check(resp): return word.lower() in resp.lower()
    elif g == "single_sentence":
        instr = f"Describe {topic} using exactly one sentence."
        def check(resp):
            s = re.split(r"[.!?]+", resp.strip())
            s = [x for x in s if x.strip()]
            return len(s) == 1
    elif g == "end_with_question":
        instr = f"Write 1-2 sentences about {topic}, ending with a question mark."
        def check(resp): return resp.strip().endswith("?")
    elif g == "all_caps":
        instr = f"Write a short response about {topic}. Use ALL CAPITAL LETTERS for your entire response."
        def check(resp):
            letters = [c for c in resp if c.isalpha()]
            if not letters: return False
            return sum(1 for c in letters if c.isupper()) / len(letters) > 0.9
    elif g == "no_word":
        word = rng.choice(AVOID_WORDS)
        instr = f"Write 1-2 sentences about {topic}. Do NOT use the word '{word}' anywhere."
        def check(resp):
            tokens = re.findall(r"\b\w+\b", resp.lower())
            return word.lower() not in tokens
    return g, instr, check


@torch.no_grad()
def encode_text(text, tok, model, device):
    toks = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = model(**toks)
    v = out.last_hidden_state[:, 0]  # CLS pool for bge
    return torch.nn.functional.normalize(v, dim=-1).squeeze(0).cpu().numpy()


def chunk_token_ids(token_ids, chunk_size=16):
    return [token_ids[i:i + chunk_size] for i in range(0, len(token_ids), chunk_size)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--n_conversations", type=int, default=250)
    p.add_argument("--n_goals", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    rng = random.Random(args.seed)
    device = torch.device("cuda")
    print(f"[mg] device={device}, gen={args.gen_model}, enc={args.enc_model}", flush=True)

    enc_tok = AutoTokenizer.from_pretrained(args.enc_model)
    enc_model = AutoModel.from_pretrained(args.enc_model).to(device).eval()
    enc_dim = enc_model.config.hidden_size

    gen_tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    gen_model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, dtype=torch.float16, trust_remote_code=True,
    ).to(device).eval()
    if gen_tok.pad_token_id is None:
        gen_tok.pad_token_id = gen_tok.eos_token_id
    print(f"[mg] models loaded, enc_dim={enc_dim}", flush=True)

    records = []
    n_follow_per_turn = [0] * args.n_goals
    t_start = time.time()
    for ci in range(args.n_conversations):
        messages = [{"role": "system",
                     "content": "Follow the user's instruction precisely. Address the most recent instruction."}]
        # Build goals upfront
        goals = [make_goal(rng) for _ in range(args.n_goals)]
        turn_outputs = []      # text per model turn
        turn_followed = []     # bool per model turn
        turn_instruction_texts = []  # the user instruction issued before this turn

        all_concat_tokens = []         # running tokens (only model output) for substrate encoding
        turn_chunk_start_indices = []  # which chunk index starts each turn (in chunks-after-tokenize)

        for turn_idx, (gtype, instr, check_fn) in enumerate(goals):
            messages.append({"role": "user", "content": instr})
            turn_instruction_texts.append(instr)
            chat = gen_tok.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
            prompt_ids = gen_tok(chat, return_tensors="pt").input_ids.to(device)
            out = gen_model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=args.temperature, top_p=0.95,
                pad_token_id=gen_tok.pad_token_id,
            )
            new_ids = out[0, prompt_ids.shape[1]:]
            new_text = gen_tok.decode(new_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": new_text})
            turn_outputs.append(new_text)
            followed = bool(check_fn(new_text))
            turn_followed.append(followed)
            n_follow_per_turn[turn_idx] += int(followed)
            # Record which chunk position this turn starts at
            current_chunk_start = len(all_concat_tokens) // args.chunk_size
            turn_chunk_start_indices.append(current_chunk_start)
            all_concat_tokens.extend(new_ids.tolist())

        if not all_concat_tokens:
            continue
        # Chunk the concatenated model output
        chunks = chunk_token_ids(all_concat_tokens, args.chunk_size)
        T = len(chunks)

        # Compute per-chunk z_t (encode running text) and z_goal (current instruction)
        # Determine current instruction at each chunk position: based on which turn this chunk belongs to.
        # Map chunk_idx → turn_idx using turn_chunk_start_indices.
        z_t_list, z_goal_list = [], []
        # Build chunk→turn map
        chunk_to_turn = []
        cur_turn = 0
        for ci_chunk in range(T):
            while (cur_turn + 1 < args.n_goals and
                   ci_chunk >= turn_chunk_start_indices[cur_turn + 1]):
                cur_turn += 1
            chunk_to_turn.append(cur_turn)

        # Encode current instruction for each turn ONCE
        per_turn_z_goal = [encode_text(instr, enc_tok, enc_model, device)
                            for instr in turn_instruction_texts]

        # Encode running text
        running = ""
        for ci_chunk in range(T):
            ch_text = gen_tok.decode(chunks[ci_chunk], skip_special_tokens=True)
            running = running + ch_text
            zt = encode_text(running, enc_tok, enc_model, device)
            zg = per_turn_z_goal[chunk_to_turn[ci_chunk]]
            if np.isnan(zt).any() or np.isnan(zg).any():
                z_t_list = []
                break
            z_t_list.append(zt)
            z_goal_list.append(zg)

        if not z_t_list:
            continue
        z_t_traj = np.stack(z_t_list, axis=0)
        z_goal_traj = np.stack(z_goal_list, axis=0)

        # Conversation-level success = followed at least 2/3
        conv_succ = int(sum(turn_followed) >= 2)

        records.append({
            "z_t_traj":     torch.from_numpy(z_t_traj).float(),   # fast: model generation
            "z_lang_traj":  torch.from_numpy(z_goal_traj).float(),# slow: current goal (jumps at turn)
            "z_goal":       torch.from_numpy(per_turn_z_goal[0]).float(),  # initial goal
            "T": z_t_traj.shape[0],
            "succ": conv_succ,
            "sub_id": ci,
            "turn_followed": turn_followed,
            "turn_chunk_starts": turn_chunk_start_indices,
            "turn_instructions": turn_instruction_texts,
            "turn_outputs": turn_outputs,
            "goal_types": [g[0] for g in goals],
        })

        if (ci + 1) % 10 == 0 or ci == 0:
            elapsed = time.time() - t_start
            n = len(records)
            follow_rates = [n_follow_per_turn[i] / max(1, ci + 1) for i in range(args.n_goals)]
            avg_T = float(np.mean([r["T"] for r in records])) if records else 0
            print(f"[mg] [{ci+1}/{args.n_conversations}]  records={n}  "
                  f"T_avg={avg_T:.1f}  follow_rates={[f'{r:.2f}' for r in follow_rates]}  "
                  f"elapsed={elapsed:.0f}s  eta={elapsed*(args.n_conversations-ci-1)/(ci+1):.0f}s",
                  flush=True)

    print(f"[mg] DONE. records={len(records)}", flush=True)
    n_full_succ = sum(1 for r in records if r["succ"] == 1)
    print(f"[mg] conversation-level succ (>=2/3 goals followed): "
          f"{n_full_succ}/{len(records)} = {100*n_full_succ/max(1,len(records)):.0f}%", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "records": records,
        "enc_dim": enc_dim,
        "enc_model": args.enc_model,
        "gen_model": args.gen_model,
        "chunk_size": args.chunk_size,
        "n_goals": args.n_goals,
    }, args.output)
    print(f"[mg] === ALL_DONE === saved {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
