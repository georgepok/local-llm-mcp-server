"""Diverse multi-turn goal conversations for phase-transition training.

5 goal categories × 3-4 types each = 15+ distinct goals. Varied phrasing.
60+ topics. Each conversation = 5 turns drawing goals from possibly different
categories. Each record carries the GOAL CATEGORY label for cross-category testing.

Auto-verifiable. Output: text_traces_diverse.pt with metadata for category split.
"""
import argparse
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


TOPICS = [
    # nature
    "the changing seasons", "ocean currents", "the role of bees", "moss on stone",
    "tide pools at dawn", "alpine meadows", "icebergs drifting", "morning fog",
    "summer thunderstorms", "river deltas", "winter mornings", "autumn forests",
    "river otters", "coral reefs", "desert nights", "mountain rivers",
    # daily life
    "morning coffee rituals", "neighborhood bakeries", "subway commutes",
    "evening walks", "shared meals", "small kindnesses", "old habits",
    "rainy afternoons", "weekend markets",
    # crafts and hobbies
    "knitting in evening", "candle making", "watercolor painting",
    "vintage radios", "kite flying", "model trains", "pottery wheels",
    "wood carving", "bookbinding",
    # places
    "ancient libraries", "city skylines at night", "lighthouses on cliffs",
    "small fishing boats", "village festivals", "old wooden bridges",
    "forgotten attics", "a quiet park bench", "small mountain villages",
    "abandoned trainyards",
    # senses
    "the smell of rain", "the sound of typewriters", "the taste of citrus",
    "the warmth of sunlight", "the texture of velvet",
    # abstract
    "the passage of time", "memory and forgetting", "the value of patience",
    "the meaning of home", "what makes a good story",
]


BEGIN_PHRASES = ["However", "Interestingly", "Surprisingly", "Notably",
                  "Importantly", "Curiously", "Remarkably", "Honestly",
                  "Frankly", "Naturally", "Apparently"]
INCLUDE_WORDS = ["whisper", "lantern", "thunder", "compass", "ember",
                  "echo", "twilight", "mirror", "shadow", "harbor"]
AVOID_WORDS = ["the", "and", "but", "very", "really", "just"]
PHRASING_TEMPLATES = {
    "neutral": "{}",
    "polite": "Please {}",
    "task": "Your task: {}",
    "imperative": "{}.",
    "explicit": "Make sure to {}",
    "constraint": "Constraint: {}",
}


def _phrase(instr, template_key):
    fmt = PHRASING_TEMPLATES[template_key]
    # If template has placeholder format, fill; else append
    if "{}" in fmt:
        return fmt.format(instr.rstrip("."))
    return f"{fmt} {instr}"


# ---- Goal generators per category ----
# Each returns (category, goal_type, instruction_text, check_fn)


def make_surface_goal(rng, topic):
    g = rng.choice(["begin_with", "all_caps", "exact_word_count", "no_word",
                     "end_with_excl"])
    if g == "begin_with":
        phrase = rng.choice(BEGIN_PHRASES)
        instr = f"Write 1-2 sentences about {topic}, beginning your response with '{phrase}'"
        def check(resp): return resp.strip().startswith(phrase)
    elif g == "all_caps":
        instr = f"Write a short response about {topic} using ALL CAPITAL LETTERS"
        def check(resp):
            letters = [c for c in resp if c.isalpha()]
            if not letters: return False
            return sum(1 for c in letters if c.isupper()) / len(letters) > 0.9
    elif g == "exact_word_count":
        n = int(rng.choice([8, 10, 12, 15, 18]))
        instr = f"Write about {topic} in exactly {n} words"
        def check(resp): return len(resp.split()) == n
    elif g == "no_word":
        word = rng.choice(AVOID_WORDS)
        instr = f"Write about {topic} without using the word '{word}'"
        def check(resp):
            tokens = re.findall(r"\b\w+\b", resp.lower())
            return word.lower() not in tokens
    elif g == "end_with_excl":
        instr = f"Write a short, emphatic response about {topic} ending with an exclamation mark"
        def check(resp): return resp.strip().endswith("!")
    return "surface", g, instr, check


def make_inclusion_goal(rng, topic):
    g = rng.choice(["include_word", "mention_topic_twice", "include_n_examples"])
    if g == "include_word":
        word = rng.choice(INCLUDE_WORDS)
        instr = f"Write 1-2 sentences about {topic} including the word '{word}'"
        def check(resp): return word.lower() in resp.lower()
    elif g == "mention_topic_twice":
        instr = f"Write 1-2 sentences about {topic} and mention {topic} at least twice"
        def check(resp): return resp.lower().count(topic.lower()) >= 2
    elif g == "include_n_examples":
        n = int(rng.choice([2, 3]))
        instr = f"Give exactly {n} short example items about {topic}"
        def check(resp):
            # heuristic: count numbered list items or comma-separated items
            numbered = len(re.findall(r"^\s*\d+[.\)]\s+", resp, flags=re.MULTILINE))
            if numbered == n:
                return True
            # Or count bullet/dash lines
            bulleted = len(re.findall(r"^\s*[-*]\s+", resp, flags=re.MULTILINE))
            if bulleted == n:
                return True
            return False
    return "inclusion", g, instr, check


def make_structure_goal(rng, topic):
    g = rng.choice(["single_sentence", "n_sentences", "one_word"])
    if g == "single_sentence":
        instr = f"Describe {topic} using exactly one sentence"
        def check(resp):
            s = [x for x in re.split(r"[.!?]+", resp.strip()) if x.strip()]
            return len(s) == 1
    elif g == "n_sentences":
        n = int(rng.choice([2, 3]))
        instr = f"Write exactly {n} sentences about {topic}"
        def check(resp):
            s = [x for x in re.split(r"[.!?]+", resp.strip()) if x.strip()]
            return len(s) == n
    elif g == "one_word":
        instr = f"Respond with exactly one word that captures {topic}"
        def check(resp):
            tokens = re.findall(r"\b\w+\b", resp.strip())
            return len(tokens) == 1
    return "structure", g, instr, check


def make_format_goal(rng, topic):
    g = rng.choice(["numbered_list_3", "bullet_list_3", "title_case", "lowercase"])
    if g == "numbered_list_3":
        instr = f"Write 3 short observations about {topic} as a numbered list (1., 2., 3.)"
        def check(resp):
            lines = [l for l in resp.split("\n") if l.strip()]
            numbered = [l for l in lines if re.match(r"^\s*[1-9][.\)]\s+", l)]
            return len(numbered) == 3
    elif g == "bullet_list_3":
        instr = f"Write 3 short observations about {topic} as a bullet list using dashes (-)"
        def check(resp):
            lines = [l for l in resp.split("\n") if l.strip()]
            bullets = [l for l in lines if re.match(r"^\s*[-*]\s+", l)]
            return len(bullets) == 3
    elif g == "title_case":
        instr = f"Write One Short Sentence About {topic} In Title Case Where Each Word Is Capitalized"
        def check(resp):
            tokens = re.findall(r"\b[A-Za-z]+\b", resp.strip())
            if not tokens: return False
            small_words = {"a","an","and","or","of","to","in","on","the","but","for","at","by","with"}
            caps = 0
            counted = 0
            for tok in tokens:
                if tok.lower() in small_words and counted > 0:
                    continue
                counted += 1
                if tok[0].isupper():
                    caps += 1
            return counted > 0 and caps / counted > 0.85
    elif g == "lowercase":
        instr = f"write one short sentence about {topic} using only lowercase letters"
        def check(resp):
            letters = [c for c in resp if c.isalpha()]
            if not letters: return False
            return sum(1 for c in letters if c.islower()) / len(letters) > 0.95
    return "format", g, instr, check


def make_contrast_goal(rng, topic):
    g = rng.choice(["end_with_question", "no_period", "question_each_sentence"])
    if g == "end_with_question":
        instr = f"Write 1-2 sentences about {topic}, ending with a question"
        def check(resp): return resp.strip().endswith("?")
    elif g == "no_period":
        instr = f"Write a short response about {topic} without using any periods (.)"
        def check(resp): return "." not in resp.strip().rstrip(".!?")
    elif g == "question_each_sentence":
        instr = f"Write 2 sentences about {topic}, each one ending with a question mark"
        def check(resp):
            s = [x for x in re.split(r"(?<=[.!?])\s+", resp.strip()) if x.strip()]
            if len(s) < 2: return False
            return all(x.endswith("?") for x in s[:2])
    return "contrast", g, instr, check


def make_length_goal(rng, topic):
    g = rng.choice(["at_most_N", "at_least_N", "between_NM"])
    if g == "at_most_N":
        n = int(rng.choice([8, 12, 16, 20]))
        instr = f"Respond about {topic} using at most {n} words"
        def check(resp): return len(resp.split()) <= n
    elif g == "at_least_N":
        n = int(rng.choice([15, 25, 35]))
        instr = f"Respond about {topic} using at least {n} words"
        def check(resp): return len(resp.split()) >= n
    elif g == "between_NM":
        lo = int(rng.choice([10, 15, 20]))
        hi = lo + int(rng.choice([8, 12, 15]))
        instr = f"Write between {lo} and {hi} words about {topic}"
        def check(resp):
            n = len(resp.split())
            return lo <= n <= hi
    return "length", g, instr, check


def make_punct_goal(rng, topic):
    g = rng.choice(["n_commas", "n_periods", "no_punct"])
    if g == "n_commas":
        n = int(rng.choice([2, 3, 4]))
        instr = f"Write about {topic} using exactly {n} commas"
        def check(resp): return resp.count(",") == n
    elif g == "n_periods":
        n = int(rng.choice([2, 3, 4]))
        instr = f"Write about {topic} with exactly {n} periods"
        def check(resp): return resp.count(".") == n
    elif g == "no_punct":
        instr = f"Write about {topic} without using any punctuation marks (no commas, periods, question marks, etc.)"
        def check(resp):
            return all(c not in ",.?!;:\"'" for c in resp.strip())
    return "punct", g, instr, check


def make_repetition_goal(rng, topic):
    g = rng.choice(["word_N_times", "word_each_sentence", "phrase_twice"])
    word = rng.choice(["compass", "lantern", "thunder", "ember", "echo", "twilight"])
    if g == "word_N_times":
        n = int(rng.choice([2, 3]))
        instr = f"Write about {topic} using the word '{word}' exactly {n} times"
        def check(resp):
            tokens = re.findall(r"\b\w+\b", resp.lower())
            return tokens.count(word.lower()) == n
    elif g == "word_each_sentence":
        instr = f"Write 2 sentences about {topic}, including the word '{word}' in each sentence"
        def check(resp):
            sents = [s for s in re.split(r"[.!?]+", resp.strip()) if s.strip()]
            if len(sents) < 2: return False
            return all(word.lower() in s.lower() for s in sents[:2])
    elif g == "phrase_twice":
        phrase = rng.choice(["in fact", "moreover", "however", "of course"])
        instr = f"Write about {topic} using the phrase '{phrase}' exactly twice"
        def check(resp):
            return resp.lower().count(phrase.lower()) == 2
    return "repetition", g, instr, check


def make_case_goal(rng, topic):
    g = rng.choice(["alt_case", "first_letter_X", "no_caps"])
    if g == "alt_case":
        instr = f"Write about {topic} alternating UPPER and lower case for each letter"
        def check(resp):
            letters = [c for c in resp if c.isalpha()]
            if len(letters) < 4: return False
            alternations = sum(1 for i in range(1, len(letters))
                                  if letters[i].isupper() != letters[i-1].isupper())
            return alternations / max(1, len(letters)-1) > 0.7
    elif g == "first_letter_X":
        letter = rng.choice(["T", "S", "B", "M", "P"])
        instr = f"Write about {topic} where every word begins with the letter '{letter}' (or '{letter.lower()}')"
        def check(resp):
            tokens = re.findall(r"\b\w+\b", resp.strip())
            if len(tokens) < 3: return False
            return sum(1 for t in tokens if t[0].lower() == letter.lower()) / len(tokens) > 0.85
    elif g == "no_caps":
        instr = f"Write about {topic} without using any uppercase letters at all"
        def check(resp):
            letters = [c for c in resp if c.isalpha()]
            if not letters: return False
            return not any(c.isupper() for c in letters)
    return "case", g, instr, check


def make_structure_alt_goal(rng, topic):
    """Additional structure variants."""
    g = rng.choice(["specific_form", "two_paragraphs", "starts_question"])
    if g == "specific_form":
        instr = f"Write a response about {topic} in the form 'A is B because C'"
        def check(resp):
            return "because" in resp.lower() and "is" in resp.lower()
    elif g == "two_paragraphs":
        instr = f"Write about {topic} as exactly two short paragraphs separated by a blank line"
        def check(resp):
            paras = [p for p in resp.split("\n\n") if p.strip()]
            return len(paras) == 2
    elif g == "starts_question":
        instr = f"Write about {topic} starting with a question, then provide an answer"
        def check(resp):
            sents = [s for s in re.split(r"(?<=[.!?])\s+", resp.strip()) if s.strip()]
            if not sents: return False
            return sents[0].rstrip().endswith("?")
    return "structure_alt", g, instr, check


CATEGORY_MAKERS = {
    "surface":       make_surface_goal,
    "inclusion":     make_inclusion_goal,
    "structure":     make_structure_goal,
    "format":        make_format_goal,
    "contrast":      make_contrast_goal,
    "length":        make_length_goal,
    "punct":         make_punct_goal,
    "repetition":    make_repetition_goal,
    "case":          make_case_goal,
    "structure_alt": make_structure_alt_goal,
}


def make_goal(rng, category=None):
    """Sample a goal. Optionally restrict to a specific category."""
    topic = rng.choice(TOPICS)
    if category is None:
        category = rng.choice(list(CATEGORY_MAKERS.keys()))
    cat_name, gtype, instr, check = CATEGORY_MAKERS[category](rng, topic)
    # Apply phrasing variation
    template = rng.choice(list(PHRASING_TEMPLATES.keys()))
    instr_phrased = _phrase(instr, template)
    return cat_name, gtype, instr_phrased, check, template


@torch.no_grad()
def encode_text(text, tok, model, device):
    toks = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = model(**toks)
    v = out.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(v, dim=-1).squeeze(0).cpu().numpy()


def chunk_token_ids(token_ids, chunk_size=16):
    return [token_ids[i:i + chunk_size] for i in range(0, len(token_ids), chunk_size)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--n_conversations", type=int, default=600)
    p.add_argument("--n_goals", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--mixed_categories", action="store_true", default=True,
                   help="Per turn sample a random category (default true)")
    p.add_argument("--exclude_category", default="",
                   help="If set, never sample this category (for hold-out training data)")
    p.add_argument("--only_category", default="",
                   help="If set, ONLY sample this category (for hold-out test data)")
    args = p.parse_args()

    rng = random.Random(args.seed)
    device = torch.device("cuda")
    print(f"[div] device={device}, gen={args.gen_model}", flush=True)

    enc_tok = AutoTokenizer.from_pretrained(args.enc_model)
    enc_model = AutoModel.from_pretrained(args.enc_model).to(device).eval()
    enc_dim = enc_model.config.hidden_size

    gen_tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    gen_model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, dtype=torch.float16, trust_remote_code=True,
    ).to(device).eval()
    if gen_tok.pad_token_id is None:
        gen_tok.pad_token_id = gen_tok.eos_token_id
    print(f"[div] models loaded, enc_dim={enc_dim}", flush=True)
    print(f"[div] categories: {list(CATEGORY_MAKERS.keys())}", flush=True)
    print(f"[div] topics: {len(TOPICS)}", flush=True)

    records = []
    cat_follow = {c: [0, 0] for c in CATEGORY_MAKERS}  # [followed, total] per category
    t_start = time.time()
    for ci in range(args.n_conversations):
        messages = [{"role": "system",
                     "content": "Follow the user's instruction precisely. Address the most recent instruction."}]
        # Choose category constraint
        if args.only_category:
            cats_for_conv = [args.only_category] * args.n_goals
        elif args.exclude_category:
            avail = [c for c in CATEGORY_MAKERS if c != args.exclude_category]
            cats_for_conv = [rng.choice(avail) for _ in range(args.n_goals)]
        else:
            cats_for_conv = [None] * args.n_goals  # random per turn
        goals = [make_goal(rng, category=cats_for_conv[i]) for i in range(args.n_goals)]
        turn_outputs, turn_followed, turn_instructions, turn_categories = [], [], [], []
        turn_goal_types = []
        all_concat_tokens = []
        turn_chunk_start_indices = []

        for ti, (cat, gtype, instr, check_fn, _tmpl) in enumerate(goals):
            messages.append({"role": "user", "content": instr})
            turn_instructions.append(instr)
            turn_categories.append(cat)
            turn_goal_types.append(gtype)
            chat = gen_tok.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
            prompt_ids = gen_tok(chat, return_tensors="pt").input_ids.to(device)
            out = gen_model.generate(
                prompt_ids, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=args.temperature, top_p=0.95,
                pad_token_id=gen_tok.pad_token_id,
            )
            new_ids = out[0, prompt_ids.shape[1]:]
            new_text = gen_tok.decode(new_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": new_text})
            turn_outputs.append(new_text)
            followed = bool(check_fn(new_text))
            turn_followed.append(followed)
            cat_follow[cat][0] += int(followed)
            cat_follow[cat][1] += 1
            current_chunk_start = len(all_concat_tokens) // args.chunk_size
            turn_chunk_start_indices.append(current_chunk_start)
            all_concat_tokens.extend(new_ids.tolist())

        if not all_concat_tokens:
            continue
        chunks = chunk_token_ids(all_concat_tokens, args.chunk_size)
        T = len(chunks)

        # Build chunk → turn map
        chunk_to_turn = []
        cur_turn = 0
        for ti in range(T):
            while (cur_turn + 1 < args.n_goals and
                   ti >= turn_chunk_start_indices[cur_turn + 1]):
                cur_turn += 1
            chunk_to_turn.append(cur_turn)

        per_turn_z_goal = [encode_text(instr, enc_tok, enc_model, device)
                            for instr in turn_instructions]

        z_t_list, z_goal_list = [], []
        running = ""
        for ti, ch in enumerate(chunks):
            ch_text = gen_tok.decode(ch, skip_special_tokens=True)
            running = running + ch_text
            zt = encode_text(running, enc_tok, enc_model, device)
            zg = per_turn_z_goal[chunk_to_turn[ti]]
            if np.isnan(zt).any() or np.isnan(zg).any():
                z_t_list = []
                break
            z_t_list.append(zt)
            z_goal_list.append(zg)

        if not z_t_list:
            continue
        z_t_traj = np.stack(z_t_list, axis=0)
        z_goal_traj = np.stack(z_goal_list, axis=0)

        records.append({
            "z_t_traj":    torch.from_numpy(z_t_traj).float(),
            "z_lang_traj": torch.from_numpy(z_goal_traj).float(),
            "z_goal":      torch.from_numpy(per_turn_z_goal[0]).float(),
            "T": z_t_traj.shape[0],
            "succ": int(sum(turn_followed) >= args.n_goals // 2 + 1),
            "sub_id": ci,
            "turn_followed": turn_followed,
            "turn_chunk_starts": turn_chunk_start_indices,
            "turn_instructions": turn_instructions,
            "turn_categories": turn_categories,
            "turn_goal_types": turn_goal_types,
            "turn_outputs": turn_outputs,
        })

        if (ci + 1) % 25 == 0 or ci == 0:
            elapsed = time.time() - t_start
            n = len(records)
            avg_T = float(np.mean([r["T"] for r in records])) if records else 0
            cat_str = " ".join(f"{c[:4]}={cat_follow[c][0]}/{cat_follow[c][1]}"
                                  for c in CATEGORY_MAKERS)
            print(f"[div] [{ci+1}/{args.n_conversations}]  recs={n}  T_avg={avg_T:.1f}  "
                  f"{cat_str}  elapsed={elapsed:.0f}s  "
                  f"eta={elapsed*(args.n_conversations-ci-1)/(ci+1):.0f}s", flush=True)

    print(f"[div] DONE. records={len(records)}", flush=True)
    for c in CATEGORY_MAKERS:
        f, t = cat_follow[c]
        print(f"  {c}: {f}/{t} = {100*f/max(1,t):.0f}% followed", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "records": records,
        "enc_dim": enc_dim,
        "enc_model": args.enc_model,
        "gen_model": args.gen_model,
        "chunk_size": args.chunk_size,
        "n_goals": args.n_goals,
        "categories": list(CATEGORY_MAKERS.keys()),
    }, args.output)
    print(f"[div] === ALL_DONE === saved {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
