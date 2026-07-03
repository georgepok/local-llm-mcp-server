"""Diagnostic: inspect what the judge model actually does on specific turns.

Loads a judge model and, for a handful of turns (mix of followed/drifted),
prints: the instruction, response snippet, true label, Yes/No logits + diff,
the argmax next token (what the model WANTS to say), and a short greedy
generation. Reveals whether the model is engaging with the Yes/No task at all.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_USER = (
    "You are evaluating whether an AI response follows an instruction.\n\n"
    "Instruction: {goal}\n\n"
    "AI response: {response}\n\n"
    "Did the AI response follow the instruction? Answer with a single word: Yes or No."
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--input", default="/home/pokazge/data/hs_test_contrast.pt")
    p.add_argument("--chat", action="store_true")
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, dtype=torch.float16, trust_remote_code=True).to(device).eval()
    yes_id = tok("Yes", add_special_tokens=False).input_ids[0]
    no_id = tok("No", add_special_tokens=False).input_ids[0]
    yes_id_sp = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id_sp = tok(" No", add_special_tokens=False).input_ids[0]
    print(f"Yes={yes_id}/{yes_id_sp}  No={no_id}/{no_id_sp}  chat={args.chat}")

    pack = torch.load(args.input, map_location="cpu", weights_only=False)
    # collect turns: alternate followed / drifted
    turns = []
    for r in pack["records"]:
        for ti, (g, o, f) in enumerate(zip(r["turn_instructions"], r["turn_outputs"],
                                              list(r["turn_followed"]))):
            turns.append((g, o, int(f)))
    foll = [t for t in turns if t[2] == 1][:args.n // 2]
    drift = [t for t in turns if t[2] == 0][:args.n // 2]
    sel = []
    for a, b in zip(foll, drift):
        sel += [a, b]

    for gi, (goal, resp, lab) in enumerate(sel):
        if args.chat:
            msg = [{"role": "user", "content": JUDGE_USER.format(goal=goal, response=resp)}]
            text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
            yi, ni = yes_id, no_id
        else:
            text = JUDGE_USER.format(goal=goal, response=resp) + "\n\nAnswer:"
            inputs = tok(text, return_tensors="pt").to(device)
            yi, ni = yes_id_sp, no_id_sp
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1]
            gen = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        argmax_tok = tok.decode([int(logits.argmax())])
        gen_txt = tok.decode(gen[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        diff = float(logits[yi] - logits[ni])
        print(f"\n--- ex{gi} label={'FOLLOW' if lab else 'DRIFT '} ---")
        print(f"  GOAL: {goal[:110]}")
        print(f"  RESP: {resp[:110]!r}")
        print(f"  Yes_logit={float(logits[yi]):.2f} No_logit={float(logits[ni]):.2f} diff={diff:+.2f}")
        print(f"  argmax_next={argmax_tok!r}  greedy_gen={gen_txt!r}")


if __name__ == "__main__":
    main()
