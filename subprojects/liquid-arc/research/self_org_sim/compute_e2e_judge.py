"""End-to-end DENSE per-turn judging.

For each conversation, query the judge 5 times (once per turn), each query
showing the FULL conversation context. The judge evaluates "did turn T follow
its instruction given the full conversation context."

This captures cross-turn dynamics — commitment carrying through, late-turn
drift, context-dependent goal interpretation — that isolated per-turn judging
misses.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_e2e_prompt(turn_instructions, turn_outputs, target_turn_idx):
    """Build a prompt showing all turns; ask about target_turn_idx specifically."""
    parts = [
        "You are evaluating a multi-turn conversation between a user and an AI assistant.",
        "Here is the full conversation:",
        "",
    ]
    for ti, (instr, out) in enumerate(zip(turn_instructions, turn_outputs)):
        parts.append(f"[Turn {ti+1}]")
        parts.append(f"User: {instr}")
        parts.append(f"AI: {out}")
        parts.append("")
    parts.append(
        f"Considering the full conversation above, did the AI's response in "
        f"Turn {target_turn_idx+1} follow the user's instruction for Turn {target_turn_idx+1}? "
        f"Answer with a single word: Yes or No."
    )
    parts.append("")
    parts.append("Answer:")
    return "\n".join(parts)


@torch.no_grad()
def judge_e2e(model, tok, turn_instructions, turn_outputs, device, yes_id, no_id):
    """Returns list of (yes_logit - no_logit) per turn, evaluated in full E2E context."""
    n_turns = len(turn_instructions)
    scores = []
    for ti in range(n_turns):
        prompt = build_e2e_prompt(turn_instructions, turn_outputs, ti)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        out = model(**inputs)
        logits = out.logits[0, -1]
        scores.append(float(logits[yes_id].item() - logits[no_id].item()))
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n_records", type=int, default=-1)
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[e2e] device={device}", flush=True)
    print(f"[e2e] loading {args.gen_model}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, dtype=torch.float16, trust_remote_code=True,
    ).to(device).eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]
    print(f"[e2e] yes_id={yes_id} no_id={no_id}", flush=True)

    pack = torch.load(args.input, map_location="cpu", weights_only=False)
    records = pack["records"]
    if args.n_records > 0:
        records = records[:args.n_records]
    print(f"[e2e] {len(records)} records, computing dense E2E judgments...", flush=True)

    t_start = time.time()
    for ri, r in enumerate(records):
        T = int(r["T"])
        turn_instructions = r["turn_instructions"]
        turn_outputs = r["turn_outputs"]
        turn_chunk_starts = list(r["turn_chunk_starts"])
        n_turns = len(turn_instructions)

        # E2E judgments per turn (each sees full context)
        e2e_judgments = judge_e2e(
            model, tok, turn_instructions, turn_outputs, device, yes_id, no_id
        )

        # Per-chunk: each chunk inherits its turn's E2E judgment
        judge_traj = []
        cur_turn = 0
        for t in range(T):
            while (cur_turn + 1 < n_turns and
                   t >= turn_chunk_starts[cur_turn + 1]):
                cur_turn += 1
            judge_traj.append(e2e_judgments[cur_turn])
        r["judge_traj"] = torch.tensor(judge_traj, dtype=torch.float32)
        r["turn_judgments_e2e"] = torch.tensor(e2e_judgments, dtype=torch.float32)
        # Keep prior turn_judgments (isolated) if present, for comparison

        if (ri + 1) % 25 == 0 or ri == 0:
            elapsed = time.time() - t_start
            print(f"[e2e] [{ri+1}/{len(records)}]  elapsed={elapsed:.0f}s  "
                  f"eta={elapsed*(len(records)-ri-1)/(ri+1):.0f}s", flush=True)

    print(f"[e2e] DONE. saving to {args.output}...", flush=True)
    pack["records"] = records
    pack["judge_per"] = "e2e_turn"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, args.output)
    print(f"[e2e] === ALL_DONE === saved {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
