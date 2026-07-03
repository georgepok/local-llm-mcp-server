"""Compute per-chunk LLM judgment scores for hidden-state trajectory data.

For each conversation:
  Reconstruct generated text up to each chunk
  Query Qwen: "did the partial response follow the instruction?"
  Save judge_score[t] alongside hidden states

Output: same data + judge_traj field, ready for Liquid+judge wrapper training.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def judge_logit_diff(model, tok, goal, response, device, yes_id, no_id):
    prompt = (
        f"You are evaluating whether an AI response follows an instruction.\n\n"
        f"Instruction: {goal}\n\n"
        f"AI response: {response}\n\n"
        f"Did the AI response follow the instruction? Answer with a single word: Yes or No.\n\n"
        f"Answer:"
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = model(**inputs)
    logits = out.logits[0, -1]
    return float(logits[yes_id].item() - logits[no_id].item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--judge_per", choices=["chunk", "turn"], default="turn",
                   help="Compute judgment per chunk (richer) or per turn (faster)")
    p.add_argument("--n_records", type=int, default=-1,
                   help="-1 for all records")
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[judge] device={device}, judge_per={args.judge_per}", flush=True)
    print(f"[judge] loading {args.gen_model}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.gen_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.gen_model, dtype=torch.float16, trust_remote_code=True,
    ).to(device).eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]
    print(f"[judge] yes_id={yes_id} no_id={no_id}", flush=True)

    pack = torch.load(args.input, map_location="cpu", weights_only=False)
    records = pack["records"]
    if args.n_records > 0:
        records = records[:args.n_records]
    print(f"[judge] {len(records)} records, computing judgments...", flush=True)

    t_start = time.time()
    for ri, r in enumerate(records):
        T = int(r["T"])
        turn_instructions = r["turn_instructions"]
        turn_outputs = r["turn_outputs"]
        turn_chunk_starts = list(r["turn_chunk_starts"])
        n_turns = len(turn_instructions)

        # Per-turn judgment: single query per turn
        turn_judgments = []
        for ti in range(n_turns):
            score = judge_logit_diff(model, tok,
                                        turn_instructions[ti], turn_outputs[ti],
                                        device, yes_id, no_id)
            turn_judgments.append(score)

        # Map per-chunk: each chunk gets the judgment of the turn it belongs to
        judge_traj = []
        cur_turn = 0
        for t in range(T):
            while (cur_turn + 1 < n_turns and
                   t >= turn_chunk_starts[cur_turn + 1]):
                cur_turn += 1
            judge_traj.append(turn_judgments[cur_turn])
        r["judge_traj"] = torch.tensor(judge_traj, dtype=torch.float32)
        r["turn_judgments"] = torch.tensor(turn_judgments, dtype=torch.float32)

        if (ri + 1) % 25 == 0 or ri == 0:
            elapsed = time.time() - t_start
            print(f"[judge] [{ri+1}/{len(records)}]  elapsed={elapsed:.0f}s  "
                  f"eta={elapsed*(len(records)-ri-1)/(ri+1):.0f}s", flush=True)

    print(f"[judge] DONE. saving to {args.output}...", flush=True)
    pack["records"] = records
    pack["judge_per"] = args.judge_per
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, args.output)
    print(f"[judge] === ALL_DONE === saved {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
