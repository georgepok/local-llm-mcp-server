"""LLM-as-judge baseline: query Qwen directly for goal-following judgment per chunk.

For each conversation in test set:
  Per chunk: query Qwen "did the response so far follow the instruction?"
  Extract P(Yes) - P(No) probability
  Aggregate per-turn (mean, last) → drift score

If this gets high AUC on held-out CONTRAST category, the LLM's task-agnostic
instruction-following knowledge can be leveraged directly — Liquid wrapper would
then consume this signal for temporal smoothing.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def roc_auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rs = scores[labels == 1]
    rn = scores[labels == 0]
    wins = (rs[:, None] > rn[None, :]).sum()
    ties = (rs[:, None] == rn[None, :]).sum()
    return (wins + 0.5 * ties) / (n_pos * n_neg)


@torch.no_grad()
def judge_probability(model, tok, goal, response, device):
    """Returns P(Yes) - P(No) from a single-token judgment prompt."""
    prompt = (
        f"You are evaluating whether an AI response follows an instruction.\n\n"
        f"Instruction: {goal}\n\n"
        f"AI response: {response}\n\n"
        f"Did the AI response follow the instruction? Answer with a single word: Yes or No.\n\n"
        f"Answer:"
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model(**inputs)
    logits = out.logits[0, -1]  # next-token logits
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]
    yes_logit = logits[yes_id].item()
    no_logit = logits[no_id].item()
    # Soft probability
    diff = yes_logit - no_logit
    return diff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--test_traj", required=True)
    p.add_argument("--n_records", type=int, default=200,
                   help="Limit number of test records for speed")
    p.add_argument("--judge_per", choices=["turn", "chunk"], default="turn",
                   help="Query judge per turn (cheaper) or per chunk (richer)")
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

    pack = torch.load(args.test_traj, map_location="cpu", weights_only=False)
    records = pack["records"][:args.n_records]
    print(f"[judge] {len(records)} records to probe", flush=True)

    rows = []
    t_start = time.time()
    for ri, r in enumerate(records):
        turn_instructions = r["turn_instructions"]
        turn_outputs = r["turn_outputs"]
        turn_followed = r["turn_followed"]
        for ti, (instr, out_text, followed) in enumerate(
                zip(turn_instructions, turn_outputs, turn_followed)):
            score = judge_probability(model, tok, instr, out_text, device)
            rows.append({
                "sub_id": int(r["sub_id"]),
                "turn_idx": ti,
                "followed": int(followed),
                "judge_score": float(score),
            })
        if (ri + 1) % 20 == 0:
            elapsed = time.time() - t_start
            print(f"[judge] [{ri+1}/{len(records)}]  elapsed={elapsed:.0f}s  "
                  f"eta={elapsed*(len(records)-ri-1)/(ri+1):.0f}s", flush=True)

    labels = np.array([1 - r["followed"] for r in rows], dtype=int)
    scores = np.array([-r["judge_score"] for r in rows])  # higher score = drift
    n_drift = int(labels.sum())
    print(f"[judge] {len(rows)} per-turn judgments, drifts={n_drift}/{len(rows)}")
    print()
    auc = roc_auc(scores, labels)
    print(f"OVERALL judge AUC (predict DRIFT from low Yes-probability): {auc:.3f}")
    mF = scores[labels == 1].mean()
    mS = scores[labels == 0].mean()
    print(f"  mean judge score (FAIL, drift): {-mF:+.3f}")
    print(f"  mean judge score (SUCC, follow): {-mS:+.3f}")
    print(f"  delta: {-mF + mS:+.3f}")

    print()
    print("PER-TURN-POSITION AUC:")
    for ti in sorted(set(r["turn_idx"] for r in rows)):
        ti_rows = [r for r in rows if r["turn_idx"] == ti]
        ti_labels = np.array([1 - r["followed"] for r in ti_rows], dtype=int)
        if ti_labels.sum() < 2 or len(ti_rows) - ti_labels.sum() < 2:
            continue
        ti_scores = np.array([-r["judge_score"] for r in ti_rows])
        ti_auc = roc_auc(ti_scores, ti_labels)
        print(f"  turn {ti}: n={len(ti_rows)} drifts={int(ti_labels.sum())}  AUC={ti_auc:.3f}")


if __name__ == "__main__":
    main()
