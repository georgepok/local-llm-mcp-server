"""Relabel a judged pack with COMPREHENSIVE turn_followed.

comprehensive_followed = narrow_followed AND (sentence-count constraint satisfied
when the instruction explicitly states one). The narrow check_fn tested only the
ONE intended constraint per goal; many instructions also state "1-2 sentences" /
"N sentences" which a faithful judge (and honest task semantics) must enforce.

Writes a new pack with overwritten turn_followed (keeps judge_traj, hidden states,
everything else). Downstream trainer/eval then measure on the HONEST task.

~20% of contrast-test "follow" turns flip under this (validated 2026-05-28).
"""
import argparse
import re
import torch


def n_sentences(text):
    return len([x for x in re.split(r"[.!?]+", text.strip()) if x.strip()])


def sentence_count_ok(instr, resp):
    il = instr.lower()
    ns = n_sentences(resp)
    if "1-2 sentences" in il or "1 - 2 sentences" in il or "one or two sentences" in il:
        return 1 <= ns <= 2
    if "exactly one sentence" in il or "single sentence" in il or "one sentence" in il:
        return ns == 1
    m = re.search(r"(?:exactly\s+)?(\d+)\s+sentences", il)
    if m:
        return ns == int(m.group(1))
    if "two sentences" in il:
        return ns == 2
    if "three sentences" in il:
        return ns == 3
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    pack = torch.load(args.input, map_location="cpu", weights_only=False)
    n_flip = 0
    n_total = 0
    for r in pack["records"]:
        instrs = r["turn_instructions"]
        outs = r["turn_outputs"]
        followed = list(r["turn_followed"])
        new_followed = []
        for ti in range(len(followed)):
            narrow = bool(followed[ti])
            comp = narrow and sentence_count_ok(instrs[ti], outs[ti])
            new_followed.append(bool(comp))
            n_total += 1
            if comp != narrow:
                n_flip += 1
        # preserve original type (list of bool / tensor)
        r["turn_followed"] = new_followed
        r["turn_followed_narrow"] = followed  # keep original for reference
    print(f"[relabel] {args.input} -> {args.output}")
    print(f"[relabel] turns: {n_total}  flipped follow->drift: {n_flip} ({100*n_flip/max(1,n_total):.1f}%)")
    torch.save(pack, args.output)
    print(f"[relabel] === ALL_DONE ===")


if __name__ == "__main__":
    main()
