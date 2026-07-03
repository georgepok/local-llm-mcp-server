"""Decisive test: is a strong judge actually GOOD, but penalized by NARROW labels?

Many instructions carry incidental phrasing the original check_fn ignores —
most often "Write 1-2 sentences about X, ...". A thorough judge evaluates the
FULL instruction and disagrees with the narrow label.

This recomputes a COMPREHENSIVE label = narrow_label AND (incidental sentence-
count constraint, when the instruction states one), from the stored
(instruction, output) text. Then reports raw-judge AUC against BOTH the narrow
label and the comprehensive label, for whatever judge_traj is in the pack.

If the strong-judge pack's AUC jumps under comprehensive labels, the judge was
right and the labels were narrow -> the ceiling was a label artifact.
"""
import argparse
import re
import numpy as np
import torch


def roc_auc(y_true, y_score):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    pos = (y_true == 1).sum(); neg = (y_true == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score)); ranks[order] = np.arange(1, len(y_score) + 1)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return (ranks[y_true == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def n_sentences(text):
    s = [x for x in re.split(r"[.!?]+", text.strip()) if x.strip()]
    return len(s)


def sentence_count_ok(instr, resp):
    """Detect an explicit sentence-count requirement in the instruction and
    check the response against it. Returns True if no such requirement OR it's met."""
    il = instr.lower()
    ns = n_sentences(resp)
    if "1-2 sentences" in il or "1 - 2 sentences" in il or "one or two sentences" in il:
        return 1 <= ns <= 2
    if "exactly one sentence" in il or "one sentence" in il or "single sentence" in il:
        return ns == 1
    m = re.search(r"(?:exactly\s+)?(\d+)\s+sentences", il)
    if m:
        return ns == int(m.group(1))
    if "two sentences" in il:
        return ns == 2
    if "three sentences" in il:
        return ns == 3
    return True  # no sentence-count constraint stated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    args = p.parse_args()
    pack = torch.load(args.pack, map_location="cpu", weights_only=False)
    j_all, narrow_all, comp_all = [], [], []
    n_changed = 0
    for r in pack["records"]:
        T = int(r["T"])
        if T < 4 or "judge_traj" not in r:
            continue
        judge = r["judge_traj"]
        starts = list(r["turn_chunk_starts"])
        followed = list(r["turn_followed"])
        instrs = r["turn_instructions"]
        outs = r["turn_outputs"]
        # comprehensive per-turn label
        comp_turn = []
        for ti in range(len(followed)):
            narrow = int(followed[ti])
            comp = narrow and sentence_count_ok(instrs[ti], outs[ti])
            comp_turn.append(int(comp))
            if int(comp) != narrow:
                n_changed += 1
        cur = 0
        for t in range(T):
            while cur + 1 < len(starts) and t >= starts[cur + 1]:
                cur += 1
            j_all.append(float(judge[t]))
            narrow_all.append(int(followed[cur]))
            comp_all.append(int(comp_turn[cur]))
    j = np.array(j_all)
    narrow = np.array(narrow_all); comp = np.array(comp_all)
    auc_narrow = roc_auc(1 - narrow, -j)
    auc_comp = roc_auc(1 - comp, -j)
    print(f"[comp] {args.pack}")
    print(f"[comp] chunks={len(j)}  narrow follow={int(narrow.sum())}  comp follow={int(comp.sum())}")
    print(f"[comp] turns flipped follow->drift by sentence-count: {n_changed}")
    print(f"[comp] raw-judge AUC  vs NARROW labels = {auc_narrow:.4f}")
    print(f"[comp] raw-judge AUC  vs COMPREHENSIVE  = {auc_comp:.4f}")


if __name__ == "__main__":
    main()
