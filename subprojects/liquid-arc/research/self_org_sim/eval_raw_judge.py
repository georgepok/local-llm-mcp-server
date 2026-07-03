"""Compute raw-judge cross-cat AUC directly from a judged pack.

The raw-judge baseline = use judge_traj (per-chunk) as the logit, no Liquid.
This is the FLOOR that JudgeLiquid refines on top of. Matches the trainer's
auc_on_set convention exactly: AUC for predicting drift (1-label) from -judge.
"""
import argparse
import numpy as np
import torch


def roc_auc_score(y_true, y_score):
    """AUC via Mann-Whitney U (rank-based), no sklearn dependency."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    sum_pos = ranks[y_true == 1].sum()
    return (sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def raw_judge_auc(pack_path):
    pack = torch.load(pack_path, map_location="cpu", weights_only=False)
    records = pack["records"]
    all_judge, all_label = [], []
    for r in records:
        T = int(r["T"])
        if T < 4 or "judge_traj" not in r:
            continue
        judge = r["judge_traj"]
        turn_starts = list(r["turn_chunk_starts"])
        turn_followed = list(r["turn_followed"])
        cur_turn = 0
        for t in range(T):
            while (cur_turn + 1 < len(turn_starts) and
                   t >= turn_starts[cur_turn + 1]):
                cur_turn += 1
            all_judge.append(float(judge[t]))
            all_label.append(int(turn_followed[cur_turn]))
    judge_np = np.array(all_judge)
    label_np = np.array(all_label)
    n_drift = int((label_np == 0).sum())
    n_follow = int((label_np == 1).sum())
    # trainer convention: roc_auc(-logits, 1-labels) -> AUC for drift detection
    auc = roc_auc_score(1 - label_np, -judge_np)
    return auc, n_drift, n_follow, len(records)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    args = p.parse_args()
    auc, nd, nf, nr = raw_judge_auc(args.pack)
    print(f"[raw-judge] {args.pack}")
    print(f"[raw-judge] records={nr}  chunks: drift={nd} follow={nf}")
    print(f"[raw-judge] cross-cat AUC = {auc:.4f}")
