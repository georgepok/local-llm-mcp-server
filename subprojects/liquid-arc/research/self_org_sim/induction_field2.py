"""INDUCTION v2 — HARD negatives (the real crux test).

v1 flaw: near-miss = generic off-topic tangent (far in rep-space) -> raw-cos already ~0.72, the test
didn't stress structure-vs-familiarity. The framework demands negatives NEAR in representation space
but differing in goal-service. Cleanest such negative: a goal-serving continuation FOR A DIFFERENT
GOAL. Positive and negative are both fluent on-task continuations; they differ ONLY in which goal they
serve. If a learned low-rank field separates these on HELD-OUT goals better than raw-cos (topic
similarity to the seed centroid), the field encodes goal-STRUCTURE that generalizes — induction.
If it ties raw-cos, the "field" is just seed-familiarity (text in disguise).

Loads cached reprs from induction_field.py (no regeneration). Reports EASY (drift) for reference and
HARD (cross-goal) as the decisive test, each for contrastive field / positive-only / raw-cos.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


class Field(nn.Module):
    def __init__(self, d, r, nonlin=False):
        super().__init__()
        if nonlin:
            self.P = nn.Sequential(nn.Linear(d, 2 * r), nn.GELU(), nn.Linear(2 * r, r))
        else:
            lin = nn.Linear(d, r, bias=False); nn.init.orthogonal_(lin.weight); self.P = lin
        self.nonlin = nonlin

    def align(self, h, z):
        ph = F.normalize(self.P(h), dim=-1); pz = F.normalize(self.P(z), dim=-1)
        return (ph * pz).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/pokazge/checkpoints/induction_repr.pt")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--n_heldout_goals", type=int, default=8)
    ap.add_argument("--seed_turns", type=int, default=2)
    ap.add_argument("--nonlin", action="store_true")
    ap.add_argument("--neg_per_pos", type=int, default=8, help="hard negatives sampled per positive during fit")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    data = torch.load(args.cache, weights_only=False)
    goals = sorted(data.keys())
    d = data[goals[0]][0][0].shape[0]
    rng = np.random.default_rng(7)
    ho = set(rng.choice(goals, size=min(args.n_heldout_goals, len(goals) // 2), replace=False).tolist())
    tr = [g for g in goals if g not in ho]
    ho = sorted(ho)

    def split(pairs):
        k = min(args.seed_turns, len(pairs) - 1) if len(pairs) > 1 else len(pairs)
        return list(range(k)), list(range(k, len(pairs)))

    pos = {g: torch.stack([p[0] for p in data[g]]) for g in goals}     # [T, d] goal-serving reps
    drift = {g: torch.stack([p[1] for p in data[g]]) for g in goals}   # [T, d] off-topic reps
    zG = {}
    for g in goals:
        si, _ = split(data[g])
        zG[g] = F.normalize(pos[g][si].mean(0), dim=0)

    # training set: positives = train goals' seed turns; z = that goal's code
    fit_h, fit_z, fit_g = [], [], []
    for g in tr:
        si, _ = split(data[g])
        for i in si:
            fit_h.append(pos[g][i]); fit_z.append(zG[g]); fit_g.append(g)
    Hp = torch.stack(fit_h).to(device); Z = torch.stack(fit_z).to(device)
    fit_g = np.array(fit_g)
    tr_pos_bank = {g: pos[g].to(device) for g in tr}

    def fit(contrastive, hard):
        fld = Field(d, args.rank, args.nonlin).to(device)
        opt = torch.optim.Adam(fld.parameters(), lr=5e-3)
        for ep in range(args.epochs):
            opt.zero_grad()
            ap_ = fld.align(Hp, Z)
            if not contrastive:
                loss = (1 - ap_).mean()
            else:
                negs = []
                for k, g in enumerate(fit_g):
                    if hard:                                            # hard neg = other goals' goal-serving reps
                        others = [gg for gg in tr if gg != g]
                        gg = others[rng.integers(len(others))]
                        bank = tr_pos_bank[gg]
                    else:                                               # easy neg = this goal's drift reps
                        bank = drift[g].to(device)
                    negs.append(bank[rng.integers(bank.shape[0])])
                Hn = torch.stack(negs)
                an_ = fld.align(Hn, Z)
                loss = F.softplus(args.margin - (ap_ - an_)).mean()
            loss.backward(); opt.step()
        return fld

    @torch.no_grad()
    def evalp(fld, hard):
        """Held-out GOALS. pos = a held-out goal's continuation; neg = another held-out goal's
        continuation (hard) or the same goal's drift (easy). acc = field prefers the matching one."""
        sp, sn = [], []
        for g in ho:
            _, hoi = split(data[g])
            idx = hoi if hoi else list(range(len(data[g])))
            z = zG[g].to(device)
            for i in idx:
                hp = pos[g][i].to(device)
                if hard:
                    others = [gg for gg in ho if gg != g]
                    gg = others[rng.integers(len(others))]
                    hn = pos[gg][rng.integers(pos[gg].shape[0])].to(device)
                else:
                    hn = drift[g][rng.integers(drift[g].shape[0])].to(device)
                if fld is None:
                    sp.append(float((F.normalize(hp, dim=0) * z).sum()))
                    sn.append(float((F.normalize(hn, dim=0) * z).sum()))
                else:
                    sp.append(float(fld.align(hp.unsqueeze(0), z.unsqueeze(0))))
                    sn.append(float(fld.align(hn.unsqueeze(0), z.unsqueeze(0))))
        sp, sn = torch.tensor(sp), torch.tensor(sn)
        return float((sp > sn).float().mean()), len(sp)

    print(f"[v2] d={d} train={len(tr)} held-out goals={len(ho)} rank={args.rank} nonlin={args.nonlin}", flush=True)
    print("\n=== INDUCTION v2  (held-out GOALS; acc = field prefers the goal-matching continuation) ===", flush=True)
    for hard, lbl in [(False, "EASY neg (off-topic drift)   "), (True, "HARD neg (other goal's task) ")]:
        fc = fit(True, hard); fp = fit(False, hard)
        ac, n = evalp(fc, hard); apv, _ = evalp(fp, hard); ar, _ = evalp(None, hard)
        print(f"  {lbl} n={n:3d} | contrastive={ac:.3f}  positive-only={apv:.3f}  raw-cos={ar:.3f}", flush=True)
    print("\n[verdict] induction iff contrastive >> raw-cos on HARD negatives (structure beyond topic-similarity).", flush=True)
    print("[ind2] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
