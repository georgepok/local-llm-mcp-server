"""INDUCTION v3 — DEFORM, don't replace.

30B finding: goal-identity is RICH in the native manifold (raw-cos HARD=0.825) but the learned
contrastive field DESTROYS it (0.35-0.60) while excelling at engagement (EASY 0.90+). The two are
complementary. A goal-field should be a LIGHT-TOUCH DEFORMATION of the native geometry: keep the
native cosine basin (carries goal-identity for free) and ADD a residual engagement pull. Test:
  combined(lambda) = cos(h, z_G) + lambda * learned_engagement(h, z_G)
Does a residual preserve identity (HARD stays ~0.825) while gaining engagement (EASY rises)?
If yes, the generalized goal-following operator = native basin + small learned deformation.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


class Field(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        lin = nn.Linear(d, r, bias=False); nn.init.orthogonal_(lin.weight); self.P = lin

    def align(self, h, z):
        ph = F.normalize(self.P(h), dim=-1); pz = F.normalize(self.P(z), dim=-1)
        return (ph * pz).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/pokazge/checkpoints/induction_repr_30b.pt")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--n_heldout_goals", type=int, default=20)
    ap.add_argument("--seed_turns", type=int, default=2)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    data = torch.load(args.cache, weights_only=False)
    goals = sorted(data.keys()); d = data[goals[0]][0][0].shape[0]
    rng = np.random.default_rng(7)
    ho = sorted(rng.choice(goals, size=min(args.n_heldout_goals, len(goals) // 2), replace=False).tolist())
    tr = [g for g in goals if g not in ho]

    def split(pairs):
        k = min(args.seed_turns, len(pairs) - 1) if len(pairs) > 1 else len(pairs)
        return list(range(k)), list(range(k, len(pairs)))

    pos = {g: torch.stack([p[0] for p in data[g]]) for g in goals}
    drift = {g: torch.stack([p[1] for p in data[g]]) for g in goals}
    zG = {g: F.normalize(pos[g][split(data[g])[0]].mean(0), dim=0) for g in goals}

    # train the engagement (easy-contrastive) field on train goals
    Hp, Hn, Z = [], [], []
    for g in tr:
        si, _ = split(data[g])
        for i in si:
            Hp.append(pos[g][i]); Hn.append(drift[g][rng.integers(drift[g].shape[0])]); Z.append(zG[g])
    Hp = torch.stack(Hp).to(dev); Hn = torch.stack(Hn).to(dev); Z = torch.stack(Z).to(dev)
    fld = Field(d, args.rank).to(dev)
    opt = torch.optim.Adam(fld.parameters(), lr=5e-3)
    for _ in range(args.epochs):
        opt.zero_grad()
        loss = F.softplus(args.margin - (fld.align(Hp, Z) - fld.align(Hn, Z))).mean()
        loss.backward(); opt.step()

    @torch.no_grad()
    def acc(score_fn, hard):
        sp, sn = [], []
        for g in ho:
            _, hoi = split(data[g]); idx = hoi if hoi else list(range(len(data[g])))
            z = zG[g].to(dev)
            for i in idx:
                hp = pos[g][i].to(dev)
                if hard:
                    others = [gg for gg in ho if gg != g]; gg = others[rng.integers(len(others))]
                    hn = pos[gg][rng.integers(pos[gg].shape[0])].to(dev)
                else:
                    hn = drift[g][rng.integers(drift[g].shape[0])].to(dev)
                sp.append(score_fn(hp, z)); sn.append(score_fn(hn, z))
        sp, sn = torch.tensor(sp), torch.tensor(sn)
        return float((sp > sn).float().mean()), len(sp)

    def native(h, z): return float((F.normalize(h, dim=0) * z).sum())
    def learned(h, z): return float(fld.align(h.unsqueeze(0), z.unsqueeze(0)))
    def combined(lam):
        return lambda h, z: native(h, z) + lam * learned(h, z)

    print(f"[v3] 30B d={d} held-out goals={len(ho)} — DEFORM (native + lambda*residual)", flush=True)
    print(f"{'field':22s} {'EASY (engage)':>14s} {'HARD (identity)':>16s}", flush=True)
    for name, fn in [("native cos", native), ("learned only", learned)] + \
                    [(f"native+{l}*learned", combined(l)) for l in (0.25, 0.5, 1.0, 2.0)]:
        ae, _ = acc(fn, hard=False); ah, n = acc(fn, hard=True)
        print(f"{name:22s} {ae:>14.3f} {ah:>16.3f}", flush=True)
    print("[v3] best = high on BOTH columns => generalized goal-following operator (deform not replace)", flush=True)
    print("[ind3] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
