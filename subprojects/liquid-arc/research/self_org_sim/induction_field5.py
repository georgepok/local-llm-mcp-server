"""INDUCTION v5 — identity-preserving deform via a GOAL-INDEPENDENT engagement potential.

v3/v4: a goal-conditioned residual boosts engagement but erodes identity (it perturbs cross-goal
geometry). Cleaner: separate the two components of the goal-field.
  Phi_G(h) = -cos(h, z_G)              # GOAL-SPECIFIC: native basin, carries identity (free, 0.85)
             - lambda * g(h)           # GOAL-AGNOSTIC: learned scalar "on-task-ness" potential
g(h) does not depend on z_G, so it adds the SAME amount to every goal -> cannot change cross-goal
RANKING -> identity preserved. But g(pos) > g(drift) -> engagement boosted. The deformation deepens
on-task regions uniformly; the native basin keeps the goal identity.

This is the formalization in two parts: attractor structure (native basin) + focus/engagement
(goal-agnostic potential). Runs on CPU (cached reps; avoids GPU contention).
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


class Engage(nn.Module):
    """Goal-agnostic on-task-ness potential g(h) -> scalar."""
    def __init__(self, d, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(F.normalize(x, dim=-1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/pokazge/checkpoints/induction_repr_30b.pt")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--n_heldout_goals", type=int, default=20)
    ap.add_argument("--seed_turns", type=int, default=2)
    args = ap.parse_args()
    dev = torch.device("cpu")                                                 # tiny; avoid GPU contention
    torch.manual_seed(0)
    data = torch.load(args.cache, weights_only=False, map_location="cpu")
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

    # train g(h) on engagement: g(pos) > g(drift), goal-agnostic
    P_, D_ = [], []
    for g in tr:
        si, _ = split(data[g])
        for i in si:
            P_.append(pos[g][i]); D_.append(drift[g][rng.integers(drift[g].shape[0])])
    P_ = torch.stack(P_); D_ = torch.stack(D_)
    g = Engage(d)
    opt = torch.optim.Adam(g.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(args.epochs):
        opt.zero_grad()
        loss = F.softplus(args.margin - (g(P_) - g(D_))).mean()
        loss.backward(); opt.step()

    def native(h, z): return (F.normalize(h, dim=-1) * z).sum(-1)

    @torch.no_grad()
    def acc(scale, hard):
        sp, sn = [], []
        for gg in ho:
            _, hoi = split(data[gg]); idx = hoi if hoi else list(range(len(data[gg])))
            z = zG[gg]
            for i in idx:
                hp = pos[gg][i]
                if hard:
                    others = [x for x in ho if x != gg]; o = others[rng.integers(len(others))]
                    hn = pos[o][rng.integers(pos[o].shape[0])]
                else:
                    hn = drift[gg][rng.integers(drift[gg].shape[0])]
                sp.append(float(native(hp.unsqueeze(0), z.unsqueeze(0)) + scale * g(hp.unsqueeze(0))))
                sn.append(float(native(hn.unsqueeze(0), z.unsqueeze(0)) + scale * g(hn.unsqueeze(0))))
        sp, sn = torch.tensor(sp), torch.tensor(sn)
        return float((sp > sn).float().mean())

    print(f"[v5] 30B d={d} held-out={len(ho)} — native basin + goal-AGNOSTIC engagement potential", flush=True)
    print(f"{'field':26s} {'EASY(engage)':>13s} {'HARD(identity)':>15s}", flush=True)
    for scale in (0.0, 0.1, 0.2, 0.4, 0.8):
        lbl = "native cos" if scale == 0 else f"native + {scale}*g(h)"
        print(f"{lbl:26s} {acc(scale, False):>13.3f} {acc(scale, True):>15.3f}", flush=True)
    print("[v5] target: identity HOLDS at native ~0.85 (g is goal-agnostic) while engage rises", flush=True)
    print("[ind5] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
