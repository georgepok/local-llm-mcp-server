"""INDUCTION v6 — FOCUS as a multiplicative gain (identity-preserving by shape).

v5 lesson: identity is a SMALL-MARGIN native signal; any ADDITIVE engagement term swamps it. So the
deformation must not be additive. Focus = a non-negative GAIN kappa(h) MULTIPLYING the native basin:
  Phi_G(h) = -kappa(h) * cos(h, z_G)
Multiplying scales pull-STRENGTH while leaving the basin DIRECTION (the ranking that carries identity)
far less disturbed than an additive term. kappa(h) is goal-agnostic 'how hard to pull here' (high when
drifting / off-task). This is the FOCUS property of the formalization as an intrinsic field gain.
Compared head-to-head with native (identity backbone) and the v3 additive residual on the same cache.
Runs on CPU.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


class Gain(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return F.softplus(self.net(F.normalize(x, dim=-1)).squeeze(-1)) + 1e-3   # kappa > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/pokazge/checkpoints/induction_repr_30b.pt")
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--n_heldout_goals", type=int, default=20)
    ap.add_argument("--seed_turns", type=int, default=2)
    args = ap.parse_args()
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

    def native_b(h, z): return (F.normalize(h, dim=-1) * z).sum(-1)

    # training pairs (engagement: pos vs drift), scored on the gained field
    Pp, Pd, Pz = [], [], []
    for g in tr:
        si, _ = split(data[g])
        for i in si:
            Pp.append(pos[g][i]); Pd.append(drift[g][rng.integers(drift[g].shape[0])]); Pz.append(zG[g])
    Pp = torch.stack(Pp); Pd = torch.stack(Pd); Pz = torch.stack(Pz)
    kap = Gain(d)
    opt = torch.optim.Adam(kap.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(args.epochs):
        opt.zero_grad()
        sp = kap(Pp) * native_b(Pp, Pz); sn = kap(Pd) * native_b(Pd, Pz)
        loss = F.softplus(args.margin - (sp - sn)).mean()
        loss.backward(); opt.step()

    @torch.no_grad()
    def acc(score_fn, hard):
        a, b = [], []
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
                a.append(float(score_fn(hp.unsqueeze(0), z.unsqueeze(0)))); b.append(float(score_fn(hn.unsqueeze(0), z.unsqueeze(0))))
        a, b = torch.tensor(a), torch.tensor(b)
        return float((a > b).float().mean())

    gained = lambda h, z: kap(h) * native_b(h, z)
    print(f"[v6] 30B d={d} held-out={len(ho)} — FOCUS as multiplicative gain kappa(h)*cos(h,z_G)", flush=True)
    print(f"{'field':22s} {'EASY(engage)':>13s} {'HARD(identity)':>15s}", flush=True)
    print(f"{'native cos':22s} {acc(native_b, False):>13.3f} {acc(native_b, True):>15.3f}", flush=True)
    print(f"{'kappa(h)*cos':22s} {acc(gained, False):>13.3f} {acc(gained, True):>15.3f}", flush=True)
    print("[v6] target: identity HOLDS near native (multiplicative preserves direction) while engage rises", flush=True)
    print("[ind6] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
