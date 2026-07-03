"""INDUCTION v4 — IDENTITY-PRESERVING deformation.

v3 showed native+0.25*residual gets engage 0.975 but identity only 0.70 (native alone 0.85): the
residual, trained for engagement only, still erodes goal-identity. Fix: train the residual on the
COMBINED field against BOTH objectives jointly — engagement (vs drift) AND identity (vs other goals'
task) — so the deformation must add following-pull WITHOUT breaking the native goal-separation.
Also test an ORTHOGONAL residual (acts only in the complement of the goal-identity subspace).

Target: engage ~0.97 AND identity ~0.85 simultaneously => the deformation preserves identity.
Runs on cached 30B reps (instant).
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


class Residual(nn.Module):
    def __init__(self, d, r, U=None):
        super().__init__()
        lin = nn.Linear(d, r, bias=False); nn.init.orthogonal_(lin.weight); self.P = lin
        self.register_buffer("U", U if U is not None else torch.zeros(0, d))  # identity subspace to remove

    def _strip(self, x):
        if self.U.numel() == 0:
            return x
        return x - (x @ self.U.t()) @ self.U                                  # remove goal-identity component

    def align(self, h, z):
        ph = F.normalize(self.P(self._strip(h)), dim=-1)
        pz = F.normalize(self.P(self._strip(z)), dim=-1)
        return (ph * pz).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/pokazge/checkpoints/induction_repr_30b.pt")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--n_heldout_goals", type=int, default=20)
    ap.add_argument("--seed_turns", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--orth_k", type=int, default=0, help="dims of goal-identity subspace to strip (0=off)")
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

    U = None
    if args.orth_k > 0:                                                       # goal-identity subspace = PCA of train centroids
        Zc = torch.stack([zG[g] for g in tr])
        U = torch.linalg.svd(Zc - Zc.mean(0), full_matrices=False)[2][:args.orth_k].to(dev)

    # train pairs: positives (seed turns), their drift, their goal code, and goal id (for cross-goal negs)
    P_, D_, Z_, G_ = [], [], [], []
    for g in tr:
        si, _ = split(data[g])
        for i in si:
            P_.append(pos[g][i]); D_.append(drift[g][rng.integers(drift[g].shape[0])]); Z_.append(zG[g]); G_.append(g)
    P_ = torch.stack(P_).to(dev); D_ = torch.stack(D_).to(dev); Z_ = torch.stack(Z_).to(dev); G_ = np.array(G_)
    pos_dev = {g: pos[g].to(dev) for g in tr}

    def native(h, z): return (F.normalize(h, dim=-1) * z).sum(-1)

    def run(beta):
        res = Residual(d, args.rank, U).to(dev)
        opt = torch.optim.Adam(res.parameters(), lr=5e-3)
        for _ in range(args.epochs):
            opt.zero_grad()
            comb = lambda h, z: native(h, z) + args.scale * res.align(h, z)
            eng = F.softplus(args.margin - (comb(P_, Z_) - comb(D_, Z_))).mean()      # engagement vs drift
            # identity: positive's own goal vs another train goal's task continuation
            negs = torch.stack([pos_dev[g2][rng.integers(pos_dev[g2].shape[0])]
                                for g2 in [tr[(tr.index(g) + 1 + rng.integers(len(tr) - 1)) % len(tr)] for g in G_]])
            idn = F.softplus(args.margin - (comb(P_, Z_) - comb(negs, Z_))).mean()    # identity vs other goal
            (eng + beta * idn).backward(); opt.step()
        return res

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
                sp.append(float(score_fn(hp.unsqueeze(0), z.unsqueeze(0)))); sn.append(float(score_fn(hn.unsqueeze(0), z.unsqueeze(0))))
        sp, sn = torch.tensor(sp), torch.tensor(sn)
        return float((sp > sn).float().mean())

    print(f"[v4] 30B d={d} held-out={len(ho)} scale={args.scale} orth_k={args.orth_k} — identity-preserving deform", flush=True)
    print(f"{'field':28s} {'EASY(engage)':>13s} {'HARD(identity)':>15s}", flush=True)
    print(f"{'native cos':28s} {acc(lambda h,z: native(h,z), False):>13.3f} {acc(lambda h,z: native(h,z), True):>15.3f}", flush=True)
    for beta in (0.0, 0.5, 1.0, 2.0, 4.0):
        res = run(beta)
        comb = lambda h, z: native(h, z) + args.scale * res.align(h, z)
        print(f"{'combined beta=' + str(beta):28s} {acc(comb, False):>13.3f} {acc(comb, True):>15.3f}", flush=True)
    print("[v4] target: engage ~0.97 AND identity ~0.85 (beta>0 should lift identity vs v3's 0.70)", flush=True)
    print("[ind4] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
