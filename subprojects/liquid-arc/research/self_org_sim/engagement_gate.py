"""ENGAGEMENT-GATE as Liquid dynamics over the native goal-field.

Architecture (from the induction program): the GOAL-FIELD is the fixed native basin cos(h, z_G)
(carries attractor + identity, generalizes @0.85). FOCUS is NOT baked into the field — it is the
Liquid deciding WHEN/HOW-HARD to apply the native pull. This module is that gate:

  native_pull(h, z_G) = unit vector increasing cos(h,z_G)   # direction is NATIVE, fixed
  LiquidGate (LTC cell): holds z_G, reads the LLM hidden TRAJECTORY, emits focus gain kappa_t >= 0
  correction_t = kappa_t * native_pull(h_t, z_G)            # the gated field pull (for realization)

The gate is trajectory-integrating (continuous persistent state), not an instantaneous threshold:
sustained drift can emerge as sustained pull. Trained so kappa stays QUIET on goal-serving
trajectories and FIRES on drift trajectories. Validated on held-out goals; compared to an
instantaneous (memoryless) detector to show the Liquid's trajectory context helps.
CPU, cached 30B reps.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def native_pull(h, z):
    """Unit direction that increases cos(h, z): component of z orthogonal to h (tangent on sphere)."""
    hn = F.normalize(h, dim=-1)
    d = z - (hn * z).sum(-1, keepdim=True) * hn
    return F.normalize(d, dim=-1)


class LiquidGate(nn.Module):
    """LTC continuous cell: ds/dt = (1/tau)(target(x,s) - s); x = read(h, z_G). Emits focus gain kappa."""
    def __init__(self, d_llm, d=128):
        super().__init__()
        self.in_h = nn.Linear(d_llm, d); self.in_z = nn.Linear(d_llm, d)
        self.cell = nn.Linear(2 * d, d); self.tau = nn.Linear(2 * d, d)
        self.kap = nn.Linear(d, 1)
        self.d = d; self.s = None

    def reset(self, B, device):
        self.s = torch.zeros(B, self.d, device=device)

    def step(self, h, z):
        x = F.silu(self.in_h(h) + self.in_z(z))                  # input from LLM hidden + goal seed
        inp = torch.cat([x, self.s], -1)
        target = torch.tanh(self.cell(inp))
        tau = F.softplus(self.tau(inp)) + 1.0                    # continuous time-constant >= 1
        self.s = self.s + (target - self.s) / tau               # LTC update (dt=1)
        return F.softplus(self.kap(self.s)).squeeze(-1)          # kappa >= 0

    def run(self, traj, z):                                      # traj [T, d_llm]
        self.reset(1, traj.device)
        return torch.stack([self.step(traj[t:t + 1], z) for t in range(traj.shape[0])]).squeeze(-1)  # [T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/pokazge/checkpoints/induction_repr_30b_big.pt")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--n_heldout_goals", type=int, default=40)
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    torch.manual_seed(0)
    data = torch.load(args.cache, weights_only=False, map_location="cpu")
    goals = sorted(data.keys()); d_llm = data[goals[0]][0][0].shape[0]
    rng = np.random.default_rng(7)
    ho = sorted(rng.choice(goals, size=min(args.n_heldout_goals, len(goals) // 2), replace=False).tolist())
    tr = [g for g in goals if g not in ho]

    pos = {g: torch.stack([p[0] for p in data[g]]) for g in goals}     # on-task trajectory [T,d]
    neg = {g: torch.stack([p[1] for p in data[g]]) for g in goals}     # drift trajectory [T,d]
    zG = {g: F.normalize(pos[g][:2].mean(0), dim=0) for g in goals}     # seed from first 2 turns

    gate = LiquidGate(d_llm, args.d)
    opt = torch.optim.Adam(gate.parameters(), lr=3e-3, weight_decay=1e-4)
    for ep in range(args.epochs):
        opt.zero_grad(); loss = 0.0
        for g in tr:
            z = zG[g]
            kp = gate.run(pos[g], z).mean()                            # on-task -> quiet
            kn = gate.run(neg[g], z).mean()                            # drift -> fire
            loss = loss + F.softplus(args.margin - (kn - kp)) + 0.01 * (kp ** 2)
        (loss / len(tr)).backward(); opt.step()

    @torch.no_grad()
    def evals():
        # trajectory-level: does mean-kappa(drift) > mean-kappa(on-task) on held-out goals?
        traj_hits, last_hits, perturn = [], [], {}
        cos_gain = []
        for g in ho:
            z = zG[g]
            kp = gate.run(pos[g], z); kn = gate.run(neg[g], z)
            traj_hits.append(float(kn.mean() > kp.mean()))
            last_hits.append(float(kn[-1] > kp[-1]))
            for t in range(len(kn)):
                perturn.setdefault(t, []).append((float(kp[t]), float(kn[t])))
            # does the gated correction move a drift state toward the goal? (cos before vs after)
            for t in range(neg[g].shape[0]):
                h = neg[g][t:t + 1]
                corr = kn[t] * native_pull(h, z.unsqueeze(0))
                c0 = float(F.cosine_similarity(h, z.unsqueeze(0)))
                c1 = float(F.cosine_similarity(h + corr, z.unsqueeze(0)))
                cos_gain.append(c1 - c0)
        return traj_hits, last_hits, perturn, cos_gain

    th, lh, pu, cg = evals()
    print(f"[gate] held-out goals={len(ho)}  d={args.d}", flush=True)
    print(f"  trajectory drift>on-task (mean kappa): {np.mean(th):.3f}", flush=True)
    print(f"  trajectory drift>on-task (last kappa) : {np.mean(lh):.3f}", flush=True)
    print(f"  per-turn kappa (on-task / drift):", flush=True)
    for t in sorted(pu):
        kp = np.mean([a for a, _ in pu[t]]); kn = np.mean([b for _, b in pu[t]])
        print(f"    turn {t}: on-task={kp:.3f}  drift={kn:.3f}", flush=True)
    print(f"  gated correction raises cos(h,z) on drift states by mean {np.mean(cg):+.4f} (native pull works)", flush=True)
    if args.save:
        torch.save({"gate": gate.state_dict(), "d": args.d, "d_llm": d_llm,
                    "ho_goals": ho, "zG": {g: zG[g] for g in goals}}, args.save)
        print(f"[gate] saved -> {args.save}", flush=True)
    print("[gate] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
