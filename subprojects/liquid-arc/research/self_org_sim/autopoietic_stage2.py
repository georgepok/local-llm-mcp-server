# AUTOPOIETIC CLOSURE — Stage 2: individuation + the decisive marker (identity-defense). SAME body W (one substrate),
# different perturbation HISTORIES -> does each settle into a DISTINCT, LOCATED (non-flat) self, and does it DEFEND its
# own attractor when pushed toward another viable self's attractor? Defense (return to own, not accept the other) is the
# cut between a self and a generic stabilizer: a stabilizer is indifferent to which viable config it occupies; a self is not.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn.functional as F, statistics as st
torch.set_grad_enabled(False)
N, TAU, RHO, E_STAR, ETA, STAB = 256, 3.0, 0.85, 0.10, 0.04, 7000
def stabilize(W, hist_seed):                                                     # internalized homeostasis under a specific history
    torch.manual_seed(2000 + hist_seed)
    s = torch.randn(N) * 0.3; g = torch.full((N,), 0.8); E = torch.zeros(N); states = []
    for t in range(STAB):
        inp = torch.randn(N) * 0.5 if (t % 150 == 0) else torch.zeros(N)
        r = torch.tanh(g * s); s = s + (-s + (W @ r + inp)) / TAU; rn = torch.tanh(g * s)
        E = 0.98 * E + 0.02 * rn.pow(2); g = (g + ETA * (E_STAR - E) * g).clamp(0.05, 25)
        if t > STAB - 1500: states.append(rn)
    S = torch.stack(states); return g.clone(), s.clone(), S.mean(0), float(S.mean(0).norm())   # gain, state, attractor-signature, locatedness
def evolve(W, g, s0, steps=2500, perturb=True):                                  # run a fixed self (its gains) from a given state
    s = s0.clone(); states = []
    for t in range(steps):
        inp = torch.randn(N) * 0.3 if (perturb and t % 150 == 0) else torch.zeros(N)
        r = torch.tanh(g * s); s = s + (-s + (W @ r + inp)) / TAU
        if t > steps - 600: states.append(torch.tanh(g * s))
    return torch.stack(states).mean(0)
torch.manual_seed(0); W = torch.randn(N, N) / N ** 0.5 * RHO                      # ONE body
M = 8; selves = [stabilize(W, h) for h in range(M)]
gs = torch.stack([x[0] for x in selves]); sigs = torch.stack([x[2] for x in selves]); loc = [x[3] for x in selves]
def xcos(X):
    Xn = F.normalize(X, dim=1); C = Xn @ Xn.t(); off = C[~torch.eye(len(X), dtype=torch.bool)]; return float(off.mean()), float(off.std())
gm, gsd = xcos(gs); sm, ssd = xcos(sigs)
print('=== MARKER 2: INDIVIDUATION (same body, %d different histories) ===' % M)
print('  locatedness (attractor-sig norm; flat≈0): mean=%.3f  per-self=%s' % (st.mean(loc), ['%.2f' % v for v in loc]))
print('  gain-config cross-cos     = %.3f ± %.3f   (1=identical/generic, <1=individuated)' % (gm, gsd))
print('  attractor-sig cross-cos   = %.3f ± %.3f   (1=same self, <1=distinct selves)' % (sm, ssd))
print('\n=== MARKER 3 (DECISIVE): IDENTITY-DEFENSE ===', flush=True)
def_, acc_ = [], []
for A in range(M):
    gA, _, sigA, _ = selves[A]
    for B in range(M):
        if A == B: continue
        _, sB, sigB, _ = selves[B]
        final = evolve(W, gA, sB.clone())                                        # A's dynamics, started AT B's attractor (non-fatal push)
        cA = float(F.cosine_similarity(final.unsqueeze(0), sigA.unsqueeze(0))); cB = float(F.cosine_similarity(final.unsqueeze(0), sigB.unsqueeze(0)))
        def_.append(cA - cB); acc_.append(cA)
print('  pushed A->B, run A: returns to A? (cosA - cosB; >0 = DEFENDS its identity)')
print('  mean(cosA-cosB) = %+.3f   frac defended = %.2f   mean cosA(return) = %+.3f' % (st.mean(def_), sum(1 for d in def_ if d > 0) / len(def_), st.mean(acc_)))
print('\nverdict: individuated + located + defends -> a self formed.  generic/flat/indifferent -> a stabilizer (the wall).')
print('=== ALL_DONE ===')
