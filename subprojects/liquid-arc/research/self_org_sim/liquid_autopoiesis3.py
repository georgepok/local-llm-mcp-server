# FAITHFUL autopoiesis, v3: make the maintenance the system's OWN. The manifold input is now SPARSE — one inject step
# per chunk, then the Liquid runs AUTONOMOUSLY for K steps and must HOLD its belief through its own recurrence (the
# compressor's real job). Now the recurrence is load-bearing: homeostatic excitability must RAISE gain to self-sustain
# (-> criticality), and the manifold history shapes which attractor. Both plasticities local soil; no controller/clamp.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn.functional as F, statistics as st
torch.set_grad_enabled(False)
D, TAU, ETA_H, ETA_G, E_STAR, LOOPS, KAUTO = 192, 3.0, 0.02, 0.05, 0.12, 240, 10
obj = torch.load('/home/pokazge/checkpoints/objective_drift.pt', weights_only=False, map_location='cpu')
data = [m for m in obj['data'] if len(m['gen']) >= 8]; d_m = data[0]['gen'][0].shape[1]
MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
for m in data: m['gist'] = [F.normalize(c.mean(0) - MU, dim=0) for c in m['gen']]
torch.manual_seed(0); Win = torch.randn(D, d_m) / d_m ** 0.5
def lyap(W, g, b0, k=20):
    b1, b2 = b0.clone(), b0 + torch.randn(D) * 1e-6; d0 = (torch.tanh(g * b1) - torch.tanh(g * b2)).norm() + 1e-12; acc = 0.0
    for _ in range(k):
        b1 = b1 + (-b1 + torch.tanh(g * (W @ b1))) / TAU; b2 = b2 + (-b2 + torch.tanh(g * (W @ b2))) / TAU
        d = (torch.tanh(g * b1) - torch.tanh(g * b2)).norm() + 1e-12; acc += float(torch.log(d / d0)); b2 = b1 + (b2 - b1) * (d0 / d)
    return acc / k
def live(W0, gists, loops):
    W = W0.clone(); fro = W0.norm(); b = torch.randn(D) * 0.1; g = torch.ones(D); E = torch.zeros(D); ly = []; late = []
    for L in range(loops):
        for gi in gists:
            b = b + (-b + torch.tanh(g * (W @ b + 3.0 * Win @ gi))) / TAU                 # SPARSE inject (perturb)
            for _ in range(KAUTO):                                                        # AUTONOMOUS holding — recurrence must self-sustain
                b = b + (-b + torch.tanh(g * (W @ b))) / TAU
                E = 0.98 * E + 0.02 * b.pow(2); g = (g + ETA_G * (E_STAR - E) * g).clamp(0.05, 25)
                W = W + ETA_H * torch.outer(b, b) / D; W = W * (fro / (W.norm() + 1e-9))
        if L % (loops // 8) == 0: ly.append(lyap(W, g, b.clone()))
        if L > loops - 20: late.append(b.clone())
    return W, g, torch.stack(late).mean(0), ly
def settle(W, g, b0, steps=300):
    b = b0.clone()
    for _ in range(steps): b = b + (-b + torch.tanh(g * (W @ b))) / TAU
    return torch.tanh(g * b)
torch.manual_seed(1); W0 = torch.randn(D, D) / D ** 0.5 * 0.9; M = min(8, len(data))
selves = [live(W0, data[i]['gist'], LOOPS) for i in range(M)]
Ws = [s[0] for s in selves]; gs = torch.stack([s[1] for s in selves]); states = [s[2] for s in selves]
dW = torch.stack([(s[0] - W0).flatten() for s in selves]); sigs = torch.stack([settle(Ws[i], gs[i], states[i]) for i in range(M)])
def xcos(X): Xn = F.normalize(X, dim=1); C = Xn @ Xn.t(); off = C[~torch.eye(len(X), dtype=torch.bool)]; return float(off.mean()), float(off.std())
ml = [st.mean([selves[s][3][i] for s in range(M)]) for i in range(len(selves[0][3]))]
print('=== FAITHFUL AUTOPOIESIS v3: sparse manifold + autonomous holding; recurrence load-bearing; both soil; NO controller ===')
print('  locatedness=%.3f | gain end mean=%.2f (now must RAISE to self-sustain)' % (st.mean([float(s.norm()) for s in sigs]), float(gs.mean())))
print('MARKER 1 transition (mean Lyapunov over run): %s' % ['%+.2f' % v for v in ml])
print('MARKER 2 individuation: dW cross-cos=%.3f±%.3f | gain cross-cos=%.3f±%.3f | attractor cross-cos=%.3f±%.3f' % (*xcos(dW), *xcos(gs), *xcos(sigs)))
df = []
for A in range(M):
    for B in range(M):
        if A == B: continue
        fromB = settle(Ws[A], gs[A], states[B]); cA = float(F.cosine_similarity(fromB.unsqueeze(0), sigs[A].unsqueeze(0))); cB = float(F.cosine_similarity(fromB.unsqueeze(0), sigs[B].unsqueeze(0)))
        df.append(cA - cB)
print('MARKER 3 identity-defense (decisive): mean(cosA-cosB)=%+.3f  frac=%.2f' % (st.mean(df), sum(1 for d in df if d > 0) / len(df)))
print('=== ALL_DONE ===')
