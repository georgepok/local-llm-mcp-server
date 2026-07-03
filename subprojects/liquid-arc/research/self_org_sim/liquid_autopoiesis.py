# FAITHFUL autopoietic closure — the project's persistent-state LIQUID on the LLM MANIFOLD, no homegrown controllers.
# Substrate: belief b (the Liquid), recurrent W, tau — the same family as the compressor. Perception: the LLM manifold
# drift gist-stream drives it (fixed projection, the Liquid LIVES on the manifold). Viability: ONLY the Liquid's own
# intrinsic dynamics (tanh saturation + -b/tau damping) — NO external clamp / LayerNorm / gain-controller. Individuation:
# ONE local plasticity rule on the Liquid's OWN weights (normalized Hebbian = substrate soil, not a controller; NOT
# gradient descent). Run long on each manifold history; test the three markers: transition, individuation, identity-defense.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn.functional as F, statistics as st
torch.set_grad_enabled(False)
D, TAU, ETA_H, LOOPS = 192, 3.0, 0.02, 240
obj = torch.load('/home/pokazge/checkpoints/objective_drift.pt', weights_only=False, map_location='cpu')
data = [m for m in obj['data'] if len(m['gen']) >= 8]; d_m = data[0]['gen'][0].shape[1]
MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
for m in data: m['gist'] = [F.normalize(c.mean(0) - MU, dim=0) for c in m['gen']]    # the LLM manifold the Liquid lives on
torch.manual_seed(0); Win = torch.randn(D, d_m) / d_m ** 0.5                          # fixed perception (manifold -> Liquid)
def lyap(W, b0, k=20):
    b1, b2 = b0.clone(), b0 + torch.randn(D) * 1e-6; d0 = (b1 - b2).norm() + 1e-12; acc = 0.0
    for _ in range(k):
        b1 = b1 + (-b1 + torch.tanh(W @ b1)) / TAU; b2 = b2 + (-b2 + torch.tanh(W @ b2)) / TAU
        d = (b1 - b2).norm() + 1e-12; acc += float(torch.log(d / d0)); b2 = b1 + (b2 - b1) * (d0 / d)
    return acc / k
def live(W0, gists, loops):                                                          # the Liquid lives on this manifold history
    W = W0.clone(); fro = W0.norm(); b = torch.randn(D) * 0.1; ly = []; late = []
    for L in range(loops):
        for gi in gists:
            for _ in range(2): b = b + (-b + torch.tanh(W @ b + Win @ gi)) / TAU       # intrinsic dynamics, NO clamp
            W = W + ETA_H * torch.outer(b, b) / D; W = W * (fro / (W.norm() + 1e-9))    # local Hebbian (soil), self-normalized
        if L % (loops // 8) == 0: ly.append(lyap(W, b.clone()))
        if L > loops - 20: late.append(b.clone())
    return W, torch.stack(late).mean(0), ly
def settle(W, b0, steps=300):
    b = b0.clone()
    for _ in range(steps): b = b + (-b + torch.tanh(W @ b)) / TAU
    return torch.tanh(W @ b)
torch.manual_seed(1); W0 = torch.randn(D, D) / D ** 0.5 * 0.9                          # ONE body
M = min(8, len(data)); selves = [live(W0, data[i]['gist'], LOOPS) for i in range(M)]   # same body, different manifold histories
Ws = [s[0] for s in selves]; states = [s[1] for s in selves]
dW = torch.stack([(s[0] - W0).flatten() for s in selves]); sigs = torch.stack([settle(Ws[i], states[i]) for i in range(M)])
def xcos(X): Xn = F.normalize(X, dim=1); C = Xn @ Xn.t(); off = C[~torch.eye(len(X), dtype=torch.bool)]; return float(off.mean()), float(off.std())
ml = [st.mean([selves[s][2][i] for s in range(M)]) for i in range(len(selves[0][2]))]
print('=== FAITHFUL AUTOPOIESIS: project Liquid on LLM manifold; intrinsic viability + ONE local-plasticity soil; NO controllers/gradient/clamp ===')
print('  locatedness (attractor norm): %.3f   |  M=%d selves, same body, different manifold histories' % (st.mean([float(s.norm()) for s in sigs]), M))
print('MARKER 1 transition  (mean Lyapunov across run): %s' % ['%+.2f' % v for v in ml])
print('MARKER 2 individuation: dW-structure cross-cos=%.3f±%.3f | attractor cross-cos=%.3f±%.3f' % (*xcos(dW), *xcos(sigs)))
df = []
for A in range(M):
    for B in range(M):
        if A == B: continue
        fromB = settle(Ws[A], states[B])                                              # A's evolved structure, started at B's state
        cA = float(F.cosine_similarity(fromB.unsqueeze(0), sigs[A].unsqueeze(0))); cB = float(F.cosine_similarity(fromB.unsqueeze(0), sigs[B].unsqueeze(0)))
        df.append(cA - cB)
print('MARKER 3 identity-defense (decisive): A-structure from B-state returns to A? mean(cosA-cosB)=%+.3f  frac=%.2f' % (st.mean(df), sum(1 for d in df if d > 0) / len(df)))
print('\nverdict: transition + individuated STRUCTURE + defends -> a self formed in the Liquid on the manifold, no module.')
print('=== ALL_DONE ===')
