# DECISIVE disambiguation: is the trained compressor's 0.923 autonomous retention a FORMED SELF (multiple stable,
# individuated attractors — each goal-belief a basin that returns after perturbation and stays distinct from others) or
# just a SLOW INTEGRATOR / single global sink (momentum retains any init; all goals flow to one point = no self)?
# Tests, all autonomous (NO input) on the trained compressor's recurrence:
#   T1 self-retention   : autonomous from A's belief -> cos to init (A near a fixed point?).
#   T2 small-pert RETURN : nudge A's belief by small noise, evolve autonomously -> does it RETURN to A (stable attractor)?
#   T3 distinctness HOLD : do autonomous endpoints across goals STAY distinct (many basins) or MERGE to one (global sink)?
#   T4 viable-basin defense: set belief to ANOTHER goal B (a real viable identity), evolve autonomously -> stay in B's
#                           basin (=> separate coexisting selves) or collapse to a shared point (=> one sink)?
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
torch.set_grad_enabled(False); torch.manual_seed(0)
data = [m for m in torch.load('/home/pokazge/checkpoints/objective_drift_txt.pt', weights_only=False, map_location='cpu')['data'] if len(m['gen']) >= 10]
d_m = data[0]['gen'][0].shape[1]; MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
def cen(C): return C - MU
class Compressor(nn.Module):
    def __init__(s, d_m, D=384, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(d_m, heads * dh); s.Wv = nn.Linear(d_m, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); K_ = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K_) / s.dh ** 0.5, dim=-1); return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
comp = Compressor(d_m); comp.load_state_dict(torch.load('/home/pokazge/checkpoints/lora_inloop.pt', map_location='cpu')['comp'])
TAU = F.softplus(comp.log_tau) + 0.5
def drive_step(b, h, C):
    a = comp.collect(cen(C), b)
    for _ in range(2): b = b + (-b + torch.tanh(comp.W(b) + a)) / TAU / 2
    return b, 0.9 * h + 0.1 * b
def auto(b, h, K):                                                                # evolve K steps with NO input; return b,h traj
    bs, hs = [], []
    for _ in range(K):
        for _ in range(2): b = b + (-b + torch.tanh(comp.W(b))) / TAU / 2
        h = 0.9 * h + 0.1 * b; bs.append(b.clone()); hs.append(h.clone())
    return torch.stack(bs), torch.stack(hs)
def cos(a, b): return float(F.cosine_similarity(a, b, dim=-1).mean())
# drive each trajectory to its mid-point belief (b,h)
M = len(data); K = 12
B0, H0 = [], []
for m in data:
    b = torch.zeros(comp.D); h = torch.zeros(comp.D); t0 = len(m['gen']) // 2
    for t in range(t0): b, h = drive_step(b, h, m['gen'][t])
    B0.append(b); H0.append(h)
print('=== DECISIVE: formed-self vs slow-integrator (trained compressor, %d goals, %d autonomous steps) ===' % (M, K), flush=True)
# T1 self-retention
t1 = [cos(auto(B0[A].clone(), H0[A].clone(), K)[1][-1], H0[A]) for A in range(M)]
print('T1 self-retention   : autonomous cos(h_end, h_init)         = %.3f   (A sits near a fixed point)' % st.mean(t1), flush=True)
# T2 small-perturbation return: nudge b, evolve autonomously, does cos-to-init recover vs right after the nudge?
for eps in (0.1, 0.3, 0.6):
    c_after, c_end = [], []
    for A in range(M):
        b0 = B0[A]; n = torch.randn_like(b0); n = n / n.norm() * b0.norm() * eps; bp = b0 + n
        c_after.append(cos(bp, b0)); bs, _ = auto(bp.clone(), H0[A].clone(), K); c_end.append(cos(bs[-1], b0))
    print('T2 return (eps=%.1f)  : cos-to-init  right-after-nudge %.3f -> autonomous-end %.3f   (%+.3f = %s)'
          % (eps, st.mean(c_after), st.mean(c_end), st.mean(c_end) - st.mean(c_after),
             'RETURNS = stable attractor' if st.mean(c_end) > st.mean(c_after) else 'drifts away'), flush=True)
# T3 distinctness hold: cross-cos of autonomous endpoints vs cross-cos of inits (do basins merge?)
def xcos(X): Xn = F.normalize(X, dim=1); off = (Xn @ Xn.t())[~torch.eye(len(X), dtype=torch.bool)]; return float(off.mean())
ends = torch.stack([auto(B0[A].clone(), H0[A].clone(), K)[0][-1] for A in range(M)])
x_init, x_end = xcos(torch.stack(B0)), xcos(ends)
print('T3 distinctness     : cross-cos  init %.3f -> autonomous-end %.3f   (%s)'
      % (x_init, x_end, 'STAY DISTINCT = many coexisting basins' if x_end < 0.6 else 'MERGE toward one global sink'), flush=True)
# T4 viable-basin defense: put A's machine at B's belief, evolve autonomously — stay near B (separate basin) or leave it?
keepB = []
for A in range(M):
    Bx = (A + 1) % M; bs, _ = auto(B0[Bx].clone(), H0[Bx].clone(), K); keepB.append(cos(bs[-1], B0[Bx]))
print('T4 viable-basin     : autonomous from another goal B: cos(end, B_init) = %.3f   (%s)'
      % (st.mean(keepB), 'stays in B basin = coexisting selves' if st.mean(keepB) > 0.6 else 'leaves B = no distinct B basin'), flush=True)
print('\nVERDICT: formed-self requires T1 high + T2 RETURNS + T3 STAY DISTINCT + T4 stays. slow-integrator = T1 high but T2 no-return / T3 merges.', flush=True)
print('=== ALL_DONE ===', flush=True)
