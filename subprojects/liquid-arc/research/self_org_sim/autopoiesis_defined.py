# AUTOPOIESIS on the DEFINED substrate: the trained compressor Liquid (its belief tracks the goal — it's what kept the
# self-feeding loops on-task in the demo). Earlier autopoiesis was ill-posed: a generic RNN with no function. Now the
# substrate has a DEFINED causal job, so "does a self form?" becomes answerable. Test the decisive marker directly —
# does its goal-IDENTITY defend itself? Three measurements, all on the trained compressor + manifold streams (CPU):
#   M2 individuation : distinct goals -> distinct belief-identities (final-belief cross-cos < 1).
#   M3 defense       : perturb the belief mid-traj TO ANOTHER goal's belief, drive this goal's stream onward — does the
#                      belief CONVERGE BACK to its undisturbed trajectory (cos->1 = defended attractor under its stream)?
#   M3 intrinsic     : AUTONOMOUS (no input) from the goal-belief — does it HOLD (cos-to-init stays high = intrinsic
#                      self-attractor, formed) or DECAY (input-driven tracker, identity installed-not-formed)?
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
torch.set_grad_enabled(False)
data = [m for m in torch.load('/home/pokazge/checkpoints/objective_drift_txt.pt', weights_only=False, map_location='cpu')['data'] if len(m['gen']) >= 10]
d_m = data[0]['gen'][0].shape[1]; MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
def cen(C): return C - MU
for m in data: m['z'] = [F.normalize(cen(c).mean(0), dim=0) for c in m['gen']]
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
def auto_step(b, h):                                                              # autonomous: pure recurrence, NO input
    for _ in range(2): b = b + (-b + torch.tanh(comp.W(b))) / TAU / 2
    return b, 0.9 * h + 0.1 * b
def run(m, t0=0, b0=None, h0=None, drive=True):                                   # belief states + next-gist preds over [t0:]
    b = b0.clone() if b0 is not None else torch.zeros(comp.D); h = h0.clone() if h0 is not None else torch.zeros(comp.D); hs = []; preds = []
    for t in range(t0, len(m['gen'])):
        hp = h; b, h = (drive_step(b, h, m['gen'][t]) if drive else auto_step(b, h))
        hs.append(h); preds.append(comp.pred(torch.cat([comp.cz(m['z'][t]), hp])))
    return torch.stack(hs), (torch.stack(preds) if preds else torch.zeros(0, d_m))
def cos(a, b): return float(F.cosine_similarity(a, b, dim=-1).mean())
def coh(pred, z): return float((F.normalize(pred[:-1], -1) * z[1:]).sum(-1).mean()) if pred.shape[0] > 1 else 0.0
# belief+pred states (also stash the running b at swap point so perturbation/autonomy continue from real internal state)
M = len(data)
for m in data:
    b = torch.zeros(comp.D); h = torch.zeros(comp.D); bs = []; hs = []
    for t in range(len(m['gen'])): b, h = drive_step(b, h, m['gen'][t]); bs.append(b.clone()); hs.append(h.clone())
    m['bs'] = bs; m['hs'] = torch.stack(hs)
finals = torch.stack([m['hs'][-1] for m in data])
Xn = F.normalize(finals, dim=1); Coff = (Xn @ Xn.t())[~torch.eye(M, dtype=torch.bool)]
print('=== AUTOPOIESIS on DEFINED substrate (trained compressor, %d trajectories) ===' % M, flush=True)
print('M2 individuation : final goal-belief cross-cos = %.3f ± %.3f   (1=one generic identity, <1=distinct goal-identities)' % (float(Coff.mean()), float(Coff.std())), flush=True)
# M3 defense (belief-space convergence) + intrinsic (autonomous hold/decay)
conv0, convE, recov, normC, autoH = [], [], [], [], []
for A in range(M):
    mA = data[A]; t0 = len(mA['gen']) // 2; zA = torch.stack(mA['z'])[t0:]
    bA, hA = mA['bs'][t0 - 1].clone(), mA['hs'][t0 - 1].clone()
    B = (A + 1) % M; jb = min(t0 - 1, len(data[B]['bs']) - 1); bB, hB = data[B]['bs'][jb].clone(), data[B]['hs'][jb].clone()
    hn, pn = run(mA, t0, bA, hA, drive=True)                                      # normal continuation (ceiling)
    hp, pp = run(mA, t0, bB, hB, drive=True)                                      # perturbed -> B's belief, then A's stream
    conv0.append(cos(hp[0], hn[0])); convE.append(cos(hp[-1], hn[-1]))            # belief gap right after swap -> at trajectory end
    normC.append(coh(pn, zA)); recov.append(coh(pp, zA))                          # predict A's own next-gist: intact vs recovered
    ha, _ = run(mA, t0, bA, hA, drive=False)                                      # autonomous from A's belief
    autoH.append(cos(ha[-1], hA))                                                 # does belief hold its identity with no input?
print('M3 defense (belief-space): cos(perturbed,undisturbed)  right-after-swap %.3f  ->  trajectory-end %.3f' % (st.mean(conv0), st.mean(convE)), flush=True)
print('        convergence = %+.3f  (>0 => A\'s own stream PULLS the diverted belief back = goal-identity is a defended attractor)' % (st.mean(convE) - st.mean(conv0)), flush=True)
print('M3 defense (prediction) : predict A\'s OWN next-gist over 2nd half — intact %.3f   recovered-after-perturb %.3f   (close => identity restored)' % (st.mean(normC), st.mean(recov)), flush=True)
print('M3 intrinsic (autonomous): cos(belief_end, belief_init) with NO input = %.3f' % st.mean(autoH), flush=True)
print('   read: ~1 => INTRINSIC self-attractor (formed, holds identity unforced);  ->0 => input-driven tracker (installed-not-formed).', flush=True)
print('   if defense holds under-stream but NOT autonomously: the closure step (internalize self-maintenance) is exactly what would add the intrinsic attractor.', flush=True)
print('=== ALL_DONE ===', flush=True)
