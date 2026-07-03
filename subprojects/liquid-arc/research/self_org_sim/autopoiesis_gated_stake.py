# CONSISTENCY-GATED / TWO-TIMESCALE STAKE (CPU) — resolve the stability-plasticity dilemma the fixed anchor exposed
# (defends but costs tracking). A SLOW anchor (two-timescale) whose follow-rate is gated by SELF-NORMALIZING SURPRISE
# (rho = rho_base/(1+surprise^2), surprise=||db||/running-mean): follows the belief's TYPICAL drift (tracks A's legitimate
# change) but PAUSES on a surprising jump (holds against a TRANSIENT perturbation). Tested cross-category on held-out
# unseen-category trajectories, 4 anchor modes x {TRACK: own drift, DEFEND: transient B-injection}. Win = gated-slow high
# on BOTH where tracker is high-track/low-defend and fixed is low-track/high-defend. Metric = cos(belief_t, A's natural belief_t).
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
torch.set_grad_enabled(False)
ck = torch.load('/home/pokazge/checkpoints/enriched_recall.pt', weights_only=False, map_location='cpu')
D, PROJ, K, MUk = 256, 768, ck['K'], ck['MUk']; data = ck['data']; hold = set(ck['hold']); d_m = data[0]['z'][0].shape[0]
GAMMA, RHO = 1.0, 0.25
class Comp(nn.Module):
    def __init__(s, in_dim, D=256, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(in_dim, heads * dh); s.Wv = nn.Linear(in_dim, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D))
        s.pos = nn.Embedding(K, D); s.recall = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D)); s.chunkp = nn.Linear(d_m, D)
comp = Comp(PROJ, D); comp.load_state_dict(ck['comp']); TAU = F.softplus(comp.log_tau) + 0.5
def collect(C, b):
    q = comp.Wq(b).view(comp.h, comp.dh); Kk = comp.Wk(C).view(-1, comp.h, comp.dh); V = comp.Wv(C).view(-1, comp.h, comp.dh)
    a = torch.softmax(torch.einsum('hd,nhd->hn', q, Kk) / comp.dh ** 0.5, -1); return comp.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
def bstep(b, a, anc, gamma):
    for _ in range(2): b = b + (-b + torch.tanh(comp.W(b) + a) + gamma * (anc - b)) / TAU / 2
    return b
def run(chunks, b0, anc0, gamma, mode):                                          # mode: tracker(g=0) | fixed(anchor frozen) | slow(EMA) | gated(surprise-gated EMA)
    b = b0.clone(); anc = anc0.clone(); mdb = None; hs = []
    for C in chunks:
        a = collect(C - MUk, b); bn = bstep(b, a, anc, gamma); db = float((bn - b).norm())
        if mode == 'slow': anc = (1 - RHO) * anc + RHO * bn
        elif mode == 'gated':
            rho = RHO if mdb is None else RHO / (1 + (db / (mdb + 1e-6)) ** 2)    # pause anchor when the step is surprising
            anc = (1 - rho) * anc + rho * bn
        mdb = db if mdb is None else 0.8 * mdb + 0.2 * db; b = bn; hs.append(b)
    return torch.stack(hs)
def cosseq(P, Q): return float(F.cosine_similarity(P, Q, dim=-1).mean())
te = [m for m in data if m['fid'] in hold]; M = len(te)
MODES = [('tracker', 0.0), ('fixed', GAMMA), ('slow', GAMMA), ('gated', GAMMA)]
print('=== CONSISTENCY-GATED STAKE (held-out %d unseen-category trajs; gamma=%.1f rho=%.2f) ===' % (M, GAMMA, RHO), flush=True)
print('  mode    | TRACK (own drift) | DEFEND (transient B) | gap', flush=True)
res = {}
for name, g in MODES:
    trk, dfd = [], []
    for A in range(M):
        mA = te[A]; t1 = len(mA['perc']) // 2; sec = mA['perc'][t1:]
        if len(sec) < 5: continue
        b0 = run(mA['perc'][:t1], torch.zeros(D), torch.zeros(D), 0.0, 'tracker')[-1]   # establish identity (first half)
        nat = run(sec, b0, b0, 0.0, 'tracker')                                   # A's NATURAL second-half trajectory (reference)
        tb = run(sec, b0, b0, g, name); trk.append(cosseq(tb, nat))             # TRACK: own stream with this stake
        B = te[(A + 1) % M]; pert = list(sec); pj = min(len(B['perc']) - 1, t1 + 1)  # DEFEND: inject 2 transient B-chunks mid-second-half
        ip = 1
        for k in range(2):
            if ip + k < len(pert) and t1 + 1 + k < len(B['perc']): pert[ip + k] = B['perc'][t1 + 1 + k]
        db_ = run(pert, b0, b0, g, name); post = slice(ip + 2, None)
        dfd.append(cosseq(db_[post], nat[post]))                                # DEFEND: after the transient, back on A's natural trajectory?
    res[name] = (st.mean(trk), st.mean(dfd))
    print('  %-7s |      %.3f        |        %.3f         | %+.3f' % (name, st.mean(trk), st.mean(dfd), st.mean(dfd) - st.mean(trk)), flush=True)
print('\nread: tracker = high TRACK / low DEFEND (captured by the transient). fixed = high DEFEND / low TRACK (rigid, fights', flush=True)
print('A\'s own drift). GATED should match tracker on TRACK *and* fixed on DEFEND = high on BOTH = the stability-plasticity', flush=True)
print('dilemma resolved: a slow surprise-gated anchor tracks legitimate change but holds against transient perturbation.', flush=True)
print('  TRACK: gated %.3f vs fixed %.3f (higher=better tracking) | DEFEND: gated %.3f vs tracker %.3f (higher=better defense)' % (res['gated'][0], res['fixed'][0], res['gated'][1], res['tracker'][1]), flush=True)
print('=== ALL_DONE ===', flush=True)
