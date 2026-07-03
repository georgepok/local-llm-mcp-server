# STAGE 1 CV — 4-fold cross-CATEGORY CV on the existing 60 trajectories, to get a ROBUST (de-noised) held-out MRR
# BEFORE committing to a ~2hr 60->200 generation run. v2 best-checkpoint 0.797 was on a single noisy n=15 held-out
# (stable plateau ~0.66). Rotating the held-out categories across 4 folds and averaging removes that selection noise.
# If mean-over-folds robustly beats 0.676 -> Stage 1 genuinely passes on 60 (go to Stage 2). If marginal -> the data
# lever (60->200) is justified. Same regularized canonical-belief architecture as v2 (d=64, drop 0.3, wd 1e-3).
import os, copy, torch, torch.nn as nn, torch.nn.functional as F, statistics as st, random
torch.set_float32_matmul_precision('high'); torch.manual_seed(0); random.seed(0)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = 64; PROJ = 768; PDIM = 128; DT = 1.0; TAUFLOOR = 1.0; CLAMP = 8.0; DROP = 0.3; WD = 1e-3
EPOCHS = 300; GATE = 0.676; K = 4
DATA = os.environ.get('DATA', '/home/pokazge/checkpoints/objective_drift60.pt')
data = torch.load(DATA, weights_only=False, map_location='cpu')['data']
d_m = data[0]['gen'][0].shape[1]; nkv_raw = data[0]['nkv'][0].shape[1]
gR = torch.Generator().manual_seed(11); Rp = F.normalize(torch.randn(nkv_raw, PROJ, generator=gR), dim=0).to(dev)
MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0).to(dev)
def gist(chunks): return F.normalize((torch.cat(chunks, 0).to(dev) - MU).mean(0), dim=0)
for m in data:
    m['perc'] = [(c.to(dev).float() @ Rp).mean(0) for c in m['nkv']]; m['goal'] = gist(m['gen']); m['nkv'] = None
cats = sorted(set(m['fid'] for m in data)); folds = [cats[i::K] for i in range(K)]   # interleaved 5-cats/fold; each cat held out once
print('STAGE1 CV | trajs=%d cats=%d | %d folds (held-cats/fold=%s) | d=%d drop=%.1f wd=%.0e gate>%.3f'
      % (len(data), len(cats), K, [len(f) for f in folds], D, DROP, WD, GATE), flush=True)
class LiquidBelief(nn.Module):
    def __init__(s, d_in, d):
        super().__init__(); s.read_in = nn.Linear(d_in, d); s.W = nn.Linear(d, d)
        s.log_tau = nn.Parameter(torch.zeros(d)); s.ln = nn.LayerNorm(d); s.idrop = nn.Dropout(DROP); s.d = d
    def step(s, h, perc):
        tau = TAUFLOOR + F.softplus(s.log_tau)
        return s.ln(h + DT * (-h / tau + torch.tanh(s.W(h) + s.read_in(s.idrop(perc))))).clamp(-CLAMP, CLAMP)
    def run(s, percs):
        h = torch.zeros(s.d, device=dev)
        for p in percs: h = s.step(h, p)
        return h
def fresh():
    b = LiquidBelief(PROJ, D).to(dev); q = nn.Sequential(nn.Dropout(DROP), nn.Linear(D, PDIM)).to(dev)
    g = nn.Sequential(nn.Dropout(DROP), nn.Linear(d_m, PDIM)).to(dev)
    o = torch.optim.Adam(list(b.parameters()) + list(q.parameters()) + list(g.parameters()), lr=3e-3, weight_decay=WD)
    return b, q, g, o
def qg(ms, b, q, g):
    Q = torch.stack([q(b.run(m['perc'])) for m in ms]); G = torch.stack([g(m['goal']) for m in ms])
    return F.normalize(Q, dim=-1), F.normalize(G, dim=-1)
def mrr(ms, b, q, g):
    b.eval(); q.eval(); g.eval()
    with torch.no_grad():
        Q, G = qg(ms, b, q, g); S = Q @ G.t(); rr = [1.0 / (1 + int((S[i] > S[i, i]).sum())) for i in range(len(ms))]
    b.train(); q.train(); g.train(); return st.mean(rr)
bests, plats = [], []
for fi, held in enumerate(folds):
    hold = set(held); tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
    b, q, g, opt = fresh(); hist = []
    for ep in range(EPOCHS):
        random.shuffle(tr); b.train(); q.train(); g.train(); Q, G = qg(tr, b, q, g)
        loss = F.cross_entropy(Q @ G.t() / 0.07, torch.arange(len(tr), device=dev))
        opt.zero_grad(); loss.backward(); opt.step(); hist.append(mrr(te, b, q, g))
    best = max(hist); plat = st.mean(hist[-25:]); bests.append(best); plats.append(plat)
    print('  fold %d  held_cats=%s n_te=%d  best %.3f @ep%d  plateau %.3f' % (fi, sorted(hold), len(te), best, hist.index(best), plat), flush=True)
print('=== STAGE1 CV (%d folds) | mean BEST %.3f (sd %.3f) | mean PLATEAU %.3f (sd %.3f) | gate %.3f ==='
      % (K, st.mean(bests), st.pstdev(bests), st.mean(plats), st.pstdev(plats), GATE), flush=True)
verdict = 'ROBUST PASS' if st.mean(plats) > GATE else ('MARGINAL (best>gate, plateau<gate -> data lever justified)' if st.mean(bests) > GATE else 'FAIL')
print('=== VERDICT: %s ===' % verdict, flush=True)
print('=== ALL_DONE ===', flush=True)
