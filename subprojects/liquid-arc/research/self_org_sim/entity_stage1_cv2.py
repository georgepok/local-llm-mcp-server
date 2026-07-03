# STAGE 1 CV v2 — FOUR FIXES from the inbox critique (each was a proximal-win-vs-distant-goal defect):
#  (1) INSTANCE MULTIPLICATION — the validated anti-memorization shape that EARNED 0.676: emit a recall query at EVERY
#      position t>=KWARM, not once per trajectory. ~10x instances from the same trajectories (45 -> ~360). v1/v2 regressed
#      this to one query/trajectory -> the train-MRR->1.0 overfit + marginal plateau are the 45-example signature.
#  (2) COMMENSURABLE GATE — MRR@15: rank the true goal among NCAND=15 sampled candidates, so the metric is candidate-set-
#      size invariant. Raw MRR over a 50-candidate held-out (200-data, 1/4 hold-out) has chance ~0.09 vs ~0.22 at 15 and
#      would MANUFACTURE a FAIL against the 0.676 gate (which was earned at ~15 candidates). Chance MRR@15 = H_15/15 ~ 0.22.
#  (3) NO TEST LEAK — three-way CATEGORY split: train / val / test. SELECT the epoch by VAL MRR@15, REPORT TEST MRR@15 at
#      that epoch (test never touched for selection). best-by-heldout-then-report-heldout is a max over evals = biased
#      (that is how the single-run 0.797 fooled us). Averaged over R random splits.
#  (4) LEAK EXPRESSIBLE — clamp-only STATE (no LayerNorm on the state path); LN moved to the READOUT. LN-on-state pins |h|
#      to unit scale so the leak -h/tau cannot decay anything -> would silently invalidate the Stage-4 leak-as-stake test.
import os, torch, torch.nn as nn, torch.nn.functional as F, statistics as st, random
torch.set_float32_matmul_precision('high'); torch.manual_seed(0); random.seed(0)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA = os.environ.get('DATA', '/home/pokazge/checkpoints/objective_drift60.pt')
D = 64; PROJ = 768; PDIM = 128; DT = 1.0; TAUFLOOR = 1.0; CLAMP = 8.0; DROP = 0.3; WD = 1e-3; KWARM = 3
EPOCHS = 300; EVAL_EVERY = 10; GATE = 0.676; NCAND = 15; NSAMP = 24; R = int(os.environ.get('R', '4'))
TRAINF, VALF = 0.6, 0.2                                                           # category fractions; remainder = test
data = torch.load(DATA, weights_only=False, map_location='cpu')['data']
d_m = data[0]['gen'][0].shape[1]; nkv_raw = data[0]['nkv'][0].shape[1]
gR = torch.Generator().manual_seed(11); Rp = F.normalize(torch.randn(nkv_raw, PROJ, generator=gR), dim=0).to(dev)
MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0).to(dev)
def gist(chunks): return F.normalize((torch.cat(chunks, 0).to(dev) - MU).mean(0), dim=0)
for m in data:
    m['perc'] = [(c.to(dev).float() @ Rp).mean(0) for c in m['nkv']]; m['goal'] = gist(m['gen']); m['nkv'] = None
cats = sorted(set(m['fid'] for m in data)); chance = sum(1.0 / r for r in range(1, NCAND + 1)) / NCAND
print('STAGE1 CV2 | data=%s | trajs=%d cats=%d | d=%d KWARM=%d MRR@%d (chance %.3f) gate %.3f | %d splits %.0f/%.0f/%.0f%%'
      % (os.path.basename(DATA), len(data), len(cats), D, KWARM, NCAND, chance, GATE, R, TRAINF * 100, VALF * 100, (1 - TRAINF - VALF) * 100), flush=True)
class LiquidBelief(nn.Module):                                                    # canonical state h; CLAMP-ONLY (no LN on state -> leak expressible)
    def __init__(s, d_in, d):
        super().__init__(); s.read_in = nn.Linear(d_in, d); s.W = nn.Linear(d, d)
        s.log_tau = nn.Parameter(torch.zeros(d)); s.idrop = nn.Dropout(DROP); s.d = d
    def step(s, h, perc):
        tau = TAUFLOOR + F.softplus(s.log_tau)
        return (h + DT * (-h / tau + torch.tanh(s.W(h) + s.read_in(s.idrop(perc))))).clamp(-CLAMP, CLAMP)
    def run_seq(s, percs):
        h = torch.zeros(s.d, device=dev); hs = []
        for p in percs: h = s.step(h, p); hs.append(h)
        return hs                                                                # ALL h_t (per-position queries)
def fresh():
    b = LiquidBelief(PROJ, D).to(dev)
    q = nn.Sequential(nn.LayerNorm(D), nn.Dropout(DROP), nn.Linear(D, PDIM)).to(dev)   # LN on the READOUT, not the state
    g = nn.Sequential(nn.Dropout(DROP), nn.Linear(d_m, PDIM)).to(dev)
    o = torch.optim.Adam(list(b.parameters()) + list(q.parameters()) + list(g.parameters()), lr=3e-3, weight_decay=WD)
    return b, q, g, o
def train_step(tr, b, q, g, opt):
    b.train(); q.train(); g.train(); Q = []; lab = []
    for i, m in enumerate(tr):
        hs = b.run_seq(m['perc'])
        for t in (list(range(KWARM, len(hs))) or [len(hs) - 1]): Q.append(q(hs[t])); lab.append(i)   # INSTANCE MULTIPLICATION
    Q = F.normalize(torch.stack(Q), dim=-1); G = F.normalize(torch.stack([g(m['goal']) for m in tr]), dim=-1)
    loss = F.cross_entropy(Q @ G.t() / 0.07, torch.tensor(lab, device=dev))
    opt.zero_grad(); loss.backward(); opt.step(); return float(loss)
@torch.no_grad()
def mrr15(eval_ms, pool_ms, b, q, g, rng):                                        # COMMENSURABLE: rank true goal among NCAND sampled candidates
    b.eval(); q.eval(); g.eval()
    poolG = F.normalize(torch.stack([g(m['goal']) for m in pool_ms]), dim=-1); pid = {id(m): j for j, m in enumerate(pool_ms)}
    rr = []
    for m in eval_ms:
        qv = F.normalize(q(b.run_seq(m['perc'])[-1]), dim=-1); tj = pid[id(m)]; others = [j for j in range(len(pool_ms)) if j != tj]
        for _ in range(NSAMP):
            idx = rng.sample(others, NCAND - 1); cand = torch.cat([poolG[tj][None], poolG[idx]], 0); sims = cand @ qv
            rr.append(1.0 / (1 + int((sims[1:] > sims[0]).sum())))
    b.train(); q.train(); g.train(); return st.mean(rr)
test_at_bestval = []; val_at_best = []
for split in range(R):
    rng = random.Random(1000 + split); sc = cats[:]; rng.shuffle(sc)
    ntr = int(TRAINF * len(sc)); nva = max(1, int(VALF * len(sc)))
    trc, vac, tec = set(sc[:ntr]), set(sc[ntr:ntr + nva]), set(sc[ntr + nva:])
    tr = [m for m in data if m['fid'] in trc]; va = [m for m in data if m['fid'] in vac]; te = [m for m in data if m['fid'] in tec]
    b, q, g, opt = fresh(); valc = []; testc = []; eps = []
    for ep in range(EPOCHS):
        train_step(tr, b, q, g, opt)
        if ep % EVAL_EVERY == 0 or ep == EPOCHS - 1:
            valc.append(mrr15(va, tr + va, b, q, g, rng)); testc.append(mrr15(te, tr + te, b, q, g, rng)); eps.append(ep)
    bi = valc.index(max(valc)); test_at_bestval.append(testc[bi]); val_at_best.append(valc[bi])
    print('  split %d | train/val/test cats %d/%d/%d | best VAL %.3f @ep%d -> TEST@bestval %.3f | test_final %.3f'
          % (split, len(trc), len(vac), len(tec), valc[bi], eps[bi], testc[bi], testc[-1]), flush=True)
mt = st.mean(test_at_bestval); sdt = st.pstdev(test_at_bestval)
print('=== STAGE1 CV2 | TEST MRR@%d (val-selected, no leak) = %.3f (sd %.3f) | mean best-VAL %.3f | chance %.3f | gate %.3f -> %s ==='
      % (NCAND, mt, sdt, st.mean(val_at_best), chance, GATE, 'PASS' if mt > GATE else 'FAIL'), flush=True)
print('=== ALL_DONE ===', flush=True)
