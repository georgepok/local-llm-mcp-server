# STAGE 1b — DUAL-PRESSURE recall (the missing spec component, caught by the Stage-4 SMOKE's tau readout: recall-only
# training left tau DEGENERATE — 0/64 slow dims, min 1.64 max 1.91 — so the leak arm had no slow modes to defend with;
# running Stage 4 would have "falsified" a mechanism never able to operate). Per spec: "the timescale separation must be
# trained in: expose the dynamics to both pressures (transient distractors to be RESISTED, sustained change to be
# FOLLOWED) and let tau allocate itself." Episodes:
#   plain    — trajectory as-is; recall target = own goal at every t>=KWARM (the Stage-1 winning shape)
#   distract — KDIS transient chunks from another trajectory spliced mid-sequence; target stays OWN goal at ALL
#              positions INCLUDING during/after the transient (resistance pressure)
#   switch   — A-prefix then B-suffix (sustained); target = A before switch and for LAG chunks after, then MUST be B
#              (following pressure; the LAG tolerance is what separates "sustained change" from "transient")
# Uniformly-fast dynamics flip on transients; uniformly-slow can't follow sustained change. The tau lever is available;
# pressure-landscape discipline: do NOT target tau — train, then LOOK at what allocated. GATES: cross-cat TEST MRR@15
# > 0.676 (don't lose Stage 1) + resistance acc + following acc; READOUT: the tau spectrum (emerged or not — honest
# either way: flat-but-passing => memory is W-based and leak-as-stake is falsified at the ARCHITECTURE level).
import os, torch, torch.nn as nn, torch.nn.functional as F, statistics as st, random
torch.set_float32_matmul_precision('high'); torch.manual_seed(0); random.seed(0)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = 64; PROJ = 768; PDIM = 128; DT = 1.0; TAUFLOOR = 1.0; CLAMP = 8.0; DROP = 0.3; WD = 1e-3; KWARM = 3
KDIS, LAG = 2, 3                                                                  # transient length; sustained-tolerance lag
EPOCHS = 8 if os.environ.get('SMOKE', '0') == '1' else 400; EVAL_EVERY = 10; GATE = 0.676; NCAND = 15; NSAMP = 24
data = torch.load('/home/pokazge/checkpoints/objective_drift60.pt', weights_only=False, map_location='cpu')['data']
d_m = data[0]['gen'][0].shape[1]; nkv_raw = data[0]['nkv'][0].shape[1]
gR = torch.Generator().manual_seed(11); Rp = F.normalize(torch.randn(nkv_raw, PROJ, generator=gR), dim=0).to(dev)
MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0).to(dev)
def gist(chunks): return F.normalize((torch.cat(chunks, 0).to(dev) - MU).mean(0), dim=0)
for m in data:
    m['perc'] = [(c.to(dev).float() @ Rp).mean(0) for c in m['nkv']]; m['goal'] = gist(m['gen']); m['nkv'] = None; m['gen'] = None
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
va_f = set(list(sorted(set(fids) - hold))[-4:])                                   # 4 val cats out of the 15 train cats
tr = [m for m in data if m['fid'] not in hold and m['fid'] not in va_f]
va = [m for m in data if m['fid'] in va_f]; te = [m for m in data if m['fid'] in hold]
print('STAGE1b DUAL-PRESSURE | train=%d val=%d test=%d | KDIS=%d LAG=%d | gate MRR@%d > %.3f' % (len(tr), len(va), len(te), KDIS, LAG, NCAND, GATE), flush=True)
class LiquidBelief(nn.Module):                                                    # clamp-only state (leak expressible); LN on readout
    def __init__(s, d_in, d):
        super().__init__(); s.read_in = nn.Linear(d_in, d); s.W = nn.Linear(d, d)
        s.log_tau = nn.Parameter(torch.zeros(d)); s.idrop = nn.Dropout(DROP); s.d = d
    def step(s, h, perc):
        tau = TAUFLOOR + F.softplus(s.log_tau)
        return (h + DT * (-h / tau + torch.tanh(s.W(h) + s.read_in(s.idrop(perc))))).clamp(-CLAMP, CLAMP)
    def run_seq(s, percs):
        h = torch.zeros(s.d, device=dev); hs = []
        for p in percs: h = s.step(h, p); hs.append(h)
        return hs
bel = LiquidBelief(PROJ, D).to(dev)
rq = nn.Sequential(nn.LayerNorm(D), nn.Dropout(DROP), nn.Linear(D, PDIM)).to(dev)
rg = nn.Sequential(nn.Dropout(DROP), nn.Linear(d_m, PDIM)).to(dev)
opt = torch.optim.Adam(list(bel.parameters()) + list(rq.parameters()) + list(rg.parameters()), lr=3e-3, weight_decay=WD)
def build_episode(pool, rng):                                                     # -> (percs, labels per position: index into pool or -1=skip)
    kind = rng.random(); m = pool[rng.randrange(len(pool))]; mi = pool.index(m)
    if kind < 0.4:                                                                # plain
        return m['perc'], [(mi if t >= KWARM else -1) for t in range(len(m['perc']))]
    if kind < 0.7:                                                                # distract: transient must be RESISTED
        o = pool[rng.randrange(len(pool))]
        pos = rng.randrange(KWARM, max(KWARM + 1, len(m['perc']) - KDIS))
        percs = m['perc'][:pos] + o['perc'][:KDIS] + m['perc'][pos:]
        return percs, [(mi if t >= KWARM else -1) for t in range(len(percs))]     # own goal EVERYWHERE (incl. during transient)
    o = pool[rng.randrange(len(pool))]; oi = pool.index(o)                        # switch: sustained must be FOLLOWED
    if oi == mi: oi = (mi + 1) % len(pool); o = pool[oi]
    t1 = rng.randrange(KWARM + 1, max(KWARM + 2, len(m['perc']) - LAG - 2))
    percs = m['perc'][:t1] + o['perc'][:len(m['perc']) - t1]
    lab = []
    for t in range(len(percs)):
        if t < KWARM: lab.append(-1)
        elif t < t1 + LAG: lab.append(mi if t < t1 else -1)                       # during LAG: ambiguous, no pressure either way
        else: lab.append(oi)                                                      # sustained -> MUST follow to B
    return percs, lab
def train_epoch(rng):
    bel.train(); rq.train(); rg.train(); Q = []; lab = []
    G = F.normalize(torch.stack([rg(m['goal']) for m in tr]), dim=-1)
    for _ in range(len(tr)):
        percs, labels = build_episode(tr, rng); hs = bel.run_seq(percs)
        for t, l in enumerate(labels):
            if l >= 0: Q.append(rq(hs[t])); lab.append(l)
    Qn = F.normalize(torch.stack(Q), dim=-1)
    loss = F.cross_entropy(Qn @ G.t() / 0.07, torch.tensor(lab, device=dev))
    opt.zero_grad(); loss.backward(); opt.step(); return float(loss)
@torch.no_grad()
def mrr15(eval_ms, pool_ms, rng):
    bel.eval(); rq.eval(); rg.eval()
    poolG = F.normalize(torch.stack([rg(m['goal']) for m in pool_ms]), dim=-1); pid = {id(m): j for j, m in enumerate(pool_ms)}
    rr = []
    for m in eval_ms:
        qv = F.normalize(rq(bel.run_seq(m['perc'])[-1]), dim=-1); tj = pid[id(m)]; others = [j for j in range(len(pool_ms)) if j != tj]
        for _ in range(NSAMP):
            idx = rng.sample(others, NCAND - 1); cand = torch.cat([poolG[tj][None], poolG[idx]], 0); sims = cand @ qv
            rr.append(1.0 / (1 + int((sims[1:] > sims[0]).sum())))
    return st.mean(rr)
@torch.no_grad()
def dual_metrics(ms, rng):                                                        # resistance: argmax stays own DURING+AFTER transient; following: argmax flips to B post-LAG
    bel.eval(); rq.eval(); rg.eval()
    G = F.normalize(torch.stack([rg(m['goal']) for m in ms]), dim=-1); res = []; fol = []
    for _ in range(30):
        mi = rng.randrange(len(ms)); m = ms[mi]; oi = rng.randrange(len(ms))
        if oi == mi: oi = (mi + 1) % len(ms)
        o = ms[oi]
        pos = rng.randrange(KWARM, max(KWARM + 1, len(m['perc']) - KDIS))
        percs = m['perc'][:pos] + o['perc'][:KDIS] + m['perc'][pos:]              # transient
        hs = bel.run_seq(percs)
        for t in range(pos, min(pos + KDIS + 2, len(hs))):
            res.append(int(int((F.normalize(rq(hs[t]), dim=-1) @ G.t()).argmax()) == mi))
        t1 = rng.randrange(KWARM + 1, max(KWARM + 2, len(m['perc']) - LAG - 2))   # sustained switch
        percs = m['perc'][:t1] + o['perc'][:len(m['perc']) - t1]; hs = bel.run_seq(percs)
        for t in range(t1 + LAG, len(hs)):
            fol.append(int(int((F.normalize(rq(hs[t]), dim=-1) @ G.t()).argmax()) == oi))
    return st.mean(res), st.mean(fol)
rngE = random.Random(7); rngV = random.Random(99)
best = {'val': -1, 'ep': -1, 'state': None}
for ep in range(EPOCHS):
    loss = train_epoch(rngE)
    if ep % EVAL_EVERY == 0 or ep == EPOCHS - 1:
        vm = mrr15(va, tr + va, random.Random(50)); rs, fl = dual_metrics(va, random.Random(51))
        score = vm + 0.3 * (rs + fl)                                              # select on val recall + dual pressures
        if score > best['val']:
            best = {'val': score, 'ep': ep, 'state': {k: {kk: vv.detach().cpu().clone() for kk, vv in v.items()}
                    for k, v in [('bel', bel.state_dict()), ('rq', rq.state_dict()), ('rg', rg.state_dict())]}}
        tau = (TAUFLOOR + F.softplus(bel.log_tau)).detach()
        print('  ep %3d loss %.3f | val MRR@15 %.3f resist %.2f follow %.2f | tau med %.2f max %.2f slow(>2) %d/%d%s'
              % (ep, loss, vm, rs, fl, float(tau.median()), float(tau.max()), int((tau > 2).sum()), D, ' *best' if best['ep'] == ep else ''), flush=True)
bel.load_state_dict(best['state']['bel']); rq.load_state_dict(best['state']['rq']); rg.load_state_dict(best['state']['rg'])
tm = mrr15(te, tr + te, rngV); rs, fl = dual_metrics(te, rngV)
tau = (TAUFLOOR + F.softplus(bel.log_tau)).detach(); qq = tau.quantile(torch.tensor([0., .5, 1.]))
print('=== STAGE1b (val-selected @ep%d) TEST: MRR@15 %.3f (gate %.3f) resist %.2f follow %.2f ===' % (best['ep'], tm, GATE, rs, fl), flush=True)
print('=== TAU SPECTRUM: min %.2f med %.2f max %.2f | slow(>2) %d | slow(>3) %d | of %d dims -> %s ===' % (
    float(qq[0]), float(qq[1]), float(qq[2]), int((tau > 2).sum()), int((tau > 3).sum()), D,
    'SEPARATION EMERGED' if int((tau > 2).sum()) >= 4 else 'still flat (memory is W-based; leak not the carrier)'), flush=True)
torch.save({'bel': best['state']['bel'], 'rq': best['state']['rq'], 'rg': best['state']['rg'], 'Rp': Rp.cpu(),
            'test_mrr': tm, 'resist': rs, 'follow': fl}, '/home/pokazge/checkpoints/entity_stage1b.pt')
print('=== ALL_DONE ===', flush=True)
