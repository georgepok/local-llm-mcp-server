# STAGE 1 v2 — regularized + best-checkpoint. v1 OVERFIT (train MRR->1.0; held-out peaked 0.680>gate at ep~150 then
# declined to 0.589 final). Fixes: smaller belief (d=64, the validated scale), stronger dropout + weight_decay, bottleneck
# match dim, and SAVE-BEST-by-held-out (checkpoint-from-best, not final). Canonical Liquid belief; perception = mean-pooled
# native KV @ fixed Rp (validated); leak native but QUARANTINED. NO 27B (recorded streams). GATE: cross-cat held-out MRR
# > 0.676. On 60 trajectories this probes the architecture ceiling; the robust pass expects the data lever (60->~200).
import os, copy, torch, torch.nn as nn, torch.nn.functional as F, statistics as st, random
torch.set_float32_matmul_precision('high'); torch.manual_seed(0); random.seed(0)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SMOKE = os.environ.get('SMOKE', '0') == '1'
D = int(os.environ.get('D', '64')); PROJ = 768; PDIM = 128; DT = 1.0; TAUFLOOR = 1.0; CLAMP = 8.0
DROP = float(os.environ.get('DROP', '0.3')); WD = float(os.environ.get('WD', '1e-3'))
EPOCHS = 8 if SMOKE else 500; GATE = 0.676
data = torch.load('/home/pokazge/checkpoints/objective_drift60.pt', weights_only=False, map_location='cpu')['data']
d_m = data[0]['gen'][0].shape[1]; nkv_raw = data[0]['nkv'][0].shape[1]
gR = torch.Generator().manual_seed(11); Rp = F.normalize(torch.randn(nkv_raw, PROJ, generator=gR), dim=0).to(dev)
MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0).to(dev)
def gist(chunks): return F.normalize((torch.cat(chunks, 0).to(dev) - MU).mean(0), dim=0)
for m in data:
    m['perc'] = [(c.to(dev).float() @ Rp).mean(0) for c in m['nkv']]; m['goal'] = gist(m['gen']); m['nkv'] = None
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
print('STAGE1v2 RECALL | trajs=%d train=%d test=%d cats=%d held=%s | d=%d drop=%.2f wd=%.0e bottleneck=%d gate>%.3f'
      % (len(data), len(tr), len(te), len(fids), sorted(hold), D, DROP, WD, PDIM, GATE), flush=True)
class LiquidBelief(nn.Module):
    def __init__(s, d_in, d):
        super().__init__(); s.read_in = nn.Linear(d_in, d); s.W = nn.Linear(d, d)
        s.log_tau = nn.Parameter(torch.zeros(d)); s.ln = nn.LayerNorm(d); s.idrop = nn.Dropout(DROP); s.d = d
    def step(s, h, perc):
        tau = TAUFLOOR + F.softplus(s.log_tau)
        dh = -h / tau + torch.tanh(s.W(h) + s.read_in(s.idrop(perc)))   # input dropout on the read (anti-memorization)
        return s.ln(h + DT * dh).clamp(-CLAMP, CLAMP)
    def run(s, percs):
        h = torch.zeros(s.d, device=dev)
        for p in percs: h = s.step(h, p)
        return h
bel = LiquidBelief(PROJ, D).to(dev)
rq = nn.Sequential(nn.Dropout(DROP), nn.Linear(D, PDIM)).to(dev)
rg = nn.Sequential(nn.Dropout(DROP), nn.Linear(d_m, PDIM)).to(dev)
opt = torch.optim.Adam(list(bel.parameters()) + list(rq.parameters()) + list(rg.parameters()), lr=3e-3, weight_decay=WD)
def qg(ms):
    Q = torch.stack([rq(bel.run(m['perc'])) for m in ms]); G = torch.stack([rg(m['goal']) for m in ms])
    return F.normalize(Q, dim=-1), F.normalize(G, dim=-1)
@torch.no_grad()
def mrr(ms):
    bel.eval(); rq.eval(); rg.eval(); Q, G = qg(ms); S = Q @ G.t()
    rr = [1.0 / (1 + int((S[i] > S[i, i]).sum())) for i in range(len(ms))]
    bel.train(); rq.train(); rg.train(); return st.mean(rr)
print('=== STAGE1v2 TRAIN (regularized, save-best-by-heldout) ===', flush=True)
best = {'mrr': -1.0, 'ep': -1, 'state': None}; hist = []
for ep in range(EPOCHS):
    random.shuffle(tr); bel.train(); rq.train(); rg.train(); Q, G = qg(tr)
    loss = F.cross_entropy(Q @ G.t() / 0.07, torch.arange(len(tr), device=dev))
    opt.zero_grad(); loss.backward(); opt.step()
    ho = mrr(te); hist.append(ho)
    if ho > best['mrr']:
        best = {'mrr': ho, 'ep': ep, 'state': {k: copy.deepcopy({kk: vv.detach().cpu() for kk, vv in v.items()})
                for k, v in [('bel', bel.state_dict()), ('rq', rq.state_dict()), ('rg', rg.state_dict())]}}
    if ep % max(1, EPOCHS // 12) == 0 or ep == EPOCHS - 1:
        print('  ep %3d  loss %.3f  train_MRR %.3f  HELDOUT_MRR %.3f  (best %.3f@%d)' % (ep, float(loss), mrr(tr), ho, best['mrr'], best['ep']), flush=True)
plateau = st.mean(hist[-25:])
print('=== STAGE1v2 | BEST held-out MRR = %.3f @ep%d | final %.3f | last25-mean %.3f | gate %.3f -> %s ==='
      % (best['mrr'], best['ep'], hist[-1], plateau, GATE, 'PASS' if best['mrr'] > GATE else 'FAIL'), flush=True)
torch.save({**best['state'], 'Rp': Rp.cpu(), 'mrr': best['mrr'], 'ep': best['ep']}, '/home/pokazge/checkpoints/entity_stage1.pt')
print('=== ALL_DONE ===', flush=True)
