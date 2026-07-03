# TRAINING-SHAPE fix (CPU): replace the memorizable PREDICT-next-gist objective with CONTRASTIVE GOAL-RECALL. Same Liquid
# + AoA read; only the pressure changes. The belief h_t must, at EVERY position (esp. late, after drift), RECALL its own
# trajectory's dropped goal (early-chunk gist) and pick it out against OTHER trajectories' goals (InfoNCE). A random/late
# query can't be served by a memorized next-step — the belief must actually RETAIN the goal, a general operation. Augmented
# over all positions; cross-category held-out. If recall generalizes (held-out >> chance) where predict overfits (~0), the
# training shape is the lever. Channel = layer-32 here to isolate the OBJECTIVE variable; port to native KV after.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
torch.manual_seed(0)
data = [m for m in torch.load('/home/pokazge/checkpoints/drift_lean.pt', weights_only=False, map_location='cpu')['data'] if len(m['gen']) >= 10]
d_m = data[0]['gen'][0].shape[1]; MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
for m in data:
    m['z'] = [F.normalize((c - MU).mean(0), dim=0) for c in m['gen']]
    m['g'] = F.normalize(torch.stack(m['z'][:3]).mean(0), dim=0)                  # the dropped goal = early-chunk gist
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
D = 256
print('RECALL vs PREDICT | trajs=%d train=%d test=%d cats=%d held-out=%s' % (len(data), len(tr), len(te), len(fids), sorted(hold)), flush=True)
class Comp(nn.Module):
    def __init__(s, D=256, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(d_m, heads * dh); s.Wv = nn.Linear(d_m, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D))
        s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))   # PREDICT head
        s.recall = nn.Linear(D, D); s.goalp = nn.Linear(d_m, D)                   # RECALL head
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); K = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K) / s.dh ** 0.5, -1); return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
    def beliefs(s, m):
        b = torch.zeros(s.D); h = torch.zeros(s.D); tau = F.softplus(s.log_tau) + 0.5; hs = []; hps = []
        for t in range(len(m['gen'])):
            hps.append(h); a = s.collect(m['gen'][t] - MU, b)
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b; hs.append(h)
        return torch.stack(hs), torch.stack(hps)
def train_predict():
    torch.manual_seed(0); c = Comp(D); opt = torch.optim.Adam(c.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(150):
        loss = 0.0
        for m in tr:
            hs, hps = c.beliefs(m); z = torch.stack(m['z']); pr = torch.stack([c.pred(torch.cat([c.cz(z[t]), hps[t]])) for t in range(len(z))])
            loss = loss + (1 - (F.normalize(pr[:-1], dim=-1) * z[1:]).sum(-1)).mean()
        opt.zero_grad(); (loss / len(tr)).backward(); opt.step()
    with torch.no_grad():
        def coh(m): hs, hps = c.beliefs(m); z = torch.stack(m['z']); pr = torch.stack([c.pred(torch.cat([c.cz(z[t]), hps[t]])) for t in range(len(z))]); return float((F.normalize(pr[:-1], dim=-1) * z[1:]).sum(-1).mean())
        return st.mean([coh(m) for m in tr]), st.mean([coh(m) for m in te])
def train_recall(tau=0.1):
    torch.manual_seed(0); c = Comp(D); opt = torch.optim.Adam(c.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(150):
        HS = [c.beliefs(m)[0] for m in tr]; GP = torch.stack([c.goalp(m['g']) for m in tr])   # goals recomputed (goalp trains)
        loss = 0.0; cnt = 0
        for mi in range(len(tr)):
            for t in range(3, HS[mi].shape[0]):                                   # recall the goal from every position >=3
                r = c.recall(HS[mi][t]); scores = (r @ GP.t()) / tau
                loss = loss + F.cross_entropy(scores.unsqueeze(0), torch.tensor([mi])); cnt += 1
        opt.zero_grad(); (loss / cnt).backward(); opt.step()
    with torch.no_grad():
        GPt = torch.stack([c.goalp(m['g']) for m in te]); cor = 0; tot = 0; rr = 0.0; trcor = 0; trtot = 0
        GPtr = torch.stack([c.goalp(m['g']) for m in tr])
        for mi, m in enumerate(te):                                              # held-out: pick own goal among held-out goals, from late positions
            hs = c.beliefs(m)[0]
            for t in range(hs.shape[0] // 2, hs.shape[0]):
                r = c.recall(hs[t]); sc = r @ GPt.t(); rank = int((sc > sc[mi]).sum()); cor += (rank == 0); rr += 1.0 / (rank + 1); tot += 1
        for mi, m in enumerate(tr):
            hs = c.beliefs(m)[0]
            for t in range(hs.shape[0] // 2, hs.shape[0]):
                r = c.recall(hs[t]); sc = r @ GPtr.t(); trcor += (int((sc > sc[mi]).sum()) == 0); trtot += 1
        return cor / tot, rr / tot, 1.0 / len(te), trcor / trtot
print('\n--- PREDICT next-gist (the overfit baseline) ---', flush=True)
trc, tec = train_predict(); print('  train coh %.3f | HELD-OUT coh %.3f  (cross-category)' % (trc, tec), flush=True)
print('--- RECALL goal, contrastive (the reshaped pressure) ---', flush=True)
acc, mrr, chance, tracc = train_recall()
print('  train recall acc %.3f | HELD-OUT recall acc %.3f | MRR %.3f | chance %.3f (n_te=%d)' % (tracc, acc, mrr, chance, len(te)), flush=True)
print('\n=== VERDICT ===', flush=True)
print('  PREDICT cross-cat coh %.3f (memorizes: ~0 held-out)' % tec, flush=True)
print('  RECALL  cross-cat acc %.3f vs chance %.3f  => %s' % (acc, chance, 'GENERALIZES (training shape is the lever)' if acc > 2 * chance else 'still not generalizing (need stronger pressure: bottleneck/augment)'), flush=True)
print('=== ALL_DONE ===', flush=True)
