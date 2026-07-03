# DIAGNOSE the overfit (CPU, on the snapshot): the compressor hits train coh 0.96 but held-out ~0 cross-category. Is that
# (a) NO generalization at all (too little data), or (b) WITHIN-category generalization that fails only CROSS-category?
# Test the SAME layer-32-AoA compressor under two splits: RANDOM-by-trajectory (held-out categories mostly still in train)
# vs CROSS-CATEGORY-by-fid (held-out categories unseen). Also a regularization sweep (weight_decay, epochs) to see if
# overfit is tamable. If random>>cross -> it's specifically cross-category transfer (the known ceiling); if both ~0 -> data scale.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
data = [m for m in torch.load('/home/pokazge/checkpoints/_snap60.pt', weights_only=False, map_location='cpu')['data'] if len(m['texts']) >= 10]
d_m = data[0]['gen'][0].shape[1]; MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
for m in data: m['z'] = [F.normalize((c - MU).mean(0), dim=0) for c in m['gen']]
class Comp(nn.Module):
    def __init__(s, in_dim, D=384, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(in_dim, heads * dh); s.Wv = nn.Linear(in_dim, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); K_ = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K_) / s.dh ** 0.5, dim=-1); return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
    def run(s, m):
        b = torch.zeros(s.D); h = torch.zeros(s.D); tau = F.softplus(s.log_tau) + 0.5; pr = []
        for t in range(len(m['gen'])):
            hp = h; a = s.collect(m['gen'][t] - MU, b)
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b; pr.append(s.pred(torch.cat([s.cz(m['z'][t]), hp])))
        return torch.stack(pr)
def coh(P, z): return float((F.normalize(P[:-1], dim=-1) * z[1:]).sum(-1).mean())
def train_eval(tr, te, wd, eps):
    torch.manual_seed(0); comp = Comp(d_m); opt = torch.optim.Adam(comp.parameters(), lr=1e-3, weight_decay=wd)
    for ep in range(eps):
        loss = 0.0
        for m in tr: pr = comp.run(m); z = torch.stack(m['z']); loss = loss + (1 - (F.normalize(pr[:-1], dim=-1) * z[1:]).sum(-1)).mean()
        opt.zero_grad(); (loss / len(tr)).backward(); opt.step()
    with torch.no_grad():
        trc = st.mean([coh(comp.run(m), torch.stack(m['z'])) for m in tr]); tec = st.mean([coh(comp.run(m), torch.stack(m['z'])) for m in te])
    return trc, tec
N = len(data); fids = sorted(set(m['fid'] for m in data))
print('layer-32-AoA compressor | %d trajs, %d categories' % (N, len(fids)), flush=True)
# RANDOM split (held-out categories mostly still represented in train)
torch.manual_seed(1); perm = torch.randperm(N).tolist(); k = max(2, N // 4)
teR = [data[i] for i in perm[:k]]; trR = [data[i] for i in perm[k:]]
# CROSS-CATEGORY split (held-out categories unseen)
hold = set(fids[-max(1, len(fids) // 4):]); trC = [m for m in data if m['fid'] not in hold]; teC = [m for m in data if m['fid'] in hold]
print('\n  split            | wd     ep  | train coh | HELD-OUT coh | n_te', flush=True)
for (name, tr, te) in [('RANDOM-by-traj', trR, teR), ('CROSS-CAT-by-fid', trC, teC)]:
    for wd, eps in [(1e-5, 180), (1e-3, 120), (1e-2, 80)]:
        trc, tec = train_eval(tr, te, wd, eps)
        print('  %-16s | %.0e %4d | %.3f     | %+.3f       | %d' % (name, wd, eps, trc, tec, len(te)), flush=True)
print('\nread: RANDOM held-out >> CROSS-CAT => within-category generalizes, cross-category does not (the ceiling — needs many', flush=True)
print('more categories). Both ~0 even with strong wd => not just overfit, the dropped-context signal itself does not transfer at this scale.', flush=True)
print('=== ALL_DONE ===', flush=True)
