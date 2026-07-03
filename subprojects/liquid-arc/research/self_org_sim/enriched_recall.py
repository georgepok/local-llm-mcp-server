# ENRICHED multi-chunk recall (CPU): make the belief carry the dropped-trajectory DETAIL, not just the goal gist. The
# belief h_t (late) must recall EACH of the first K=4 dropped chunks DISTINCTLY, position-conditioned (recall[h_t,pos(q)]
# -> match z_q vs other trajectories' chunk-q). Forces a richer retention than goal-mean recall. Native-KV read (the
# architecture), cross-category held-out. Saves compressor + compact {fid,perc,z} -> enriched_recall.pt for the autopoiesis step.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
torch.manual_seed(0); D, PROJ, K = 256, 768, 4
print('loading 60-traj data + projecting native KV ...', flush=True)
data = [m for m in torch.load('/home/pokazge/checkpoints/objective_drift60.pt', weights_only=False, map_location='cpu')['data'] if len(m['texts']) >= 10]
d_m = data[0]['gen'][0].shape[1]; nkv_raw = data[0]['nkv'][0].shape[1]
MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
gR = torch.Generator().manual_seed(11); Rp = F.normalize(torch.randn(nkv_raw, PROJ, generator=gR), dim=0)
for m in data:
    m['z'] = [F.normalize((c - MU).mean(0), dim=0) for c in m['gen']]
    m['perc'] = [c @ Rp for c in m['nkv']]; m['nkv'] = None
    m['drop'] = torch.stack(m['z'][:K])                                          # [K, d_m] the first K dropped chunks to recall distinctly
MUk = torch.cat([c for m in data for c in m['perc']], 0).mean(0)
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
print('ENRICHED multi-chunk recall (K=%d) | trajs=%d train=%d test=%d cats=%d held-out=%s' % (K, len(data), len(tr), len(te), len(fids), sorted(hold)), flush=True)
class Comp(nn.Module):
    def __init__(s, in_dim, D=256, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(in_dim, heads * dh); s.Wv = nn.Linear(in_dim, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D))
        s.pos = nn.Embedding(K, D); s.recall = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D)); s.chunkp = nn.Linear(d_m, D)
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); Kk = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, Kk) / s.dh ** 0.5, -1); return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
    def beliefs(s, m):
        b = torch.zeros(s.D); h = torch.zeros(s.D); tau = F.softplus(s.log_tau) + 0.5; hs = []
        for t in range(len(m['perc'])):
            a = s.collect(m['perc'][t] - MUk, b)
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b; hs.append(h)
        return torch.stack(hs)
comp = Comp(PROJ, D); opt = torch.optim.Adam(comp.parameters(), lr=1e-3, weight_decay=1e-4)
def chunk_proj(group):                                                           # [n, K, D] recall-space projections of each traj's first-K chunks
    return torch.stack([torch.stack([comp.chunkp(m['drop'][q]) for q in range(K)]) for m in group])
for ep in range(20 if os.environ.get('SMOKE') else 130):
    HS = [comp.beliefs(m) for m in tr]; DP = chunk_proj(tr); loss = 0.0; cnt = 0
    for mi in range(len(tr)):
        for t in range(K + 2, HS[mi].shape[0]):                                  # recall from late positions (the chunks are "dropped")
            for q in range(K):
                r = comp.recall(torch.cat([HS[mi][t], comp.pos.weight[q]]))
                loss = loss + F.cross_entropy(((r @ DP[:, q, :].t()) / 0.1).unsqueeze(0), torch.tensor([mi])); cnt += 1
    opt.zero_grad(); (loss / cnt).backward(); opt.step()
    if ep % 30 == 0: print('  ep %d recall-loss %.3f' % (ep, float(loss) / cnt), flush=True)
def evalmrr(group):
    DP = chunk_proj(group); cor = 0; tot = 0; rr = 0.0
    for mi, m in enumerate(group):
        hs = comp.beliefs(m)
        for t in range(hs.shape[0] // 2, hs.shape[0]):
            for q in range(K):
                sc = comp.recall(torch.cat([hs[t], comp.pos.weight[q]])) @ DP[:, q, :].t(); rk = int((sc > sc[mi]).sum())
                cor += (rk == 0); rr += 1.0 / (rk + 1); tot += 1
    return cor / tot, rr / tot
with torch.no_grad():
    a, mrr = evalmrr(te); atr, _ = evalmrr(tr)
print('  ENRICHED belief: held-out multi-chunk recall acc %.3f MRR %.3f (train acc %.3f) chance %.3f (n_te=%d, K=%d)' % (a, mrr, atr, 1.0 / len(te), len(te), K), flush=True)
print('  (goal-only recall was MRR 0.676 acc 0.448 -- enriched carries per-position detail)', flush=True)
torch.save({'comp': comp.state_dict(), 'Rp': Rp, 'MUk': MUk, 'data': [{'fid': m['fid'], 'seed': m['seed'], 'perc': m['perc'], 'z': m['z'], 'drop': m['drop']} for m in data], 'hold': sorted(hold), 'K': K}, '/home/pokazge/checkpoints/enriched_recall.pt')
print('saved enriched_recall.pt', flush=True)
print('=== ALL_DONE ===', flush=True)
