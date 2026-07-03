# STRUCTURE-vs-CONTENT recall (CPU, concurrent with the GPU write-strength run). Tests the program's through-line: does
# RELATIONAL structure transfer where CONTENT doesn't? Same Liquid+AoA belief, three position-conditioned contrastive
# recall targets, identical cross-category split: CONTENT (absolute gist z_q — the enriched one that generalized WORSE),
# RELATIONAL (the transition z_{q+1}-z_q, the CHANGE, invariant to absolute content), GOAL (abstract intent, the baseline
# that worked 0.676). Held-out MRR each. If RELATIONAL > CONTENT and ~>= GOAL, the fuller belief is built from relations,
# not content. Layer-32 channel (drift_lean.pt) for fast iteration; the OBJECTIVE is the variable, not the read.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
torch.manual_seed(0)
data = [m for m in torch.load('/home/pokazge/checkpoints/drift_lean.pt', weights_only=False, map_location='cpu')['data'] if len(m['gen']) >= 10]
d_m = data[0]['gen'][0].shape[1]; MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0); D, K = 256, 4
for m in data:
    m['z'] = [F.normalize((c - MU).mean(0), dim=0) for c in m['gen']]
    m['goal'] = F.normalize(torch.stack(m['z'][:3]).mean(0), dim=0)
    m['content'] = torch.stack(m['z'][:K])                                       # absolute content of first K chunks
    m['rel'] = torch.stack([F.normalize(m['z'][q + 1] - m['z'][q], dim=0) for q in range(K)])  # transitions = relations (the CHANGE)
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
print('STRUCTURE vs CONTENT recall | trajs=%d train=%d test=%d cats=%d held-out=%s' % (len(data), len(tr), len(te), len(fids), sorted(hold)), flush=True)
class Comp(nn.Module):
    def __init__(s, D=256, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(d_m, heads * dh); s.Wv = nn.Linear(d_m, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D))
        s.pos = nn.Embedding(K, D); s.recall = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D)); s.tp = nn.Linear(d_m, D)
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); Kk = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, Kk) / s.dh ** 0.5, -1); return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
    def beliefs(s, m):
        b = torch.zeros(s.D); h = torch.zeros(s.D); tau = F.softplus(s.log_tau) + 0.5; hs = []
        for t in range(len(m['gen'])):
            a = s.collect(m['gen'][t] - MU, b)
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b; hs.append(h)
        return torch.stack(hs)
def run_target(key, goalmode, label):                                            # key in {'content','rel'} (position-conditioned) or goalmode
    torch.manual_seed(0); c = Comp(D); opt = torch.optim.Adam(c.parameters(), lr=1e-3, weight_decay=1e-4)
    def proj(group): return torch.stack([torch.stack([c.tp(m[key][q]) for q in range(K)]) for m in group]) if not goalmode else torch.stack([c.tp(m['goal']) for m in group])
    for ep in range(120):
        HS = [c.beliefs(m) for m in tr]; P = proj(tr); loss = 0.0; cnt = 0
        for mi in range(len(tr)):
            for t in range(K + 2, HS[mi].shape[0]):
                if goalmode:
                    r = c.recall(torch.cat([HS[mi][t], c.pos.weight[0]])); loss = loss + F.cross_entropy(((r @ P.t()) / 0.1).unsqueeze(0), torch.tensor([mi])); cnt += 1
                else:
                    for q in range(K):
                        r = c.recall(torch.cat([HS[mi][t], c.pos.weight[q]])); loss = loss + F.cross_entropy(((r @ P[:, q, :].t()) / 0.1).unsqueeze(0), torch.tensor([mi])); cnt += 1
        opt.zero_grad(); (loss / cnt).backward(); opt.step()
    with torch.no_grad():
        def mrr(group):
            P = proj(group); rr = 0.0; ac = 0.0; tot = 0
            for mi, m in enumerate(group):
                hs = c.beliefs(m)
                for t in range(hs.shape[0] // 2, hs.shape[0]):
                    qs = [0] if goalmode else range(K)
                    for q in qs:
                        r = c.recall(torch.cat([hs[t], c.pos.weight[q]])); sc = r @ (P.t() if goalmode else P[:, q, :].t())
                        rk = int((sc > sc[mi]).sum()); rr += 1.0 / (rk + 1); ac += (rk == 0); tot += 1
            return ac / tot, rr / tot
        a, r = mrr(te)
    print('  %-22s held-out acc %.3f MRR %.3f (chance %.3f)' % (label, a, r, 1.0 / len(te)), flush=True)
    return r
print('held-out recall (does the belief, late in an UNSEEN-category trajectory, recall this target?):', flush=True)
g = run_target(None, True, 'GOAL (abstract intent)')
co = run_target('content', False, 'CONTENT (absolute z_q)')
re = run_target('rel', False, 'RELATIONAL (transitions)')
print('\n=== STRUCTURE vs CONTENT ===', flush=True)
print('  GOAL %.3f | CONTENT %.3f | RELATIONAL %.3f' % (g, co, re), flush=True)
print('  => %s' % ('RELATIONAL > CONTENT: structure transfers better than content (hypothesis CONFIRMED)' if re > co + 0.02 else 'relational ~<= content here (transitions not more transferable on this channel)'), flush=True)
print('=== ALL_DONE ===', flush=True)
