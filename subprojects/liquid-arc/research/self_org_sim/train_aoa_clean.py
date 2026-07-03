# CLEAN disambiguation of the AoA result. Two fixes: (1) the prediction target is now QUERY-INDEPENDENT — a fixed
# compact latent of the next turn's formation (mean of the centered developmental stream), identical for both readers,
# so AoA's self-referential advantage is gone and only the BELIEF-BUILDING mechanism differs. (2) Add NON-PREDICTIVE
# structural measures: identity STABILITY within a mission (does the belief converge to a persistent attractor) and
# DISTINCTNESS across missions (is the identity individuated) — the attractor-identity property, no prediction at all.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_value_gen.pt', weights_only=False, map_location='cpu')
data = obj['data']; d_m = data[0]['gen'][0].shape[1]
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
MU = torch.cat([t for m in data for t in m['gen']], 0).mean(0)
def cen(G): return G - MU
for m in data: m['z'] = [F.normalize(cen(t).mean(0), dim=0) for t in m['gen']]   # FIXED query-independent target per turn
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
print('d_m=%d  train=%d test=%d' % (d_m, len(tr), len(te)), flush=True)

class AoA(nn.Module):
    def __init__(self, d_m, D=384, heads=6, dh=64, use_aoa=True, use_slow=True):
        super().__init__(); self.D = D; self.h = heads; self.dh = dh; self.use_aoa = use_aoa; self.use_slow = use_slow
        self.Wq = nn.Linear(D, heads * dh); self.Wk = nn.Linear(d_m, heads * dh); self.Wv = nn.Linear(d_m, heads * dh); self.Wo = nn.Linear(heads * dh, D)
        self.snap = nn.Linear(d_m, D)
        self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D)); self.slow = nn.Linear(D, D)
        self.pred = nn.Linear(D, d_m)                                          # belief -> next formation gist (FIXED target)
    def collect(self, G, b):
        if not self.use_aoa: return self.snap(G[-1])
        q = self.Wq(b).view(self.h, self.dh); K = self.Wk(G).view(-1, self.h, self.dh); V = self.Wv(G).view(-1, self.h, self.dh)
        attn = torch.softmax(torch.einsum('hd,nhd->hn', q, K) / self.dh ** 0.5, dim=-1)
        return self.Wo(torch.einsum('hn,nhd->hd', attn, V).reshape(-1))
    def beliefs(self, m):
        b = torch.zeros(self.D); s = torch.zeros(self.D); tau = F.softplus(self.log_tau) + 0.5; bs = []
        for t in range(len(m['gen'])):
            a = self.collect(cen(m['gen'][t]), b); sd = self.slow(s) if self.use_slow else torch.zeros(self.D)
            for _ in range(2): b = b + (-b + torch.tanh(self.W(b) + a + sd)) / tau / 2
            s = 0.9 * s + 0.1 * b; bs.append(b)
        return torch.stack(bs)
    def coherence(self, m):                                                    # predict FIXED next-formation-gist from belief (query-independent)
        bs = self.beliefs(m)
        if bs.shape[0] < 2: return None
        z = torch.stack(m['z'])
        return (F.normalize(self.pred(bs[:-1]), dim=-1) * z[1:]).sum(-1)
def train_eval(use_aoa, use_slow, epochs=120):
    torch.manual_seed(0); net = AoA(d_m, use_aoa=use_aoa, use_slow=use_slow).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        opt.zero_grad(); loss = 0.0; n = 0
        for m in tr:
            c = net.coherence(m)
            if c is not None: loss = loss + (1 - c).mean(); n += 1
        (loss / n).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    with torch.no_grad():
        real = torch.cat([net.coherence(m) for m in te if net.coherence(m) is not None])
        prs = torch.cat([net.pred(net.beliefs(m)[:-1]) for m in te]); zs = torch.cat([torch.stack(m['z'])[1:] for m in te])
        shuf = (F.normalize(prs, dim=-1) * zs.roll(7, 0)).sum(-1)
        # NON-PREDICTIVE structure: within-mission belief stability, across-mission final-belief distinctness
        Bs = [net.beliefs(m) for m in te]
        stab = torch.cat([(F.normalize(B[:-1], dim=-1) * F.normalize(B[1:], dim=-1)).sum(-1) for B in Bs]).mean()
        fin = F.normalize(torch.stack([B[-1] for B in Bs]), dim=-1); G = fin @ fin.t()
        cross = G[~torch.eye(len(Bs), dtype=torch.bool)].mean()                 # lower = more individuated identities
    return float(real.mean()), float(shuf.mean()), float(stab), float(cross)
print('\n%-22s  coherence  shuffle   within-stab  cross-sim' % 'config')
R = {}
for aoa in (True, False):
    for slw in (True, False):
        co, sh, st, cr = train_eval(aoa, slw); R[(aoa, slw)] = (co, st, cr)
        print('%-22s   %+.3f     %+.3f     %+.3f      %+.3f' % (('AoA' if aoa else 'snapshot') + ('+slow' if slw else '-slow'), co, sh, st, cr))
print('\nFAIR co-development (query-independent target):  AoA %+.3f  vs  snapshot %+.3f' % (R[(True, True)][0], R[(False, True)][0]))
print('slow coupling lift:  AoA %+.3f   snapshot %+.3f' % (R[(True, True)][0] - R[(True, False)][0], R[(False, True)][0] - R[(False, False)][0]))
print('identity (want high within-stab, low cross-sim):  AoA stab %+.3f cross %+.3f | snapshot stab %+.3f cross %+.3f'
      % (R[(True, True)][1], R[(True, True)][2], R[(False, True)][1], R[(False, True)][2]))
print('=== ALL_DONE ===')
