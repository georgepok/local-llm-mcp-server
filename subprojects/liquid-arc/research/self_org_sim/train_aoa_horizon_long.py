# Horizon grounding sweep on LONG (16-turn) missions. With real horizon, identity should own the long range:
# the grounding-gap (prediction collapse when identity is ablated) should grow with k far past the 6-turn ceiling.
# Same local architecture (per-turn-reset working state + identity as sole persistent carrier), self-supervised gist.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_value_genlong.pt', weights_only=False, map_location='cpu')
data = obj['data']; d_m = data[0]['gen'][0].shape[1]
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
MU = torch.cat([t for m in data for t in m['gen']], 0).mean(0)
def cen(G): return G - MU
for m in data: m['z'] = [F.normalize(cen(t).mean(0), dim=0) for t in m['gen']]
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
print('d_m=%d train=%d test=%d  turns/mission=%d' % (d_m, len(tr), len(te), len(data[0]['gen'])), flush=True)
for k in (1, 2, 4, 6, 8):
    sims = [(torch.stack(m['z'])[:-k] * torch.stack(m['z'])[k:]).sum(-1) for m in te if len(m['gen']) > k]
    print('  gist persistence cos(z[t],z[t+%d]) = %+.3f' % (k, float(torch.cat(sims).mean())), flush=True)
class LocalGrounded(nn.Module):
    def __init__(self, d_m, D=384, heads=6, dh=64):
        super().__init__(); self.D = D; self.h = heads; self.dh = dh
        self.Wq = nn.Linear(D, heads * dh); self.Wk = nn.Linear(d_m, heads * dh); self.Wv = nn.Linear(d_m, heads * dh); self.Wo = nn.Linear(heads * dh, D)
        self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D)); self.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))
    def collect(self, G, b):
        q = self.Wq(b).view(self.h, self.dh); K = self.Wk(G).view(-1, self.h, self.dh); V = self.Wv(G).view(-1, self.h, self.dh)
        attn = torch.softmax(torch.einsum('hd,nhd->hn', q, K) / self.dh ** 0.5, dim=-1)
        return self.Wo(torch.einsum('hn,nhd->hd', attn, V).reshape(-1))
    def run(self, m, ground=True):
        s = torch.zeros(self.D); tau = F.softplus(self.log_tau) + 0.5; pr = []
        for t in range(len(m['gen'])):
            b = torch.zeros(self.D); a = self.collect(cen(m['gen'][t]), b)
            for _ in range(2): b = b + (-b + torch.tanh(self.W(b) + a)) / tau / 2
            s = 0.9 * s + 0.1 * b
            pr.append(self.pred(torch.cat([b, s if ground else torch.zeros(self.D)])))
        return torch.stack(pr)
def coh_at(pr, z, k): return None if pr.shape[0] <= k else (F.normalize(pr[:-k], dim=-1) * z[k:]).sum(-1)
def train_eval(k, epochs=140):
    torch.manual_seed(0); net = LocalGrounded(d_m).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        loss, n = 0.0, 0
        for m in tr:
            c = coh_at(net.run(m, True), torch.stack(m['z']), k)
            if c is not None: loss = loss + (1 - c).mean(); n += 1
        opt.zero_grad(); (loss / n).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    with torch.no_grad():
        g = torch.cat([c for m in te for c in [coh_at(net.run(m, True), torch.stack(m['z']), k)] if c is not None])
        g0 = torch.cat([c for m in te for c in [coh_at(net.run(m, False), torch.stack(m['z']), k)] if c is not None])
    return float(g.mean()), float(g0.mean())
print('\n%-10s  pred-coh  identity=0   GROUNDING-GAP' % 'horizon')
for k in (1, 2, 4, 6, 8):
    ch, ch0 = train_eval(k); print('  k=%-7d  %+.3f     %+.3f       %+.3f' % (k, ch, ch0, ch - ch0), flush=True)
print('\n6-turn ceiling was: k=1 +0.041, k=2 +0.079. Does grounding KEEP growing with horizon now?')
print('=== ALL_DONE ===')
