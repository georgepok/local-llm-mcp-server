# ATTENTION-ON-ATTENTION co-development. The Liquid does NOT hand-craft derivatives (those compete with the
# model's own attention and lose). It REUSES attention as a 2nd-order operation: the Liquid's belief is a QUERY
# that attends back over the model's generation developmental stream (how the answer FORMED, token by token),
# collecting NEW attributes the model's next-token attention never surfaced (goal-relevant, not next-token-relevant).
# Two-flow (fast belief + slow goal) co-develops across turns. Self-supervised coupling objective — NO value labels.
# Measure = slow-channel COUPLING LIFT (the validated dynamical-goal-object signal), and AoA(process) vs snapshot(product);
# NOT prediction of any external value.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_value_gen.pt', weights_only=False, map_location='cpu')
data = obj['data']; d_m = data[0]['gen'][0].shape[1]
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
MU = torch.cat([t for m in data for t in m['gen']], 0).mean(0)
def cen(G): return G - MU
print('d_m=%d  train=%d test=%d' % (d_m, len(tr), len(te)), flush=True)

class AoACoDevelop(nn.Module):
    def __init__(self, d_m, D=384, heads=6, dh=64, use_aoa=True, use_slow=True):
        super().__init__(); self.D = D; self.h = heads; self.dh = dh; self.use_aoa = use_aoa; self.use_slow = use_slow
        self.Wq = nn.Linear(D, heads * dh); self.Wk = nn.Linear(d_m, heads * dh); self.Wv = nn.Linear(d_m, heads * dh); self.Wo = nn.Linear(heads * dh, D)
        self.snap = nn.Linear(d_m, D)                                          # ablation reader: product snapshot (last token)
        self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D)); self.slow = nn.Linear(D, D)
        self.pred = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)) # co-development: fast belief -> next collected latent
    def collect(self, G, b):                                                   # ATTENTION-ON-ATTENTION: belief queries the developmental stream
        if not self.use_aoa: return self.snap(G[-1])                           # ablation: the product (endpoint) only
        q = self.Wq(b).view(self.h, self.dh); K = self.Wk(G).view(-1, self.h, self.dh); V = self.Wv(G).view(-1, self.h, self.dh)
        attn = torch.softmax(torch.einsum('hd,nhd->hn', q, K) / self.dh ** 0.5, dim=-1)
        return self.Wo(torch.einsum('hn,nhd->hd', attn, V).reshape(-1))
    def rollout(self, m):
        b = torch.zeros(self.D); s = torch.zeros(self.D); tau = F.softplus(self.log_tau) + 0.5
        col, pr = [], []
        for t in range(len(m['gen'])):
            a = self.collect(cen(m['gen'][t]), b)                              # collect attributes from HOW it developed
            sd = self.slow(s) if self.use_slow else torch.zeros(self.D)
            for _ in range(2): b = b + (-b + torch.tanh(self.W(b) + a + sd)) / tau / 2
            s = 0.9 * s + 0.1 * b
            col.append(a); pr.append(self.pred(b))
        return torch.stack(col), torch.stack(pr)
    def coherence(self, m):                                                    # self-supervised co-development: belief[t] anticipates next collected attribute
        col, pr = self.rollout(m)
        if col.shape[0] < 2: return None
        return (F.normalize(pr[:-1], dim=-1) * F.normalize(col[1:].detach(), dim=-1)).sum(-1)
def train_eval(use_aoa, use_slow, epochs=120):
    torch.manual_seed(0); net = AoACoDevelop(d_m, use_aoa=use_aoa, use_slow=use_slow).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for ep in range(epochs):
        opt.zero_grad(); loss = 0.0; n = 0
        for m in tr:
            c = net.coherence(m)
            if c is not None: loss = loss + (1 - c).mean(); n += 1
        (loss / n).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    with torch.no_grad():
        real = torch.cat([net.coherence(m) for m in te if net.coherence(m) is not None])
        # shuffle control: predictions vs MISMATCHED next-attributes (roll) -> should collapse to ~0 if coherence is real
        cols = [net.rollout(m)[0] for m in te]; prs = [net.rollout(m)[1] for m in te]
        allcol = torch.cat([c[1:] for c in cols]); allpr = torch.cat([p[:-1] for p in prs])
        shuf = (F.normalize(allpr, dim=-1) * F.normalize(allcol.roll(7, 0), dim=-1)).sum(-1)
    return float(real.mean()), float(shuf.mean())
print('\n%-26s  coherence   shuffle-ctrl' % 'config')
res = {}
for aoa in (True, False):
    for slw in (True, False):
        co, sh = train_eval(aoa, slw)
        res[(aoa, slw)] = co
        print('%-26s    %+.3f       %+.3f' % (('AoA(process)' if aoa else 'snapshot(product)') + (' +slow' if slw else ' -slow'), co, sh))
print('\nslow COUPLING LIFT  (dynamical-goal-object signal: how much the slow goal-channel improves co-development)')
print('  AoA(process)     : %+.3f' % (res[(True, True)] - res[(True, False)]))
print('  snapshot(product): %+.3f' % (res[(False, True)] - res[(False, False)]))
print('process vs product (full): AoA %+.3f  vs  snapshot %+.3f' % (res[(True, True)], res[(False, True)]))
print('=== ALL_DONE ===')
