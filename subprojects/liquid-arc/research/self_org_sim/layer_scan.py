# Which DEPTH is best for the identity? Train the single-layer identity at EACH captured layer
# [4,8,..,64] (data already has all 16) and compare cross-frame corr. Answers: is layer-32 optimal,
# is the LAST layer (64) better, or another? Validates/corrects the unvalidated layer-32 heuristic.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_value_multi.pt', weights_only=False, map_location='cpu')
data = obj['data']; MULTI_LAYERS = list(range(4, 65, 4))
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
ALL = torch.stack([d['mseq'] for d in data]); n_layers, d_m = ALL.shape[2], ALL.shape[3]
GMEAN = ALL.reshape(-1, n_layers, d_m).mean(0)                        # per-layer mean [n_layers, d_m]
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]

class IdentityLiquid(nn.Module):
    def __init__(self, d_m, d=128, K=8, n=4):
        super().__init__(); self.K, self.d, self.n = K, d, n; D = K * d; self.D = D
        self.in_m = nn.Linear(d_m, D); self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D))
        self.slow = nn.Linear(D, D); self.value_head = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1)); self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.D, device=dev); self.s = torch.zeros(1, self.D, device=dev)
    def step(self, h):
        x = self.in_m(h) + self.slow(self.s); tau = F.softplus(self.log_tau) + 0.5
        for _ in range(self.n): self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x)) / tau / self.n
        self.s = 0.9 * self.s + 0.1 * self.b
    def value(self): return torch.sigmoid(self.value_head(self.s).squeeze())
def corr(a, b): a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))

def train_layer(li):
    gm = GMEAN[li]
    def cen(x): return x - gm
    net = IdentityLiquid(d_m).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    def roll(m):
        net.reset(); vals = []
        for t in range(m['mseq'].shape[0]):
            net.s = net.s; net.step(cen(m['mseq'][t, li]).unsqueeze(0)); vals.append(net.value())
        return torch.stack(vals)
    for ep in range(90):
        opt.zero_grad(); loss = 0.0
        for m in tr: loss = loss + F.mse_loss(roll(m), m['val'].float() / 9.0)
        (loss / len(tr)).backward(); opt.step()
    with torch.no_grad():
        P = torch.cat([roll(m) for m in te]); Y = torch.cat([m['val'].float() / 9.0 for m in te])
    return corr(P, Y)

print('per-DEPTH identity corr (EMPOWER, held-out frames):', flush=True)
res = {}
for li in range(n_layers):
    c = train_layer(li); res[MULTI_LAYERS[li]] = c
    print('  layer %2d : corr=%.3f%s' % (MULTI_LAYERS[li], c, '   <- current (32)' if MULTI_LAYERS[li] == 32 else ('   <- LAST' if MULTI_LAYERS[li] == 64 else '')), flush=True)
best = max(res, key=res.get)
print('\nBEST layer = %d (corr=%.3f) | layer-32=%.3f | last-layer-64=%.3f' % (best, res[best], res[32], res[64]))
print('=== ALL_DONE ===')
