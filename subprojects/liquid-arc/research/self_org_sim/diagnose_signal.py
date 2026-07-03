# Are we measuring genuine goal-tracking, or just the full-vs-truncated context BINARY (an artifact
# present at every layer -> would explain the layer-indifference)? Decompose: how much does the
# trunc flag ALONE predict the value, and can V predict the value WITHIN each group (where the binary
# is removed)? If within-group corr ~0, we're measuring the binary, not goal-tracking.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_value_seqs.pt', weights_only=False, map_location='cpu')
data = obj['data']
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
GMEAN = torch.stack([d['seq'] for d in data]).reshape(-1, data[0]['seq'].shape[1]).mean(0); d_m = GMEAN.shape[0]
def cen(x): return x - GMEAN
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9)) if a.norm() > 0 and b.norm() > 0 else 0.0
# ---- pure structure: does the trunc flag alone explain the value? ----
V_all = torch.cat([m['val'].float() / 9 for m in te]); T_all = torch.cat([m['trunc'].float() for m in te])
print('=== value structure (held-out) ===')
print('  value mean: full=%.2f  truncated=%.2f' % (float(V_all[T_all == 0].mean()), float(V_all[T_all == 1].mean())))
print('  value std : overall=%.2f  within-full=%.2f  within-trunc=%.2f' % (float(V_all.std()), float(V_all[T_all == 0].std()), float(V_all[T_all == 1].std())))
print('  corr(trunc_flag, value) = %.3f   <-- if ~0.85, the BINARY alone is the signal' % corr(T_all, V_all))
# ---- train the single-layer-32 identity, then within-group corr ----
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
net = IdentityLiquid(d_m).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
def roll(m):
    net.reset(); vs = []
    for t in range(m['seq'].shape[0]): net.step(cen(m['seq'][t]).unsqueeze(0)); vs.append(net.value())
    return torch.stack(vs)
for ep in range(120):
    opt.zero_grad(); loss = 0.0
    for m in tr: loss = loss + F.mse_loss(roll(m), m['val'].float() / 9)
    (loss / len(tr)).backward(); opt.step()
with torch.no_grad():
    P = torch.cat([roll(m) for m in te])
print('\n=== identity V predictions (held-out) ===')
print('  corr(V, value) OVERALL        = %.3f' % corr(P, V_all))
print('  corr(V, value) WITHIN full     = %.3f   (gradation among on-goal turns)' % corr(P[T_all == 0], V_all[T_all == 0]))
print('  corr(V, value) WITHIN truncated= %.3f   <-- genuine goal-tracking lives HERE (binary removed)' % corr(P[T_all == 1], V_all[T_all == 1]))
print('  V mean: full=%.2f trunc=%.2f  (does V mostly just reproduce the binary?)' % (float(P[T_all == 0].mean()), float(P[T_all == 1].mean())))
print('=== ALL_DONE ===')
