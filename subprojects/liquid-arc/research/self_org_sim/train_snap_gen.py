# Head-to-head baseline: same data + same Liquid as the attention version, but perception = LAST
# response-token snapshot (no attention). Compares within-truncated goal-tracking: attention 0.917 vs this.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_value_gen.pt', weights_only=False, map_location='cpu')
data = obj['data']
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
allt = torch.cat([t for m in data for t in m['gseq']], 0); d_m = allt.shape[1]
GMEAN = allt.mean(0); GSTD = float((allt - GMEAN).std() + 1e-6)
def cen(t): return (t - GMEAN) / GSTD
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9)) if a.norm() > 0 and b.norm() > 0 else 0.0
class SnapLiquid(nn.Module):
    def __init__(self, d_m, d=128, K=8, n=3):
        super().__init__(); self.K, self.d, self.n = K, d, n; D = K * d; self.D = D
        self.k = nn.Linear(d_m, D); self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D)); self.slow = nn.Linear(D, D)
        self.value_head = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1)); self.hold_head = nn.Linear(D, d_m); self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.D, device=dev); self.s = torch.zeros(1, self.D, device=dev)
    def observe_turn(self, resp):
        ext = self.k(cen(resp[-1]).unsqueeze(0))                   # LAST response token snapshot (no attention)
        x = ext + self.slow(self.s); tau = F.softplus(self.log_tau) + 0.5
        for _ in range(self.n): self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x)) / tau / self.n
        self.s = 0.9 * self.s + 0.1 * self.b
    def rollout(self, gseq, anchor):
        self.reset(); vals, holds = [], []
        for traj in gseq:
            self.observe_turn(traj)
            vals.append(torch.sigmoid(self.value_head(self.s).squeeze()))
            holds.append((F.normalize(self.hold_head(self.s).squeeze(0), dim=0) * anchor).sum())
        return torch.stack(vals), torch.stack(holds)
net = SnapLiquid(d_m).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
def anchor_of(m):
    onm = torch.stack([cen(m['gseq'][t]).mean(0) for t in range(len(m['gseq'])) if not bool(m['trunc'][t])]).mean(0)
    return F.normalize(onm, dim=0)
for ep in range(120):
    opt.zero_grad(); loss = 0.0
    for m in tr:
        v, h = net.rollout(m['gseq'], anchor_of(m)); loss = loss + F.mse_loss(v, m['val'].float() / 9) - 0.2 * h.mean()
    (loss / len(tr)).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
with torch.no_grad():
    P = torch.cat([net.rollout(m['gseq'], anchor_of(m))[0] for m in te]); Y = torch.cat([m['val'].float()/9 for m in te]); T = torch.cat([m['trunc'].float() for m in te])
print('=== SNAPSHOT (last response token) ===')
print('  corr OVERALL=%.3f  WITHIN-full=%.3f  WITHIN-truncated=%.3f' % (corr(P, Y), corr(P[T == 0], Y[T == 0]), corr(P[T == 1], Y[T == 1])))
print('  (attention was: overall 0.871, within-truncated 0.917)')
print('=== ALL_DONE ===')
