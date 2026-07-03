# ATTENTION-BASED perception. Instead of a fixed last-token snapshot (context-indifferent), the
# Liquid's belief (holding the goal) issues a QUERY that attends over the LLM's response-token hidden
# states and EXTRACTS the goal-relevant features — the way attention itself extracts what matters for
# the context. Then integrates across turns. Reuses the captured response-token hiddens (gseq).
# CRITICAL metric: corr WITHIN truncated turns (binary removed) = genuine goal-tracking. Compared
# against the last-token snapshot's within-truncated corr (is attention capturing what snapshot blurs?).
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

class AttnLiquid(nn.Module):
    def __init__(self, d_m, d=128, K=8, n=3):
        super().__init__(); self.K, self.d, self.n = K, d, n; D = K * d; self.D = D; self.scale = D ** -0.5
        self.k = nn.Linear(d_m, D); self.v = nn.Linear(d_m, D)
        self.q = nn.Linear(D, D); self.q0 = nn.Parameter(torch.randn(D) * 0.02)   # base + belief-conditioned query
        self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D)); self.slow = nn.Linear(D, D)
        self.value_head = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1)); self.hold_head = nn.Linear(D, d_m)
        self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.D, device=dev); self.s = torch.zeros(1, self.D, device=dev)
    def observe_turn(self, resp):                                  # resp [n_resp, d_m]
        h = cen(resp); Kk = self.k(h); Vv = self.v(h)             # [n_resp, D]
        q = self.q(self.s) + self.q0                              # [1, D] belief (goal) queries
        attn = F.softmax((q @ Kk.t()) * self.scale, dim=-1)       # [1, n_resp] goal-relevance weights
        ext = attn @ Vv                                          # [1, D] extracted goal-relevant features
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
net = AttnLiquid(d_m).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
def anchor_of(m):
    onm = torch.stack([cen(m['gseq'][t]).mean(0) for t in range(len(m['gseq'])) if not bool(m['trunc'][t])]).mean(0)
    return F.normalize(onm, dim=0)
LAM_HOLD = 0.2
for ep in range(120):
    opt.zero_grad(); loss = 0.0
    for m in tr:
        v, h = net.rollout(m['gseq'], anchor_of(m)); loss = loss + F.mse_loss(v, m['val'].float() / 9) - LAM_HOLD * h.mean()
    (loss / len(tr)).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if ep % 30 == 0:
        with torch.no_grad():
            P = torch.cat([net.rollout(m['gseq'], anchor_of(m))[0] for m in te]); Y = torch.cat([m['val'].float()/9 for m in te])
            print('ep %d  corr(V_attn, value) overall=%.3f' % (ep, corr(P, Y)), flush=True)
with torch.no_grad():
    P = torch.cat([net.rollout(m['gseq'], anchor_of(m))[0] for m in te]); Y = torch.cat([m['val'].float()/9 for m in te])
    T = torch.cat([m['trunc'].float() for m in te])
    # last-token snapshot baseline (within-truncated) from the SAME data
    snap = torch.stack([cen(t[-1]) for m in te for t in m['gseq']])     # last response token per turn
print('\n=== ATTENTION perception vs the binary ===')
print('  corr(trunc_flag, value)            = %.3f   (the binary baseline)' % corr(T, Y))
print('  corr(V_attn, value) OVERALL        = %.3f' % corr(P, Y))
print('  corr(V_attn, value) WITHIN full     = %.3f' % corr(P[T == 0], Y[T == 0]))
print('  corr(V_attn, value) WITHIN truncated= %.3f   <-- genuine goal-tracking (binary removed)' % corr(P[T == 1], Y[T == 1]))
torch.save({'net': net.state_dict(), 'gmean': GMEAN, 'gstd': GSTD, 'd_m': d_m}, '/home/pokazge/checkpoints/attn_identity.pt')
print('=== ALL_DONE ===')
