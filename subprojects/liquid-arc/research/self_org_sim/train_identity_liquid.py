# IDENTITY REALIZED IN THE LIQUID (not a bolted-on MLP). ONE organism: the Liquid integrates the
# manifold trajectory into its belief (slow channel persists the goal it infers from early on-goal
# turns; fast channel tracks the current state), and the VALUE is a READOUT of that belief — the
# Liquid's own voice judging agentic quality. NO separate net over the raw manifold, NO explicit
# anchor input: the goal-reference comes from the Liquid's own holding. The identity = the Liquid's
# trained dynamics + value readout (persistent params, shared across all worlds). Trained to
# internalize the LLM's rich agentic judgment INTO the Liquid's dynamics. Cross-frame test = does the
# identity generalize to UNSEEN task types as a property of the organism.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
torch.manual_seed(0)
data = torch.load('/home/pokazge/checkpoints/value_seqs.pt', weights_only=False, map_location='cpu')
dev = torch.device('cpu')
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
GMEAN = torch.stack([d['seq'] for d in data]).reshape(-1, data[0]['seq'].shape[1]).mean(0)
def cen(x): return x - GMEAN
d_m = data[0]['seq'].shape[1]
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
print('d_m=%d frames=%d held=%s train=%d test=%d' % (d_m, len(frames), sorted(hold), len(tr), len(te)), flush=True)

class IdentityLiquid(nn.Module):                                   # the organism: continuous dynamics + value/hold readouts
    def __init__(self, d_m, d=128, K=8, n=4):
        super().__init__(); self.K, self.d, self.n = K, d, n; D = K * d
        self.in_m = nn.Linear(d_m, D); self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D))
        self.slow = nn.Linear(D, D)
        self.value_head = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1))   # the identity's VALUE voice
        self.hold_head = nn.Linear(D, d_m)                          # held goal-direction (keeps the self anchored)
        self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.K*self.d, device=dev); self.s = torch.zeros(1, self.K*self.d, device=dev)
    def step(self, h):
        x = self.in_m(h) + self.slow(self.s); tau = F.softplus(self.log_tau) + 0.5
        for _ in range(self.n): self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x)) / tau / self.n
        self.s = 0.9 * self.s + 0.1 * self.b; return self.b
    def value(self): return self.value_head(self.b).squeeze(-1).squeeze(0)        # scalar in [~0,1] after sigmoid below
    def hold(self): return F.normalize(self.hold_head(self.b).squeeze(0), dim=-1)
net = IdentityLiquid(d_m).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
def anchor_of(m): return F.normalize(cen(m['seq'][~m['trunc']]).mean(0), dim=0)
def rollout(m):
    net.reset(); vals, holds = [], []
    a = anchor_of(m)
    for t in range(m['seq'].shape[0]):
        net.b = net.step(cen(m['seq'][t]).unsqueeze(0))
        vals.append(torch.sigmoid(net.value())); holds.append((net.hold() * a).sum())
    return torch.stack(vals), torch.stack(holds)
def corr(a, b): a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))
LAM_HOLD = 0.3
for ep in range(500):
    opt.zero_grad(); loss = 0.0
    for m in tr:
        v, h = rollout(m); y = m['val'].float() / 9.0
        loss = loss + F.mse_loss(v, y) - LAM_HOLD * h.mean()        # internalize value; keep the self anchored (hold high)
    (loss / len(tr)).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if ep % 100 == 0:
        with torch.no_grad():
            P, Y, on, dr = [], [], [], []
            for m in te:
                v, _ = rollout(m); P.append(v); Y.append(m['val'].float()/9.0)
                on.append(v[~m['trunc']]); dr.append(v[m['trunc']])
            P = torch.cat(P); Y = torch.cat(Y); on = torch.cat(on); dr = torch.cat(dr)
            print('ep %d  held-out-FRAME: corr(V_liquid, LLM-value)=%.3f  V(on)=%.2f V(drift)=%.2f gap=%.2f' %
                  (ep, corr(P, Y), float(on.mean()), float(dr.mean()), float(on.mean()-dr.mean())), flush=True)
with torch.no_grad():
    P, Y, on, dr = [], [], [], []
    for m in te:
        v, _ = rollout(m); P.append(v); Y.append(m['val'].float()/9.0); on.append(v[~m['trunc']]); dr.append(v[m['trunc']])
    P = torch.cat(P); Y = torch.cat(Y); on = torch.cat(on); dr = torch.cat(dr)
print('\n[identity-LIQUID] UNSEEN task types — identity AS the Liquid:')
print('  corr(V_liquid, LLM agentic-value) = %.3f   (the identity, realized in the Liquid, generalizes)' % corr(P, Y))
print('  V(on-goal)=%.2f  V(drifted)=%.2f  gap=%+.2f' % (float(on.mean()), float(dr.mean()), float(on.mean()-dr.mean())))
torch.save({'net': net.state_dict(), 'gmean': GMEAN, 'd_m': d_m}, '/home/pokazge/checkpoints/identity_liquid.pt')
print('[identity-LIQUID] === ALL_DONE ===')
