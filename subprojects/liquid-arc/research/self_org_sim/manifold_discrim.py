# Guard against the trivial-constant-direction failure: is the Liquid's HELD manifold goal-direction
# MISSION-SPECIFIC, or just a generic "on-task" vector that scores high vs every anchor? Check on
# held-out frames: (1) mutual similarity of mission anchors, (2) does the held belief retrieve its
# OWN mission's anchor among all held-out anchors (top-1).
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
data = torch.load('/home/pokazge/checkpoints/manifold_seqs.pt', weights_only=False, map_location='cpu')
ck = torch.load('/home/pokazge/checkpoints/manifold_holder.pt', weights_only=False, map_location='cpu')
GMEAN = ck['gmean']; d_m = ck['d_m']
def cen(x): return x - GMEAN
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames)//4):])
te = [d for d in data if d['fid'] in hold]

class ManifoldHolder(nn.Module):
    def __init__(self, d_m, d=128, K=4, n_steps=3):
        super().__init__(); self.K, self.d, self.n = K, d, n_steps; D = K * d
        self.in_m = nn.Linear(d_m, D); self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D))
        self.slow = nn.Linear(D, D); self.out = nn.Linear(D, d_m); self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.K*self.d); self.s = torch.zeros(1, self.K*self.d)
    def step(self, h):
        x = self.in_m(h) + self.slow(self.s); tau = F.softplus(self.log_tau) + 0.5
        for _ in range(self.n): self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x))/tau/self.n
        self.s = 0.9*self.s + 0.1*self.b; return self.b
    def readout(self): return F.normalize(self.out(self.b), dim=-1)
net = ManifoldHolder(d_m); net.load_state_dict(ck['net']); net.eval()
def anchor(m): return F.normalize(cen(m['seq'][~m['trunc']]).mean(0), dim=0)
anchors = torch.stack([anchor(m) for m in te])                        # [T, d_m]
# (1) anchor diversity
S = anchors @ anchors.T; off = S[~torch.eye(len(te), dtype=bool)]
print('held-out mission anchors: mean pairwise cos=%.3f (low => anchors are mission-distinct, not a constant)' % float(off.mean()))
# (2) does the held belief at truncated turns retrieve its OWN anchor?
hit = tot = 0; ownc = []; bestother = []
with torch.no_grad():
    for i, m in enumerate(te):
        net.reset()
        for t in range(m['seq'].shape[0]):
            net.b = net.step(cen(m['seq'][t]).unsqueeze(0))
            if bool(m['trunc'][t]):
                held = net.readout().squeeze(0); sims = anchors @ held
                j = int(sims.argmax()); hit += (j == i); tot += 1
                ownc.append(float(sims[i])); o = sims.clone(); o[i] = -9; bestother.append(float(o.max()))
print('held belief @ truncated turns: retrieves OWN mission anchor top-1 = %.0f%% (%d/%d, chance=%.0f%%)' % (100*hit/tot, hit, tot, 100.0/len(te)))
print('  cos(held, OWN anchor)=%.3f   cos(held, best OTHER anchor)=%.3f   margin=%+.3f' % (np.mean(ownc), np.mean(bestother), np.mean(ownc)-np.mean(bestother)))
print('=== ALL_DONE ===')
