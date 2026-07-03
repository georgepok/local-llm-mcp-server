# Manifold-native goal-trajectory holder, v2 with CENTERING (removes transformer hidden-state
# anisotropy — the narrow-cone common component that saturates raw cosine ~0.93; manifold analog of
# the validated relative-cosine fix). The Liquid (a continuous cell + slow channel) lives ENTIRELY
# on the CENTERED manifold: reads centered hidden-state positions, maintains a goal-direction, reads
# back the held centered goal-direction. NO text. Goal SELF-EXTRACTED = centroid of early on-goal
# (centered) positions. CROSS-TASK test: hold out whole frames (unseen task types) = genericity.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
torch.manual_seed(0)
data = torch.load('/home/pokazge/checkpoints/manifold_seqs.pt', weights_only=False, map_location='cpu')
dev = torch.device('cpu')
frames = sorted(set(d['fid'] for d in data))
n_hold = max(3, len(frames) // 4); hold_frames = set(frames[-n_hold:])
GMEAN = torch.stack([d['seq'] for d in data]).reshape(-1, data[0]['seq'].shape[1]).mean(0)   # anisotropy common component
def cen(x): return x - GMEAN                                          # center -> exposes real goal-direction
tr = [d for d in data if d['fid'] not in hold_frames]
te = [d for d in data if d['fid'] in hold_frames]
d_m = data[0]['seq'].shape[1]
print('frames=%d train_frames=%d held_out=%s | train=%d test=%d d_m=%d' %
      (len(frames), len(frames) - n_hold, sorted(hold_frames), len(tr), len(te), d_m), flush=True)

def anchor(m):
    early = cen(m['seq'][~m['trunc']]); return F.normalize(early.mean(0), dim=0)   # centered on-goal centroid

class ManifoldHolder(nn.Module):
    def __init__(self, d_m, d=128, K=4, n_steps=3):
        super().__init__(); self.K, self.d, self.n = K, d, n_steps; D = K * d
        self.in_m = nn.Linear(d_m, D); self.W = nn.Linear(D, D)
        self.log_tau = nn.Parameter(torch.zeros(D)); self.slow = nn.Linear(D, D)
        self.out = nn.Linear(D, d_m); self.b = None; self.s = None
    def reset(self, dev): self.b = torch.zeros(1, self.K * self.d, device=dev); self.s = torch.zeros(1, self.K * self.d, device=dev)
    def step(self, h_m):
        x = self.in_m(h_m) + self.slow(self.s); tau = F.softplus(self.log_tau) + 0.5
        for _ in range(self.n):
            self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x)) / tau / self.n
        self.s = 0.9 * self.s + 0.1 * self.b
        return self.b
    def readout(self): return F.normalize(self.out(self.b), dim=-1)

net = ManifoldHolder(d_m).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
def rollout(m):
    a = anchor(m); net.reset(dev); ests = []
    for t in range(m['seq'].shape[0]):
        net.b = net.step(cen(m['seq'][t]).unsqueeze(0))              # Liquid reads CENTERED manifold position
        ests.append(net.readout().squeeze(0))
    return torch.stack(ests), a, m['trunc']
def raw_cos(m, a):                                                    # centered raw-manifold drift (baseline)
    c = F.normalize(cen(m['seq']), dim=1); return (c * a).sum(-1)
for ep in range(300):
    opt.zero_grad(); loss = 0.0
    for m in tr:
        e, a, tm = rollout(m); w = torch.where(tm, 2.5, 1.0)
        loss = loss + ((1 - (e * a).sum(-1)) * w).mean()
    (loss / len(tr)).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if ep % 60 == 0:
        with torch.no_grad():
            L, R = [], []
            for m in te:
                e, a, tm = rollout(m); L.append(float((e[tm] * a).sum(-1).mean())); R.append(float(raw_cos(m, a)[tm].mean()))
            print('ep %d  held-out-FRAME truncated (CENTERED): LIQUID-held cos=%.3f  RAW-manifold cos=%.3f' % (ep, np.mean(L), np.mean(R)), flush=True)
with torch.no_grad():
    Lf, Rf, Lt, Rt = [], [], [], []
    for m in te:
        e, a, tm = rollout(m); rc = raw_cos(m, a)
        Lf.append(float((e[~tm] * a).sum(-1).mean())); Rf.append(float(rc[~tm].mean()))
        Lt.append(float((e[tm] * a).sum(-1).mean())); Rt.append(float(rc[tm].mean()))
print('\n[manifold-holder] CROSS-TASK held-out frames %s (CENTERED manifold cosine):' % sorted(hold_frames))
print('  full-context turns : LIQUID=%.3f  RAW-manifold=%.3f' % (np.mean(Lf), np.mean(Rf)))
print('  TRUNCATED turns    : LIQUID=%.3f  RAW-manifold=%.3f   (Liquid navigates to goal-attractor on UNSEEN task types: %s)'
      % (np.mean(Lt), np.mean(Rt), 'YES' if np.mean(Lt) > np.mean(Rt) + 0.03 else 'no'))
torch.save({'net': net.state_dict(), 'd_m': d_m, 'gmean': GMEAN}, '/home/pokazge/checkpoints/manifold_holder.pt')
print('[manifold-holder] === ALL_DONE ===')
