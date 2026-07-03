# GENERATION-TRAJECTORY identity in the Liquid. The Liquid integrates the PER-TOKEN representational
# flow of each response (how the representation MOVES while the answer is produced) — the temporal
# dimension every snapshot-variant (single/multi/flow, all ~0.89) discarded. If the uniform ceiling
# was the suboptimal capture (not data), THIS should break it. Two timescales: within a turn it
# integrates the generation token-flow; across turns the slow channel carries. Identity = value/hold
# readouts of the belief. Cross-frame test vs the 0.89 snapshot ceiling.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_value_gen.pt', weights_only=False, map_location='cpu')
data = obj['data']
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
allt = torch.cat([t for m in data for t in m['gseq']], 0)            # [total_resp_tokens, d_m]
d_m = allt.shape[1]; GMEAN = allt.mean(0); GSTD = float((allt - GMEAN).std() + 1e-6)
def cen(t): return (t - GMEAN) / GSTD
CAP = 24                                                            # cap response trajectory length (CPU tractability)
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
print('d_m=%d total_resp_tokens=%d GSTD=%.2f | train=%d test=%d' % (d_m, allt.shape[0], GSTD, len(tr), len(te)), flush=True)

class GenLiquid(nn.Module):
    def __init__(self, d_m, d=128, K=6, n_ode=2, cap=CAP):
        super().__init__(); self.K, self.d, self.n = K, d, n_ode; D = K * d; self.D = D
        self.in_m = nn.Linear(d_m, D); self.tok_emb = nn.Embedding(cap, D)
        self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D)); self.slow = nn.Linear(D, D)
        self.value_head = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1))
        self.hold_head = nn.Linear(D, d_m); self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.D, device=dev); self.s = torch.zeros(1, self.D, device=dev)
    def observe_turn(self, traj):                                  # traj [n_resp, d_m] -> integrate the generation token-flow
        self.b = torch.zeros(1, self.D, device=dev); tau = F.softplus(self.log_tau) + 0.5
        traj = traj[:CAP]
        for i in range(traj.shape[0]):
            x = self.in_m(cen(traj[i]).unsqueeze(0)) + self.tok_emb(torch.tensor(min(i, CAP - 1))) + self.slow(self.s)
            for _ in range(self.n): self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x)) / tau / self.n
        self.s = 0.9 * self.s + 0.1 * self.b
    def rollout(self, gseq, anchor):
        self.reset(); vals, holds = [], []
        for traj in gseq:
            self.observe_turn(traj)
            vals.append(torch.sigmoid(self.value_head(self.s).squeeze()))
            holds.append((F.normalize(self.hold_head(self.s).squeeze(0), dim=0) * anchor).sum())
        return torch.stack(vals), torch.stack(holds)
net = GenLiquid(d_m).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
def anchor_of(m):
    onmean = torch.stack([cen(m['gseq'][t][:CAP]).mean(0) for t in range(len(m['gseq'])) if not bool(m['trunc'][t])]).mean(0)
    return F.normalize(onmean, dim=0)
def corr(a, b): a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))
LAM_HOLD = 0.2
for ep in range(100):
    opt.zero_grad(); loss = 0.0
    for m in tr:
        v, h = net.rollout(m['gseq'], anchor_of(m)); y = m['val'].float() / 9.0
        loss = loss + F.mse_loss(v, y) - LAM_HOLD * h.mean()
    (loss / len(tr)).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if ep % 25 == 0:
        with torch.no_grad():
            P, Y = [], []
            for m in te:
                v, _ = net.rollout(m['gseq'], anchor_of(m)); P.append(v); Y.append(m['val'].float()/9.0)
            P = torch.cat(P); Y = torch.cat(Y)
            print('ep %d  held-out-FRAME: corr(V_gen, LLM-value)=%.3f' % (ep, corr(P, Y)), flush=True)
with torch.no_grad():
    P, Y, on, dr = [], [], [], []
    for m in te:
        v, _ = net.rollout(m['gseq'], anchor_of(m)); P.append(v); Y.append(m['val'].float()/9.0)
        on.append(v[~m['trunc']]); dr.append(v[m['trunc']])
    P = torch.cat(P); Y = torch.cat(Y); on = torch.cat(on); dr = torch.cat(dr)
print('\n[gen-identity] UNSEEN task types — GENERATION-TRAJECTORY identity (the process the snapshots discarded):')
print('  corr(V_gen, LLM agentic-value) = %.3f  (vs snapshot ceiling ~0.89)' % corr(P, Y))
print('  V(on-goal)=%.2f V(drifted)=%.2f gap=%+.2f' % (float(on.mean()), float(dr.mean()), float(on.mean()-dr.mean())))
torch.save({'net': net.state_dict(), 'gmean': GMEAN, 'gstd': GSTD, 'd_m': d_m, 'cap': CAP}, '/home/pokazge/checkpoints/gen_identity.pt')
print('[gen-identity] === ALL_DONE ===')
