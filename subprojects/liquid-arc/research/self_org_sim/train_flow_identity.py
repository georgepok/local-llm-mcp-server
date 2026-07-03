# FLOW-based identity in the Liquid. The Liquid integrates the INTER-LAYER FLOW (residual-stream
# deltas = what each block ADDS) — the genuine computational dynamics, with MAGNITUDE kept. Per-
# position centering removes the AVERAGE flow at each block (exposing the goal-specific deviation in
# the flow); a single global scale keeps relative magnitudes. Identity = value/hold readouts of the
# Liquid belief built by integrating the flow. Cross-frame test vs multi-layer (silhouette) baseline.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_value_flow.pt', weights_only=False, map_location='cpu')
data = obj['data']
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
ALL = torch.stack([d['fseq'] for d in data])                         # [M, turns, n_deltas, d_m]
n_d, d_m = ALL.shape[2], ALL.shape[3]
GMEAN = ALL.reshape(-1, n_d, d_m).mean(0)                            # per-position mean flow [n_deltas, d_m]
GSTD = float((ALL.reshape(-1, n_d, d_m) - GMEAN).std() + 1e-6)       # single global scale (keeps relative magnitudes)
def norm_flow(fseq): return (fseq - GMEAN) / GSTD
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
print('n_deltas=%d d_m=%d GSTD=%.2f | train=%d test=%d' % (n_d, d_m, GSTD, len(tr), len(te)), flush=True)

class FlowLiquid(nn.Module):
    def __init__(self, d_m, n_d, d=128, K=6, n_ode=2):
        super().__init__(); self.K, self.d, self.n = K, d, n_ode; D = K * d; self.D = D
        self.in_m = nn.Linear(d_m, D); self.pos_emb = nn.Embedding(n_d, D)
        self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D)); self.slow = nn.Linear(D, D)
        self.value_head = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1))
        self.hold_head = nn.Linear(D, d_m); self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.D, device=dev); self.s = torch.zeros(1, self.D, device=dev)
    def observe_turn(self, flow):                                   # flow [n_deltas, d_m] -> integrate the residual-stream flow
        self.b = torch.zeros(1, self.D, device=dev); tau = F.softplus(self.log_tau) + 0.5
        for i in range(flow.shape[0]):
            x = self.in_m(flow[i].unsqueeze(0)) + self.pos_emb(torch.tensor(i)) + self.slow(self.s)
            for _ in range(self.n): self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x)) / tau / self.n
        self.s = 0.9 * self.s + 0.1 * self.b
    def rollout(self, fseq, anchor):
        self.reset(); vals, holds = [], []
        for t in range(fseq.shape[0]):
            self.observe_turn(fseq[t])
            vals.append(torch.sigmoid(self.value_head(self.s).squeeze()))
            holds.append((F.normalize(self.hold_head(self.s).squeeze(0), dim=0) * anchor).sum())
        return torch.stack(vals), torch.stack(holds)
net = FlowLiquid(d_m, n_d).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
def anchor_of(m): c = norm_flow(m['fseq']); return F.normalize(c[~m['trunc']].mean(0).mean(0), dim=0)
def corr(a, b): a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))
LAM_HOLD = 0.2
for ep in range(120):
    opt.zero_grad(); loss = 0.0
    for m in tr:
        v, h = net.rollout(norm_flow(m['fseq']), anchor_of(m)); y = m['val'].float() / 9.0
        loss = loss + F.mse_loss(v, y) - LAM_HOLD * h.mean()
    (loss / len(tr)).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if ep % 30 == 0:
        with torch.no_grad():
            P, Y = [], []
            for m in te:
                v, _ = net.rollout(norm_flow(m['fseq']), anchor_of(m)); P.append(v); Y.append(m['val'].float()/9.0)
            P = torch.cat(P); Y = torch.cat(Y)
            print('ep %d  held-out-FRAME: corr(V_flow, LLM-value)=%.3f' % (ep, corr(P, Y)), flush=True)
with torch.no_grad():
    P, Y, on, dr = [], [], [], []
    for m in te:
        v, _ = net.rollout(norm_flow(m['fseq']), anchor_of(m)); P.append(v); Y.append(m['val'].float()/9.0)
        on.append(v[~m['trunc']]); dr.append(v[m['trunc']])
    P = torch.cat(P); Y = torch.cat(Y); on = torch.cat(on); dr = torch.cat(dr)
print('\n[flow-identity] UNSEEN task types — FLOW (inter-layer dynamics) identity in the Liquid:')
print('  corr(V_flow, LLM agentic-value) = %.3f  (vs multi-layer silhouette, vs single-layer 0.959)' % corr(P, Y))
print('  V(on-goal)=%.2f V(drifted)=%.2f gap=%+.2f' % (float(on.mean()), float(dr.mean()), float(on.mean()-dr.mean())))
torch.save({'net': net.state_dict(), 'gmean': GMEAN, 'gstd': GSTD, 'd_m': d_m, 'n_d': n_d}, '/home/pokazge/checkpoints/flow_identity.pt')
print('[flow-identity] === ALL_DONE ===')
