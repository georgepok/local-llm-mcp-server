# MULTI-LAYER, PROCESS-AWARE identity in the Liquid. No single fixed layer: the Liquid reads the
# DEPTH-TRAJECTORY (last-token hidden across a span of layers, as a sequence) — perceiving HOW the LLM
# builds its representation through its computation, not a static layer-32 snapshot. Learned depth-
# position encoding lets the species weight depths. Two timescales: within a turn it integrates the
# depth-process (fast b); across turns it carries the belief (slow s). Value+hold are readouts of the
# persistent belief = the identity, in the Liquid. Per-layer centering (each depth has its own
# anisotropy). Trained on the EMPOWER objective values. Cross-frame test vs single-layer 0.959.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_value_multi.pt', weights_only=False, map_location='cpu')
data = obj['data']
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
ALL = torch.stack([d['mseq'] for d in data])                         # [M, turns, n_layers, d_m]
n_layers, d_m = ALL.shape[2], ALL.shape[3]
GMEAN = ALL.reshape(-1, n_layers, d_m).mean(0)                        # PER-LAYER mean [n_layers, d_m]
def cen(mseq): return mseq - GMEAN                                    # center each depth by its own mean
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
print('n_layers=%d d_m=%d | train=%d test=%d held=%s' % (n_layers, d_m, len(tr), len(te), sorted(hold)), flush=True)

class MultiLayerLiquid(nn.Module):
    def __init__(self, d_m, n_layers, d=128, K=6, n_ode=2):
        super().__init__(); self.K, self.d, self.n = K, d, n_ode; D = K * d; self.D = D
        self.in_m = nn.Linear(d_m, D); self.layer_emb = nn.Embedding(n_layers, D)
        self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D)); self.slow = nn.Linear(D, D)
        self.value_head = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1))
        self.hold_head = nn.Linear(D, d_m); self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.D, device=dev); self.s = torch.zeros(1, self.D, device=dev)
    def observe_turn(self, depth_stack):                             # depth_stack [n_layers, d_m] -> integrate the depth-process
        self.b = torch.zeros(1, self.D, device=dev); tau = F.softplus(self.log_tau) + 0.5
        for li in range(depth_stack.shape[0]):
            x = self.in_m(depth_stack[li].unsqueeze(0)) + self.layer_emb(torch.tensor(li)) + self.slow(self.s)
            for _ in range(self.n): self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x)) / tau / self.n
        self.s = 0.9 * self.s + 0.1 * self.b                          # carry the turn's depth-perception across turns
    def rollout(self, mseq, anchor):                                 # mseq [turns, n_layers, d_m]
        self.reset(); vals, holds = [], []
        for t in range(mseq.shape[0]):
            self.observe_turn(mseq[t])
            vals.append(torch.sigmoid(self.value_head(self.s).squeeze()))
            holds.append((F.normalize(self.hold_head(self.s).squeeze(0), dim=0) * anchor).sum())
        return torch.stack(vals), torch.stack(holds)
net = MultiLayerLiquid(d_m, n_layers).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
def anchor_of(m): c = cen(m['mseq']); return F.normalize(c[~m['trunc']].mean(0).mean(0), dim=0)   # on-goal, mean over layers
def corr(a, b): a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))
LAM_HOLD = 0.2
for ep in range(120):
    opt.zero_grad(); loss = 0.0
    for m in tr:
        v, h = net.rollout(cen(m['mseq']), anchor_of(m)); y = m['val'].float() / 9.0
        loss = loss + F.mse_loss(v, y) - LAM_HOLD * h.mean()
    (loss / len(tr)).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if ep % 30 == 0:
        with torch.no_grad():
            P, Y, on, dr = [], [], [], []
            for m in te:
                v, _ = net.rollout(cen(m['mseq']), anchor_of(m)); P.append(v); Y.append(m['val'].float()/9.0)
                on.append(v[~m['trunc']]); dr.append(v[m['trunc']])
            P = torch.cat(P); Y = torch.cat(Y); on = torch.cat(on); dr = torch.cat(dr)
            print('ep %d  held-out-FRAME: corr(V_multilayer, LLM-value)=%.3f  V(on)=%.2f V(drift)=%.2f gap=%.2f' %
                  (ep, corr(P, Y), float(on.mean()), float(dr.mean()), float(on.mean()-dr.mean())), flush=True)
with torch.no_grad():
    P, Y, on, dr = [], [], [], []
    for m in te:
        v, _ = net.rollout(cen(m['mseq']), anchor_of(m)); P.append(v); Y.append(m['val'].float()/9.0); on.append(v[~m['trunc']]); dr.append(v[m['trunc']])
    P = torch.cat(P); Y = torch.cat(Y); on = torch.cat(on); dr = torch.cat(dr)
print('\n[multilayer-identity] UNSEEN task types — MULTI-LAYER process-aware identity:')
print('  corr(V, LLM agentic-value) = %.3f  (vs single-layer-32 = 0.959)' % corr(P, Y))
print('  V(on-goal)=%.2f  V(drifted)=%.2f  gap=%+.2f' % (float(on.mean()), float(dr.mean()), float(on.mean()-dr.mean())))
torch.save({'net': net.state_dict(), 'gmean': GMEAN, 'd_m': d_m, 'n_layers': n_layers}, '/home/pokazge/checkpoints/multilayer_identity.pt')
print('[multilayer-identity] === ALL_DONE ===')
