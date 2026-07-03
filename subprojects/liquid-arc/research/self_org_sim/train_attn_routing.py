# ROUTING identity: the Liquid integrates HOW the model attended (routing features per full-attn
# layer) — the selection DYNAMICS, NOT the selected features. Pure routing (4 feats x 16 layers/turn),
# no hidden-state content. If the goal-signal lives in the ROUTING (where/how the model looked), this
# predicts the value despite seeing NO content. Tests the user's hypothesis: snapshot of features !=
# dynamics of how features were chosen. Compared to content reads (~0.89).
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_value_attn.pt', weights_only=False, map_location='cpu')
data = obj['data']
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
ALL = torch.stack([d['aseq'] for d in data])                         # [M, turns, 16, 4]
n_L, n_F = ALL.shape[2], ALL.shape[3]
GMEAN = ALL.reshape(-1, n_L, n_F).mean(0); GSTD = ALL.reshape(-1, n_L, n_F).std(0) + 1e-6   # per (layer,feature)
def cen(a): return (a - GMEAN) / GSTD
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9)) if a.norm() > 0 and b.norm() > 0 else 0.0
print('n_layers=%d n_feats=%d | train=%d test=%d' % (n_L, n_F, len(tr), len(te)), flush=True)

class RoutingLiquid(nn.Module):
    def __init__(self, n_F, n_L, d=128, K=4, n_ode=2):
        super().__init__(); self.K, self.d, self.n = K, d, n_ode; D = K * d; self.D = D
        self.in_m = nn.Linear(n_F, D); self.layer_emb = nn.Embedding(n_L, D)
        self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D)); self.slow = nn.Linear(D, D)
        self.value_head = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1)); self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.D, device=dev); self.s = torch.zeros(1, self.D, device=dev)
    def observe_turn(self, route):                                 # route [n_L, n_F] -> integrate the routing depth-trajectory
        self.b = torch.zeros(1, self.D, device=dev); tau = F.softplus(self.log_tau) + 0.5
        for li in range(route.shape[0]):
            x = self.in_m(route[li].unsqueeze(0)) + self.layer_emb(torch.tensor(li)) + self.slow(self.s)
            for _ in range(self.n): self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x)) / tau / self.n
        self.s = 0.9 * self.s + 0.1 * self.b
    def rollout(self, aseq):
        self.reset(); vals = []
        for t in range(aseq.shape[0]):
            self.observe_turn(aseq[t]); vals.append(torch.sigmoid(self.value_head(self.s).squeeze()))
        return torch.stack(vals)
net = RoutingLiquid(n_F, n_L).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
for ep in range(150):
    opt.zero_grad(); loss = 0.0
    for m in tr: loss = loss + F.mse_loss(net.rollout(cen(m['aseq'])), m['val'].float() / 9)
    (loss / len(tr)).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if ep % 30 == 0:
        with torch.no_grad():
            P = torch.cat([net.rollout(cen(m['aseq'])) for m in te]); Y = torch.cat([m['val'].float()/9 for m in te])
            print('ep %d  corr(V_routing, value) overall=%.3f' % (ep, corr(P, Y)), flush=True)
with torch.no_grad():
    P = torch.cat([net.rollout(cen(m['aseq'])) for m in te]); Y = torch.cat([m['val'].float()/9 for m in te]); T = torch.cat([m['trunc'].float() for m in te])
print('\n=== ROUTING-ONLY identity (HOW the model attended, no content) ===')
print('  corr OVERALL=%.3f  WITHIN-full=%.3f  WITHIN-truncated=%.3f' % (corr(P, Y), corr(P[T == 0], Y[T == 0]), corr(P[T == 1], Y[T == 1])))
print('  (content snapshot 0.899 | content-attn within-trunc 0.917 | binary corr -0.346)')
torch.save({'net': net.state_dict(), 'gmean': GMEAN, 'gstd': GSTD}, '/home/pokazge/checkpoints/routing_identity.pt')
print('=== ALL_DONE ===')
