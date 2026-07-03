# COMBINED + MULTI-LEVEL DERIVATIVES. The attention DYNAMICS = derivatives of the routing across the
# layer trajectory: 1st (focusing vs diffusing), 2nd (accelerating vs stalling). Augment each layer's
# routing [4] with its 1st + 2nd layer-derivatives -> [12]. The Liquid integrates content (end result)
# + routing-with-derivatives (multi-level dynamics). Tests whether the SHAPE OF THE EVOLUTION carries
# goal-signal beyond raw routing + content.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_value_attn.pt', weights_only=False, map_location='cpu')
data = obj['data']
def derivs(route):                                                   # [n_L, 4] -> [n_L, 12] : raw, 1st, 2nd layer-derivatives
    d1 = torch.zeros_like(route); d1[1:] = route[1:] - route[:-1]
    d2 = torch.zeros_like(route); d2[1:-1] = route[2:] - 2 * route[1:-1] + route[:-2]
    return torch.cat([route, d1, d2], dim=-1)
for m in data:
    m['adv'] = torch.stack([derivs(m['aseq'][t]) for t in range(m['aseq'].shape[0])])   # [turns, n_L, 12]
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
C = torch.stack([d['seq'] for d in data]); d_m = C.shape[2]; GMEAN_C = C.reshape(-1, d_m).mean(0)
A = torch.stack([d['adv'] for d in data]); n_L, n_F = A.shape[2], A.shape[3]
GMEAN_A = A.reshape(-1, n_L, n_F).mean(0); GSTD_A = A.reshape(-1, n_L, n_F).std(0) + 1e-6
def cenc(x): return x - GMEAN_C
def cena(a): return (a - GMEAN_A) / GSTD_A
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9)) if a.norm() > 0 and b.norm() > 0 else 0.0
print('d_m=%d n_layers=%d routing-deriv-feats=%d | train=%d test=%d' % (d_m, n_L, n_F, len(tr), len(te)), flush=True)
class CombinedLiquid(nn.Module):
    def __init__(self, d_m, n_F, n_L, d=128, K=6, n_ode=2):
        super().__init__(); self.K, self.d, self.n = K, d, n_ode; D = K * d; self.D = D
        self.in_route = nn.Linear(n_F, D); self.route_emb = nn.Embedding(n_L, D); self.in_content = nn.Linear(d_m, D)
        self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D)); self.slow = nn.Linear(D, D)
        self.value_head = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1)); self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.D, device=dev); self.s = torch.zeros(1, self.D, device=dev)
    def step(self, x, tau):
        for _ in range(self.n): self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x)) / tau / self.n
    def observe_turn(self, content, routing):
        self.b = torch.zeros(1, self.D, device=dev); tau = F.softplus(self.log_tau) + 0.5
        for li in range(routing.shape[0]):
            self.step(self.in_route(routing[li].unsqueeze(0)) + self.route_emb(torch.tensor(li)) + self.slow(self.s), tau)
        self.step(self.in_content(content.unsqueeze(0)) + self.slow(self.s), tau)
        self.s = 0.9 * self.s + 0.1 * self.b
    def rollout(self, m):
        self.reset(); vals = []
        for t in range(m['seq'].shape[0]):
            self.observe_turn(cenc(m['seq'][t]), cena(m['adv'][t])); vals.append(torch.sigmoid(self.value_head(self.s).squeeze()))
        return torch.stack(vals)
net = CombinedLiquid(d_m, n_F, n_L).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
for ep in range(150):
    opt.zero_grad(); loss = 0.0
    for m in tr: loss = loss + F.mse_loss(net.rollout(m), m['val'].float() / 9)
    (loss / len(tr)).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if ep % 30 == 0:
        with torch.no_grad():
            P = torch.cat([net.rollout(m) for m in te]); Y = torch.cat([m['val'].float()/9 for m in te])
            print('ep %d  corr overall=%.3f' % (ep, corr(P, Y)), flush=True)
with torch.no_grad():
    P = torch.cat([net.rollout(m) for m in te]); Y = torch.cat([m['val'].float()/9 for m in te]); T = torch.cat([m['trunc'].float() for m in te])
print('\n=== COMBINED + ROUTING DERIVATIVES (content + 1st/2nd-deriv attention dynamics) ===')
print('  corr OVERALL=%.3f  WITHIN-full=%.3f  WITHIN-truncated=%.3f' % (corr(P, Y), corr(P[T == 0], Y[T == 0]), corr(P[T == 1], Y[T == 1])))
print('  (content-only 0.899 | content-attn within-trunc 0.917)')
torch.save({'net': net.state_dict()}, '/home/pokazge/checkpoints/combined_deriv_identity.pt')
print('=== ALL_DONE ===')
