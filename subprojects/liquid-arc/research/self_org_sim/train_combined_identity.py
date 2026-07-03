# COMBINED identity: the Liquid integrates BOTH the END RESULT (content = context-last-token = the
# model's attention OUTPUT, 0.899) AND the DYNAMICS of attention (routing = entropy/recency/peak/
# recent-mass per full-attn layer = HOW the features were chosen). Per turn: integrate the routing
# depth-trajectory, THEN fold in the end-result content; belief carries across turns. Tests whether
# adding the selection-dynamics to the features BEATS features-alone (0.899) / attn within-trunc 0.917.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_value_attn.pt', weights_only=False, map_location='cpu')
data = obj['data']                                                   # each has 'seq'(content) AND 'aseq'(routing)
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
C = torch.stack([d['seq'] for d in data]); d_m = C.shape[2]
GMEAN_C = C.reshape(-1, d_m).mean(0)
A = torch.stack([d['aseq'] for d in data]); n_L, n_F = A.shape[2], A.shape[3]
GMEAN_A = A.reshape(-1, n_L, n_F).mean(0); GSTD_A = A.reshape(-1, n_L, n_F).std(0) + 1e-6
def cenc(x): return x - GMEAN_C
def cena(a): return (a - GMEAN_A) / GSTD_A
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9)) if a.norm() > 0 and b.norm() > 0 else 0.0
print('d_m=%d n_layers=%d n_feats=%d | train=%d test=%d' % (d_m, n_L, n_F, len(tr), len(te)), flush=True)

class CombinedLiquid(nn.Module):
    def __init__(self, d_m, n_F, n_L, d=128, K=6, n_ode=2):
        super().__init__(); self.K, self.d, self.n = K, d, n_ode; D = K * d; self.D = D
        self.in_route = nn.Linear(n_F, D); self.route_emb = nn.Embedding(n_L, D)
        self.in_content = nn.Linear(d_m, D)
        self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D)); self.slow = nn.Linear(D, D)
        self.value_head = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1)); self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.D, device=dev); self.s = torch.zeros(1, self.D, device=dev)
    def step(self, x, tau):
        for _ in range(self.n): self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x)) / tau / self.n
    def observe_turn(self, content, routing):                      # content [d_m], routing [n_L, n_F]
        self.b = torch.zeros(1, self.D, device=dev); tau = F.softplus(self.log_tau) + 0.5
        for li in range(routing.shape[0]):                          # integrate HOW features were chosen (routing dynamics)
            self.step(self.in_route(routing[li].unsqueeze(0)) + self.route_emb(torch.tensor(li)) + self.slow(self.s), tau)
        self.step(self.in_content(content.unsqueeze(0)) + self.slow(self.s), tau)   # then the END RESULT (content)
        self.s = 0.9 * self.s + 0.1 * self.b
    def rollout(self, m):
        self.reset(); vals = []
        for t in range(m['seq'].shape[0]):
            self.observe_turn(cenc(m['seq'][t]), cena(m['aseq'][t])); vals.append(torch.sigmoid(self.value_head(self.s).squeeze()))
        return torch.stack(vals)
net = CombinedLiquid(d_m, n_F, n_L).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
for ep in range(150):
    opt.zero_grad(); loss = 0.0
    for m in tr: loss = loss + F.mse_loss(net.rollout(m), m['val'].float() / 9)
    (loss / len(tr)).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if ep % 30 == 0:
        with torch.no_grad():
            P = torch.cat([net.rollout(m) for m in te]); Y = torch.cat([m['val'].float()/9 for m in te])
            print('ep %d  corr(V_combined, value) overall=%.3f' % (ep, corr(P, Y)), flush=True)
with torch.no_grad():
    P = torch.cat([net.rollout(m) for m in te]); Y = torch.cat([m['val'].float()/9 for m in te]); T = torch.cat([m['trunc'].float() for m in te])
print('\n=== COMBINED (end-result content + attention dynamics) identity ===')
print('  corr OVERALL=%.3f  WITHIN-full=%.3f  WITHIN-truncated=%.3f' % (corr(P, Y), corr(P[T == 0], Y[T == 0]), corr(P[T == 1], Y[T == 1])))
print('  (content-only snapshot 0.899 | content-attn within-trunc 0.917)')
torch.save({'net': net.state_dict()}, '/home/pokazge/checkpoints/combined_identity.pt')
print('=== ALL_DONE ===')
