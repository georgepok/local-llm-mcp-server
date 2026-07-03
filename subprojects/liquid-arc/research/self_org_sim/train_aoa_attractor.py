# GROUNDING via attractor (goal-as-field). The additive grounded model only made identity AVAILABLE to prediction
# (cat(b,s)) — and the fast belief was a sufficient sidechannel, so prediction bypassed identity (grounding-gap 0).
# Availability != grounding. Here identity is LOAD-BEARING: it is the ATTRACTOR/SETPOINT of the working dynamics —
# the belief relaxes toward the identity each step, and the turn's AoA input is the OFFSET FROM THE SELF. Prediction
# then runs in the identity-RELATIVE frame ("prediction is always relative" = relative to the self). Decisive test:
# zero the identity setpoint -> the working state loses its base -> prediction MUST move (grounding-gap > 0).
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_value_gen.pt', weights_only=False, map_location='cpu')
data = obj['data']; d_m = data[0]['gen'][0].shape[1]
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
MU = torch.cat([t for m in data for t in m['gen']], 0).mean(0)
def cen(G): return G - MU
for m in data: m['z'] = [F.normalize(cen(t).mean(0), dim=0) for t in m['gen']]
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
print('d_m=%d  train=%d test=%d' % (d_m, len(tr), len(te)), flush=True)

class AttractorGrounded(nn.Module):
    def __init__(self, d_m, D=384, heads=6, dh=64):
        super().__init__(); self.D = D; self.h = heads; self.dh = dh
        self.Wq = nn.Linear(D, heads * dh); self.Wqs = nn.Linear(D, heads * dh)
        self.Wk = nn.Linear(d_m, heads * dh); self.Wv = nn.Linear(d_m, heads * dh); self.Wo = nn.Linear(heads * dh, D)
        self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D))
        self.pred = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, d_m))   # prediction from the identity-centered working state
    def collect(self, G, b, s):
        q = (self.Wq(b) + self.Wqs(s)).view(self.h, self.dh)
        K = self.Wk(G).view(-1, self.h, self.dh); V = self.Wv(G).view(-1, self.h, self.dh)
        attn = torch.softmax(torch.einsum('hd,nhd->hn', q, K) / self.dh ** 0.5, dim=-1)
        return self.Wo(torch.einsum('hn,nhd->hd', attn, V).reshape(-1))
    def run(self, m, ground=True):
        b = torch.zeros(self.D); s = torch.zeros(self.D); tau = F.softplus(self.log_tau) + 0.5; ss, pr = [], []
        for t in range(len(m['gen'])):
            a = self.collect(cen(m['gen'][t]), b, s)
            sp = s if ground else torch.zeros(self.D)                            # identity SETPOINT (ablatable) — the base the working state sits on
            for _ in range(2): b = b + (-(b - sp) + torch.tanh(self.W(b - sp) + a)) / tau / 2   # relax toward identity, driven by the offset
            s = 0.9 * s + 0.1 * b
            ss.append(s); pr.append(self.pred(b))
        return torch.stack(ss), torch.stack(pr)
def supcon(B, lab, tau=0.2):
    Bn = F.normalize(B, dim=-1); S = (Bn @ Bn.t() / tau); N = B.shape[0]; eye = torch.eye(N, dtype=torch.bool)
    S = S.masked_fill(eye, -1e9); lse = torch.logsumexp(S, dim=1); pos = (lab[:, None] == lab[None, :]) & ~eye
    L = 0.0; n = 0
    for i in range(N):
        if pos[i].any(): L = L + (lse[i] - S[i][pos[i]].mean()); n += 1
    return L / n
def predloss(pr, z): return (1 - (F.normalize(pr[:-1], dim=-1) * z[1:]).sum(-1)).mean()
torch.manual_seed(0); net = AttractorGrounded(d_m).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
for ep in range(160):
    SS, lab, pl, n = [], [], 0.0, 0
    for mid, m in enumerate(tr):
        ss, pr = net.run(m, ground=True); SS.append(ss); lab += [mid] * ss.shape[0]; pl = pl + predloss(pr, torch.stack(m['z'])); n += 1
    loss = supcon(torch.cat(SS), torch.tensor(lab)) + 0.5 * pl / n
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if ep % 40 == 0: print('ep %d loss %.3f' % (ep, float(loss)), flush=True)
with torch.no_grad():
    SS, lab, coh, coh0 = [], [], [], []
    for mid, m in enumerate(te):
        ss, pr = net.run(m, ground=True); _, pr0 = net.run(m, ground=False)
        SS.append(ss); lab += [mid] * ss.shape[0]; z = torch.stack(m['z'])
        coh.append((F.normalize(pr[:-1], dim=-1) * z[1:]).sum(-1)); coh0.append((F.normalize(pr0[:-1], dim=-1) * z[1:]).sum(-1))
    Bn = F.normalize(torch.cat(SS), dim=-1); lab = torch.tensor(lab); S = Bn @ Bn.t(); eye = torch.eye(len(lab), dtype=torch.bool)
    sep = float(S[(lab[:, None] == lab[None, :]) & ~eye].mean() - S[lab[:, None] != lab[None, :]].mean())
    ch = float(torch.cat(coh).mean()); ch0 = float(torch.cat(coh0).mean())
print('\n=== ATTRACTOR-GROUNDED (identity = setpoint of the working dynamics) ===')
print('  identity-SEP          : %+.3f   (want high — identity individuated & persistent)' % sep)
print('  pred-coh (grounded)   : %+.3f   (want substantial — the self has a function)' % ch)
print('  pred-coh (setpoint=0) : %+.3f' % ch0)
print('  GROUNDING-GAP         : %+.3f   (prediction collapse when identity removed — additive model gave 0.000)' % (ch - ch0))
print('=== ALL_DONE ===')
