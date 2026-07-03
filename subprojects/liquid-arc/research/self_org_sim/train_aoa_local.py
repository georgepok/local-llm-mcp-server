# Grounding kept failing because the working belief ACCUMULATES across turns -> it independently re-derives whatever
# the identity holds -> identity is redundant FOR PREDICTION -> gradient ignores it (gap ~0). The fix is real timescale
# SEPARATION: the working state b is LOCAL (reset every turn, integrates only the current turn's formation); the
# identity s is the SOLE persistent carrier across turns (EMA of the local beliefs, individuated by its own objective).
# Now any cross-turn anticipation MUST route through identity — there is no other persistent path — so prediction is
# forced to STAND ON identity. Decisive test: ablate s from the prediction path -> next-turn prediction must collapse
# to within-turn-only -> grounding-gap > 0 at last.
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

class LocalGrounded(nn.Module):
    def __init__(self, d_m, D=384, heads=6, dh=64):
        super().__init__(); self.D = D; self.h = heads; self.dh = dh
        self.Wq = nn.Linear(D, heads * dh); self.Wk = nn.Linear(d_m, heads * dh); self.Wv = nn.Linear(d_m, heads * dh); self.Wo = nn.Linear(heads * dh, D)
        self.b0 = nn.Parameter(torch.zeros(D))                                    # learned per-turn reset state (local working start)
        self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D))
        self.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))
    def collect(self, G, b):
        q = self.Wq(b).view(self.h, self.dh); K = self.Wk(G).view(-1, self.h, self.dh); V = self.Wv(G).view(-1, self.h, self.dh)
        attn = torch.softmax(torch.einsum('hd,nhd->hn', q, K) / self.dh ** 0.5, dim=-1)
        return self.Wo(torch.einsum('hn,nhd->hd', attn, V).reshape(-1))
    def run(self, m, ground=True):
        s = torch.zeros(self.D); tau = F.softplus(self.log_tau) + 0.5; ss, pr = [], []
        for t in range(len(m['gen'])):
            b = self.b0                                                          # LOCAL: reset every turn (no cross-turn accumulation in b)
            a = self.collect(cen(m['gen'][t]), b)
            for _ in range(2): b = b + (-b + torch.tanh(self.W(b) + a)) / tau / 2  # integrate ONLY this turn's formation
            s = 0.9 * s + 0.1 * b                                                # identity = the SOLE persistent cross-turn carrier
            sg = s if ground else torch.zeros(self.D)
            ss.append(s); pr.append(self.pred(torch.cat([b, sg])))              # prediction = local working state grounded on persistent identity
        return torch.stack(ss), torch.stack(pr)
def supcon(B, lab, tau=0.2):
    Bn = F.normalize(B, dim=-1); S = (Bn @ Bn.t() / tau); N = B.shape[0]; eye = torch.eye(N, dtype=torch.bool)
    S = S.masked_fill(eye, -1e9); lse = torch.logsumexp(S, dim=1); pos = (lab[:, None] == lab[None, :]) & ~eye
    L = 0.0; n = 0
    for i in range(N):
        if pos[i].any(): L = L + (lse[i] - S[i][pos[i]].mean()); n += 1
    return L / n
def predloss(pr, z): return (1 - (F.normalize(pr[:-1], dim=-1) * z[1:]).sum(-1)).mean()
torch.manual_seed(0); net = LocalGrounded(d_m).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
for ep in range(160):
    SS, lab, pl, n = [], [], 0.0, 0
    for mid, m in enumerate(tr):
        ss, pr = net.run(m, True); SS.append(ss); lab += [mid] * ss.shape[0]; pl = pl + predloss(pr, torch.stack(m['z'])); n += 1
    loss = supcon(torch.cat(SS), torch.tensor(lab)) + 0.5 * pl / n
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if ep % 40 == 0: print('ep %d loss %.3f' % (ep, float(loss)), flush=True)
with torch.no_grad():
    SS, lab, coh, coh0 = [], [], [], []
    for mid, m in enumerate(te):
        ss, pr = net.run(m, True); _, pr0 = net.run(m, False); SS.append(ss); lab += [mid] * ss.shape[0]; z = torch.stack(m['z'])
        coh.append((F.normalize(pr[:-1], dim=-1) * z[1:]).sum(-1)); coh0.append((F.normalize(pr0[:-1], dim=-1) * z[1:]).sum(-1))
    Bn = F.normalize(torch.cat(SS), dim=-1); lab = torch.tensor(lab); S = Bn @ Bn.t(); eye = torch.eye(len(lab), dtype=torch.bool)
    sep = float(S[(lab[:, None] == lab[None, :]) & ~eye].mean() - S[lab[:, None] != lab[None, :]].mean())
    ch = float(torch.cat(coh).mean()); ch0 = float(torch.cat(coh0).mean())
print('\n=== LOCAL working state + identity as SOLE persistent carrier ===')
print('  identity-SEP          : %+.3f' % sep)
print('  pred-coh (grounded)   : %+.3f' % ch)
print('  pred-coh (identity=0) : %+.3f' % ch0)
print('  GROUNDING-GAP         : %+.3f   (prior architectures: additive 0.000, attractor -0.022)' % (ch - ch0))
print('=== ALL_DONE ===')
