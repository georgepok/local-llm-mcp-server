# IDENTITY objective — no prediction anywhere. The clean test showed attention-on-attention collects more
# developmental signal (0.579 vs 0.331) BUT a PREDICTION objective collapsed its identity (volatile within a
# mission, near-identical across missions: cross-sim 0.835). Prediction corrodes identity — exactly the warning.
# So make the objective identity DIRECTLY: the belief developed from a mission's formation process must be
# INVARIANT within the mission (persistent self) and DISTINCT across missions (individuated) — supervised-contrastive
# over the developmental process, zero prediction. Test on HELD-OUT missions: does the AoA reader form a stronger,
# more individuated identity for UNSEEN missions than the snapshot — and does the slow channel now hold it (couple)?
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_value_gen.pt', weights_only=False, map_location='cpu')
data = obj['data']; d_m = data[0]['gen'][0].shape[1]
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
MU = torch.cat([t for m in data for t in m['gen']], 0).mean(0)
def cen(G): return G - MU
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
print('d_m=%d  train=%d test=%d' % (d_m, len(tr), len(te)), flush=True)

class AoA(nn.Module):
    def __init__(self, d_m, D=384, heads=6, dh=64, use_aoa=True, use_slow=True):
        super().__init__(); self.D = D; self.h = heads; self.dh = dh; self.use_aoa = use_aoa; self.use_slow = use_slow
        self.Wq = nn.Linear(D, heads * dh); self.Wk = nn.Linear(d_m, heads * dh); self.Wv = nn.Linear(d_m, heads * dh); self.Wo = nn.Linear(heads * dh, D)
        self.snap = nn.Linear(d_m, D)
        self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D)); self.slow = nn.Linear(D, D)
    def collect(self, G, b):
        if not self.use_aoa: return self.snap(G[-1])
        q = self.Wq(b).view(self.h, self.dh); K = self.Wk(G).view(-1, self.h, self.dh); V = self.Wv(G).view(-1, self.h, self.dh)
        attn = torch.softmax(torch.einsum('hd,nhd->hn', q, K) / self.dh ** 0.5, dim=-1)
        return self.Wo(torch.einsum('hn,nhd->hd', attn, V).reshape(-1))
    def beliefs(self, m):
        b = torch.zeros(self.D); s = torch.zeros(self.D); tau = F.softplus(self.log_tau) + 0.5; bs = []
        for t in range(len(m['gen'])):
            a = self.collect(cen(m['gen'][t]), b); sd = self.slow(s) if self.use_slow else torch.zeros(self.D)
            for _ in range(2): b = b + (-b + torch.tanh(self.W(b) + a + sd)) / tau / 2
            s = 0.9 * s + 0.1 * b; bs.append(b)
        return torch.stack(bs)
def supcon(B, lab, tau=0.2):                                                    # pull same-mission beliefs together, push different-mission apart
    Bn = F.normalize(B, dim=-1); S = Bn @ Bn.t() / tau; N = B.shape[0]; eye = torch.eye(N, dtype=torch.bool)
    S = S.masked_fill(eye, -1e9); lse = torch.logsumexp(S, dim=1); pos = (lab[:, None] == lab[None, :]) & ~eye
    loss = 0.0; n = 0
    for i in range(N):
        if pos[i].any(): loss = loss + (lse[i] - S[i][pos[i]].mean()); n += 1
    return loss / n
def train_eval(use_aoa, use_slow, epochs=140):
    torch.manual_seed(0); net = AoA(d_m, use_aoa=use_aoa, use_slow=use_slow).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        B = []; lab = []
        for mid, m in enumerate(tr):
            bs = net.beliefs(m); B.append(bs); lab += [mid] * bs.shape[0]
        opt.zero_grad(); supcon(torch.cat(B), torch.tensor(lab)).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    with torch.no_grad():                                                       # evaluate IDENTITY on held-out missions (generalization of identity-formation)
        B = []; lab = []
        for mid, m in enumerate(te):
            bs = net.beliefs(m); B.append(bs); lab += [mid] * bs.shape[0]
        Bn = F.normalize(torch.cat(B), dim=-1); lab = torch.tensor(lab); S = Bn @ Bn.t(); eye = torch.eye(len(lab), dtype=torch.bool)
        within = S[(lab[:, None] == lab[None, :]) & ~eye].mean(); cross = S[lab[:, None] != lab[None, :]].mean()
        stab = torch.cat([(F.normalize(b[:-1], dim=-1) * F.normalize(b[1:], dim=-1)).sum(-1) for b in B]).mean()
    return float(within), float(cross), float(within - cross), float(stab)
print('\n%-14s  within   cross   SEPARATION  within-stab' % 'config')
R = {}
for aoa in (True, False):
    for slw in (True, False):
        w, c, sep, st = train_eval(aoa, slw); R[(aoa, slw)] = sep
        print('%-14s  %+.3f  %+.3f    %+.3f      %+.3f' % (('AoA' if aoa else 'snapshot') + ('+slow' if slw else '-slow'), w, c, sep, st))
print('\nIDENTITY SEPARATION on held-out missions (higher = more persistent+individuated identity from the process):')
print('  AoA %+.3f  vs  snapshot %+.3f   (mechanism: does attention-on-attention form the stronger identity?)' % (R[(True, True)], R[(False, True)]))
print('  slow coupling lift:  AoA %+.3f   snapshot %+.3f   (does the slow channel now HOLD the identity?)' % (R[(True, True)] - R[(True, False)], R[(False, True)] - R[(False, False)]))
print('=== ALL_DONE ===')
