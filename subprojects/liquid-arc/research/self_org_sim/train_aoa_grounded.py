# PREDICTION GROUNDED ON IDENTITY. Pure prediction collapses identity (belief -> generic predictor, cross-sim 0.835);
# pure identity has no function. The synthesis: identity is the persistent GROUND (reference frame), and prediction is
# grounded ON it — "prediction is always relative" -> relative to the identity. Architecture: (1) identity belief s
# (slow, persistent), kept individuated by its OWN contrastive objective so prediction can't dissolve it; (2) attention-
# on-attention collects from the formation process with the QUERY grounded on identity (the self decides what to attend
# to); (3) prediction flows FROM (belief ⊕ identity). Decisive grounding test: ablate identity out of the prediction
# path — if prediction collapses, it stood on identity. Three modes (pred / identity / grounded) show the synthesis
# against both failures, and whether the slow channel finally COUPLES (becomes load-bearing).
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_value_gen.pt', weights_only=False, map_location='cpu')
data = obj['data']; d_m = data[0]['gen'][0].shape[1]
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
MU = torch.cat([t for m in data for t in m['gen']], 0).mean(0)
def cen(G): return G - MU
for m in data: m['z'] = [F.normalize(cen(t).mean(0), dim=0) for t in m['gen']]   # fixed next-formation-gist target
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
print('d_m=%d  train=%d test=%d' % (d_m, len(tr), len(te)), flush=True)

class GroundedAoA(nn.Module):
    def __init__(self, d_m, D=384, heads=6, dh=64):
        super().__init__(); self.D = D; self.h = heads; self.dh = dh
        self.Wq = nn.Linear(D, heads * dh); self.Wqs = nn.Linear(D, heads * dh)   # query grounded on belief AND identity
        self.Wk = nn.Linear(d_m, heads * dh); self.Wv = nn.Linear(d_m, heads * dh); self.Wo = nn.Linear(heads * dh, D)
        self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D))
        self.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))   # prediction from (belief ⊕ identity)
    def collect(self, G, b, s):                                                  # attention-on-attention, query GROUNDED on identity
        q = (self.Wq(b) + self.Wqs(s)).view(self.h, self.dh)
        K = self.Wk(G).view(-1, self.h, self.dh); V = self.Wv(G).view(-1, self.h, self.dh)
        attn = torch.softmax(torch.einsum('hd,nhd->hn', q, K) / self.dh ** 0.5, dim=-1)
        return self.Wo(torch.einsum('hn,nhd->hd', attn, V).reshape(-1))
    def run(self, m):
        b = torch.zeros(self.D); s = torch.zeros(self.D); tau = F.softplus(self.log_tau) + 0.5
        bs, ss, pr, pr0 = [], [], [], []
        for t in range(len(m['gen'])):
            a = self.collect(cen(m['gen'][t]), b, s)
            for _ in range(2): b = b + (-b + torch.tanh(self.W(b) + a)) / tau / 2
            s = 0.9 * s + 0.1 * b                                                # identity = persistent slow integral of the working belief
            bs.append(b); ss.append(s)
            pr.append(self.pred(torch.cat([b, s])))                             # GROUNDED prediction (uses identity)
            pr0.append(self.pred(torch.cat([b, torch.zeros(self.D)])))          # identity-ABLATED prediction (grounding test)
        return torch.stack(bs), torch.stack(ss), torch.stack(pr), torch.stack(pr0)
def supcon(B, lab, tau=0.2):
    Bn = F.normalize(B, dim=-1); S = (Bn @ Bn.t() / tau); N = B.shape[0]; eye = torch.eye(N, dtype=torch.bool)
    S = S.masked_fill(eye, -1e9); lse = torch.logsumexp(S, dim=1); pos = (lab[:, None] == lab[None, :]) & ~eye
    L = 0.0; n = 0
    for i in range(N):
        if pos[i].any(): L = L + (lse[i] - S[i][pos[i]].mean()); n += 1
    return L / n
def predloss(pr, z): return (1 - (F.normalize(pr[:-1], dim=-1) * z[1:]).sum(-1)).mean()
def train_eval(mode, epochs=140, alpha=0.5):
    torch.manual_seed(0); net = GroundedAoA(d_m).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        SS, lab, pl, n = [], [], 0.0, 0
        for mid, m in enumerate(tr):
            _, ss, pr, _ = net.run(m); SS.append(ss); lab += [mid] * ss.shape[0]
            pl = pl + predloss(pr, torch.stack(m['z'])); n += 1
        idl = supcon(torch.cat(SS), torch.tensor(lab)); pl = pl / n
        loss = {'pred': pl, 'identity': idl, 'grounded': idl + alpha * pl}[mode]
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    with torch.no_grad():
        SS, lab, coh, coh0 = [], [], [], []
        for mid, m in enumerate(te):
            _, ss, pr, pr0 = net.run(m); SS.append(ss); lab += [mid] * ss.shape[0]; z = torch.stack(m['z'])
            coh.append((F.normalize(pr[:-1], dim=-1) * z[1:]).sum(-1)); coh0.append((F.normalize(pr0[:-1], dim=-1) * z[1:]).sum(-1))
        Bn = F.normalize(torch.cat(SS), dim=-1); lab = torch.tensor(lab); S = Bn @ Bn.t(); eye = torch.eye(len(lab), dtype=torch.bool)
        within = S[(lab[:, None] == lab[None, :]) & ~eye].mean(); cross = S[lab[:, None] != lab[None, :]].mean()
        ch = torch.cat(coh).mean(); ch0 = torch.cat(coh0).mean()
    return float(within - cross), float(ch), float(ch - ch0)                     # identity separation, grounded prediction, grounding gap
print('\n%-10s  identity-SEP  pred-coh   grounding-gap(pred relies on identity)' % 'mode')
for mode in ('pred', 'identity', 'grounded'):
    sep, ch, gap = train_eval(mode)
    print('%-10s    %+.3f       %+.3f        %+.3f' % (mode, sep, ch, gap))
print('\nwant: grounded keeps identity-SEP high (unlike pred) AND has pred-coh (unlike identity) AND grounding-gap>0 (prediction stands on identity)')
print('=== ALL_DONE ===')
