# LIQUID-AS-CONTEXT-COMPRESSOR. The self-feeding trajectory drifts; a sliding window keeps only the most recent chunk
# (z_t) and DROPS the earlier history (which still influences the next chunk, since generation was full-context). The
# Liquid runs attention-on-attention over every chunk's developmental stream, compressing the history into a persistent
# state h. Predict the next chunk's gist z_{t+1} from [recent window z_t  +  compressed dropped-history h_{t-1}].
# Ablate the compression -> gap = does the Liquid keep prediction ON TRACK beyond the window? And does that value GROW
# with position (more history dropped)? That growth is the whole hypothesis: the Liquid holds what the window can't.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, collections
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load(os.environ.get('DRIFT_DATA', '/home/pokazge/checkpoints/objective_drift.pt'), weights_only=False, map_location='cpu')
data = [m for m in obj['data'] if len(m['gen']) >= 8]
d_m = data[0]['gen'][0].shape[1]
MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
def cen(C): return C - MU
for m in data: m['z'] = [F.normalize(cen(c).mean(0), dim=0) for c in m['gen']]
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(2, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
print('d_m=%d trajs train=%d test=%d steps~%d' % (d_m, len(tr), len(te), len(data[0]['gen'])), flush=True)
class Compressor(nn.Module):
    def __init__(self, d_m, D=384, heads=6, dh=64):
        super().__init__(); self.D = D; self.h = heads; self.dh = dh
        self.Wq = nn.Linear(D, heads * dh); self.Wk = nn.Linear(d_m, heads * dh); self.Wv = nn.Linear(d_m, heads * dh); self.Wo = nn.Linear(heads * dh, D)
        self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D))
        self.cz = nn.Linear(d_m, D)                                            # encode the recent in-window gist
        self.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))   # [window, compression] -> next gist
    def collect(self, C, b):
        q = self.Wq(b).view(self.h, self.dh); K = self.Wk(C).view(-1, self.h, self.dh); V = self.Wv(C).view(-1, self.h, self.dh)
        attn = torch.softmax(torch.einsum('hd,nhd->hn', q, K) / self.dh ** 0.5, dim=-1)
        return self.Wo(torch.einsum('hn,nhd->hd', attn, V).reshape(-1))
    def run(self, m, use_comp=True):
        b = torch.zeros(self.D); h = torch.zeros(self.D); tau = F.softplus(self.log_tau) + 0.5; preds = []
        for t in range(len(m['gen'])):
            h_prev = h                                                         # compression of chunks 0..t-1 = the DROPPED history
            a = self.collect(cen(m['gen'][t]), b)
            for _ in range(2): b = b + (-b + torch.tanh(self.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b                                              # running compression 0..t
            hc = h_prev if use_comp else torch.zeros(self.D)
            preds.append(self.pred(torch.cat([self.cz(m['z'][t]), hc])))       # predict z_{t+1} from [window z_t, dropped-history compression]
        return torch.stack(preds)
    def coh(self, m, use_comp=True):
        pr = self.run(m, use_comp)
        if pr.shape[0] < 2: return None
        return (F.normalize(pr[:-1], dim=-1) * torch.stack(m['z'])[1:]).sum(-1)
net = Compressor(d_m).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
for ep in range(160):
    loss, n = 0.0, 0
    for m in tr:
        c = net.coh(m, True)
        if c is not None: loss = loss + (1 - c).mean(); n += 1
    opt.zero_grad(); (loss / n).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if ep % 40 == 0: print('ep %d loss %.3f' % (ep, float(loss / n)), flush=True)
with torch.no_grad():
    C = torch.cat([net.coh(m, True) for m in te]); C0 = torch.cat([net.coh(m, False) for m in te])
    posg = collections.defaultdict(list)
    for m in te:
        cc, c0 = net.coh(m, True), net.coh(m, False)
        for t in range(cc.shape[0]): posg[t].append(float(cc[t] - c0[t]))
print('\n=== LIQUID CONTEXT-COMPRESSOR ===')
print('  next-gist coh WITH compression : %+.3f' % float(C.mean()))
print('  next-gist coh WITHOUT (window) : %+.3f' % float(C0.mean()))
print('  COMPRESSION VALUE (gap)        : %+.3f' % float(C.mean() - C0.mean()))
print('  gap by position (does compression value GROW as more history is dropped?):')
for t in sorted(posg): print('    t=%2d  gap=%+.3f' % (t, sum(posg[t]) / len(posg[t])))
torch.save({'net': net.state_dict()}, '/home/pokazge/checkpoints/context_compressor.pt')
print('=== ALL_DONE ===')
