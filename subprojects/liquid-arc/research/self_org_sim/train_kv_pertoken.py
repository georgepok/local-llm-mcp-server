# PER-TOKEN KV compressor — the fair test. Same structure as the +0.104 hidden compressor, but the Liquid does
# attention-on-attention over each chunk's PER-TOKEN K/V (flattened across the 16 full-attn layers -> [n_tok, 8192]),
# the real attention substrate at token granularity, instead of per-token layer-32 hiddens. Compress the dropped
# history, predict the next chunk's gist from [window + compression], ablate -> gap, gap-by-position. Head-to-head vs
# layer-32 hidden (+0.104).
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, collections
torch.manual_seed(0); dev = torch.device('cpu')
obj = torch.load('/home/pokazge/checkpoints/objective_drift_kv.pt', weights_only=False, map_location='cpu')
data = [m for m in obj['data'] if len(m['kv']) >= 8]
n_L, kvd = data[0]['kv'][0].shape[1], data[0]['kv'][0].shape[2]; KVF = n_L * kvd; d_m = data[0]['gist'].shape[1]
for m in data: m['kvf'] = [k.reshape(k.shape[0], -1).float() for k in m['kv']]   # per chunk: [n_tok, 16*512]
allkv = torch.cat([t for m in data for t in m['kvf']], 0); KMU = allkv.mean(0); KSD = allkv.std(0) + 1e-5
for m in data:
    m['kvf'] = [(t - KMU) / KSD for t in m['kvf']]
MU = torch.cat([m['gist'] for m in data], 0).mean(0)
for m in data: m['z'] = F.normalize(m['gist'] - MU, dim=1)
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(2, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
print('trajs train=%d test=%d  per-token KV feat=%d (16 layers x %d)' % (len(tr), len(te), KVF, kvd), flush=True)
class KVPC(nn.Module):
    def __init__(s, KVF, d_m, D=384, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(KVF, heads * dh); s.Wv = nn.Linear(KVF, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))
    def collect(s, kvf, b):                                                      # attention-on-attention over per-token KV
        q = s.Wq(b).view(s.h, s.dh); K = s.Wk(kvf).view(-1, s.h, s.dh); V = s.Wv(kvf).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K) / s.dh ** 0.5, dim=-1)
        return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
    def run(s, m, use_comp=True):
        b = torch.zeros(s.D); h = torch.zeros(s.D); tau = F.softplus(s.log_tau) + 0.5; preds = []
        for t in range(len(m['kvf'])):
            hp = h; a = s.collect(m['kvf'][t], b)
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b
            preds.append(s.pred(torch.cat([s.cz(m['z'][t]), hp if use_comp else torch.zeros(s.D)])))
        return torch.stack(preds)
    def coh(s, m, uc=True):
        pr = s.run(m, uc); return None if pr.shape[0] < 2 else (F.normalize(pr[:-1], dim=-1) * m['z'][1:]).sum(-1)
net = KVPC(KVF, d_m); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
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
print('\n=== PER-TOKEN KV-CACHE COMPRESSOR (attention-on-attention over the real KV substrate) ===')
print('  next-gist coh WITH KV-compression : %+.3f' % float(C.mean()))
print('  next-gist coh WITHOUT (window)    : %+.3f' % float(C0.mean()))
print('  KV-COMPRESSION VALUE (gap)        : %+.3f   (layer-32 hidden baseline +0.104)' % float(C.mean() - C0.mean()))
print('  gap by position:'); [print('    t=%2d  gap=%+.3f' % (t, sum(posg[t]) / len(posg[t]))) for t in sorted(posg)]
torch.save({'net': net.state_dict()}, '/home/pokazge/checkpoints/kv_pertoken_compressor.pt')
print('=== ALL_DONE ===')
