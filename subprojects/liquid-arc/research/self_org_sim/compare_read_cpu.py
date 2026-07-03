# CPU-only READ comparison (runs concurrently with the GPU generation, on the incrementally-saved data). Trains the SAME
# Liquid+AoA compressor on two read channels — (A) layer-32 stream, (B) broad native KV (all full-attn layers) — and
# compares HELD-OUT next-gist coh on a proper cross-category split. This is the perception question the n=2 eval couldn't
# answer; no 27B needed. Reports train+test coh (overfit check) and n. Re-run as generation accumulates more trajectories.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
torch.manual_seed(0)
data = [m for m in torch.load('/home/pokazge/checkpoints/objective_drift60.pt', weights_only=False, map_location='cpu')['data'] if len(m['texts']) >= 10]
d_m = data[0]['gen'][0].shape[1]; nkv_dim = data[0]['nkv'][0].shape[1]
MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0); MUk = torch.cat([c for m in data for c in m['nkv']], 0).mean(0)
for m in data: m['z'] = [F.normalize((c - MU).mean(0), dim=0) for c in m['gen']]
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
print('READ compare (CPU) | trajs=%d (%d fids) train=%d test=%d (held-out fids=%s)' % (len(data), len(fids), len(tr), len(te), sorted(hold)), flush=True)
class Comp(nn.Module):
    def __init__(s, in_dim, mu, D=384, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh; s.register_buffer('mu', mu)
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(in_dim, heads * dh); s.Wv = nn.Linear(in_dim, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); K_ = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K_) / s.dh ** 0.5, dim=-1); return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
    def run(s, m, key):
        b = torch.zeros(s.D); h = torch.zeros(s.D); tau = F.softplus(s.log_tau) + 0.5; pr = []
        for t in range(len(m[key])):
            hp = h; a = s.collect(m[key][t] - s.mu, b)
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b; pr.append(s.pred(torch.cat([s.cz(m['z'][t]), hp])))
        return torch.stack(pr)
def coh(P, z): return float((F.normalize(P[:-1], dim=-1) * z[1:]).sum(-1).mean())
def winonly(m):                                                                  # ablate held memory: predict next gist from window z_t alone (chance-ish baseline)
    return st.mean([coh(torch.stack([m['z'][t] for t in range(len(m['z']))]), torch.stack(m['z'])) for _ in [0]])
def run_channel(key, in_dim, mu, label):
    comp = Comp(in_dim, mu); opt = torch.optim.Adam(comp.parameters(), lr=1e-3, weight_decay=1e-5)
    for ep in range(180):
        loss = 0.0
        for m in tr: pr = comp.run(m, key); z = torch.stack(m['z']); loss = loss + (1 - (F.normalize(pr[:-1], dim=-1) * z[1:]).sum(-1)).mean()
        opt.zero_grad(); (loss / len(tr)).backward(); opt.step()
    with torch.no_grad():
        trc = st.mean([coh(comp.run(m, key), torch.stack(m['z'])) for m in tr]); tec = st.mean([coh(comp.run(m, key), torch.stack(m['z'])) for m in te])
    print('  %-16s read dim=%5d | train coh %.3f | HELD-OUT coh %.3f (n=%d trajs)' % (label, in_dim, trc, tec, len(te)), flush=True)
    return tec
print('held-out next-gist coh (higher = the read carries more dropped-context signal):', flush=True)
a = run_channel('gen', d_m, MU, 'LAYER-32 tap')
b = run_channel('nkv', nkv_dim, MUk, 'NATIVE broad-KV')
print('  => NATIVE %s LAYER-32 by %+.3f on held-out (n=%d). %s' % ('beats' if b > a else 'trails', b - a, len(te), 'native read is at least as good' if b >= a - 0.01 else 'native read costs signal'), flush=True)
print('=== ALL_DONE ===', flush=True)
