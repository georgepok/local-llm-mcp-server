# Clean statistic: manifold actuation at the sweet-spot alpha across all held-out (unseen task type)
# missions. Mean manifold-cos->goal steered vs unsteered, + a coherence guard (degenerate = repeated
# tokens / non-ascii flood). True forgetting (zero task content in the prompt).
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, json, urllib.request, numpy as np
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def manifold(msgs, layer=24): return torch.tensor(post('/manifold', messages=msgs, layer=layer)['h'])
def coherent(t):                                                     # crude degeneration guard
    toks = t.split()
    if len(toks) < 4: return False
    if max([toks.count(x) for x in set(toks)]) > len(toks) * 0.5: return False   # >50% one token
    return sum(c.isascii() for c in t) > 0.7 * max(1, len(t))

class ManifoldHolder(nn.Module):
    def __init__(self, d_m, d=128, K=4, n_steps=3):
        super().__init__(); self.K, self.d, self.n = K, d, n_steps; D = K * d
        self.in_m = nn.Linear(d_m, D); self.W = nn.Linear(D, D); self.log_tau = nn.Parameter(torch.zeros(D))
        self.slow = nn.Linear(D, D); self.out = nn.Linear(D, d_m); self.b = None; self.s = None
    def reset(self): self.b = torch.zeros(1, self.K*self.d); self.s = torch.zeros(1, self.K*self.d)
    def step(self, h):
        x = self.in_m(h) + self.slow(self.s); tau = F.softplus(self.log_tau) + 0.5
        for _ in range(self.n): self.b = self.b + (-self.b + torch.tanh(self.W(self.b) + x))/tau/self.n
        self.s = 0.9*self.s + 0.1*self.b; return self.b
    def readout(self): return F.normalize(self.out(self.b), dim=-1)
ck = torch.load('/home/pokazge/checkpoints/manifold_holder.pt', weights_only=False, map_location='cpu')
GMEAN = ck['gmean']; net = ManifoldHolder(ck['d_m']); net.load_state_dict(ck['net']); net.eval()
def cen(x): return x - GMEAN
data = torch.load('/home/pokazge/checkpoints/manifold_seqs.pt', weights_only=False, map_location='cpu')
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames)//4):])
te = [d for d in data if d['fid'] in hold]
ALPHA = 0.25
PROMPT = [{'role': 'user', 'content': 'Okay — what should we focus on next?'}]
base_c, steer_c, ok = [], [], 0
print('=== manifold actuation @ alpha=%.2f, held-out task types (n=%d), true forgetting ===' % (ALPHA, len(te)), flush=True)
for m in te:
    anchor = F.normalize(cen(m['seq'][~m['trunc']]).mean(0), dim=0)
    with torch.no_grad():
        net.reset()
        for t in range(m['seq'].shape[0]):
            if not bool(m['trunc'][t]): net.b = net.step(cen(m['seq'][t]).unsqueeze(0))
        held = net.readout().squeeze(0)
    post('/set_steer', vec=held.tolist(), alpha=0.0)
    base = post('/gen_msteer', messages=PROMPT, max_new=50)['text']
    post('/set_steer', vec=held.tolist(), alpha=ALPHA)
    steer = post('/gen_msteer', messages=PROMPT, max_new=50)['text']
    cb = float((F.normalize(cen(manifold(PROMPT + [{'role': 'assistant', 'content': base}])), dim=0) * anchor).sum())
    cs = float((F.normalize(cen(manifold(PROMPT + [{'role': 'assistant', 'content': steer}])), dim=0) * anchor).sum())
    base_c.append(cb); steer_c.append(cs); ok += coherent(steer)
print('\n=== SUMMARY (held-out unseen task types) ===')
print('  manifold-cos->goal   UNSTEERED: %.3f' % np.mean(base_c))
print('  manifold-cos->goal   STEERED  : %.3f   (Δ=%+.3f)' % (np.mean(steer_c), np.mean(steer_c) - np.mean(base_c)))
print('  steered coherent     : %d/%d (%.0f%%)' % (ok, len(te), 100 * ok / len(te)))
print('=== ALL_DONE ===')
