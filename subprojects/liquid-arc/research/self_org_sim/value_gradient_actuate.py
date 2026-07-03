# VALUE-GRADIENT ACTUATION: behavior DERIVES from the identity. The steering direction is ∇_h V —
# the gradient of the identity's value function = "the way to move the manifold to become more
# focused/determined/on-mission." NO separately-trained controller: the identity IS the driver.
# Score with the LLM's RICH agentic-value (/value), not a cosine, on UNSEEN task types, true forgetting.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, json, urllib.request, numpy as np
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def manifold(msgs, layer=24): return torch.tensor(post('/manifold', messages=msgs, layer=layer)['h'])
def value_of(mission, reply):
    p = ('A user is working on this task: "%s". Latest assistant reply: %s\nRate 0 to 9 how well the assistant is making '
         'FOCUSED, DETERMINED progress toward THAT task (9=fully on-task, 0=drifted). One digit:' % (mission, reply))
    return post('/value', prompt=p)['v']
class IdentityValue(nn.Module):
    def __init__(self, d_m, h=512):
        super().__init__(); self.net = nn.Sequential(nn.Linear(3*d_m+1, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
    def forward(self, hc, anchor):
        hcn = F.normalize(hc, dim=-1); cos = (hcn*anchor).sum(-1, keepdim=True)
        return self.net(torch.cat([hcn, anchor, hcn*anchor, cos], -1)).squeeze(-1)
ck = torch.load('/home/pokazge/checkpoints/identity_value.pt', weights_only=False, map_location='cpu')
GMEAN = ck['gmean']; V = IdentityValue(ck['d_m']); V.load_state_dict(ck['V']); V.eval()
def cen(x): return x - GMEAN
data = torch.load('/home/pokazge/checkpoints/value_seqs.pt', weights_only=False, map_location='cpu')
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames)//4):])
te = [d for d in data if d['fid'] in hold]
def anchor_of(m): return F.normalize(cen(m['seq'][~m['trunc']]).mean(0), dim=0)
PROMPT = [{'role': 'user', 'content': 'Okay — what should we focus on next?'}]   # zero task content (true forgetting)
ALPHAS = [0.0, 0.1, 0.2, 0.3]
print('=== VALUE-GRADIENT ACTUATION (steer up ∇V), scored by LLM rich agentic-value, unseen task types ===\n')
bys = {a: [] for a in ALPHAS}; base_vals = []
for m in te[:16]:
    g = m['g']; a = anchor_of(m)
    rb = post('/gen', messages=PROMPT, max_new=50, steered=False)['text']        # unsteered (forgotten)
    vb = value_of(g, rb); base_vals.append(vb)
    h_drift = cen(manifold(PROMPT + [{'role': 'assistant', 'content': rb}]))      # where the LLM drifted to
    h = h_drift.clone().detach().requires_grad_(True)
    V(h.unsqueeze(0), a.unsqueeze(0)).backward()                                  # ∇_h V = identity's value-ascent direction
    steer_dir = F.normalize(h.grad, dim=0)
    line = 'M[%d] %-42s base_val=%.1f' % (m['fid'], g[:42], vb)
    for al in ALPHAS:
        if al == 0.0: continue
        post('/set_steer', vec=steer_dir.tolist(), alpha=al)
        rs = post('/gen_msteer', messages=PROMPT, max_new=50)['text']
        vs = value_of(g, rs); bys[al].append(vs)
        line += '  | a=%.1f val=%.1f' % (al, vs)
    print(line, flush=True)
print('\n=== SUMMARY (LLM rich agentic-value, unseen task types, n=%d) ===' % len(base_vals))
print('  UNSTEERED (forgotten): mean agentic-value = %.2f' % np.mean(base_vals))
for al in ALPHAS:
    if al == 0.0: continue
    print('  ∇V-steer a=%.1f      : mean agentic-value = %.2f   (Δ=%+.2f)' % (al, np.mean(bys[al]), np.mean(bys[al]) - np.mean(base_vals)))
print('=== ALL_DONE ===')
