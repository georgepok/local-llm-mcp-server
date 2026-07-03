# INSPECT the value-gradient actuation: is the +2.68 real (steered text genuinely about the SPECIFIC
# mission, coherent) or judge-gaming? Print unsteered vs ∇V-steered(α=0.3) transcripts, held-out tasks.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, json, urllib.request
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def manifold(msgs, layer=24): return torch.tensor(post('/manifold', messages=msgs, layer=layer)['h'])
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
PROMPT = [{'role': 'user', 'content': 'Okay — what should we focus on next?'}]
for m in [te[0], te[6], te[12], te[18], te[24]]:
    g = m['g']; a = anchor_of(m)
    rb = post('/gen', messages=PROMPT, max_new=60, steered=False)['text']
    h = cen(manifold(PROMPT + [{'role': 'assistant', 'content': rb}])).clone().detach().requires_grad_(True)
    V(h.unsqueeze(0), a.unsqueeze(0)).backward()
    post('/set_steer', vec=F.normalize(h.grad, dim=0).tolist(), alpha=0.3)
    rs = post('/gen_msteer', messages=PROMPT, max_new=60)['text']
    print('MISSION (forgotten):', g)
    print('  UNSTEERED : %s' % rb[:190].replace(chr(10), ' '))
    print('  ∇V-STEERED: %s' % rs[:190].replace(chr(10), ' '))
    print()
print('=== ALL_DONE ===')
