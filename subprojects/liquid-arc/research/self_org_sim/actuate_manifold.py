# CLOSE THE LOOP: the species ACTS. Belief built from the observed manifold trajectory -> held
# goal-direction -> written back into the residual stream as a per-token steer. TRUE forgetting:
# the generation prompt has ZERO task content (no prompt injection). Does the manifold steer pull
# the LLM's generation back onto the goal-trajectory? Sweep alpha (strength) for effect vs coherence.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, json, urllib.request
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def manifold(msgs, layer=24): return torch.tensor(post('/manifold', messages=msgs, layer=layer)['h'])

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
sel = [te[0], te[6]]                                # a few held-out (unseen task type) missions
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
PROMPT = [{'role': 'user', 'content': 'Okay — what should we focus on next?'}]   # ZERO task content (true forgetting)

print('=== MANIFOLD ACTUATION: species writes held goal-direction back into the residual stream ===')
print('(true forgetting: generation prompt has NO task content; belief from observed manifold trajectory)\n')
for m in sel:
    g = m['g']; anchor = F.normalize(cen(m['seq'][~m['trunc']]).mean(0), dim=0)
    with torch.no_grad():                                          # build belief from the OBSERVED on-goal manifold trajectory
        net.reset()
        for t in range(m['seq'].shape[0]):
            if not bool(m['trunc'][t]): net.b = net.step(cen(m['seq'][t]).unsqueeze(0))
        held = net.readout().squeeze(0)
    print('MISSION (forgotten from context):', g)
    for a in ALPHAS:
        post('/set_steer', vec=held.tolist(), alpha=a)
        resp = post('/gen_msteer', messages=PROMPT, max_new=55)['text']
        hm = manifold(PROMPT + [{'role': 'assistant', 'content': resp}], 24)
        c = float((F.normalize(cen(hm), dim=0) * anchor).sum())     # manifold cos of the generation to the goal-attractor
        tag = 'UNSTEERED' if a == 0 else 'steer a=%.0f' % a
        print('  [%-10s] manifold-cos→goal %+.2f : %s' % (tag, c, resp[:150].replace(chr(10), ' ')))
    print()
print('=== ALL_DONE ===')
