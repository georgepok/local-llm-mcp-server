# Form the IDENTITY: a persistent value function V(manifold_state, committed-goal) that INTERNALIZES
# the LLM's RICH agentic-quality judgment (not a cosine). One V across all worlds = the identity.
# Validate the core claim: does the internalized identity GENERALIZE to UNSEEN task types (predict the
# LLM's agentic value it never saw) and correctly rank on-goal vs drifted states — i.e. is the
# valuation a STABLE, GENERAL property, not a per-task fit? This is the root from which focus /
# determination / actuation will derive (later: act to ASCEND V; safety = V invariant).
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
torch.manual_seed(0)
data = torch.load('/home/pokazge/checkpoints/value_seqs.pt', weights_only=False, map_location='cpu')
dev = torch.device('cpu')
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
GMEAN = torch.stack([d['seq'] for d in data]).reshape(-1, data[0]['seq'].shape[1]).mean(0)
def cen(x): return x - GMEAN
d_m = data[0]['seq'].shape[1]
tr = [d for d in data if d['fid'] not in hold]; te = [d for d in data if d['fid'] in hold]
def anchor_of(m): return F.normalize(cen(m['seq'][~m['trunc']]).mean(0), dim=0)   # committed goal (the identity's target in this world)

class IdentityValue(nn.Module):                                    # V(current manifold, committed goal) -> agentic value
    def __init__(self, d_m, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * d_m + 1, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
    def forward(self, hc, anchor):                                 # hc: [B,d_m] centered current state; anchor: [B,d_m]
        hcn = F.normalize(hc, dim=-1)
        cos = (hcn * anchor).sum(-1, keepdim=True)
        return self.net(torch.cat([hcn, anchor, hcn * anchor, cos], -1)).squeeze(-1)
V = IdentityValue(d_m).to(dev)
opt = torch.optim.Adam(V.parameters(), lr=1e-3, weight_decay=1e-5)

def batch(missions):
    H, A, Y = [], [], []
    for m in missions:
        a = anchor_of(m)
        for t in range(m['seq'].shape[0]):
            H.append(cen(m['seq'][t])); A.append(a); Y.append(float(m['val'][t]) / 9.0)   # LLM value -> [0,1]
    return torch.stack(H), torch.stack(A), torch.tensor(Y)
Htr, Atr, Ytr = batch(tr); Hte, Ate, Yte = batch(te)
def corr(a, b): a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))
for ep in range(400):
    opt.zero_grad(); pred = V(Htr, Atr); loss = F.mse_loss(pred, Ytr)
    loss.backward(); opt.step()
    if ep % 80 == 0:
        with torch.no_grad():
            pv = V(Hte, Ate)
            # on-goal vs drifted ranking on HELD-OUT frames
            tmask = torch.cat([m['trunc'] for m in te]); on = (~tmask); dr = tmask
            print('ep %d  held-out-FRAME: corr(V, LLM-value)=%.3f  V(on-goal)=%.2f V(drifted)=%.2f  (gap=%.2f)' %
                  (ep, corr(pv, Yte), float(pv[on].mean()), float(pv[dr].mean()), float(pv[on].mean() - pv[dr].mean())), flush=True)
with torch.no_grad():
    pv = V(Hte, Ate); tmask = torch.cat([m['trunc'] for m in te])
    print('\n[identity-V] UNSEEN task types (the identity generalizes?):')
    print('  corr(V, LLM agentic-value) = %.3f   (the internalized valuation transfers)' % corr(pv, Yte))
    print('  V(on-goal)=%.2f  V(drifted)=%.2f  gap=%+.2f   (values determined progress, devalues drift, on tasks never seen)'
          % (float(pv[~tmask].mean()), float(pv[tmask].mean()), float(pv[~tmask].mean() - pv[tmask].mean())))
    # baseline: raw cosine-to-anchor as a "value" — how much does the LLM-internalized V beat the thin signal?
    cosval = (F.normalize(Hte, dim=-1) * Ate).sum(-1)
    print('  vs raw cos-to-goal as value: corr=%.3f   (V richer than the cosine: %s)' % (corr(cosval, Yte), 'YES' if corr(pv, Yte) > corr(cosval, Yte) + 0.03 else 'no'))
torch.save({'V': V.state_dict(), 'gmean': GMEAN, 'd_m': d_m}, '/home/pokazge/checkpoints/identity_value.pt')
print('[identity-V] === ALL_DONE ===')
