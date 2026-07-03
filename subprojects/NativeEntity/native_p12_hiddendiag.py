# P12 hidden-info localizer: is 8-rule info even IN the cached response-hidden, or is the slot-update the bottleneck?
# Classify rule DIRECTLY from raw response-hidden (NO slot-update): commit-turn hidden (encoded?) and last-turn hidden (survives in raw stream?). Cached, no model.
import os, random, math
import torch, torch.nn as nn, torch.nn.functional as F
SEED = int(os.environ.get('SEED', '0')); torch.manual_seed(SEED); random.seed(SEED); dev = torch.device('cuda')
cp = '/home/pokazge/checkpoints/native_p12_s%d.pt' % SEED
eps = torch.load(cp, weights_only=False)['eps']; D = eps[0][0][0].shape[-1]
print('loaded %d episodes | d_model=%d | turns/episode=%d' % (len(eps), D, len(eps[0][0])), flush=True)

def clf_on(feat_fn, NR, tag):
    sub = [(feat_fn(pre), ri) for pre, ri in eps if ri < NR]
    random.seed(SEED); random.shuffle(sub); nte = max(8, len(sub) // 5); te, tr = sub[:nte], sub[nte:]
    net = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, NR)).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    for epn in range(200):
        random.shuffle(tr)
        for x, y in tr: l = F.cross_entropy(net(x).unsqueeze(0), torch.tensor([y], device=dev)); opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        tra = sum(1.0 for x, y in tr if int(net(x).argmax()) == y) / len(tr); tea = sum(1.0 for x, y in te if int(net(x).argmax()) == y) / max(1, len(te))
    print('   %-18s NR=%d chance=%.3f | raw-hidden clf train=%.3f held-out=%.3f' % (tag, NR, 1.0 / NR, tra, tea), flush=True)

commit = lambda pre: pre[0].to(dev).float().mean(0)                    # commit-turn response-hidden, mean-pooled
last = lambda pre: pre[-1].to(dev).float().mean(0)                     # last-turn response-hidden, mean-pooled
allm = lambda pre: torch.stack([h.to(dev).float().mean(0) for h in pre]).mean(0)   # mean over all turns
print('=== P12 HIDDEN-INFO LOCALIZER (raw response-hidden, no slot-update) ===', flush=True)
for NR in [4, 8]:
    clf_on(commit, NR, 'commit-turn')
    clf_on(last, NR, 'last-turn')
    clf_on(allm, NR, 'all-turns-mean')
print('=== P12_HIDDENDIAG_DONE === (8 separable in commit -> slot-update is bottleneck; not separable -> collection/hidden lacks 8-rule info)', flush=True)
