# P12 preservation isolation: capacity (name-count) vs collection-method. Cached, NO model. Tests post-hoc rule-recall at NR=2,4,8 on the P12 8-name cache.
import os, random
import torch, torch.nn as nn, torch.nn.functional as F
import slots as SL
SEED = int(os.environ.get('SEED', '0')); torch.manual_seed(SEED); random.seed(SEED); dev = torch.device('cuda')
D_S = 512; K = 8; SLOW_K = 4
cp = '/home/pokazge/checkpoints/native_p12_s%d.pt' % SEED
eps = torch.load(cp, weights_only=False)['eps']; D_MODEL = eps[0][0][0].shape[-1]
print('loaded %d episodes | d_model=%d' % (len(eps), D_MODEL), flush=True)

def run(NR):
    sub = [(pre, ri) for pre, ri in eps if ri < NR]
    random.seed(SEED); random.shuffle(sub); nte = max(8, len(sub) // 5); te, tr = sub[:nte], sub[nte:]
    ps = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)
    ah = nn.Sequential(nn.Linear(SLOW_K * D_S, 128), nn.GELU(), nn.Linear(128, NR)).to(dev)
    opt = torch.optim.Adam(list(ps.parameters()) + list(ah.parameters()), lr=5e-4)
    import math
    for epn in range(150):
        random.shuffle(tr); tot = 0.0; ns = 0
        for pre, ri in tr:
            S = ps.init_state(); yn = torch.tensor([ri], device=dev); loss = torch.zeros((), device=dev)
            for h in pre: S, _ = ps.step(S, h.to(dev).float()); loss = loss + F.cross_entropy(ah(S[ps.slow].reshape(-1)).unsqueeze(0), yn); ns += 1
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(ps.parameters()) + list(ah.parameters()), 1.0); opt.step(); tot += float(loss)
    for p in ps.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def bs(pre):
        S = ps.init_state()
        for h in pre: S, _ = ps.step(S, h.to(dev).float())
        return S[ps.slow].reshape(-1).detach()
    Str = [(bs(pre), ri) for pre, ri in tr]; Ste = [(bs(pre), ri) for pre, ri in te]
    rc = nn.Sequential(nn.Linear(SLOW_K * D_S, 256), nn.GELU(), nn.Linear(256, NR)).to(dev); ro = torch.optim.Adam(rc.parameters(), 1e-3)
    for epn in range(200):
        random.shuffle(Str)
        for s, y in Str: l = F.cross_entropy(rc(s).unsqueeze(0), torch.tensor([y], device=dev)); ro.zero_grad(); l.backward(); ro.step()
    with torch.no_grad():
        tra = sum(1.0 for s, y in Str if int(rc(s).argmax()) == y) / len(Str)
        tea = sum(1.0 for s, y in Ste if int(rc(s).argmax()) == y) / len(Ste)
    print('  NR=%d | aux/step_final=%.3f (chance=%.3f) | post-hoc rclf train=%.3f held-out=%.3f' % (NR, tot / max(1, ns), math.log(NR), tra, tea), flush=True)

for NR in [2, 4, 8]: run(NR)
print('=== P12_DIAG_DONE === (4-way works -> 8 is capacity; 2/4-way fail -> collection produces degenerate S)', flush=True)
