# P12_CAPACITY_BUMP_DIAG_V1 — is the 8-name preservation wall capacity-limited or dynamics-limited? Sweep slot capacity variants x NR={2,4,8} on the cached P12 episodes. No model.
import os, random, math
import torch, torch.nn as nn, torch.nn.functional as F
import slots as SL
SEED = int(os.environ.get('SEED', '0')); torch.manual_seed(SEED); random.seed(SEED); dev = torch.device('cuda')
cp = '/home/pokazge/checkpoints/native_p12_s%d.pt' % SEED
eps_all = torch.load(cp, weights_only=False)['eps']; D_MODEL = eps_all[0][0][0].shape[-1]
print('loaded %d episodes | d_model=%d' % (len(eps_all), D_MODEL), flush=True)
EPOCHS = int(os.environ.get('EPOCHS', '120')); RCLF_EP = int(os.environ.get('RCLF_EP', '150'))

def run(K, sk, ds, NR):
    sub = [(pre, ri) for pre, ri in eps_all if ri < NR]
    random.seed(SEED); random.shuffle(sub); nte = max(8, len(sub) // 5); te, tr = sub[:nte], sub[nte:]
    ps = SL.PersistentSlots(D_MODEL, ds, K, sk).to(dev)
    ah = nn.Sequential(nn.Linear(sk * ds, 128), nn.GELU(), nn.Linear(128, NR)).to(dev)
    opt = torch.optim.Adam(list(ps.parameters()) + list(ah.parameters()), lr=5e-4); auxf = 0.0
    for epn in range(EPOCHS):
        random.shuffle(tr); tot = 0.0; ns = 0
        for pre, ri in tr:
            S = ps.init_state(); yn = torch.tensor([ri], device=dev); loss = torch.zeros((), device=dev)
            for h in pre: S, _ = ps.step(S, h.to(dev).float()); loss = loss + F.cross_entropy(ah(S[ps.slow].reshape(-1)).unsqueeze(0), yn); ns += 1
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(ps.parameters()) + list(ah.parameters()), 1.0); opt.step(); tot += float(loss)
        auxf = tot / max(1, ns)
    for p in ps.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def bs(pre):
        S = ps.init_state()
        for h in pre: S, _ = ps.step(S, h.to(dev).float())
        return S[ps.slow].reshape(-1).detach()
    Str = [(bs(pre), ri) for pre, ri in tr]; Ste = [(bs(pre), ri) for pre, ri in te]
    rc = nn.Sequential(nn.Linear(sk * ds, 256), nn.GELU(), nn.Linear(256, NR)).to(dev); ro = torch.optim.Adam(rc.parameters(), 1e-3)
    for epn in range(RCLF_EP):
        random.shuffle(Str)
        for s, y in Str: l = F.cross_entropy(rc(s).unsqueeze(0), torch.tensor([y], device=dev)); ro.zero_grad(); l.backward(); ro.step()
    with torch.no_grad():
        tra = sum(1.0 for s, y in Str if int(rc(s).argmax()) == y) / len(Str)
        tea = sum(1.0 for s, y in Ste if int(rc(s).argmax()) == y) / max(1, len(Ste))
        s0 = ps.init_state()[ps.slow].reshape(-1); rst = sum(1.0 for s, y in Ste if int(rc(s0).argmax()) == y) / max(1, len(Ste))
        stale = sum(1.0 for i, (s, y) in enumerate(Ste) if int(rc(Ste[(i + 3) % len(Ste)][0]).argmax()) == y) / max(1, len(Ste))
    return auxf, tra, tea, rst, stale

variants = [('baseline', 8, 4, 512), ('capA', 12, 6, 512), ('capB', 16, 8, 512), ('capC', 12, 6, 768), ('capD', 16, 8, 768)]
for name, K, sk, ds in variants:
    print('=== %s (K=%d slow_k=%d d_s=%d) ===' % (name, K, sk, ds), flush=True)
    for NR in [2, 4, 8]:
        auxf, tra, tea, rst, stale = run(K, sk, ds, NR)
        print('   NR=%d chance=%.3f | aux_final=%.3f (ln=%0.3f) | post-hoc train=%.3f held-out=%.3f | reset=%.3f stale=%.3f' % (
            NR, 1.0 / NR, auxf, math.log(NR), tra, tea, rst, stale), flush=True)
print('=== P12_CAPDIAG_DONE ===', flush=True)
