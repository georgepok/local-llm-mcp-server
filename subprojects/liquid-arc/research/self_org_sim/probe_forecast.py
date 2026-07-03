# FORECAST probe — motion's TRUE observable is the FUTURE, not contemporaneous change. Δval[t]=val[t]-val[t-1]
# is backward-looking and collinear with val[t] (the judge scores the current end-state), so position owns it.
# The derivative perspective = where the trajectory is HEADING. Test: from rep[t], forecast val[t+1] (next turn,
# before its content exists), and the FORWARD change Δf=val[t+1]-val[t] (persistence removed by construction).
# If the attention's MOTION anticipates the next value beyond what POSITION already encodes, the perspective is real.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch
torch.manual_seed(0)
obj = torch.load('/home/pokazge/checkpoints/objective_value_attn.pt', weights_only=False, map_location='cpu')
data = obj['data']
def derivs(r):
    d1 = torch.zeros_like(r); d1[1:] = r[1:] - r[:-1]
    d2 = torch.zeros_like(r); d2[1:-1] = r[2:] - 2 * r[1:-1] + r[:-2]
    return torch.cat([r, d1, d2], -1)
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
def build(is_test):
    Xc, Xd, Xr, Vt, Vn = [], [], [], [], []                 # Vt=val[t], Vn=val[t+1]
    for m in data:
        if (m['fid'] in hold) != is_test: continue
        v = m['val'].float()
        for t in range(m['seq'].shape[0] - 1):              # need t+1 to exist
            Xc.append(m['seq'][t]); Xd.append(derivs(m['aseq'][t]).flatten()); Xr.append(m['aseq'][t].flatten())
            Vt.append(v[t]); Vn.append(v[t + 1])
    return torch.stack(Xc), torch.stack(Xd), torch.stack(Xr), torch.tensor(Vt), torch.tensor(Vn)
Xc_tr, Xd_tr, Xr_tr, Vt_tr, Vn_tr = build(False)
Xc_te, Xd_te, Xr_te, Vt_te, Vn_te = build(True)
Fwd_tr, Fwd_te = Vn_tr - Vt_tr, Vn_te - Vt_te               # forward change = where the value is HEADING next
def std(a, b): mu = a.mean(0); sd = a.std(0) + 1e-6; return (a - mu) / sd, (b - mu) / sd
def ridge_pred(Xtr, Xte, ytr, c=0.1):
    Xtr, Xte = std(Xtr, Xte); K = Xtr @ Xtr.t(); n = K.shape[0]; lam = c * K.diag().mean()
    alpha = torch.linalg.solve(K + lam * torch.eye(n), ytr - ytr.mean())
    return Xte @ Xtr.t() @ alpha + ytr.mean()
def cc(p, y): a = p - p.mean(); b = y - y.mean(); return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))
print('forecast samples: train=%d test=%d' % (len(Vn_tr), len(Vn_te)), flush=True)
print('\npersistence  corr(val[t], val[t+1])               : %+.3f   (trivial baseline)' % cc(Vt_te, Vn_te))
print('\n-- forecast NEXT value val[t+1] from rep[t] --')
print('  position[t] -> val[t+1] : %+.3f' % cc(ridge_pred(Xc_tr, Xc_te, Vn_tr), Vn_te))
print('  motion[t]   -> val[t+1] : %+.3f' % cc(ridge_pred(Xd_tr, Xd_te, Vn_tr), Vn_te))
print('\n-- forecast FORWARD-CHANGE Δf=val[t+1]-val[t]  (persistence removed; pure "where is it heading") --')
print('  position[t] -> Δf : %+.3f' % cc(ridge_pred(Xc_tr, Xc_te, Fwd_tr), Fwd_te))
print('  motion[t]   -> Δf : %+.3f' % cc(ridge_pred(Xd_tr, Xd_te, Fwd_tr), Fwd_te))
print('  routing-raw -> Δf : %+.3f' % cc(ridge_pred(Xr_tr, Xr_te, Fwd_tr), Fwd_te))
pf_tr, pf_te = ridge_pred(Xc_tr, Xc_tr, Fwd_tr), ridge_pred(Xc_tr, Xc_te, Fwd_tr)
pd_te = ridge_pred(Xd_tr, Xd_te, Fwd_tr - pf_tr)            # motion on the forward-change position could NOT explain
print('  Δf residual-after-position, by MOTION : %+.3f   (anticipation motion adds beyond position)' % cc(pd_te, Fwd_te - pf_te))
print('=== ALL_DONE ===')
