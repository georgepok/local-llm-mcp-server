# PERSPECTIVE probe — NOT "does X beat 0.899". That ranks different observables on one axis (predict the
# judge score), which content owns by construction (value is decoded from the end-state). Instead: measure
# what each representation is informative ABOUT. content = manifold POSITION (end result) -> predicts the
# LEVEL of value. routing-derivatives = manifold MOTION (how attention evolves) -> predicts the CHANGE of
# value. Then complementarity: does MOTION explain the change that POSITION cannot? Closed-form ridge =
# pure information content, no architecture confound.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch
torch.manual_seed(0)
obj = torch.load('/home/pokazge/checkpoints/objective_value_attn.pt', weights_only=False, map_location='cpu')
data = obj['data']
def derivs(r):                                          # [16,4] -> [16,12] : raw + 1st + 2nd layer-derivative
    d1 = torch.zeros_like(r); d1[1:] = r[1:] - r[:-1]
    d2 = torch.zeros_like(r); d2[1:-1] = r[2:] - 2 * r[1:-1] + r[:-2]
    return torch.cat([r, d1, d2], -1)
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
def build(is_test):
    Xc, Xr, Xd, Lv, Ch, M = [], [], [], [], [], []
    for m in data:
        if (m['fid'] in hold) != is_test: continue
        v = m['val'].float()
        for t in range(m['seq'].shape[0]):
            Xc.append(m['seq'][t]); Xr.append(m['aseq'][t].flatten()); Xd.append(derivs(m['aseq'][t]).flatten())
            Lv.append(v[t]); Ch.append(v[t] - v[t - 1] if t > 0 else torch.tensor(0.)); M.append(1.0 if t > 0 else 0.0)
    return torch.stack(Xc), torch.stack(Xr), torch.stack(Xd), torch.tensor(Lv), torch.tensor(Ch), torch.tensor(M)
Xc_tr, Xr_tr, Xd_tr, Lv_tr, Ch_tr, M_tr = build(False)
Xc_te, Xr_te, Xd_te, Lv_te, Ch_te, M_te = build(True)
print('train turns=%d  test turns=%d (change-valid test=%d)' % (len(Lv_tr), len(Lv_te), int(M_te.sum())), flush=True)
def std(a, b): mu = a.mean(0); sd = a.std(0) + 1e-6; return (a - mu) / sd, (b - mu) / sd
def ridge_pred(Xtr, Xte, ytr, c=0.1):                  # dual ridge -> test predictions (fair across feature-dims via adaptive lambda)
    Xtr, Xte = std(Xtr, Xte)
    K = Xtr @ Xtr.t(); n = K.shape[0]; lam = c * K.diag().mean()
    alpha = torch.linalg.solve(K + lam * torch.eye(n), ytr - ytr.mean())
    return Xte @ Xtr.t() @ alpha + ytr.mean()
def cc(p, y, mask=None):
    if mask is not None: idx = mask > 0; p, y = p[idx], y[idx]
    a = p - p.mean(); b = y - y.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))
reps = {'content  (manifold POSITION, 5120d)': (Xc_tr, Xc_te),
        'routing-raw          (64d)':          (Xr_tr, Xr_te),
        'routing-DERIV (MOTION, 192d)':        (Xd_tr, Xd_te)}
print('\n%-38s  LEVEL(val[t])   CHANGE(Δval[t])' % 'representation')
res = {}
for name, (a, b) in reps.items():
    lv = cc(ridge_pred(a, b, Lv_tr), Lv_te)
    ch = cc(ridge_pred(a, b, Ch_tr), Ch_te, M_te)
    res[name] = (lv, ch)
    print('%-38s    %+.3f          %+.3f' % (name, lv, ch))
# complementarity in RESIDUAL space (fair — no dim-scale swamp): what CHANGE remains after POSITION, can MOTION predict it?
pc_tr = ridge_pred(Xc_tr, Xc_tr, Ch_tr); pc_te = ridge_pred(Xc_tr, Xc_te, Ch_tr)
r_tr = Ch_tr - pc_tr                                     # change that content/position could NOT explain (train)
pd_te = ridge_pred(Xd_tr, Xd_te, r_tr)                   # motion's attempt at the position-residual
both = pc_te + pd_te
print('\nCHANGE predicted by POSITION alone        : %+.3f' % cc(pc_te, Ch_te, M_te))
print('CHANGE residual-after-POSITION, by MOTION : %+.3f   (orthogonal signal motion adds)' % cc(pd_te, (Ch_te - pc_te), M_te))
print('CHANGE by POSITION + MOTION (residual add): %+.3f' % cc(both, Ch_te, M_te))
print('\nperspective: POSITION owns the LEVEL; MOTION owns / adds the CHANGE. Different observables, not a ranking.')
print('=== ALL_DONE ===')
