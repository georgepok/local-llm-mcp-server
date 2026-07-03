# The derivative principle on the RICH signal. Routing-derivatives lost because routing (entropy/recency/peak/
# recent-mass) is a LOSSY sketch of attention; the hidden POSITION already integrated it. So apply the user's
# 1st/2nd-derivative idea to the POSITION itself: manifold VELOCITY = content[t]-content[t-1], ACCELERATION =
# content[t]-2content[t-1]+content[t-2]. Does the position's OWN motion add anticipation (forecast forward-change
# Δf=val[t+1]-val[t]) beyond the position snapshot? This is the fair test of "dynamics matter" on the signal that
# actually carries information.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch
torch.manual_seed(0)
obj = torch.load('/home/pokazge/checkpoints/objective_value_attn.pt', weights_only=False, map_location='cpu')
data = obj['data']
frames = sorted(set(d['fid'] for d in data)); hold = set(frames[-max(3, len(frames) // 4):])
def build(is_test):
    P, V, A, Vt, Vn = [], [], [], [], []
    for m in data:
        if (m['fid'] in hold) != is_test: continue
        s = m['seq']; v = m['val'].float()
        for t in range(s.shape[0] - 1):
            vel = (s[t] - s[t - 1]) if t >= 1 else torch.zeros_like(s[t])               # manifold VELOCITY (1st deriv of position)
            acc = (s[t] - 2 * s[t - 1] + s[t - 2]) if t >= 2 else torch.zeros_like(s[t]) # manifold ACCELERATION (2nd deriv)
            P.append(s[t]); V.append(vel); A.append(acc); Vt.append(v[t]); Vn.append(v[t + 1])
    return torch.stack(P), torch.stack(V), torch.stack(A), torch.tensor(Vt), torch.tensor(Vn)
P_tr, V_tr, A_tr, Vt_tr, Vn_tr = build(False)
P_te, V_te, A_te, Vt_te, Vn_te = build(True)
Fwd_tr, Fwd_te = Vn_tr - Vt_tr, Vn_te - Vt_te
def std(a, b): mu = a.mean(0); sd = a.std(0) + 1e-6; return (a - mu) / sd, (b - mu) / sd
def ridge_pred(Xtr, Xte, ytr, c=0.1):
    Xtr, Xte = std(Xtr, Xte); K = Xtr @ Xtr.t(); n = K.shape[0]; lam = c * K.diag().mean()
    alpha = torch.linalg.solve(K + lam * torch.eye(n), ytr - ytr.mean()); return Xte @ Xtr.t() @ alpha + ytr.mean()
def cc(p, y): a = p - p.mean(); b = y - y.mean(); return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))
print('samples train=%d test=%d' % (len(Vn_tr), len(Vn_te)), flush=True)
print('\n-- derivative principle on the RICH position: forecast forward-change Δf=val[t+1]-val[t] --')
print('  position only            : %+.3f' % cc(ridge_pred(P_tr, P_te, Fwd_tr), Fwd_te))
print('  velocity only (Δposition): %+.3f' % cc(ridge_pred(V_tr, V_te, Fwd_tr), Fwd_te))
pf_tr, pf_te = ridge_pred(P_tr, P_tr, Fwd_tr), ridge_pred(P_tr, P_te, Fwd_tr)            # forward-change position CAN explain
rv = ridge_pred(V_tr, V_te, Fwd_tr - pf_tr)                                              # velocity on what position could NOT
rva = ridge_pred(torch.cat([V_tr, A_tr], 1), torch.cat([V_te, A_te], 1), Fwd_tr - pf_tr)
print('  Δf resid-after-position, by VELOCITY       : %+.3f   (anticipation the position MOTION adds)' % cc(rv, Fwd_te - pf_te))
print('  Δf resid-after-position, by VELOCITY+ACCEL : %+.3f' % cc(rva, Fwd_te - pf_te))
print('=== ALL_DONE ===')
