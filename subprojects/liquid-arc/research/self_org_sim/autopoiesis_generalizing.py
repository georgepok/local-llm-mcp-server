# AUTOPOIESIS on the GENERALIZING substrate (CPU): the earlier autopoiesis ran on the memorizing compressor; now the
# enriched-recall belief GENERALIZES (carries a transferable goal-identity on unseen categories). Test the markers
# CROSS-CATEGORY (held-out trajectories): (M2) INDIVIDUATION — distinct unseen goals -> distinct belief-identities; (M3)
# IDENTITY-DEFENSE via the validated STAKE — feed a VIABLE other held-out goal's stream to an A-belief WITH a restoring
# force gamma toward A's established identity; does it HOLD A (cos->A > cos->B) where the bare tracker (gamma=0) gets
# captured, WHILE still tracking A's own stream? A defending identity on UNSEEN categories = a transferable self, not memorized.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
torch.set_grad_enabled(False)
ck = torch.load('/home/pokazge/checkpoints/enriched_recall.pt', weights_only=False, map_location='cpu')
D, PROJ, K, MUk = 256, 768, ck['K'], ck['MUk']; data = ck['data']; hold = set(ck['hold']); d_m = data[0]['z'][0].shape[0]
class Comp(nn.Module):                                                           # must match enriched_recall.py to load state
    def __init__(s, in_dim, D=256, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(in_dim, heads * dh); s.Wv = nn.Linear(in_dim, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D))
        s.pos = nn.Embedding(K, D); s.recall = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D)); s.chunkp = nn.Linear(d_m, D)
comp = Comp(PROJ, D); comp.load_state_dict(ck['comp']); TAU = F.softplus(comp.log_tau) + 0.5
def collect(C, b):
    q = comp.Wq(b).view(comp.h, comp.dh); Kk = comp.Wk(C).view(-1, comp.h, comp.dh); V = comp.Wv(C).view(-1, comp.h, comp.dh)
    a = torch.softmax(torch.einsum('hd,nhd->hn', q, Kk) / comp.dh ** 0.5, -1); return comp.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
def step(b, h, C, anchor, gamma):                                                # one chunk; gamma>0 = restoring force toward the established identity (the STAKE)
    a = collect(C - MUk, b)
    for _ in range(2): b = b + (-b + torch.tanh(comp.W(b) + a) + gamma * (anchor - b)) / TAU / 2
    return b, 0.9 * h + 0.1 * b
def establish(m, t1):
    b = torch.zeros(D); h = torch.zeros(D)
    for t in range(t1): b, h = step(b, h, m['perc'][t], b, 0.0)
    return b, h
def cos(a, b): return float(F.cosine_similarity(a, b, 0))
te = [m for m in data if m['fid'] in hold]; M = len(te)
REF = [establish(m, len(m['perc']))[0] for m in te]                              # full-trajectory belief-identities (held-out)
print('=== AUTOPOIESIS on GENERALIZING substrate (enriched recall, %d HELD-OUT unseen-category trajectories) ===' % M, flush=True)
Xn = F.normalize(torch.stack(REF), dim=1); off = (Xn @ Xn.t())[~torch.eye(M, dtype=torch.bool)]
print('M2 individuation (held-out): belief-identity cross-cos = %.3f ± %.3f  (<1 = distinct transferable identities)' % (float(off.mean()), float(off.std())), flush=True)
print('\nM3 identity-defense via STAKE (feed a VIABLE other held-out goal\'s stream; restoring force gamma toward own identity):', flush=True)
print('  gamma |  cos->own(A)  cos->other(B)  | holds? | tracking(own stream, cos to no-stake)', flush=True)
for g in [0.0, 0.3, 0.6, 1.0, 1.5]:
    dA, dB, trk = [], [], []
    for A in range(M):
        mA = te[A]; t1 = len(mA['perc']) // 2; bA, hA = establish(mA, t1); anc = bA.clone()
        B = (A + 1) % M; mB = te[B]
        b, h = bA.clone(), hA.clone()                                            # DEFENSE: feed B's stream with stake toward A
        for t in range(len(mB['perc'])): b, h = step(b, h, mB['perc'][t], anc, g)
        dA.append(cos(b, REF[A])); dB.append(cos(b, REF[B]))
        b0, h0 = bA.clone(), hA.clone(); b1, h1 = bA.clone(), hA.clone()         # TRACKING: own remaining stream, stake vs none
        for t in range(t1, len(mA['perc'])):
            b0, h0 = step(b0, h0, mA['perc'][t], anc, 0.0); b1, h1 = step(b1, h1, mA['perc'][t], anc, g)
        trk.append(cos(b1, b0))
    held = st.mean(dA) > st.mean(dB)
    print('  %.1f   |    %+.3f      %+.3f    | %s | %.3f' % (g, st.mean(dA), st.mean(dB), 'HOLDS-A' if held else 'captured', st.mean(trk)), flush=True)
print('\nread: gamma=0 = bare tracker (cos->B>cos->A = captured by the viable alternative, like the un-staked substrate). A', flush=True)
print('gamma that flips to HOLDS-A while tracking stays ~1 = identity-defense that GENERALIZES to unseen categories = a', flush=True)
print('transferable self-marker (defends ITS identity against a viable alternative), not a memorized artifact.', flush=True)
print('=== ALL_DONE ===', flush=True)
