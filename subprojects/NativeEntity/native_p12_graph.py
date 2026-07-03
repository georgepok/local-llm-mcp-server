# P12_SLOT_GRAPH_CONTRASTIVE_V1 — fix the dynamics wall (capacity ruled out). Slot-slot self-attention + supervised-contrastive rule-identity loss + 2->4->6->8 curriculum.
# Cached P12 episodes, NO model. Goal: form a stable 8-rule identity manifold so post-hoc preservation holds at NR=8.
import os, random, math
import torch, torch.nn as nn, torch.nn.functional as F
SEED = int(os.environ.get('SEED', '0')); torch.manual_seed(SEED); random.seed(SEED); dev = torch.device('cuda')
cp = '/home/pokazge/checkpoints/native_p12_s%d.pt' % SEED
eps_all = torch.load(cp, weights_only=False)['eps']; D_MODEL = eps_all[0][0][0].shape[-1]
K = int(os.environ.get('K', '12')); SLOW_K = int(os.environ.get('SLOW_K', '6')); D_S = int(os.environ.get('D_S', '768'))
TAU = float(os.environ.get('TAU', '0.1')); CW = float(os.environ.get('CW', '1.0')); STAGE_EP = int(os.environ.get('STAGE_EP', '80'))
print('P12-GRAPH | episodes=%d d_model=%d | K=%d slow_k=%d d_s=%d | SupCon tau=%.2f w=%.1f curriculum 2->4->6->8' % (len(eps_all), D_MODEL, K, SLOW_K, D_S, TAU, CW), flush=True)


class SlotGraph(nn.Module):                                            # slot-update with slot-slot SELF-ATTENTION (slots exchange info -> structured rule manifold)
    def __init__(s, d_model, d_s, K, slow_k, heads=4):
        super().__init__(); s.d_s, s.K, s.slow_k, s.heads, s.dh = d_s, K, slow_k, heads, d_s // heads
        s.read_in = nn.Linear(d_model, d_s)
        s.qc, s.kc, s.vc = nn.Linear(d_s, d_s), nn.Linear(d_s, d_s), nn.Linear(d_s, d_s)     # cross-attn slots<-hidden
        s.qs, s.ks, s.vs = nn.Linear(d_s, d_s), nn.Linear(d_s, d_s), nn.Linear(d_s, d_s)     # slot-slot self-attn
        s.gru = nn.GRUCell(d_s, d_s); s.ln1, s.ln2 = nn.LayerNorm(d_s), nn.LayerNorm(d_s)
        g0 = torch.cat([torch.full((slow_k,), -1.5), torch.zeros(K - slow_k)]); s.gate = nn.Parameter(g0)
        s.S0 = nn.Parameter(torch.randn(K, d_s) * 0.02)

    def _attn(s, q, k, v):
        T = k.shape[0]; Q = q.view(-1, s.heads, s.dh).transpose(0, 1); Kk = k.view(T, s.heads, s.dh).transpose(0, 1); Vv = v.view(T, s.heads, s.dh).transpose(0, 1)
        a = torch.softmax((Q @ Kk.transpose(-1, -2)) / (s.dh ** 0.5), dim=-1); return (a @ Vv).transpose(0, 1).reshape(-1, s.d_s)

    def init(s): return s.S0.clone()
    def step(s, S, H):
        Hp = s.read_in(H.float())
        ctx = s._attn(s.qc(S), s.kc(Hp), s.vc(Hp)); Sa = s.ln1(S + ctx)                       # absorb hidden
        ss = s._attn(s.qs(Sa), s.ks(Sa), s.vs(Sa))                                            # slots exchange info
        cand = s.gru(ss, Sa); g = torch.sigmoid(s.gate).unsqueeze(-1)
        return s.ln2(Sa + g * (cand - Sa))
    @property
    def slow(s): return slice(0, s.slow_k)


def build_eval(NR):
    sub = [(pre, ri) for pre, ri in eps_all if ri < NR]; random.seed(SEED); random.shuffle(sub)
    nte = max(8, len(sub) // 5); return sub[nte:], sub[:nte]


def supcon(Z, y):                                                      # supervised contrastive on L2-normed slow-pooled reps
    Z = F.normalize(Z, dim=-1); sim = (Z @ Z.t()) / TAU; n = Z.shape[0]
    mask = (y.unsqueeze(0) == y.unsqueeze(1)).float(); eye = torch.eye(n, device=Z.device); mask = mask - eye
    sim = sim - eye * 1e9; logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (mask * logp).sum(1) / mask.sum(1).clamp(min=1); return -(pos[mask.sum(1) > 0]).mean() if (mask.sum(1) > 0).any() else torch.zeros((), device=Z.device)


def main():
    g = SlotGraph(D_MODEL, D_S, K, SLOW_K).to(dev)
    ah = nn.Sequential(nn.Linear(SLOW_K * D_S, 256), nn.GELU(), nn.Linear(256, 8)).to(dev)
    opt = torch.optim.Adam(list(g.parameters()) + list(ah.parameters()), lr=5e-4)
    def Zslow(pre):                                                    # pooled slow-slot rep after full-episode stepping
        S = g.init()
        for h in pre: S = g.step(S, h.to(dev).float())
        return S[g.slow].reshape(-1)
    for stage, NR in enumerate([2, 4, 6, 8]):                          # CURRICULUM
        tr = [(pre, ri) for pre, ri in eps_all if ri < NR]; bs = int(os.environ.get('BS', '16'))
        for epn in range(STAGE_EP):
            random.shuffle(tr); tot = 0.0; nb = 0
            for i in range(0, len(tr), bs):
                batch = tr[i:i + bs];
                if len(batch) < 2: continue
                Z = torch.stack([Zslow(pre) for pre, ri in batch]); y = torch.tensor([ri for pre, ri in batch], device=dev)
                lc = supcon(Z, y); la = F.cross_entropy(ah(Z), y); loss = la + CW * lc
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(ah.parameters()), 1.0); opt.step(); tot += float(loss); nb += 1
            if epn % 40 == 0 or epn == STAGE_EP - 1: print('  stage%d NR<=%d ep %3d | loss=%.4f' % (stage, NR, epn, tot / max(1, nb)), flush=True)
    for p in g.parameters(): p.requires_grad_(False)
    # POST-HOC preservation at each NR
    @torch.no_grad()
    def zof(pre): return Zslow(pre).detach()
    print('=== P12_GRAPH_REPORT ===', flush=True)
    for NR in [2, 4, 8]:
        tr, te = build_eval(NR); Str = [(zof(pre), ri) for pre, ri in tr]; Ste = [(zof(pre), ri) for pre, ri in te]
        rc = nn.Sequential(nn.Linear(SLOW_K * D_S, 256), nn.GELU(), nn.Linear(256, NR)).to(dev); ro = torch.optim.Adam(rc.parameters(), 1e-3)
        for epn in range(200):
            random.shuffle(Str)
            for z, y in Str: l = F.cross_entropy(rc(z).unsqueeze(0), torch.tensor([y], device=dev)); ro.zero_grad(); l.backward(); ro.step()
        with torch.no_grad():
            tra = sum(1.0 for z, y in Str if int(rc(z).argmax()) == y) / len(Str); tea = sum(1.0 for z, y in Ste if int(rc(z).argmax()) == y) / max(1, len(Ste))
            z0 = g.init()[g.slow].reshape(-1); rst = sum(1.0 for z, y in Ste if int(rc(z0).argmax()) == y) / max(1, len(Ste))
            stale = sum(1.0 for i, (z, y) in enumerate(Ste) if int(rc(Ste[(i + 3) % len(Ste)][0]).argmax()) == y) / max(1, len(Ste))
        print('   NR=%d chance=%.3f | post-hoc train=%.3f held-out=%.3f | reset=%.3f stale=%.3f' % (NR, 1.0 / NR, tra, tea, rst, stale), flush=True)
    print('=== P12_GRAPH_DONE === (gate: NR=8 train>=0.85 held-out>=0.50, reset/stale near chance)', flush=True)


main()
