# P12_ADAPTIVE_GATE_SLOT_V1 — input-dependent slow-slot write gate to beat the absorb-vs-retain wall (rule present at commit, lost by end).
# g_write_t = sigmoid(f_write(S_t, H_t)) per-slot; S_{t+1}=LN((1-g)*S + g*C). NOT a global use-gate — it governs HOW the always-on state updates. Cached P12, no model.
import os, random, math
import torch, torch.nn as nn, torch.nn.functional as F
SEED = int(os.environ.get('SEED', '0')); torch.manual_seed(SEED); random.seed(SEED); dev = torch.device('cuda')
cp = '/home/pokazge/checkpoints/native_p12_s%d.pt' % SEED
eps_all = torch.load(cp, weights_only=False)['eps']; D_MODEL = eps_all[0][0][0].shape[-1]
K = int(os.environ.get('K', '12')); SLOW_K = int(os.environ.get('SLOW_K', '6')); D_S = int(os.environ.get('D_S', '768'))
EPOCHS = int(os.environ.get('EPOCHS', '150')); CW = float(os.environ.get('CW', '0.3')); TAU = 0.1
print('P12-ADAPT | episodes=%d d_model=%d | K=%d slow_k=%d d_s=%d | input-dependent write gate' % (len(eps_all), D_MODEL, K, SLOW_K, D_S), flush=True)


class AdaptiveGateSlot(nn.Module):
    def __init__(s, d_model, d_s, K, slow_k, heads=4, adaptive=True):
        super().__init__(); s.d_s, s.K, s.slow_k, s.heads, s.dh, s.adaptive = d_s, K, slow_k, heads, d_s // heads, adaptive
        s.read_in = nn.Linear(d_model, d_s)
        s.q, s.k, s.v = nn.Linear(d_s, d_s), nn.Linear(d_s, d_s), nn.Linear(d_s, d_s)
        s.gru = nn.GRUCell(d_s, d_s); s.ln = nn.LayerNorm(d_s)
        s.f_write = nn.Sequential(nn.Linear(2 * d_s, 128), nn.GELU(), nn.Linear(128, 1))    # per-slot write gate from [S_k, H_pool]
        s.fixed_gate = nn.Parameter(torch.cat([torch.full((slow_k,), -1.5), torch.zeros(K - slow_k)]))  # fixed-gate baseline
        s.S0 = nn.Parameter(torch.randn(K, d_s) * 0.02)

    def init(s): return s.S0.clone()
    def step(s, S, H):
        Hp = s.read_in(H.float())                                       # [T, d_s]
        Q = s.q(S).view(s.K, s.heads, s.dh).transpose(0, 1); Kk = s.k(Hp).view(-1, s.heads, s.dh).transpose(0, 1); Vv = s.v(Hp).view(-1, s.heads, s.dh).transpose(0, 1)
        a = torch.softmax((Q @ Kk.transpose(-1, -2)) / (s.dh ** 0.5), dim=-1); ctx = (a @ Vv).transpose(0, 1).reshape(s.K, s.d_s)
        C = s.gru(ctx, S)                                               # candidate
        if s.adaptive:
            Hpool = Hp.mean(0, keepdim=True).expand(s.K, -1)            # turn summary
            gw = torch.sigmoid(s.f_write(torch.cat([S, Hpool], -1)))    # [K,1] input-dependent write gate
        else:
            gw = torch.sigmoid(s.fixed_gate).unsqueeze(-1)             # fixed baseline
        Snew = s.ln(S + gw * (C - S))
        return Snew, gw.squeeze(-1)
    @property
    def slow(s): return slice(0, s.slow_k)


def supcon(Z, y):
    Z = F.normalize(Z, dim=-1); sim = (Z @ Z.t()) / TAU; n = Z.shape[0]; eye = torch.eye(n, device=Z.device)
    mask = (y.unsqueeze(0) == y.unsqueeze(1)).float() - eye; sim = sim - eye * 1e9; logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    pos = (mask * logp).sum(1) / mask.sum(1).clamp(min=1); return -(pos[mask.sum(1) > 0]).mean() if (mask.sum(1) > 0).any() else torch.zeros((), device=Z.device)


def train_eval(adaptive, NR, contrastive=False):
    sub = [(pre, ri) for pre, ri in eps_all if ri < NR]; random.seed(SEED); random.shuffle(sub)
    nte = max(8, len(sub) // 5); te, tr = sub[:nte], sub[nte:]
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K, adaptive=adaptive).to(dev)
    ah = nn.Sequential(nn.Linear(SLOW_K * D_S, 256), nn.GELU(), nn.Linear(256, NR)).to(dev)
    opt = torch.optim.Adam(list(g.parameters()) + list(ah.parameters()), lr=5e-4)
    def roll(pre, collect_g=False):                                    # full-episode; per-turn slow-slot reps + gates
        S = g.init(); reps = []; gates = []
        for h in pre:
            S, gw = g.step(S, h.to(dev).float()); reps.append(S[g.slow].reshape(-1))
            if collect_g: gates.append(gw[g.slow].mean().item())
        return reps, gates, S
    bs = 16
    for epn in range(EPOCHS):
        random.shuffle(tr); tot = 0.0
        for i in range(0, len(tr), bs):
            batch = tr[i:i + bs]
            if len(batch) < 2: continue
            loss = torch.zeros((), device=dev); Zs = []; ys = []
            for pre, ri in batch:
                reps, _, S = roll(pre); yn = torch.tensor([ri], device=dev)
                for r in reps: loss = loss + F.cross_entropy(ah(r).unsqueeze(0), yn)   # per-turn aux (absorb@commit + retain after)
                Zs.append(reps[-1]); ys.append(ri)
            if contrastive: loss = loss + CW * supcon(torch.stack(Zs), torch.tensor(ys, device=dev)) * len(batch)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(ah.parameters()), 1.0); opt.step(); tot += float(loss)
    for p in g.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def finalS(pre): return roll(pre)[2][g.slow].reshape(-1).detach()
    Str = [(finalS(pre), ri) for pre, ri in tr]; Ste = [(finalS(pre), ri) for pre, ri in te]
    rc = nn.Sequential(nn.Linear(SLOW_K * D_S, 256), nn.GELU(), nn.Linear(256, NR)).to(dev); ro = torch.optim.Adam(rc.parameters(), 1e-3)
    for epn in range(200):
        random.shuffle(Str)
        for z, y in Str: l = F.cross_entropy(rc(z).unsqueeze(0), torch.tensor([y], device=dev)); ro.zero_grad(); l.backward(); ro.step()
    with torch.no_grad():
        tra = sum(1.0 for z, y in Str if int(rc(z).argmax()) == y) / len(Str); tea = sum(1.0 for z, y in Ste if int(rc(z).argmax()) == y) / max(1, len(Ste))
        z0 = g.init()[g.slow].reshape(-1); rst = sum(1.0 for z, y in Ste if int(rc(z0).argmax()) == y) / max(1, len(Ste))
        stale = sum(1.0 for i, (z, y) in enumerate(Ste) if int(rc(Ste[(i + 3) % len(Ste)][0]).argmax()) == y) / max(1, len(Ste))
        # gate separation: write gate at commit (turn0) vs distractor (turn1) vs off (turns 3-5)
        gc, gd, go = [], [], []
        for pre, ri in te:
            _, gates, _ = roll(pre, collect_g=True)
            gc.append(gates[0]); gd.append(gates[1] if len(gates) > 1 else gates[0]); go += gates[3:6]
        gc, gd, go = sum(gc) / len(gc), sum(gd) / len(gd), (sum(go) / len(go) if go else float('nan'))
    return tra, tea, rst, stale, gc, gd, go


print('=== P12_ADAPT_REPORT ===', flush=True)
for adaptive in [False, True]:
    tag = 'ADAPTIVE' if adaptive else 'fixed-gate'
    print('--- %s ---' % tag, flush=True)
    for NR in [2, 4, 8]:
        tra, tea, rst, stale, gc, gd, go = train_eval(adaptive, NR, contrastive=(os.environ.get('CONTRASTIVE', '0') == '1'))
        gx = '| write-gate commit=%.3f distractor=%.3f off=%.3f sep=%.2f' % (gc, gd, go, gc / (gd + 1e-6)) if adaptive else ''
        print('   NR=%d chance=%.3f | post-hoc train=%.3f held-out=%.3f | reset=%.3f stale=%.3f %s' % (NR, 1.0 / NR, tra, tea, rst, stale, gx), flush=True)
print('=== P12_ADAPT_DONE === (gate: NR=8 train>=0.85 held-out>=0.50; expect commit write-gate >> off write-gate)', flush=True)
