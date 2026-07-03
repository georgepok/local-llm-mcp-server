# THE STAKE probe (CPU, belief-level, fast): the closed-loop runs proved the trained compressor-actuator does NOT defend
# a singular identity — it TRACKS input (input-driven) and the actuator AMPLIFIES whatever the belief holds => positive-
# feedback capture (wrong sign for defense). The theory's fix: internalize the viability hand — give the belief its OWN
# equilibrium-seeking so maintenance is its own, with a real restoring force toward its established identity = the STAKE.
# Cleanest no-controller realization: self-attraction toward the belief's running identity (EMA of its history),
#   b <- b + (-b + tanh(Wb + a) + gamma*(ema - b))/tau .   gamma=0 reproduces the tracker. Sweep gamma and ask the two
# questions that define a SELF vs a frozen block:
#   DEFENSE : feed a VIABLE OTHER goal B's stream to an A-belief -> does it HOLD A (cos->A high, cos->B low) or get captured?
#   TRACKING: feed A's OWN continued stream -> is next-gist prediction (the compressor's validated job) PRESERVED?
# A real self needs BOTH (defends viable perturbation AND still tracks legitimate change). That balance point is the stake.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
torch.set_grad_enabled(False)
data = [m for m in torch.load('/home/pokazge/checkpoints/objective_drift_txt.pt', weights_only=False, map_location='cpu')['data'] if len(m['gen']) >= 10]
d_m = data[0]['gen'][0].shape[1]; MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
def cen(C): return C - MU
for m in data: m['z'] = [F.normalize(cen(c).mean(0), dim=0) for c in m['gen']]
class Compressor(nn.Module):
    def __init__(s, d_m, D=384, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(d_m, heads * dh); s.Wv = nn.Linear(d_m, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); K_ = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K_) / s.dh ** 0.5, dim=-1); return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
comp = Compressor(d_m); comp.load_state_dict(torch.load('/home/pokazge/checkpoints/lora_inloop.pt', map_location='cpu')['comp'])
TAU = F.softplus(comp.log_tau) + 0.5
def step(b, h, C, anchor, gamma, rho):                                           # gamma = stake strength toward anchor; rho = anchor drift rate
    a = comp.collect(cen(C), b)
    for _ in range(2): b = b + (-b + torch.tanh(comp.W(b) + a) + gamma * (anchor - b)) / TAU / 2
    anchor = (1 - rho) * anchor + rho * b; h = 0.9 * h + 0.1 * b                  # rho=0 => FIXED anchor (real restoring force); rho>0 => slowly follows
    return b, h, anchor
def establish(m, t1):                                                            # drive first t1 chunks, no stake -> established belief identity
    b = torch.zeros(comp.D); h = torch.zeros(comp.D)
    for t in range(t1): b, h, _ = step(b, h, m['gen'][t], b, 0.0, 0.0)
    return b, h
def cos(a, b): return float(F.cosine_similarity(a, b, dim=-1).mean())
M = len(data); REF = [establish(m, len(m['gen']))[0] for m in data]              # reference identities (full-trajectory beliefs)
CONFIGS = [(0.0, 0.0), (0.3, 0.0), (0.6, 0.0), (1.0, 0.0), (0.6, 0.02), (1.0, 0.02), (1.0, 0.1)]  # (gamma, rho); first is the bare tracker
print('=== THE STAKE: restoring-force toward established identity (trained compressor, %d goals) ===' % M, flush=True)
print('  DEFENSE = feed VIABLE goal-B stream to an A-belief; TRACKING = feed A own stream, next-gist coh. A self needs BOTH.', flush=True)
print('  gamma rho |  defense cos->A   cos->B  (hold A?)  |  tracking coh  | net', flush=True)
for g, rho in CONFIGS:
    defA, defB, trk = [], [], []
    for A in range(M):
        B = (A + 1) % M; mB, mA = data[B], data[A]; t1 = len(mA['gen']) // 2
        bA, hA = establish(mA, t1)                                               # established A identity = the anchor
        b, h, anc = bA.clone(), hA.clone(), bA.clone()                           # DEFENSE: feed B's stream, stake toward A
        for t in range(len(mB['gen'])): b, h, anc = step(b, h, mB['gen'][t], anc, g, rho)
        defA.append(cos(b, REF[A])); defB.append(cos(b, REF[B]))
        b, h, anc = bA.clone(), hA.clone(), bA.clone(); preds, zs = [], []        # TRACKING: feed A's own stream, stake toward A
        for t in range(t1, len(mA['gen'])):
            hp = h; b, h, anc = step(b, h, mA['gen'][t], anc, g, rho)
            preds.append(comp.pred(torch.cat([comp.cz(mA['z'][t]), hp]))); zs.append(mA['z'][t])
        P = F.normalize(torch.stack(preds), dim=-1); Z = torch.stack(zs)
        trk.append(float((P[:-1] * Z[1:]).sum(-1).mean()) if len(preds) > 1 else 0.0)
    held = st.mean(defA) > st.mean(defB)
    net = 'DEFENDS+TRACKS' if (held and st.mean(trk) > 0.012) else ('frozen-block' if held else 'tracker(captured)')
    print('  %.1f  %.2f |     %+.3f     %+.3f   %s  |    %+.4f    | %s'
          % (g, rho, st.mean(defA), st.mean(defB), 'HOLDS-A' if held else 'captured', st.mean(trk), net), flush=True)
print('\nread: (0.0,0.0) = bare tracker (expect captured, cos->B>cos->A). A config that flips to HOLDS-A while keeping tracking', flush=True)
print('coh near the tracker baseline = the STAKE that makes a self (defends viable perturbation, still tracks legitimate change).', flush=True)
print('HOLDS-A only with collapsed coh = a frozen block, not a self -> the stake must be consistency-gated, not constant.', flush=True)
print('=== ALL_DONE ===', flush=True)
