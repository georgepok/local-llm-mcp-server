# AUTOPOIETIC CLOSURE — Stage 1 (recalibrated): the cleanest stake. W is SUBCRITICAL (spectral radius <1), so activity
# DECAYS to silence (death) unless the system raises its own gain to self-sustain. The external hand we normally apply
# is exactly "keep it active"; here we test internalizing it. Modes:
#   noreg : fixed gain, no maintenance      -> activity decays -> SILENT death
#   clamp : EXTERNAL hand (we rescale to target energy each step) -> alive by our hand
#   homeo : INTERNALIZED (per-unit gain self-raises toward a target activity) -> must IGNITE and HOLD itself
# Perturbations provide the kicks that let a high-enough gain ignite self-sustaining dynamics. Watch: viability over
# time (lives by its own gain?), and the Lyapunov signature crossing toward ~0 (self-sustaining edge) — a discontinuous
# ignition = crystallization; gradual/none = not.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, statistics as st
torch.set_grad_enabled(False)
N, STEPS, TAU, RHO = 256, 9000, 3.0, 0.85
E_STAR, E_LO, E_HI, ETA, G0 = 0.10, 0.012, 0.55, 0.04, 0.8
WARMUP, KDEATH = 400, 100
def lyap(W, g, s, k=25):
    s1, s2 = s.clone(), s + torch.randn(N) * 1e-6
    d0 = (torch.tanh(g * s1) - torch.tanh(g * s2)).norm() + 1e-12; acc = 0.0
    for _ in range(k):
        s1 = s1 + (-s1 + W @ torch.tanh(g * s1)) / TAU; s2 = s2 + (-s2 + W @ torch.tanh(g * s2)) / TAU
        d = (torch.tanh(g * s1) - torch.tanh(g * s2)).norm() + 1e-12; acc += float(torch.log(d / d0)); s2 = s1 + (s2 - s1) * (d0 / d)
    return acc / k
def run(mode, seed=0):
    torch.manual_seed(seed)
    W = torch.randn(N, N) / N ** 0.5 * RHO; s = torch.randn(N) * 0.3; g = torch.full((N,), G0); E = torch.zeros(N)
    H = {'E': [], 'gain': [], 'lyap': [], 'alive': []}; alive = True; death = None; oob = 0
    for t in range(STEPS):
        inp = torch.randn(N) * 0.5 if (t % 150 == 0) else torch.zeros(N)         # perturbation kicks (let high gain ignite)
        r = torch.tanh(g * s); s = s + (-s + (W @ r + inp)) / TAU; rn = torch.tanh(g * s)
        E = 0.98 * E + 0.02 * rn.pow(2)
        if mode == 'homeo': g = (g + ETA * (E_STAR - E) * g).clamp(0.05, 25)      # internalized: raise gain to self-sustain
        elif mode == 'clamp': s = s * (E_STAR ** 0.5 / (rn.pow(2).mean().sqrt() + 1e-6))  # external hand
        popE = float(rn.pow(2).mean()); inband = E_LO < popE < E_HI
        if t > WARMUP:
            oob = 0 if inband else oob + 1
            if oob >= KDEATH and alive: death = t; alive = False
        H['E'].append(popE); H['gain'].append(float(g.mean())); H['alive'].append(float(alive))
        if t % 200 == 0: H['lyap'].append(lyap(W, g, s.clone()))
    return H, death
print('N=%d steps=%d  rho=%.2f (SUBCRITICAL) tau=%.1f  E*=%.2f  band=[%.3f,%.2f]' % (N, STEPS, RHO, TAU, E_STAR, E_LO, E_HI), flush=True)
for mode in ['noreg', 'clamp', 'homeo']:
    H, death = run(mode); ly = H['lyap']
    seg = lambda a, i, j: st.mean(a[i:j])
    print('\nmode=%-6s  alive_frac=%.2f  death_step=%s' % (mode, sum(H['alive']) / STEPS, death))
    print('   energy:  early=%.3f  mid=%.3f  late=%.3f' % (seg(H['E'], 0, 300), seg(H['E'], STEPS // 2 - 150, STEPS // 2 + 150), seg(H['E'], STEPS - 300, STEPS)))
    print('   gain  :  early=%.2f   late=%.2f   (homeo should self-raise toward self-sustaining)' % (seg(H['gain'], 0, 100), seg(H['gain'], STEPS - 200, STEPS)))
    print('   LYAP over run:', ['%+.2f' % v for v in ly[::4]])
print('\nread: noreg silent-dies; clamp lives by our hand; homeo must raise its OWN gain to ignite & hold — watch for')
print('a discontinuous Lyapunov/energy ignition (crystallization) vs gradual. Next stages: individuation + identity-defense.')
print('=== ALL_DONE ===')
