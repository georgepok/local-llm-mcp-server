import os, math, random
os.environ['ME_MODE']='none'
import numpy as np

# COHERENCE_SUBSTRATE_EVOLUTION_V1 — Phase 1 (impl-order steps 1-6). Quantum-INSPIRED, classically simulated.
# NOT a claim about microtubules/Orch-OR/consciousness. Tests whether a transient phase-coherence substrate
# gives a CAUSAL advantage on distributed relational binding, vanishing under phase-scramble + matched capacity.
# Task E3-PLUS = contextual-XOR delayed credit: reward iff action == (context XOR observation) -> requires binding
# a HELD context with a CURRENT observation (non-linearly separable; single factor insufficient).
np.seterr(over='ignore', invalid='ignore')

D_IN=8   # ctx(0/1), f1(2/3), f2(4/5), distractor(6), reward-flag(7)
D_ACT=2

def obs_ctx(c):    v=np.zeros(D_IN,np.float32); v[c]=1.0; return v
def obs_f1(f):     v=np.zeros(D_IN,np.float32); v[2+f]=1.0; return v
def obs_f2(f):     v=np.zeros(D_IN,np.float32); v[4+f]=1.0; return v
def obs_o(o):      v=np.zeros(D_IN,np.float32); v[2+o]=1.0; return v    # (legacy easy task)
def obs_dist(rng): v=np.zeros(D_IN,np.float32); v[6]=1.0; v[:6]=0.15*rng.randn(6); return v
def obs_reward():  v=np.zeros(D_IN,np.float32); v[7]=1.0; return v

class Organism:
    """Classical recurrent + eligibility-trace plastic readout, with an optional coherence layer whose
       global order parameter MODULATES plasticity (coherence-gated consolidation). Conditions differ only
       in the coherence layer + how the plasticity gate is computed (matched param/state/update budget)."""
    def __init__(self, cond, H=24, N=16, seed=0):
        self.cond=cond; self.H=H; self.N=N; g=np.random.RandomState(seed)
        self.Win=(g.randn(H,D_IN)*0.6).astype(np.float32)
        Wr=g.randn(H,H).astype(np.float32); Wr*= 0.95/(np.max(np.abs(np.linalg.eigvals(Wr)))+1e-6); self.Wrec=Wr
        self.Wout=(g.randn(D_ACT,H)*0.05).astype(np.float32)
        self.leak=0.2; self.gain=1.3
        self.eta=0.08; self.edecay=0.6
        # coherence / matched-capacity layer
        self.uses_phase = cond in ('phase','phase_highdephase','phase_scramble','global_lock')
        self.uses_extra = cond=='extra_real'
        if self.uses_phase or self.uses_extra:
            self.Wh2c=(g.randn(N,H)*0.5).astype(np.float32)           # neural->coherence projection
            self.omega=(g.uniform(-0.3,0.3,N)).astype(np.float32)
            self.K={'phase':1.2,'phase_highdephase':1.2,'phase_scramble':1.2,'global_lock':6.0}.get(cond,1.2)
            self.dephase={'phase':0.15,'phase_highdephase':1.5,'phase_scramble':0.15,'global_lock':0.005}.get(cond,0.15)
        self.g=g; self.reset()
    def reset(self):
        self.h=np.zeros(self.H,np.float32); self.e=np.zeros_like(self.Wout)
        if self.uses_phase: self.phi=self.g.uniform(0,2*math.pi,self.N).astype(np.float32)
        if self.uses_extra: self.hx=np.zeros(self.N,np.float32)
    def step(self, obs, rng):
        self.h=(1-self.leak)*self.h+self.leak*np.tanh(self.gain*(self.Wrec@self.h)+self.Win@obs)
        self.h=np.clip(self.h,-8,8)
        C=1.0
        if self.uses_phase:
            drive=self.Wh2c@self.h
            for _ in range(3):                                        # 3 sub-steps of Kuramoto
                mf=np.mean(np.exp(1j*self.phi)); R=np.abs(mf); psi=np.angle(mf)
                dphi=self.omega + self.K*R*np.sin(psi-self.phi) + 0.4*drive + self.dephase*rng.randn(self.N)
                self.phi=(self.phi+0.25*dphi)%(2*math.pi)
            phi_read=self.phi
            if self.cond=='phase_scramble': phi_read=rng.uniform(0,2*math.pi,self.N)   # B6: scramble phases at read
            C=float(np.abs(np.mean(np.exp(1j*phi_read))))            # global order parameter as plasticity gate
        elif self.uses_extra:                                        # B1: matched extra REAL units (no phase), matched gate
            self.hx=0.8*self.hx+0.2*np.tanh(self.Wh2c@self.h+self.dephase_noise(rng))
            C=float(np.mean(np.abs(self.hx)))                        # a scalar summary, matched compute, NO phase relations
        self.C=C; return self.Wout@self.h, C
    def dephase_noise(self, rng): return 0.15*rng.randn(self.N).astype(np.float32)
    def update(self, cons, action, C):
        oa=np.zeros(D_ACT,np.float32); oa[action]=1.0
        self.e=self.edecay*self.e+np.outer(oa,self.h)               # eligibility of (action x state)
        cf=C if self.cond!='classical' else 1.0                     # coherence-modulated consolidation
        if self.cond in ('classical','extra_real'): cf=C if self.uses_extra else 1.0
        self.Wout += self.eta*cons*cf*self.e
        np.clip(self.Wout,-4,4,out=self.Wout)

def rollout_e3plus(org, rng, ntrials=24):                            # legacy EASY task (contextual XOR, per-channel)
    c=rng.randint(0,1); org.step(obs_ctx(c), rng); corr=[]
    for t in range(ntrials):
        o=rng.randint(0,1); out,C=org.step(obs_o(o), rng)
        a=int(np.argmax(out[:D_ACT])); target=c ^ o; cons=1.0 if a==target else -1.0
        corr.append(int(a==target)); _,C2=org.step(obs_reward(), rng); org.update(cons, a, C2)
    return corr

def rollout_hard(org, rng, ntrials=36, ndist=3):
    # 3-factor DISTRIBUTED parity: context c (held all episode) + f1 + f2 separated by distractors; delayed reward.
    # reward iff action == (c XOR f1 XOR f2). Requires holding+binding 3 factors across interference w/ limited state.
    c=rng.randint(0,1); org.step(obs_ctx(c), rng); corr=[]
    for t in range(ntrials):
        f1=rng.randint(0,1); org.step(obs_f1(f1), rng)
        for _ in range(ndist): org.step(obs_dist(rng), rng)
        f2=rng.randint(0,1); out,C=org.step(obs_f2(f2), rng)
        a=int(np.argmax(out[:D_ACT])); target=c ^ f1 ^ f2; cons=1.0 if a==target else -1.0
        corr.append(int(a==target))
        for _ in range(ndist): org.step(obs_dist(rng), rng)         # delay before reward (eligibility must bridge)
        _,C2=org.step(obs_reward(), rng); org.update(cons, a, C2)
    return corr

def eval_cond(cond, task='hard', neps=200, H=12, N=16, ndist=3):
    accs_late=[]; accs_all=[]; Cs=[]; unstable=0
    for ep in range(neps):
        org=Organism(cond, H=H, N=N, seed=1000+(ep%17)); rng=np.random.RandomState(7000+ep)
        corr=rollout_hard(org, rng, ndist=ndist) if task=='hard' else rollout_e3plus(org, rng)
        if not np.all(np.isfinite(org.Wout)): unstable+=1; continue
        accs_late.append(np.mean(corr[-10:])); accs_all.append(np.mean(corr)); Cs.append(getattr(org,'C',1.0))
    return {'late':round(float(np.mean(accs_late)),3),'all':round(float(np.mean(accs_all)),3),
            'coh':round(float(np.mean(Cs)),3),'nan':round(unstable/neps,3),'n':len(accs_late)}

if __name__=='__main__':
    print("=== COHERENCE_SUBSTRATE_EVOLUTION_V1 Phase-1: E3-PLUS HARD (3-factor DISTRIBUTED parity, H=12, distractors) ===", flush=True)
    print("  chance=0.50; bind context(held) XOR f1 XOR f2 across distractors+delay w/ limited state", flush=True)
    conds=[('B0 classical','classical'),('B1 extra_real (matched cap)','extra_real'),
           ('B2 phase-coherence','phase'),('B4 high-dephasing','phase_highdephase'),
           ('B5 global-lock','global_lock'),('B6 PHASE-SCRAMBLE','phase_scramble')]
    for label,cond in conds:
        r=eval_cond(cond)
        print(f"  {label:28s}: late_acc={r['late']:.3f} all_acc={r['all']:.3f} coh={r['coh']:.2f} nan={r['nan']} n={r['n']}", flush=True)
    print("=== DECISIVE: does B2 phase-coherence > B0/B1 (capacity), AND does B6 phase-scramble collapse it? ===", flush=True)
    print("=== COHERENCE_SMOKE_DONE ===", flush=True)
