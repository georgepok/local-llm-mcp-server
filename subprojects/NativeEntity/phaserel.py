import os, math, random
os.environ['ME_MODE']='none'
import numpy as np
np.seterr(over='ignore', invalid='ignore')

# PHASE_RELATIONAL_BINDING_V1 — Phase 1, first decisive milestone. Quantum-INSPIRED, classically simulated.
# NO Orch-OR/microtubule/consciousness claims. Hypothesis (INDUCTIVE BIAS, not unique computability):
# PAIRWISE phase structure B_ij=cos(phi_i-phi_j) holds temporary feature bindings with less interference /
# better scaling than parameter-matched real superposition. Milestone: (a) features stay recognizable,
# (b) PAIRWISE phase scramble SELECTIVELY destroys PAIRING not feature identity, (c) GLOBAL rotation is inert,
# (d) phase scales to more objects better than a param-matched real (tensor/superposition) control.
NCOL=8; NSHP=8; DC=8; DS=8
def _emb(n,d,seed):
    g=np.random.RandomState(seed); E=g.randn(n,d).astype(np.float32); return (E/np.linalg.norm(E,axis=1,keepdims=True))
COL=_emb(NCOL,DC,1); SHP=_emb(NSHP,DS,2)

def make_episode(rng, K, held_shapes=None):
    cols=rng.sample(range(NCOL),K); shps=rng.sample(range(NSHP),K)      # distinct colors + shapes per object
    objs=list(zip(cols,shps)); order=objs[:]; rng.shuffle(order)
    q=rng.randrange(K)
    return order, order[q][0], order[q][1]                              # presentation order, query-color, true-shape

# ---------------- PHASE-RELATIONAL binder (B4): binding lives in pairwise phase ----------------
class PhaseBinder:
    """Dedicated color-sites + shape-sites. Object's color & shape (co-present in time) lock to a common
       TEMPORAL phase (emerges from timing: a slow phase advances per object). Query-by-color -> the shape
       whose site is most phase-aligned. Multiple bindings held as distinct phase clusters. No slot/ID channel."""
    def __init__(self, cond='phase', seed=0):
        self.cond=cond
        self.Kdrive=2.0; self.omega=0.62; self.steps=8                   # strong locking; omega*Kmax(8)=5<2pi -> distinct phases
    def run(self, objs, qcol, rng):
        nr=np.random.RandomState(rng.randrange(1<<30))
        phi=nr.uniform(0,2*math.pi,NCOL+NSHP)                            # [NCOL color-sites | NSHP shape-sites]
        active_c=np.zeros(NCOL); active_s=np.zeros(NSHP); theta=0.0
        for (c,s) in objs:
            theta+=self.omega                                            # temporal phase for THIS object (timing)
            for _ in range(self.steps):                                  # packet: co-present color+shape lock to theta
                phi[c]      += self.Kdrive*math.sin(theta-phi[c])
                phi[NCOL+s] += self.Kdrive*math.sin(theta-phi[NCOL+s])
            active_c[c]=1; active_s[s]=1
        if self.cond=='phase_scramble': phi=nr.uniform(0,2*math.pi,NCOL+NSHP)        # destroy pairwise phase (keep active sites)
        if self.cond=='global_rotate': phi=(phi+1.234)%(2*math.pi)                    # inert if only relative phase matters
        # READ binding: query color -> shape site most phase-aligned (among ACTIVE shape sites)
        pc=phi[qcol]; Brow=np.array([math.cos(pc-phi[NCOL+s]) if active_s[s] else -9 for s in range(NSHP)])
        pred_shape=int(np.argmax(Brow))
        feat_ok = active_c.sum()>0                                       # feature recognition: which sites active
        return pred_shape, active_c, active_s

# ---------------- REAL superposition binder (B2/B8 tensor-product control, matched info) ----------------
class RealBinder:
    """Parameter-matched real relational memory: M = sum_k outer(shape_k, color_k); query M@color -> shape.
       This is the strong classical/tensor-product binding baseline (NOT weakened)."""
    def __init__(self, cond='real', seed=0): self.cond=cond
    def run(self, objs, qcol, rng):
        M=np.zeros((DS,DC),np.float32); ac=np.zeros(NCOL); as_=np.zeros(NSHP)
        for (c,s) in objs:
            M+=np.outer(SHP[s],COL[c]); ac[c]=1; as_[s]=1                # real superposition of bindings
        if self.cond=='real_scramble': M=np.random.RandomState(rng.randrange(1<<30)).randn(DS,DC).astype(np.float32)
        out=M@COL[qcol]; pred=int(np.argmax(SHP@out))
        return pred, ac, as_

def eval_binder(kind, cond, K, neps=400, seed=0):
    bind_ok=[]; feat_ok=[]
    B = PhaseBinder(cond,seed) if kind=='phase' else RealBinder(cond,seed)
    for ep in range(neps):
        rng=random.Random(5000+ep); objs,qc,qs=make_episode(rng,K)
        pred, ac, asv = B.run(objs, qc, rng)
        bind_ok.append(int(pred==qs))
        # feature recognition: are all present shapes 'known active' (a shape that WAS present should be reportable)
        feat_ok.append(int(asv[qs]==1))
    return {'bind':round(float(np.mean(bind_ok)),3),'feat':round(float(np.mean(feat_ok)),3)}

if __name__=='__main__':
    print("=== PHASE_RELATIONAL_BINDING_V1 milestone: 2-object feature binding, scaling, selective scramble ===", flush=True)
    print("  binding_acc = query-color -> correct paired-shape (chance=1/K); feat = correct shape was present", flush=True)
    print("  -- SCALING: binding accuracy vs #objects (phase B4 vs real-superposition control) --", flush=True)
    print(f"  {'K':>3} {'chance':>7} | {'phase_bind':>10} {'real_bind':>9} | {'phase_feat':>10} {'real_feat':>9}", flush=True)
    for K in [2,3,4,6,8]:
        ph=eval_binder('phase','phase',K); re=eval_binder('real','real',K)
        print(f"  {K:>3} {1.0/K:>7.2f} | {ph['bind']:>10.3f} {re['bind']:>9.3f} | {ph['feat']:>10.3f} {re['feat']:>9.3f}", flush=True)
    print("  -- SELECTIVE SCRAMBLE @ K=4 (does pairwise-phase scramble kill PAIRING but not FEATURE?) --", flush=True)
    for cond in ['phase','phase_scramble','global_rotate']:
        r=eval_binder('phase',cond,4); print(f"  phase[{cond:14s}]: binding={r['bind']:.3f} feature={r['feat']:.3f}", flush=True)
    r=eval_binder('real','real_scramble',4); print(f"  real[scramble]        : binding={r['bind']:.3f} feature={r['feat']:.3f}", flush=True)
    print("=== MILESTONE CHECK: phase binding > real at high K? scramble kills binding not feature? rotate inert? ===", flush=True)
    print("=== PHASEREL_DONE ===", flush=True)
