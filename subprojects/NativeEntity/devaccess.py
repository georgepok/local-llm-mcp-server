import os, math, random
os.environ['ME_MODE']='none'
import numpy as np
np.seterr(over='ignore', invalid='ignore')

# DEVELOPMENTAL_ACCESSIBILITY_AND_LOCALITY_V1 — Q1: is a PHASE-relational binder easier to EVOLVE than a
# real associative binder at MATCHED real-scalar STATE budget? Both EVOLVED from weak random genomes (Part 2:
# no hand-coded binding rule; evolution must discover projections/coeffs/dynamics). Quantum-INSPIRED, classical.
NCOL=8; NSHP=8; DC=8; DS=8
def _emb(n,d,seed):
    g=np.random.RandomState(seed); E=g.randn(n,d).astype(np.float32); return E/np.linalg.norm(E,axis=1,keepdims=True)
COL=_emb(NCOL,DC,101); SHP=_emb(NSHP,DS,202)
def make_ep(rng,K):
    cols=rng.sample(range(NCOL),K); shps=rng.sample(range(NSHP),K); objs=list(zip(cols,shps)); rng.shuffle(objs)
    q=rng.randrange(K); return objs, objs[q][0], objs[q][1]

# ---- PHASE binder: B phase sites; each feature drives a (seeded) site-subset; co-present features lock to a
#      temporal phase; readout = phase alignment of query-color sites vs each shape's sites. Evolved: proj/spf/omega/drive.
def run_phase(gen, objs, qcol, rng):
    B=gen['B']; spf=gen['spf']; gp=np.random.RandomState(gen['proj_seed'])
    Pc=[gp.choice(B,spf,replace=False) for _ in range(NCOL)]; Ps=[gp.choice(B,spf,replace=False) for _ in range(NSHP)]
    phi=np.random.RandomState(rng.randrange(1<<30)).uniform(0,2*math.pi,B); theta=0.0; present_s=set()
    for (c,s) in objs:
        theta+=gen['omega']
        for _ in range(gen['lock']):
            for i in Pc[c]: phi[i]+=gen['drive']*math.sin(theta-phi[i])
            for i in Ps[s]: phi[i]+=gen['drive']*math.sin(theta-phi[i])
        present_s.add(s)
    def mph(idx): z=np.mean(np.exp(1j*phi[idx])); return np.angle(z)
    pcq=mph(Pc[qcol]); best=-9; bs=0
    for s in present_s:
        sc=math.cos(pcq-mph(Ps[s]))
        if sc>best: best=sc; bs=s
    return bs

# ---- REAL binder: d x d associative memory (d^2 <= B state); generic Hebbian storage of evolved feature
#      projections; readout = memory-retrieve by query color. Evolved: proj seeds / store coef / decay / d.
def run_real(gen, objs, qcol, rng):
    d=gen['d']; gp=np.random.RandomState(gen['proj_seed'])
    Ce=gp.randn(NCOL,d).astype(np.float32); Ce/=np.linalg.norm(Ce,axis=1,keepdims=True)
    Se=gp.randn(NSHP,d).astype(np.float32); Se/=np.linalg.norm(Se,axis=1,keepdims=True)
    M=np.zeros((d,d),np.float32); present_s=[]
    for (c,s) in objs:
        M=(1-gen['decay'])*M + gen['store']*np.outer(Se[s],Ce[c]); present_s.append(s)
    out=M@Ce[qcol]; best=-9; bs=0
    for s in set(present_s):
        sc=float(Se[s]@out)
        if sc>best: best=sc; bs=s
    return bs

def fitness(gen, K=2, neps=60):
    ok=[]
    for ep in range(neps):
        rng=random.Random(90000+ep); objs,qc,qs=make_ep(rng,K)
        try: pred = run_phase(gen,objs,qc,rng) if gen['mode']=='phase' else run_real(gen,objs,qc,rng)
        except Exception: return 0.0
        ok.append(int(pred==qs))
    return round(float(np.mean(ok)),3)

def rand_gen(mode, B, g):
    if mode=='phase':
        return {'mode':'phase','B':B,'proj_seed':g.randint(0,1<<30),'spf':g.randint(1,max(2,B//4)),
                'omega':round(g.uniform(0.2,1.4),2),'drive':round(g.uniform(0.2,1.5),2),'lock':g.randint(2,8)}
    d=max(2,int(math.floor(math.sqrt(B))))
    return {'mode':'real','B':B,'proj_seed':g.randint(0,1<<30),'d':d,
            'store':round(g.uniform(0.1,1.5),2),'decay':round(g.uniform(0.0,0.3),2)}

def mutate(gen, g):
    import copy; x=copy.deepcopy(gen)
    if x['mode']=='phase':
        k=g.choice(['proj_seed','spf','omega','drive','lock'])
        if k=='proj_seed': x['proj_seed']=g.randint(0,1<<30)
        elif k=='spf': x['spf']=max(1,min(x['B']//2,x['spf']+g.choice([-1,1])))
        elif k=='omega': x['omega']=round(min(1.6,max(0.1,x['omega']+g.uniform(-0.2,0.2))),2)
        elif k=='drive': x['drive']=round(min(2.0,max(0.1,x['drive']+g.uniform(-0.3,0.3))),2)
        elif k=='lock': x['lock']=max(1,min(10,x['lock']+g.choice([-1,1])))
    else:
        k=g.choice(['proj_seed','store','decay'])
        if k=='proj_seed': x['proj_seed']=g.randint(0,1<<30)
        elif k=='store': x['store']=round(min(2.0,max(0.05,x['store']*g.uniform(0.6,1.6))),2)
        elif k=='decay': x['decay']=round(min(0.6,max(0.0,x['decay']+g.uniform(-0.1,0.1))),2)
    return x

def evolve(mode, B, seed, N=64, gens=50, thresholds=(0.6,0.75,0.9)):
    g=random.Random(seed); pop=[rand_gen(mode,B,g) for _ in range(N)]
    hit={t:None for t in thresholds}; evalcount=0; besttraj=[]
    for gen in range(gens):
        fits=[fitness(x) for x in pop]; evalcount+=N; best=max(fits); besttraj.append(best)
        for t in thresholds:
            if hit[t] is None and best>=t: hit[t]=evalcount
        order=sorted(range(N),key=lambda i:-fits[i]); elite=[pop[i] for i in order[:max(1,N//10)]]
        off=[]
        while len(off)<N-len(elite):
            p=pop[order[g.randrange(min(N//2,N))]] if g.random()<0.7 else g.choice(pop); off.append(mutate(p,g))
        pop=elite+off
    fits=[fitness(x) for x in pop]; bi=int(np.argmax(fits)); bg=pop[bi]
    transfer={K:fitness(bg,K=K,neps=60) for K in (2,3,4,6,8)}
    return {'best':round(max(fits),3),'median':round(float(np.median(fits)),3),'hit':hit,'evals':evalcount,'transfer':transfer}

if __name__=='__main__':
    SEEDS=int(os.environ.get('DA_SEEDS','20')); GENS=int(os.environ.get('DA_GENS','50'))
    print(f"=== Q1 EVOLVABILITY: phase vs real binder, matched STATE budget, {SEEDS} seeds x {GENS} gens ===", flush=True)
    print("  (both evolved from weak random genomes; fitness=binding acc K=2; chance=0.50)", flush=True)
    for B in [16,32,64]:
        for mode in ['phase','real']:
            R=[evolve(mode,B,s,gens=GENS) for s in range(SEEDS)]
            import numpy as _np
            fb=_np.mean([r['best'] for r in R]); fm=_np.mean([r['median'] for r in R])
            reach={t:_np.mean([1 if r['hit'][t] else 0 for r in R]) for t in (0.6,0.75,0.9)}
            e75=[r['hit'][0.75] for r in R if r['hit'][0.75]]; e75m=_np.median(e75) if e75 else None
            tr={K:round(float(_np.mean([r['transfer'][K] for r in R])),2) for K in (2,4,8)}
            print(f"  B={B:2d} {mode:5s}: final_best={fb:.3f} final_med={fm:.3f} | reach%(.6/.75/.9)={reach[0.6]:.2f}/{reach[0.75]:.2f}/{reach[0.9]:.2f} | median_evals_to_.75={e75m} | transfer K2/4/8={tr[2]}/{tr[4]}/{tr[8]}", flush=True)
    print("=== DECISION: Case A (phase<=real discovery AND asymptotic -> close) / B (phase faster, worse) / C (phase faster+transfers) ===", flush=True)
    print("=== DEVACCESS_Q1_DONE ===", flush=True)
