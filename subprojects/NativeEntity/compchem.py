import os, random, math
os.environ['ME_MODE']='none'
from collections import defaultdict

# PRE_INDIVIDUAL_COMPOSITIONAL_EVOLUTION_V1 — regions are FRAGMENT MIXTURES (no individual objects/SPAWN/lineage,
# Part 1 disabled). Population-level reaction (generic, token-driven): copiers produce mutated copies of present
# templates; MATCH-selectivity creates ENDOGENOUS dependency (a copier that only reproduces symbol-bearing
# fragments depends on them). Generic environmental cycle: concentrate(react) -> dilute(stochastic partial
# retention) -> redistribute. Measure whether region COMPOSITION reconstructs across disruption cycles via its
# INTERNAL network vs null controls (complete-mixing / shuffled-inheritance / no-reaction=pure trapping). Part 8:
# seed only short generic fragments, NEVER a complete reciprocal set.
SYM='abcd'; OPS='CMTV'; TOKENS=list(SYM+OPS)          # C=copy M=match T=transfer/retain V=activate ; a-d = symbols/data
def randfrag(g): return tuple(g.choice(TOKENS) for _ in range(g.randint(2,5)))
def req_symbol(F):                                    # if fragment has M followed by a symbol -> copier is SELECTIVE for that symbol
    for i,t in enumerate(F):
        if t=='M' and i+1<len(F) and F[i+1] in SYM: return F[i+1]
    return None
def mutate(F,g,mr):
    F=list(F)
    for i in range(len(F)):
        if g.random()<mr: F[i]=g.choice(TOKENS)
    if g.random()<mr and len(F)<6: F.insert(g.randrange(len(F)+1),g.choice(TOKENS))
    if g.random()<mr and len(F)>2: del F[g.randrange(len(F))]
    return tuple(F)

BASE_ACT=0.25; ACT_GAIN=0.6; COST=0.10; INFLOW=60.0; DECAY_FR=0.02; RET=0.35; REACT_STEPS=200
def wsample(items, g):                                 # weighted sample (item,count) pairs
    tot=sum(c for _,c in items); r=g.random()*tot; c=0
    for it,w in items:
        c+=w
        if r<=c: return it
    return items[-1][0]

def react(reg, R, steps, g, mr, do_react=True):
    if not do_react: return R
    tot=sum(reg.values())
    if tot==0: return R
    nact=sum(c for f,c in reg.items() if 'V' in f)
    activity=min(1.0, BASE_ACT + ACT_GAIN*nact/max(tot,1))
    items=list(reg.items())
    for _ in range(steps):
        if R<=COST or not items: break
        F=wsample(items,g)
        if g.random()>activity: continue
        if 'C' in F:
            req=req_symbol(F)
            cands=[(t,c) for t,c in reg.items() if (req is None or req in t)] if req else items
            if not cands: continue
            Tp=mutate(wsample(cands,g),g,mr); reg[Tp]=reg.get(Tp,0)+1; R-=COST
            if g.random()<DECAY_FR*2:                  # occasional degradation keeps it non-trivial
                d=wsample(items,g); reg[d]-=1
                if reg[d]<=0: del reg[d]
            items=list(reg.items())
    return R

def comp_vec(reg):
    tot=sum(reg.values()) or 1
    return {f:c/tot for f,c in reg.items()}
def cos(a,b):
    keys=set(a)|set(b); na=math.sqrt(sum(v*v for v in a.values()))or 1; nb=math.sqrt(sum(v*v for v in b.values()))or 1
    return sum(a.get(k,0)*b.get(k,0) for k in keys)/(na*nb)

def cycle(grid, R, g, mr, cond):
    H=len(grid); W=len(grid[0]); parents=[[comp_vec(grid[y][x]) for x in range(W)] for y in range(H)]
    for y in range(H):
        for x in range(W): R[y][x]=react(grid[y][x], R[y][x]+INFLOW, REACT_STEPS, g, mr, do_react=(cond!='noreact'))
    # DILUTE: stochastic partial retention (low -> reconstruction must regrow, not trap)
    retained=[[defaultdict(int) for _ in range(W)] for _ in range(H)]
    for y in range(H):
        for x in range(W):
            for f,c in grid[y][x].items():
                k=sum(1 for _ in range(c) if g.random()<RET)
                if k>0: retained[y][x][f]+=k
    # REDISTRIBUTE
    if cond=='mixing':                                  # C3: pool everything, redistribute uniformly (no local heredity)
        pool=defaultdict(int)
        for y in range(H):
            for x in range(W):
                for f,c in retained[y][x].items(): pool[f]+=c
        pk=list(pool.items())
        for y in range(H):
            for x in range(W):
                grid[y][x]=defaultdict(int)
                for _ in range(max(1,sum(pool.values())//(H*W))): grid[y][x][wsample(pk,g)]+=1 if pk else 0
    elif cond=='shuffle':                                # shuffled region inheritance (break region lineage)
        cells=[retained[y][x] for y in range(H) for x in range(W)]; g.shuffle(cells)
        i=0
        for y in range(H):
            for x in range(W): grid[y][x]=defaultdict(int,cells[i]); i+=1
    else:                                                # C1 (and noreact): descendant = region's OWN retained subset
        for y in range(H):
            for x in range(W): grid[y][x]=defaultdict(int,retained[y][x])
    return parents

def seed(H,W,g,perc=14):
    grid=[[defaultdict(int) for _ in range(W)] for _ in range(H)]
    for y in range(H):
        for x in range(W):
            for _ in range(perc): grid[y][x][randfrag(g)]+=1
    R=[[INFLOW for _ in range(W)] for _ in range(H)]
    return grid,R

def run(cond, seedv, H=6,W=6,cycles=40,mr=0.03):
    g=random.Random(seedv); grid,R=seed(H,W,g)
    her=[]  # EXCESS LOCAL heredity: self-lineage similarity MINUS cross-region similarity (homogenization -> 0)
    prev_parent=None; cellids=[(y,x) for y in range(H) for x in range(W)]
    for c in range(cycles):
        parents=cycle(grid,R,g,mr,cond)
        if prev_parent is not None:
            ex=[]
            for (y,x) in cellids:
                child=comp_vec(grid[y][x]); self_s=cos(prev_parent[y][x], child)
                others=[(oy,ox) for (oy,ox) in cellids if (oy,ox)!=(y,x)]; g.shuffle(others)
                cross=[cos(prev_parent[oy][ox], child) for (oy,ox) in others[:5]]
                ex.append(self_s - (sum(cross)/len(cross) if cross else 0))
            her.append(sum(ex)/len(ex))
        prev_parent=parents
    alive=sum(1 for y in range(H) for x in range(W) if sum(grid[y][x].values())>0)
    types=len(set(f for y in range(H) for x in range(W) for f in grid[y][x]))
    return dict(cond=cond, heredity_mid=round(sum(her[len(her)//2:])/max(1,len(her)-len(her)//2),3),
                heredity_late=round(sum(her[-8:])/max(1,min(8,len(her))),3), alive_regions=alive, types=types)

if __name__=='__main__':
    print("=== PRE_INDIVIDUAL_COMPOSITIONAL_EVOLUTION_V1 — compositional heredity vs null controls ===", flush=True)
    print("  (regions = fragment mixtures; no individual reproduction; low retention -> reconstruction must REGROW composition)", flush=True)
    print(f"  {'condition':28s} {'heredity_mid':>12s} {'heredity_late':>13s} {'alive_reg':>9s} {'types':>6s}", flush=True)
    for cond,label in [('C1','C1 compositional-cycling'),('mixing','C3 complete-mixing'),('shuffle','shuffled-inheritance'),('noreact','no-reaction (pure trapping)')]:
        rs=[run(cond,s) for s in (0,1)]
        import statistics as st
        hm=st.mean(r['heredity_mid'] for r in rs); hl=st.mean(r['heredity_late'] for r in rs)
        ar=st.mean(r['alive_regions'] for r in rs); ty=st.mean(r['types'] for r in rs)
        print(f"  {label:28s} {hm:>12.3f} {hl:>13.3f} {ar:>9.1f} {ty:>6.0f}", flush=True)
    print("=== C1 heredity >> mixing/shuffle/noreact => compositional heredity from INTERNAL dynamics (not trapping/physics) ===", flush=True)
    print("=== COMPCHEM_DONE ===", flush=True)

# ---------------- CAUSAL VALIDATION (Part 11 / Milestone 1): does reconstruction depend on the INTERNAL network? ----------------
def causal_test(seedv=0, H=6, W=6, warm=30, mr=0.03):
    g=random.Random(seedv); grid,R=seed(H,W,g)
    for _ in range(warm): cycle(grid,R,g,mr,'C1')                          # reach quasi-stable diverse lineages
    regs=[(y,x) for y in range(H) for x in range(W) if sum(grid[y][x].values())>=6]
    if not regs: return None
    def reconstruct(reg0, R0, remove_fn):
        reg=defaultdict(int, {f:c for f,c in reg0.items()}); Rr=R0
        pre=comp_vec(reg)
        # DISRUPT: retain stochastic subset, then remove per intervention, then reconstruct via reaction over a few cycles
        ret=defaultdict(int)
        for f,c in reg.items():
            k=sum(1 for _ in range(c) if g.random()<RET)
            if k>0: ret[f]+=k
        remove_fn(ret)
        for _ in range(3): Rr=react(ret,Rr+INFLOW,REACT_STEPS,g,mr)         # region regrows from the (intervened) retained seed
        return cos(pre, comp_vec(ret))
    rec_none=[]; rec_copier=[]; rec_rand=[]; rec_scram=[]
    for (y,x) in regs[:8]:
        reg0=dict(grid[y][x]); R0=R[y][x]
        rec_none.append(reconstruct(reg0,R0,lambda r: None))                                   # no removal (baseline reconstruction)
        rec_copier.append(reconstruct(reg0,R0,lambda r: [r.pop(f) for f in list(r) if 'C' in f]))  # remove COPIER class
        def rmrand(r):
            fs=list(r); g.shuffle(fs); nrm=sum(1 for f in fs if 'C' in f)                       # match count of copier removal
            for f in fs[:nrm]: r.pop(f)
        rec_rand.append(reconstruct(reg0,R0,rmrand))                                            # remove equal # of RANDOM fragments (control)
        def scram(r):
            tot=sum(r.values()); r.clear()
            for _ in range(tot): r[randfrag(g)]+=1                                              # preserve count, randomize identity
        rec_scram.append(reconstruct(reg0,R0,scram))
    import statistics as st
    return dict(n=len(rec_none), baseline=round(st.mean(rec_none),3), remove_copier=round(st.mean(rec_copier),3),
                remove_random=round(st.mean(rec_rand),3), scramble=round(st.mean(rec_scram),3))

if __name__=='__main__' and os.environ.get('CAUSAL'):
    print("=== CAUSAL VALIDATION (Milestone 1): does composition reconstruction depend on the COPIER class / specific identities? ===", flush=True)
    import statistics as st
    rs=[causal_test(s) for s in (0,1,2)]; rs=[r for r in rs if r]
    for k in ('baseline','remove_copier','remove_random','scramble'):
        print(f"  {k:16s}: reconstruction-fidelity = {round(st.mean(r[k] for r in rs),3)}", flush=True)
    print("  Milestone-1 signature: remove_copier << baseline AND << remove_random (copier class causally needed);", flush=True)
    print("                          scramble << baseline (specific fragment identities matter, not just count)", flush=True)
    print("=== CAUSAL_DONE ===", flush=True)
