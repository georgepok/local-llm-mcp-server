import os, random, math
os.environ['ME_MODE']='none'
from collections import Counter
import statistics as st

# ENDOGENOUS_REACTION_TOPOLOGY_V1 — reaction graph is INITIALLY ACYCLIC (a chain 0->1->...->N-1->W; no cycle => no
# static closure possible at t=0) and MUTABLE: a reaction X->Y can endogenously produce a LINKER ('L',Y,X) that ADDS
# a back-edge Y->X to the LOCAL effective graph. A back-edge closes a cycle; the cycle survives ONLY if it keeps
# producing the linker that enables it (self-referential endogenous construction). DECISIVE: closure must appear
# ONLY with linkers-ON, NOT in frozen-acyclic OFF; and ablating linker production must COLLAPSE it. No hand-built cycle.
N=12; W='W'
COST=1.0; DEATH=0.12; SRC=0.05; LAMBDA=0.08; LINK_THRESH=2; MUT=0.0
def base_prod(X): return (X+1) if X<N-1 else W                        # acyclic base chain (edges only increase index)

def react(reg, P, steps, rng, linkers_on, neutral=False, ablate_link=False):
    reg=Counter(reg)
    for _ in range(steps):
        if P<COST or not reg: break
        if rng.random()<SRC: reg[0]+=1; P-=COST; continue            # primitives -> molecule 0 (chain source)
        mols=[(t,c) for t,c in reg.items() if isinstance(t,int)]
        if not mols: break
        tot=sum(c for _,c in mols); r=rng.random()*tot; c=0; X=None
        for t,w in mols:
            c+=w
            if r<=c: X=t; break
        if X is None: break
        prods=[base_prod(X)]
        if not ablate_link:                                          # effective back-edges from present linkers
            for i in range(N):
                if reg.get(('L',X,i),0)>LINK_THRESH: prods.append(i)
        Y=rng.choice(range(N)) if neutral else rng.choice(prods)     # neutral null: random molecule product (matched rate)
        reg[Y]+=1; P-=COST
        if linkers_on and not ablate_link and isinstance(Y,int) and rng.random()<LAMBDA:
            reg[('L',Y,X)] += 1                                      # ENDOGENOUS topology mutation: reverse linker Y->X
        if rng.random()<DEATH:                                       # matched death (molecules AND linkers decay)
            k=rng.choice(list(reg)); reg[k]-=1
            if reg[k]<=0: del reg[k]
    return reg,P

def eff_edges(reg):
    ab=[m for m in range(N) if reg.get(m,0)>2]
    E={m:set() for m in ab}
    for m in ab:
        bp=base_prod(m)
        if isinstance(bp,int) and bp in E: E[m].add(bp)
        for i in ab:
            if reg.get(('L',m,i),0)>LINK_THRESH: E[m].add(i)
    return ab,E
def cycle_nodes(reg):                                                # molecules that lie on a directed cycle of the effective graph
    ab,E=eff_edges(reg); onc=set()
    for s in ab:
        stack=[(s,[s])];
        while stack:
            u,path=stack.pop()
            for v in E.get(u,()):
                if v==s: onc.update(path)
                elif v not in path and len(path)<N: stack.append((v,path+[v]))
    return onc
def closure_occ(reg):
    onc=cycle_nodes(reg); mm=sum(reg.get(m,0) for m in range(N)) or 1
    return sum(reg.get(m,0) for m in onc)/mm, len(onc)>0

def restoration_gain(pre, seedv, linkers_on, ablate_link=False):
    onc=cycle_nodes(pre)
    target=max(onc,key=lambda m:pre.get(m,0)) if onc else max(range(N),key=lambda m:pre.get(m,0))
    def vec(reg): return {m:reg.get(m,0) for m in range(N)}
    def cos(a,b):
        na=math.sqrt(sum(v*v for v in a.values()))or 1; nb=math.sqrt(sum(v*v for v in b.values()))or 1
        return sum(a.get(k,0)*b.get(k,0) for k in set(a)|set(b))/(na*nb)
    post0=Counter(pre); post0.pop(target,None); S0=cos(vec(pre),vec(post0))
    rng=random.Random(seedv*7+3); post1,_=react(post0,2000.0,300,rng,linkers_on,ablate_link=ablate_link); S1=cos(vec(pre),vec(post1))
    rec=post1.get(target,0)/(sum(post1.get(m,0) for m in range(N))or 1)
    return S1-S0, rec

def warm(seedv, linkers_on, steps=4000, neutral=False):
    g=random.Random(seedv); reg=Counter({0:20})
    reg,_=react(reg,20000.0,steps,g,linkers_on,neutral=neutral); return reg

def ci(xs,B=2000):
    r=random.Random(0); ms=sorted(st.mean(r.choice(xs) for _ in range(len(xs))) for _ in range(B))
    return st.mean(xs), ms[int(0.025*B)], ms[int(0.975*B)]

if __name__=='__main__':
    SEEDS=list(range(16))
    print("=== ENDOGENOUS_REACTION_TOPOLOGY_V1 — can closure be CONSTRUCTED from an initially ACYCLIC graph? ===", flush=True)
    ON=[warm(s,True) for s in SEEDS]; OFF=[warm(s,False) for s in SEEDS]; NE=[warm(s,True,neutral=True) for s in SEEDS]
    def occs(regs): return [closure_occ(r) for r in regs]
    on_o=occs(ON); off_o=occs(OFF); ne_o=occs(NE)
    def rate(o): return st.mean(1 if c else 0 for _,c in o)
    def mocc(o): return ci([v for v,_ in o])
    mo,lo,ho=mocc(on_o); mf,lf,hf=mocc(off_o); mn,ln,hn=mocc(ne_o)
    print(f"  closure OCCUPANCY (mass on an effective cycle):", flush=True)
    print(f"    linkers-ON:        {mo:.3f} [{lo:.3f},{ho:.3f}]   has-cycle rate {rate(on_o):.2f}", flush=True)
    print(f"    frozen-acyclic OFF:{mf:.3f} [{lf:.3f},{hf:.3f}]   has-cycle rate {rate(off_o):.2f}  (must be ~0 -> acyclic)", flush=True)
    print(f"    neutral null:      {mn:.3f} [{ln:.3f},{hn:.3f}]   has-cycle rate {rate(ne_o):.2f}", flush=True)
    # RESTORATION GAIN (reconstruction of a deleted cycle-molecule via the CONSTRUCTED back-edge)
    rgON=[restoration_gain(ON[i],SEEDS[i],True)[0] for i in range(len(SEEDS))]
    rgOFF=[restoration_gain(OFF[i],SEEDS[i],False)[0] for i in range(len(SEEDS))]
    mg,lg,hg=ci([a-b for a,b in zip(rgON,rgOFF)])
    print(f"  restoration gain G: ON {st.mean(rgON):+.3f} vs OFF {st.mean(rgOFF):+.3f} | paired diff {mg:+.3f} [{lg:.3f},{hg:.3f}]", flush=True)
    # CAUSAL: ablate linker PRODUCTION after closure forms -> does closure COLLAPSE?
    before=[]; after=[]
    for i,s in enumerate(SEEDS):
        reg=warm(s,True); b,_=closure_occ(reg); g=random.Random(s*3+1)
        reg2,_=react(reg,20000.0,3000,g,linkers_on=False,ablate_link=True)          # linkers can no longer be produced/used
        a,_=closure_occ(reg2); before.append(b); after.append(a)
    md,ld,hd=ci([b-a for b,a in zip(before,after)])
    print(f"  CAUSAL linker-ablation: closure_occ before {st.mean(before):.3f} -> after {st.mean(after):.3f} | paired DROP {md:+.3f} [{ld:.3f},{hd:.3f}]", flush=True)
    success = (mo>0.1) and (mf<0.05) and (lg>0) and (ld>0)
    print(f"  ENDOGENOUS CONSTRUCTION {'SUPPORTED' if success else 'NOT supported'}: closure ON>>OFF-acyclic, ON-restoration>OFF, ablation collapses closure", flush=True)
    print("=== ENDOTOPO_DONE ===", flush=True)
