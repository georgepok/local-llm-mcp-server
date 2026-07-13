import os, random, math
os.environ['ME_MODE']='none'
from collections import Counter
import statistics as st

# PRE_INDIVIDUAL rebuilt assay (corrects retracted compchem.py). SINGLE-catalyst cross-catalytic chemistry, NO
# self-copy: catalyst X + primitives -> product P[X] with P[X]!=X (no type reproduces itself; a deleted type is
# regenerated ONLY by its producer, a DIFFERENT type). Reconstruction tested by STRUCTURED ABLATION (delete a type,
# measure recovery + restoration gain G=S1-S0), NOT random-subset retention (cosine is scale-invariant -> no damage).
# Mandatory positive control: hand-built reciprocal 2-cycle P[A]=B,P[B]=A the harness MUST detect, else STOP.
K=4; L=3
TYPES=[(a,b,c) for a in range(K) for b in range(K) for c in range(K)]
def make_pmap(seed, plant=None):
    g=random.Random(seed); P={}
    for x in TYPES:
        z=g.choice(TYPES)
        while z==x: z=g.choice(TYPES)                    # no fixed point (no self-reproduction)
        P[x]=z
    if plant: A,B=plant; P[A]=B; P[B]=A                  # reciprocal 2-cycle
    return P
def sim(a,b):
    ka=set(a)|set(b); na=math.sqrt(sum(v*v for v in a.values()))or 1; nb=math.sqrt(sum(v*v for v in b.values()))or 1
    return sum(a.get(k,0)*b.get(k,0) for k in ka)/(na*nb)
COST=1.0; MUT=0.02; DEATH=0.15
def react(reg, P, Pmap, steps, rng, neutral=False):
    reg=Counter(reg)
    for _ in range(steps):
        if P<COST or not reg: break
        tot=sum(reg.values()); r=rng.random()*tot; c=0; X=None
        for t,w in reg.items():
            c+=w
            if r<=c: X=t; break
        if X is None: break
        Z=rng.choice(TYPES) if neutral else Pmap[X]      # neutral null: random product (opcode-blind), matched rate
        if rng.random()<MUT: Z=rng.choice(TYPES)
        reg[Z]+=1; P-=COST
        if rng.random()<DEATH:                            # matched death (birth~death, pop bounded)
            tot2=sum(reg.values()); r2=rng.random()*tot2; c2=0
            for t,w in list(reg.items()):
                c2+=w
                if r2<=c2:
                    reg[t]-=1
                    if reg[t]<=0: del reg[t]
                    break
    return reg,P
def warm(Pmap, seedv, seed_types, per=20, steps=1200, neutral=False):
    g=random.Random(seedv); reg=Counter()
    for t in seed_types:
        for _ in range(per): reg[t]+=1
    reg,_=react(reg,4000.0,Pmap,steps,g,neutral); return reg
def frac(reg,t): s=sum(reg.values()); return reg.get(t,0)/s if s else 0.0

def ablation_gain(pre, Pmap, target, seedv, steps=500, neutral=False):
    """Structured disruption: delete `target` type; run dynamics; measure S0/S1/G + target recovery.
       Cloned RNG so arms are comparable & order-invariant."""
    post0=Counter(pre); post0.pop(target,None); S0=sim(pre,post0)
    rng=random.Random(seedv*7+13)                        # cloned dynamics RNG (identical across arms)
    post1,_=react(post0,3000.0,Pmap,steps,rng,neutral); S1=sim(pre,post1)
    return S0,S1,S1-S0, frac(post1,target)

if __name__=='__main__':
    print("=== PRE_INDIVIDUAL rebuilt assay — POSITIVE CONTROL gate (single-catalyst; structured ablation) ===", flush=True)
    A,B=(0,0,0),(3,3,3)
    GA=[]; RA=[]; GC=[]; RC=[]
    for sv in range(20):
        Pmap=make_pmap(1000+sv, plant=(A,B))
        reg=warm(Pmap, sv, [A,B])
        # need a matched non-cycle control type C present with abundance similar to A
        cand=[t for t in reg if t not in (A,B)]
        C=max(cand,key=lambda t:reg[t]) if cand else None
        s0,s1,g,rec = ablation_gain(reg,Pmap,A,sv);  GA.append(g); RA.append(rec)               # delete A: does B->A regenerate?
        if C is not None:
            s0c,s1c,gc,recc = ablation_gain(reg,Pmap,C,sv); GC.append(gc); RC.append(recc)        # delete a non-cycle type (control)
    def ci(xs): m=st.mean(xs); s=(st.stdev(xs)/math.sqrt(len(xs))) if len(xs)>1 else 0; return m,s
    ga,sga=ci(GA); ra,sra=ci(RA); gc,sgc=ci(GC); rc,src=ci(RC)
    print(f"  [POS-CTRL] delete reciprocal type A -> restoration gain G = {ga:+.3f} ± {sga:.3f}; A-fraction regenerated = {ra:.3f} ± {sra:.3f}", flush=True)
    print(f"  [POS-CTRL] delete non-cycle type C -> restoration gain G = {gc:+.3f} ± {sgc:.3f}; C-fraction regenerated = {rc:.3f} ± {src:.3f}", flush=True)
    ok = (ga>0.02) and (ra>0.05) and (ra > rc+0.03)
    print(f"  POSITIVE CONTROL {'PASSES' if ok else 'FAILS'}: G_A>0={ga>0.02}, A regenerates(B->A)={ra>0.05}, selective A>>C={ra>rc+0.03}", flush=True)
    if not ok: print("  !!! HARNESS INVALID — STOP before any C1-vs-null comparison", flush=True)
    else: print("  >>> harness valid — proceed to C1 (random-seed) vs matched-neutral null, >=20 seeds, structured ablation", flush=True)
    print("=== POSCTRL_DONE ===", flush=True)

# ---------------- MAIN: C1 (random coded worlds) vs matched-neutral null (>=20 world seeds, paired) ----------------
def world_best_gain(seedv, neutral, M=6):
    Pmap=make_pmap(2000+seedv)                                          # RANDOM chemistry, NO planted organism
    reg=warm(Pmap, seedv, [random.Random(seedv).choice(TYPES) for _ in range(14)], neutral=neutral)
    top=[t for t,_ in Counter(reg).most_common(M)]
    gains=[]; recs=[]
    for t in top:
        s0,s1,g,rec=ablation_gain(reg,Pmap,t,seedv,neutral=neutral); gains.append(g); recs.append((t,rec,g))
    if not gains: return 0.0,0.0,0.0
    best=max(recs,key=lambda r:r[1]); T,recT,gT=best
    # abundance-matched selective control: remove an OTHER top type of similar abundance, measure ITS regeneration
    others=[t for t in top if t!=T]
    matched=min(others,key=lambda t:abs(reg[t]-reg[T])) if others else None
    selG=0.0
    if matched is not None:
        _,_,gm,recm=ablation_gain(reg,Pmap,matched,seedv,neutral=neutral); selG=recT-recm
    return max(gains), st.mean(gains), selG

if __name__=='__main__' and os.environ.get('MAIN'):
    print("=== MAIN: C1 random coded worlds vs matched-neutral null (20 world seeds, structured ablation) ===", flush=True)
    cb=[]; cm=[]; cs=[]; nb=[]; nm=[]; ns=[]
    for sv in range(20):
        b,m,s=world_best_gain(sv,neutral=False); cb.append(b); cm.append(m); cs.append(s)
        b2,m2,s2=world_best_gain(sv,neutral=True); nb.append(b2); nm.append(m2); ns.append(s2)
    def ci(xs): mu=st.mean(xs); se=(st.stdev(xs)/math.sqrt(len(xs))) if len(xs)>1 else 0; return mu,se
    def paired(a,b): d=[x-y for x,y in zip(a,b)]; return ci(d)
    mcb,scb=ci(cb); mnb,snb=ci(nb); pdb,psb=paired(cb,nb)
    mcs,scs=ci(cs); mns,sns=ci(ns); pds,pss=paired(cs,ns)
    print(f"  best restoration gain G:  coded {mcb:+.3f}±{scb:.3f} | neutral {mnb:+.3f}±{snb:.3f} | paired diff {pdb:+.3f}±{psb:.3f}", flush=True)
    print(f"  selective ablation (regen T* minus abundance-matched): coded {mcs:+.3f}±{scs:.3f} | neutral {mns:+.3f}±{sns:.3f} | paired {pds:+.3f}±{pss:.3f}", flush=True)
    win = pdb-1.96*psb>0
    print(f"  PRE-REGISTERED: coded>neutral restoration gain (paired CI excludes 0)? {win}; selective>neutral? {pds-1.96*pss>0}", flush=True)
    print("  NOTE: coded 'closure' = producer-cycles of the FIXED reaction map (static chemistry structure), NOT yet accumulation of closure OVER cycles.", flush=True)
    print("=== MAIN_DONE ===", flush=True)
