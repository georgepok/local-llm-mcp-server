import os, random, math
os.environ['ME_MODE']='none'
import statistics as st
import compchem2 as cc                                    # FROZEN harness under audit

# PART 1 — independent audit of the frozen compchem2 supportability/reconstruction harness.
def audit():
    print("=== PART-1 AUDIT of frozen compchem2.py ===", flush=True)
    R={}
    # 1 no self-copy: P[X]!=X for all X, across maps
    ok=True
    for s in range(30):
        P=cc.make_pmap(1000+s)
        if any(P[x]==x for x in cc.TYPES): ok=False; break
    R['no self-copy (P[X]!=X all X, 30 maps)']=ok
    # 2 planted reciprocal positive control still detectable
    A,B=(0,0,0),(3,3,3); GA=[]; RA=[]; RC=[]
    for s in range(20):
        P=cc.make_pmap(1000+s, plant=(A,B)); reg=cc.warm(P,s,[A,B])
        cand=[t for t in reg if t not in (A,B)]; C=max(cand,key=lambda t:reg[t]) if cand else None
        _,_,g,ra=cc.ablation_gain(reg,P,A,s); GA.append(g); RA.append(ra)
        if C is not None: _,_,_,rcc=cc.ablation_gain(reg,P,C,s); RC.append(rcc)
    R['positive control detectable (G_A>0 & A-regen>>C-regen)']=(st.mean(GA)>0.02 and st.mean(RA)>st.mean(RC)+0.03)
    # 3 cloned-RNG order-invariance: run two ablation arms in both orders -> identical results
    P=cc.make_pmap(2000); reg=cc.warm(P,0,[cc.TYPES[i] for i in range(14)])
    top=[t for t,_ in __import__('collections').Counter(reg).most_common(2)]
    r1a=cc.ablation_gain(reg,P,top[0],0); r1b=cc.ablation_gain(reg,P,top[1],0)
    r2b=cc.ablation_gain(reg,P,top[1],0); r2a=cc.ablation_gain(reg,P,top[0],0)
    R['ablation order-invariant (cloned RNG)']=(r1a==r2a and r1b==r2b)
    # 4 paired bootstrap 95% CI recompute (compchem2 MAIN reported SEM; here give the actual 95% CI, labeled)
    cb=[]; nb=[]
    for s in range(20):
        cb.append(cc_world_best(s,False)); nb.append(cc_world_best(s,True))
    d=[a-b for a,b in zip(cb,nb)]
    rng=random.Random(0); ms=sorted(st.mean(rng.choice(d) for _ in range(len(d))) for _ in range(3000))
    mean_d=st.mean(d); sem=st.stdev(d)/math.sqrt(len(d)); lo,hi=ms[75],ms[2924]
    R['coded>neutral (paired bootstrap 95% CI excludes 0)']=(lo>0)
    print(f"  coded-minus-neutral best-G: mean {mean_d:+.3f}  SEM(labeled SEM) {sem:.3f}  bootstrap95%CI [{lo:+.3f},{hi:+.3f}]", flush=True)
    # 5 neutral recovery baseline reported: yes (nb above). all abundant types evaluable, not best-only:
    reg=cc.warm(cc.make_pmap(2000),0,[cc.TYPES[i] for i in range(14)])
    from collections import Counter
    allg=[cc.ablation_gain(reg,cc.make_pmap(2000),t,0)[2] for t,_ in Counter(reg).most_common(8)]
    R['all abundant types evaluable (not best-only)']=(len(allg)>=5)
    for k,v in R.items(): print(f"  [{'PASS' if v else 'FAIL'}] {k}", flush=True)
    print(f"=== AUDIT {sum(R.values())}/{len(R)} checks pass ===", flush=True)
    print("=== AUDITFREEZE_DONE ===", flush=True)

def cc_world_best(seedv, neutral, M=6):
    from collections import Counter
    P=cc.make_pmap(2000+seedv); reg=cc.warm(P,seedv,[random.Random(seedv).choice(cc.TYPES) for _ in range(14)],neutral=neutral)
    top=[t for t,_ in Counter(reg).most_common(M)]
    return max((cc.ablation_gain(reg,P,t,seedv,neutral=neutral)[2] for t in top), default=0.0)

if __name__=='__main__': audit()
