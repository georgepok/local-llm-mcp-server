import os, random, math
os.environ['ME_MODE']='none'
from collections import Counter
import statistics as st
import compchem2 as cc                                    # FROZEN validated harness (do not modify)
TYPES=cc.TYPES

# STATIC_CLOSURE_ENRICHMENT_V1 — Question: does repeated generic environmental cycling ENRICH initially RANDOM
# regional compositions for the FIXED map's reconstructive cycles? Random init independent of map; NO planted pair.
# Track cycle-occupancy + aggregate leave-one-out reconstruction OVER cycles. Controls: local-ancestry, shuffled-
# ancestry, map-shuffled, neutral, no-cycle. Cross-foster + evolved-vs-abundance-matched-random. World seed = unit,
# paired bootstrap 95% CIs. Do NOT add dependencies or seed reciprocal organizations.
def map_cycles(P):                                        # types ON a cycle of the functional graph x->P[x]
    cyc=set()
    for s in TYPES:
        seen={}; x=s; path=[]
        while x not in seen: seen[x]=len(path); path.append(x); x=P[x]
        for c in path[seen[x]:]: cyc.add(c)
    return cyc
def occupancy(reg, cyc):
    tot=sum(reg.values()); return (sum(reg.get(t,0) for t in cyc)/tot) if tot else 0.0
def agg_recon(reg, P, seedv, neutral=False, topM=3):      # aggregate leave-one-out restoration gain over ABUNDANT types
    top=[t for t,_ in Counter(reg).most_common(topM)]
    gs=[cc.ablation_gain(reg,P,t,seedv,steps=300,neutral=neutral)[2] for t in top]
    return st.mean(gs) if gs else 0.0

INFLOW=400.0; GROW=250; RET=0.30
def env_step(regions, P, g, cond, orig_P):
    H=len(regions); W=len(regions[0])
    for y in range(H):
        for x in range(W):
            rng=random.Random(g.random()*1e9)
            regions[y][x],_=cc.react(regions[y][x], INFLOW, P, GROW, rng, neutral=(cond=='neutral'))
    if cond=='nocycle': return P                          # NO bottleneck/dilution — continuous growth only
    # BOTTLENECK + DILUTION
    ret=[[Counter() for _ in range(W)] for _ in range(H)]
    for y in range(H):
        for x in range(W):
            for f,c in regions[y][x].items():
                k=sum(1 for _ in range(c) if g.random()<RET)
                if k>0: ret[y][x][f]=k
    if cond=='shuffle':
        cells=[ret[y][x] for y in range(H) for x in range(W)]; g.shuffle(cells); i=0
        for y in range(H):
            for x in range(W): regions[y][x]=Counter(cells[i]); i+=1
    else:
        for y in range(H):
            for x in range(W): regions[y][x]=Counter(ret[y][x])
    if cond=='mapshuffle': return cc.make_pmap(int(g.random()*1e9))   # re-randomize map each cycle (cycles change)
    return P

def run_world(seedv, cond, H=4,W=4,cycles=20):
    g=random.Random(seedv*131+7); P0=cc.make_pmap(3000+seedv); cyc=map_cycles(P0)   # map independent of init
    regions=[[Counter() for _ in range(W)] for _ in range(H)]
    for y in range(H):                                    # RANDOM initial composition (uniform types, independent of map)
        for x in range(W):
            for _ in range(16): regions[y][x][g.choice(TYPES)]+=1
    P=P0; occ=[]; rec=[]
    for c in range(cycles):
        occ.append(st.mean(occupancy(regions[y][x],cyc) for y in range(H) for x in range(W)))
        if c in (0,cycles-1):
            samp=[(y,x) for y in range(H) for x in range(W)][:6]
            rec.append(st.mean(agg_recon(regions[y][x],P0,seedv*100+y*4+x,neutral=(cond=='neutral')) for (y,x) in samp))
        P=env_step(regions,P,g,cond,P0)
    return dict(occ0=occ[0], occ_end=occ[-1], d_occ=occ[-1]-occ[0], recon0=rec[0], recon_end=rec[-1], d_recon=rec[-1]-rec[0],
                regions=regions, P0=P0, cyc=cyc)

def bootstrap_ci(xs, B=2000):
    n=len(xs); rng=random.Random(0); ms=[]
    for _ in range(B): ms.append(st.mean(rng.choice(xs) for _ in range(n)))
    ms.sort(); return st.mean(xs), ms[int(0.025*B)], ms[int(0.975*B)]

if __name__=='__main__':
    SEEDS=list(range(12))
    print("=== STATIC_CLOSURE_ENRICHMENT_V1 — does cycling enrich random compositions for the map's cycles? (12 world seeds) ===", flush=True)
    print(f"  {'condition':16s} {'occ0':>5s} {'occ_end':>7s} {'d_occ [95% CI]':>22s}   {'d_recon [95% CI]':>22s}", flush=True)
    store={}
    for cond in ['coded_local','coded_shuffle','mapshuffle','neutral','nocycle']:
        rs=[run_world(s,cond) for s in SEEDS]; store[cond]=rs
        do=[r['d_occ'] for r in rs]; dr=[r['d_recon'] for r in rs]
        mo,lo,ho=bootstrap_ci(do); mr,lr,hr=bootstrap_ci(dr)
        o0=st.mean(r['occ0'] for r in rs); oe=st.mean(r['occ_end'] for r in rs)
        print(f"  {cond:16s} {o0:>5.2f} {oe:>7.2f}   {mo:+.3f} [{lo:+.3f},{ho:+.3f}]   {mr:+.3f} [{lr:+.3f},{hr:+.3f}]", flush=True)
    # CROSS-FOSTER: evolved-in-own-map reconstruction vs same composition under a FOREIGN map
    own=[]; foreign=[]
    for i,s in enumerate(SEEDS):
        r=store['coded_local'][i]; reg=r['regions'][0][0]
        own.append(agg_recon(reg, r['P0'], s*100))
        foreignP=store['coded_local'][(i+1)%len(SEEDS)]['P0']
        foreign.append(agg_recon(reg, foreignP, s*100))
    mo2,lo2,ho2=bootstrap_ci([o-f for o,f in zip(own,foreign)])
    print(f"  CROSS-FOSTER: recon(own map)={st.mean(own):+.3f} vs recon(foreign map)={st.mean(foreign):+.3f} | paired diff {mo2:+.3f} [{lo2:+.3f},{ho2:+.3f}]", flush=True)
    # EVOLVED vs ABUNDANCE-MATCHED RANDOM composition
    ev=[]; rnd=[]
    for i,s in enumerate(SEEDS):
        r=store['coded_local'][i]; reg=r['regions'][0][0]; P0=r['P0']
        ev.append(agg_recon(reg,P0,s*100))
        tot=sum(reg.values()); rr=Counter()                                  # matched mass, RANDOM identities
        gg=random.Random(s);
        for _ in range(tot): rr[gg.choice(TYPES)]+=1
        rnd.append(agg_recon(rr,P0,s*100))
    me,le,he=bootstrap_ci([e-x for e,x in zip(ev,rnd)])
    print(f"  EVOLVED vs MATCHED-RANDOM: recon(evolved)={st.mean(ev):+.3f} vs recon(random)={st.mean(rnd):+.3f} | paired diff {me:+.3f} [{le:+.3f},{he:+.3f}]", flush=True)
    print("  SUCCESS = occupancy & reconstruction RISE over history ONLY for coded_local (CI>0) not controls; cross-foster own>foreign; evolved>random.", flush=True)
    print("=== STATICENRICH_DONE ===", flush=True)
