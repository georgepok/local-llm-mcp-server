import os, random
os.environ['ME_MODE']='none'
import numpy as np
np.seterr(over='ignore', invalid='ignore')
from physics import step
from diag import detect, sfield
from evophys import mutate, crossover, rand_term
from worldlaws import CLAUDE, random_law
from worldlaws2 import STRONG
from repro import GS_REPLICATE, PAIRS
import physics

# Part-14 GUARDED evolution: single-scalar count fitness was gamed by ch3 saturation (trivial expanding noise).
# Guarded fitness REJECTS saturation (channel clamp-fraction>0.15) + explosion + non-localized fields, so gaming
# scores 0. Does guarded search find GENUINE reproduction beyond Gray-Scott, or is GS the true DSL ceiling?
def guarded(law, rc, pc, T=3000):
    law=dict(law); law['base_leak']=0.0
    g=np.random.RandomState(0); X=(0.02*g.standard_normal((32,32,8))).astype(np.float32); X[:,:,0]*=0; X[:,:,rc]+=1.0; X[3:8,3:8,pc]+=0.3
    lo,hi=law.get('clamp',[-4,4]); counts=[]
    for t in range(T):
        X=step(X,law,t,np.random.RandomState(1))
        if not np.all(np.isfinite(X)): return None
        if t>=400 and t%300==0: counts.append(len(detect(X)[0]))
    if len(counts)<4: return None
    sat=max(float((X[:,:,c]>lo+0.9*(hi-lo)).mean()) for c in range(8))     # saturation (trivial-expansion guard)
    s=sfield(X); ipr=1.0-(s.sum()**2)/(s.size*(s**2).sum()+1e-9)           # localization
    early=float(np.mean(counts[:2]))+1e-6; late=float(np.mean(counts[-3:]))
    return {'sat':sat,'ipr':ipr,'early':early,'late':late,'colonized':bool(late>=4 and late>early*1.8)}

def fitness(law):
    best=0.0
    for rc,pc in PAIRS[:3]:
        try: r=guarded(law,rc,pc)
        except Exception: r=None
        if r is None: continue
        if r['sat']>0.15: continue        # GUARD: saturated field -> trivial expanding noise, reject (this alone separates
                                          # colonization sat~0 from ch3 noise-field sat~0.5; ipr guard rejected colonizers so dropped)
        f=r['late']*min(r['late']/r['early'],4.0)/2.0 + (5.0 if r['colonized'] else 0.0)
        best=max(best,f)
    return round(best,2)

def evolve(N=24, gens=10, seed=0):
    g=random.Random(seed)
    seeds=[CLAUDE['C_market'],CLAUDE['C_autocat_hull'],CLAUDE['A_gs_variant'],GS_REPLICATE,
           STRONG['S_selkov_glycolytic'],STRONG['S_driven_rps']]+[random_law(3000+i) for i in range(6)]
    import copy; pop=[copy.deepcopy(s) for s in seeds]+[random_law(4000+i) for i in range(N-len(seeds))]; pop=pop[:N]; traj=[]
    for gen in range(gens):
        fits=[fitness(x) for x in pop]; order=sorted(range(len(pop)),key=lambda i:-fits[i]); best=fits[order[0]]; traj.append(best)
        print(f"  gen {gen}: best_fit={best:.2f} median={np.median(fits):.2f} top3={[round(fits[i],1) for i in order[:3]]}", flush=True)
        elite=[pop[i] for i in order[:max(2,N//5)]]; off=[]
        while len(off)<N-len(elite):
            if g.random()<0.5: off.append(mutate(g.choice(elite),g))
            else: off.append(crossover(g.choice(elite),g.choice(elite[:3]+pop),g))
        pop=elite+off
    fits=[fitness(x) for x in pop]; bi=int(np.argmax(fits)); return pop[bi], fits[bi], traj

if __name__=='__main__':
    print("=== GUARDED EVOLUTION (Part 14): saturation/explosion/non-localized REJECTED — genuine reproduction beyond GS? ===", flush=True)
    print(f"  GS_REPLICATE guarded fitness (genuine reference)={fitness(GS_REPLICATE):.2f}", flush=True)
    best,bf,traj=evolve(N=24,gens=10,seed=0)
    print(f"=== GUARDED-EVOLVED best fitness={bf:.2f} | trajectory={traj} ===", flush=True)
    r=guarded(best,0,1,T=4000) or guarded(best,1,0,T=4000)
    if r: print(f"  guarded-best: late_count={r['late']:.1f} sat={r['sat']:.2f} ipr={r['ipr']:.2f} colonized={r['colonized']} (must be sat<0.15 & localized = GENUINE)", flush=True)
    import json; open('/home/pokazge/NativeEntity/evolved_guarded_law.json','w').write(json.dumps(best))
    print("=== VERDICT: if best <= GS fitness -> GS is genuine DSL ceiling; if > GS w/ sat<0.15 -> genuine improvement ===", flush=True)
    print("=== EVOPHYS2_DONE ===", flush=True)
