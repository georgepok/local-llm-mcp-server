import os, copy, random
os.environ['ME_MODE']='none'
import numpy as np
np.seterr(over='ignore', invalid='ignore')
from worldlaws import CLAUDE, random_law
from worldlaws2 import STRONG
from repro import run_repro, GS_REPLICATE, PAIRS

# Part 13/15/16: EVOLUTIONARY world-law search. LLM≈random for proposal, so SEARCH is the world-discovery engine.
# Unit of selection = the world LAW. Fitness = reproduction/colonization strength (the frontier metric no single law
# passed strongly). Seed pop with best-known laws + random; mutate/crossover DSL; track whether search pushes PAST the
# weak Case-E ceiling (2 borderline colonizers -> stronger colonizers / Case-F open-ended novelty).
OPSA=['diffuse','decay','react','supply','inflow','nonlin','catalyze','transport','exchange']
def rand_term(g):
    op=g.choice(OPSA); tgt=g.randint(0,3)
    if op=='diffuse': return {'tgt':tgt,'op':'diffuse','coef':round(g.uniform(-0.15,0.5),2)}
    if op=='decay': return {'tgt':tgt,'op':'decay','coef':round(g.uniform(0.01,0.2),2)}
    if op=='react': return {'tgt':tgt,'op':'react','src':[g.randint(0,3) for _ in range(g.randint(1,3))],'coef':round(g.uniform(-1,1),2)}
    if op=='supply': return {'tgt':tgt,'op':'supply','coef':round(g.uniform(0.02,0.2),2),'target':round(g.choice([0.0,0.0,1.0])*g.uniform(0.5,1.0),2)}
    if op=='inflow': return {'tgt':tgt,'op':'inflow','coef':round(g.uniform(0.02,0.12),2),'grad':g.choice(['x','y','r'])}
    if op=='nonlin': return {'tgt':tgt,'op':'nonlin','src':[g.randint(0,3)],'f':g.choice(['tanh','sigmoid','relu']),'coef':round(g.uniform(-1,1),2)}
    if op=='catalyze': return {'tgt':tgt,'op':'catalyze','cat':g.randint(0,3),'src':[g.randint(0,3)],'coef':round(g.uniform(-0.7,0.7),2)}
    if op=='transport': return {'tgt':tgt,'op':'transport','src':[g.randint(0,3)],'coef':round(g.uniform(-0.3,0.3),2)}
    return {'tgt':tgt,'op':'exchange','src':[g.randint(0,3),g.randint(0,3)],'coef':round(g.uniform(0.05,0.25),2)}

def mutate(law, g):
    l=copy.deepcopy(law); t=l['terms']; r=g.random()
    if r<0.4 and t: x=g.choice(t); x['coef']=round(float(x.get('coef',0))*g.uniform(0.6,1.6)+g.uniform(-0.05,0.05),3)
    elif r<0.6: t.append(rand_term(g))
    elif r<0.75 and len(t)>4: t.pop(g.randrange(len(t)))
    elif r<0.9 and t:
        x=g.choice(t); x['tgt']=g.randint(0,3)
        if 'src' in x: x['src']=[g.randint(0,3) for _ in x['src']]
        if 'cat' in x: x['cat']=g.randint(0,3)
    else: l['dt']=round(min(1.0,max(0.15,float(l.get('dt',0.4))*g.uniform(0.7,1.4))),2)
    return l
def crossover(a,b,g):
    ta,tb=a['terms'],b['terms']; k=g.randint(1,max(1,len(ta)-1))
    child=copy.deepcopy(ta[:k])+copy.deepcopy(g.sample(tb,min(len(tb),g.randint(1,max(1,len(tb)//2)))))
    return {'D':8,'dt':a.get('dt',0.4),'noise':a.get('noise',0.003),'clamp':a.get('clamp',[-4,4]),'terms':child}

def fitness(law):
    best=0.0
    for rc,pc in PAIRS[:3]:                       # fast: 3 seed pairs, single seed
        try:
            r=run_repro(law,rc,pc,T=3000)
        except Exception: r=None
        if r is None: continue
        # reward colonization: final structure count growth + spread (Case-E/F frontier)
        f=r['final_count']*min(r['repro_ratio'],4.0)/2.0 + 3.0*r['spread_gain'] + (5.0 if r['colonized'] else 0.0)
        best=max(best,f)
    return round(best,2)

def evolve(N=24, gens=8, seed=0):
    g=random.Random(seed)
    seeds=[CLAUDE['C_market'],CLAUDE['C_autocat_hull'],CLAUDE['A_gs_variant'],CLAUDE['B_delayed_bistable'],
           STRONG['S_selkov_glycolytic'],STRONG['S_driven_rps'],GS_REPLICATE]+[random_law(3000+i) for i in range(6)]
    pop=[copy.deepcopy(s) for s in seeds]+[random_law(4000+i) for i in range(N-len(seeds))]
    pop=pop[:N]; traj=[]
    for gen in range(gens):
        fits=[fitness(x) for x in pop]
        order=sorted(range(len(pop)),key=lambda i:-fits[i]); best=fits[order[0]]; traj.append(round(best,2))
        print(f"  gen {gen}: best_fit={best:.2f} median={np.median(fits):.2f} top3={[round(fits[i],1) for i in order[:3]]}", flush=True)
        elite=[pop[i] for i in order[:max(2,N//5)]]
        off=[]
        while len(off)<N-len(elite):
            if g.random()<0.5: off.append(mutate(g.choice(elite),g))
            else: off.append(crossover(g.choice(elite),g.choice(elite[:3]+pop),g))
        pop=elite+off
    fits=[fitness(x) for x in pop]; bi=int(np.argmax(fits))
    return pop[bi], fits[bi], traj

if __name__=='__main__':
    print("=== EVOLUTIONARY WORLD-LAW SEARCH (Part 13/15): can search push PAST the weak Case-E ceiling? ===", flush=True)
    print(f"  GS_REPLICATE seed fitness (reference strong colonizer)={fitness(GS_REPLICATE):.2f}", flush=True)
    print(f"  C_market seed fitness (best LLM colonizer)={fitness(CLAUDE['C_market']):.2f}", flush=True)
    best,bf,traj=evolve(N=24,gens=8,seed=0)
    print(f"=== EVOLVED best fitness={bf:.2f} | fitness trajectory over gens={traj} ===", flush=True)
    r=run_repro(best,0,1,T=4000) or run_repro(best,1,0,T=4000)
    print(f"  evolved-best reproduction: final_count={r['final_count'] if r else 'NA'} repro_ratio={r['repro_ratio'] if r else 'NA'} colonized={r['colonized'] if r else 'NA'}", flush=True)
    import json; open('/home/pokazge/NativeEntity/evolved_best_law.json','w').write(json.dumps(best))
    print("  (best evolved law saved to evolved_best_law.json)", flush=True)
    print("=== EVOPHYS_DONE ===", flush=True)
