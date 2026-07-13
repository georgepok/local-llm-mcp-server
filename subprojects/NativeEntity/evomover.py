import os, math, random, copy
os.environ['ME_MODE']='none'
import numpy as np
np.seterr(over='ignore', invalid='ignore')
from physics2 import step2, blank, DEF_D, H, W
from diag import detect, sfield
from richworld import RICH, rand_rich, base, strip_drive

# DIRECTED search for a MOBILE DISSIPATIVE SOLITON (Part-17 genuine surprise): jointly optimize localization +
# persistence + COHERENT mobility (net/path straight-line motion of a compact structure) + throughput-dependence.
# Generic sampling found these DISJOINT; directed fitness requires ALL simultaneously. Seeds = dissipative+advection.
def analyze(law, seed=0, T=1500, warm=300):
    D=law.get('D',DEF_D); g=np.random.RandomState(seed); X=blank(D,seed)
    X[:,:,0]+=1.0; m=(g.random((H,W))<0.08).astype(np.float32); X[:,:,1]+=0.4*m    # resource + sparse nucleation
    rng=np.random.RandomState(1000+seed); cents=[]; sizes=[]; amps=[]
    for t in range(T):
        X=step2(X,law,t,rng)
        if not np.all(np.isfinite(X)): return None
        if t>=warm and t%25==0:
            bl,s=detect(X)
            if bl: big=max(bl,key=len); cents.append((float(np.mean([c[0] for c in big])),float(np.mean([c[1] for c in big])),len(big))); sizes.append(len(big))
            else: cents.append(None)
            amps.append(float(sfield(X).mean()))
    pts=[c for c in cents if c]
    if len(pts)<6: return None
    persist=float(np.mean([1 if c else 0 for c in cents]))
    # coherent mobility: path (real steps <5 cells) and net; a compact-size-stable structure moving straight
    path=0.0; moves=0
    for a,b in zip(pts[:-1],pts[1:]):
        d=math.hypot(b[0]-a[0],b[1]-a[1])
        if d<5.0: path+=d; moves+=1
    net=math.hypot(pts[-1][0]-pts[0][0],pts[-1][1]-pts[0][1])
    coh=net/(path+1e-6) if path>2 else 0.0
    size_cv=float(np.std(sizes)/(np.mean(sizes)+1e-6)) if sizes else 9   # compact & stable size -> low CV
    loc=1.0-(sfield(X).sum()**2)/(sfield(X).size*(sfield(X)**2).sum()+1e-9)
    amp0=float(np.mean(amps[-4:]))
    Xcut=X.copy(); rng2=np.random.RandomState(77); ld=strip_drive(law)
    for t in range(400):
        Xcut=step2(Xcut,ld,10000+t,rng2)
        if not np.all(np.isfinite(Xcut)): break
    infdep=float(np.clip(1-float(sfield(Xcut).mean())/(amp0+1e-6),0,1))
    return {'persist':round(persist,2),'net':round(net,1),'path':round(path,1),'coh':round(min(coh,1),2),
            'loc':round(float(loc),2),'size_cv':round(size_cv,2),'infdep':round(infdep,2),'msize':round(float(np.mean(sizes)),1)}

def fitness(law):
    best=0.0
    for sd in (0,1):
        try: r=analyze(law,sd)
        except Exception: r=None
        if r is None: continue
        if r['persist']<0.6 or r['loc']<0.15 or r['msize']<6 or r['msize']>300: continue   # localized persistent structure gate
        # mobile dissipative soliton: net displacement * coherence * throughput-dependence, compact-stable size
        f=r['net']*r['coh']*r['infdep']/(1.0+r['size_cv'])
        best=max(best,round(f,2))
    return best

OPS2=['diffuse','react','supply','inflow','nonlin','catalyze','transport','lenia','advect','delay','decay']
def rand_term(g):
    t=rand_rich(g.randint(0,10**6))['terms']; return t[g.randrange(len(t))]
def mutate(law,g):
    l=copy.deepcopy(law); t=l['terms']; r=g.random()
    if r<0.4 and t: x=g.choice(t); x['coef']=round(float(x.get('coef',0))*g.uniform(0.6,1.6)+g.uniform(-0.05,0.05),3)
    elif r<0.6: t.append(rand_term(g))
    elif r<0.72 and len(t)>3: t.pop(g.randrange(len(t)))
    elif r<0.85 and t:
        x=g.choice(t); x['tgt']=g.randint(0,4)
        if 'src' in x: x['src']=[g.randint(0,4) for _ in x['src']]
    elif r<0.95 and t:
        x=g.choice(t)
        if x['op']=='advect': x['coef']=round(g.uniform(-0.8,0.8),2)      # tune movers
        if x['op']=='lenia': x['gmu']=round(g.uniform(0.1,0.3),2)
    else: l['dt']=round(min(1.0,max(0.1,float(l.get('dt',0.3))*g.uniform(0.7,1.4))),2)
    return l
def crossover(a,b,g):
    ta,tb=a['terms'],b['terms']; k=g.randint(1,max(1,len(ta)-1))
    return base(copy.deepcopy(ta[:k])+copy.deepcopy(g.sample(tb,min(len(tb),g.randint(1,max(1,len(tb)))))), dt=a.get('dt',0.3), clamp=tuple(a.get('clamp',[0,2])))

if __name__=='__main__':
    print("=== DIRECTED MOBILE-DISSIPATIVE-SOLITON SEARCH ===", flush=True)
    seeds0=[RICH['L_gs_advect'],RICH['L_advect_react'],RICH['L_lenia_rd'],RICH['L_lenia_advect']]+[rand_rich(6000+i) for i in range(6)]
    for nm in ['L_gs_advect','L_advect_react']: print(f"  seed {nm} fitness={fitness(RICH[nm]):.2f}", flush=True)
    g=random.Random(0); N=24; pop=[copy.deepcopy(s) for s in seeds0]+[rand_rich(7000+i) for i in range(N-len(seeds0))]; pop=pop[:N]; traj=[]
    for gen in range(10):
        fits=[fitness(x) for x in pop]; order=sorted(range(len(pop)),key=lambda i:-fits[i]); best=fits[order[0]]; traj.append(best)
        print(f"  gen {gen}: best={best:.2f} median={np.median(fits):.2f} top3={[round(fits[i],2) for i in order[:3]]}", flush=True)
        elite=[pop[i] for i in order[:max(2,N//5)]]; off=[]
        while len(off)<N-len(elite):
            off.append(mutate(g.choice(elite),g) if g.random()<0.55 else crossover(g.choice(elite),g.choice(pop),g))
        pop=elite+off
    fits=[fitness(x) for x in pop]; bi=int(np.argmax(fits)); best=pop[bi]
    r=analyze(best,0)
    print(f"=== BEST fitness={fits[bi]:.2f} traj={traj} | profile={r} ===", flush=True)
    import json; open('/home/pokazge/NativeEntity/evolved_mover_law.json','w').write(json.dumps(best))
    print("=== if best>0 w/ net>4 coh>0.5 infdep>0.3 loc>0.2 -> GENUINE mobile dissipative soliton (surprise!) ===", flush=True)
    print("=== EVOMOVER_DONE ===", flush=True)
