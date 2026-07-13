import os, math, random
os.environ['ME_MODE']='none'
import numpy as np
np.seterr(over='ignore', invalid='ignore')
from physics2 import step2, blank, DEF_D, H, W
from diag import detect, sfield

# RICHER-SUBSTRATE experiment: does search over {Lenia, advection, delay, 16ch} produce a MOBILE self-maintaining
# structure (glider) — qualitatively BEYOND the static Gray-Scott ceiling (Part-17 surprise)? Metrics: MOBILITY
# (persistent structure translates), PERSISTENCE, SELF-MAINT (inflow-cut amplitude decay = dissipative Case-D).
def strip_drive(law):
    keep=[t for t in law['terms'] if not (t['op']=='inflow' or (t['op']=='supply' and float(t.get('target',0))>0.05))]
    l=dict(law); l['terms']=keep; return l

def analyze(law, seed=0, T=1500, warm=300, init='noise'):
    D=law.get('D',DEF_D); g=np.random.RandomState(seed)
    X=blank(D,seed)
    if init=='noise': X[:,:,0]=(g.random((H,W))<0.3)*g.random((H,W))            # Lenia self-org family
    else:                                                                        # soup: resource ch0 filled + sparse nucleation (RD family)
        X[:,:,0]+=1.0; m=(g.random((H,W))<0.1).astype(np.float32); X[:,:,1]+=0.3*m
    rng=np.random.RandomState(1000+seed); cents=[]; sizes=[]; amps=[]
    for t in range(T):
        X=step2(X,law,t,rng)
        if not np.all(np.isfinite(X)): return None
        if t>=warm and t%30==0:
            bl,s=detect(X)
            if bl:
                big=max(bl,key=len); ys=[c[0] for c in big]; xs=[c[1] for c in big]
                cents.append((float(np.mean(ys)),float(np.mean(xs)))); sizes.append(len(big))
            else: cents.append(None); sizes.append(0)
            amps.append(float(sfield(X).mean()))
    if len(cents)<6: return None
    # MOBILITY: path length of the dominant structure (count only real motion <6 cells/interval, not teleport)
    path=0.0; moves=0
    for a,b in zip(cents[:-1],cents[1:]):
        if a and b:
            d=math.hypot(b[0]-a[0],b[1]-a[1])
            if d<6.0: path+=d; moves+=1
    mob=round(path/max(moves,1),2)
    persist=round(float(np.mean([1 if c else 0 for c in cents])),2)
    amp0=float(np.mean(amps[-4:]))
    # SELF-MAINT (dissipative Case-D): cut drive from settled state, does structure amplitude decay?
    Xcut=X.copy(); rng2=np.random.RandomState(77); ld=strip_drive(law)
    for t in range(500):
        Xcut=step2(Xcut,ld,10000+t,rng2)
        if not np.all(np.isfinite(Xcut)): break
    infdep=round(float(np.clip(1-float(sfield(Xcut).mean())/(amp0+1e-6),0,1)),2)
    loc=1.0-(sfield(X).sum()**2)/(sfield(X).size*(sfield(X)**2).sum()+1e-9)
    return {'mobility':mob,'persist':persist,'localization':round(float(loc),2),'infdep':infdep,'mean_size':round(float(np.mean(sizes)),1)}

# ---------- authored richer laws ----------
def base(terms,dt=0.3,noise=0.003,clamp=(0,2),D=DEF_D): return {'D':D,'dt':dt,'noise':noise,'clamp':list(clamp),'terms':terms}
RICH={}
RICH['L_lenia_selforg']=base([{'tgt':0,'op':'lenia','src':[0],'R':6,'kmu':0.5,'ksig':0.15,'gmu':0.26,'gsig':0.1,'coef':1.0},
    {'tgt':0,'op':'supply','coef':0.02,'target':0.0}],dt=0.2,clamp=(0,1))
RICH['L_lenia_advect']=base([{'tgt':0,'op':'lenia','src':[0],'R':6,'kmu':0.5,'ksig':0.15,'gmu':0.2,'gsig':0.08,'coef':1.0},
    {'tgt':0,'op':'advect','src':[1],'coef':-0.5},{'tgt':1,'op':'supply','coef':0.03,'target':1.0},{'tgt':1,'op':'diffuse','coef':0.1}],dt=0.15,clamp=(0,1.5))
RICH['L_gs_advect']=base([{'tgt':0,'op':'diffuse','coef':0.16},{'tgt':1,'op':'diffuse','coef':0.08},
    {'tgt':0,'op':'react','src':[0,1,1],'coef':-1.0},{'tgt':1,'op':'react','src':[0,1,1],'coef':1.0},
    {'tgt':0,'op':'supply','coef':0.037,'target':1.0},{'tgt':1,'op':'supply','coef':0.097,'target':0.0},
    {'tgt':1,'op':'advect','src':[2],'coef':-0.3},{'tgt':2,'op':'inflow','coef':0.04,'grad':'x'}],dt=1.0,clamp=(0,1.3))
RICH['L_lenia_2ch']=base([{'tgt':0,'op':'lenia','src':[0],'R':5,'kmu':0.5,'ksig':0.15,'gmu':0.22,'gsig':0.09,'coef':1.0},
    {'tgt':1,'op':'lenia','src':[0],'R':8,'kmu':0.5,'ksig':0.15,'gmu':0.28,'gsig':0.1,'coef':0.5},
    {'tgt':0,'op':'react','src':[1],'coef':-0.2},{'tgt':0,'op':'supply','coef':0.02,'target':0.0}],dt=0.2,clamp=(0,1))
RICH['L_delay_osc']=base([{'tgt':0,'op':'diffuse','coef':0.1},{'tgt':0,'op':'nonlin','src':[0],'f':'tanh','coef':1.0},
    {'tgt':0,'op':'react','src':[2],'coef':-1.0},{'tgt':2,'op':'delay','src':[0],'rate':0.05},
    {'tgt':0,'op':'inflow','coef':0.03,'grad':'r'},{'tgt':0,'op':'decay','coef':0.05}],dt=0.3,clamp=(-2,2))
RICH['L_advect_react']=base([{'tgt':0,'op':'inflow','coef':0.06,'grad':'r'},{'tgt':0,'op':'react','src':[0,1],'coef':-0.6},
    {'tgt':1,'op':'react','src':[0,1],'coef':0.6},{'tgt':1,'op':'advect','src':[0],'coef':-0.4},
    {'tgt':1,'op':'decay','coef':0.05},{'tgt':1,'op':'diffuse','coef':0.05}],dt=0.3,clamp=(0,2))
RICH['L_lenia_rd']=base([{'tgt':0,'op':'lenia','src':[0],'R':6,'kmu':0.5,'ksig':0.15,'gmu':0.24,'gsig':0.09,'coef':0.8},
    {'tgt':0,'op':'diffuse','coef':0.05},{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':0.2},{'tgt':1,'op':'diffuse','coef':0.3},
    {'tgt':0,'op':'react','src':[1],'coef':-0.3},{'tgt':1,'op':'supply','coef':0.04,'target':0.0}],dt=0.2,clamp=(0,1.5))

OPS2=['diffuse','decay','react','supply','inflow','nonlin','catalyze','transport','lenia','advect','delay','diffuse2']
def rand_rich(seed):
    g=random.Random(seed); nt=g.randint(4,8); terms=[]; hi=False
    for _ in range(nt):
        op=g.choice(OPS2); tgt=g.randint(0,4); c=round(g.uniform(-1,1),2)
        if op=='lenia': terms.append({'tgt':tgt,'op':'lenia','src':[g.randint(0,4)],'R':g.randint(4,8),'kmu':round(g.uniform(0.3,0.6),2),'ksig':0.15,'gmu':round(g.uniform(0.1,0.3),2),'gsig':round(g.uniform(0.05,0.12),3),'coef':round(g.uniform(0.5,1.2),2)})
        elif op=='advect': terms.append({'tgt':tgt,'op':'advect','src':[g.randint(0,4)],'coef':round(g.uniform(-0.6,0.6),2)})
        elif op=='delay': terms.append({'tgt':tgt,'op':'delay','src':[g.randint(0,4)],'rate':round(g.uniform(0.01,0.1),3)})
        elif op=='diffuse': terms.append({'tgt':tgt,'op':'diffuse','coef':round(g.uniform(-0.1,0.4),2)})
        elif op=='diffuse2': terms.append({'tgt':tgt,'op':'diffuse2','coef':round(g.uniform(0.0,0.2),2)})
        elif op=='decay': terms.append({'tgt':tgt,'op':'decay','coef':round(g.uniform(0.01,0.15),2)})
        elif op=='react': terms.append({'tgt':tgt,'op':'react','src':[g.randint(0,4) for _ in range(g.randint(1,3))],'coef':c})
        elif op=='supply': terms.append({'tgt':tgt,'op':'supply','coef':round(g.uniform(0.02,0.15),2),'target':round(g.choice([0.0,1.0]),2)})
        elif op=='inflow': terms.append({'tgt':tgt,'op':'inflow','coef':round(g.uniform(0.03,0.09),2),'grad':g.choice(['x','y','r'])}); hi=True
        elif op=='nonlin': terms.append({'tgt':tgt,'op':'nonlin','src':[g.randint(0,4)],'f':g.choice(['tanh','sigmoid']),'coef':c})
        elif op=='catalyze': terms.append({'tgt':tgt,'op':'catalyze','cat':g.randint(0,4),'src':[g.randint(0,4)],'coef':c})
        elif op=='transport': terms.append({'tgt':tgt,'op':'transport','src':[g.randint(0,4)],'coef':round(g.uniform(-0.3,0.3),2)})
    if not hi: terms.append({'tgt':0,'op':'inflow','coef':0.04,'grad':'r'})
    terms.append({'tgt':g.randint(0,4),'op':'decay','coef':0.04})
    return base(terms,dt=round(g.uniform(0.15,0.4),2),clamp=(0,2))

def best(law,seeds=(0,1)):
    rs=[analyze(law,s,init=k) for s in seeds for k in ('noise','soup')]; rs=[r for r in rs if r]
    if not rs: return None
    return max(rs,key=lambda r:(r['mobility'] if r['persist']>0.5 else 0)+r['infdep'])

if __name__=='__main__':
    print("=== RICHER-SUBSTRATE SEARCH: MOBILE self-maintaining organization beyond static Gray-Scott? ===", flush=True)
    print(f"  {'law':20s} {'mob':>4s} {'persist':>7s} {'loc':>4s} {'infdep':>6s} {'size':>5s}  MOBILE-SELFMAINT?", flush=True)
    def row(nm,r):
        if not r: print(f"  {nm:20s}  (died/exploded)", flush=True); return False
        mobself=bool(r['mobility']>=1.5 and r['persist']>=0.6 and r['infdep']>=0.3)
        print(f"  {nm:20s} {r['mobility']:>4.1f} {r['persist']:>7.2f} {r['localization']:>4.2f} {r['infdep']:>6.2f} {r['mean_size']:>5.1f}  {'*** MOBILE+SELFMAINT' if mobself else ('mobile' if r['mobility']>=1.5 and r['persist']>=0.6 else '')}", flush=True)
        return mobself
    lm=0
    for nm,law in RICH.items(): lm+=row(nm,best(law))
    print("  -- RANDOM richer laws (20) --", flush=True)
    rm=0
    for s in range(20):
        r=best(rand_rich(5000+s))
        if r and r['mobility']>=1.5 and r['persist']>=0.6: rm+= row(f'rand_{s}',r)
    print(f"=== RESULT: authored mobile+selfmaint={lm}/{len(RICH)} | random mobile+selfmaint={rm}/20 ===", flush=True)
    print("=== if any MOBILE self-maintaining structure -> genuine surprise BEYOND reaction-diffusion ceiling ===", flush=True)
    print("=== RICHWORLD_DONE ===", flush=True)
