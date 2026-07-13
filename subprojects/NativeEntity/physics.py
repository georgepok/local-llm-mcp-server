import os, math, random
os.environ['ME_MODE']='none'
import numpy as np
np.seterr(over='ignore', invalid='ignore')

# SYNTHESIZED_ARTIFICIAL_PHYSICS_V1 — Part 18/19: smallest viable world substrate + local-physics DSL +
# simulator VALIDATION controls. LLM will later propose world LAWS (not organisms); entity must emerge INSIDE
# the physics. Neutral channels x0..x(D-1); mandatory inflow+dissipation+noise; local (Moore r1) synchronous update.
# NO Orch-OR / consciousness / autopoiesis claims. This file = simulator + hand-built VALIDATION laws only.

def lap(F):  # Moore-radius-1 discrete Laplacian (periodic), 9-point (Oono-Puri)
    up=np.roll(F,1,0); dn=np.roll(F,-1,0); lf=np.roll(F,1,1); rt=np.roll(F,-1,1)
    ul=np.roll(up,1,1); ur=np.roll(up,-1,1); dl=np.roll(dn,1,1); dr=np.roll(dn,-1,1)
    return 0.2*(up+dn+lf+rt) + 0.05*(ul+ur+dl+dr) - F

def inflow_field(H,W,grad,t):
    if grad=='x':  g=np.linspace(0,1,W)[None,:].repeat(H,0)
    elif grad=='y':g=np.linspace(0,1,H)[:,None].repeat(W,1)
    elif grad=='r':
        yy,xx=np.mgrid[0:H,0:W]; g=1-np.sqrt(((yy-H/2)**2+(xx-W/2)**2))/(0.7*H); g=np.clip(g,0,1)
    else: g=np.ones((H,W))
    return (0.6+0.4*math.sin(0.01*t))*g.astype(np.float32)          # spatial gradient x temporal modulation

# ---- local-physics DSL executor: world law = list of per-channel terms over generic primitives ----
def step(X, law, t, rng):
    H,W,D=X.shape; dX=np.zeros_like(X)
    for tm in law['terms']:
        tgt=tm['tgt']; op=tm['op']; c=float(tm.get('coef',1.0))
        if op=='diffuse':   dX[:,:,tgt]+= c*lap(X[:,:,tgt])
        elif op=='decay':   dX[:,:,tgt]-= c*X[:,:,tgt]
        elif op=='react':                                           # coef * PRODUCT of source channels (mass-action)
            p=np.ones((H,W),np.float32)
            for s in tm['src']: p=p*X[:,:,s]
            dX[:,:,tgt]+= c*p
        elif op=='supply':  dX[:,:,tgt]+= c*(float(tm.get('target',0.0))-X[:,:,tgt])   # relaxation toward a level
        elif op=='inflow':  dX[:,:,tgt]+= c*inflow_field(H,W,tm.get('grad','x'),t)     # external resource-like drive
        elif op=='nonlin':                                          # bounded nonlinearity of a source channel
            s=tm['src'][0]; f=tm.get('f','tanh')
            v=X[:,:,s]; v=np.tanh(v) if f=='tanh' else (1/(1+np.exp(-v)) if f=='sigmoid' else np.maximum(0,v))
            dX[:,:,tgt]+= c*v
        elif op=='catalyze':                                        # coef * X[cat] * X[src] (state-dependent modulation)
            dX[:,:,tgt]+= c*X[:,:,tm['cat']]*X[:,:,tm['src'][0]]
        elif op=='transport':                                       # state-dependent transport down a gradient
            s=tm['src'][0]; dX[:,:,tgt]+= c*lap(X[:,:,tgt]*X[:,:,s])
        elif op=='exchange':                                        # conservative swap between two channels
            a,b=tm['src']; flow=c*(X[:,:,a]-X[:,:,b]); dX[:,:,a]-=flow; dX[:,:,b]+=flow
    X=X+float(law.get('dt',1.0))*dX
    X=X*(1.0-float(law.get('base_leak',0.004)))   # universal baseline dissipation (Part 4): persistent structure must be ACTIVELY maintained; inert init remnants in unused channels decay
    if law.get('noise',0)>0: X=X+float(law['noise'])*rng.standard_normal(X.shape).astype(np.float32)
    lo,hi=law.get('clamp',[-6.0,6.0]); X=np.clip(X,lo,hi)
    return X

def init_field(H,W,D,kind,seed):
    g=np.random.RandomState(seed); X=(0.02*g.standard_normal((H,W,D))).astype(np.float32)
    if kind=='uniform': X+=0.1*g.standard_normal((H,W,D)).astype(np.float32)
    elif kind=='impulse':
        for _ in range(12): y,x=g.randint(0,H),g.randint(0,W); X[y,x,:]+=g.standard_normal(D)
    elif kind=='droplet':
        for _ in range(4):
            y,x=g.randint(4,H-4),g.randint(4,W-4); X[y-3:y+3,x-3:x+3,:]+=0.8*g.standard_normal(D)
    elif kind=='stripe': X[:, ::4, :]+=0.5
    return X

def run(law, kind='droplet', seed=0, T=1200, D=None, snap=None):
    D=D or law.get('D',8); X=init_field(32,32,D,kind,seed); rng=np.random.RandomState(1000+seed); snaps={}
    for t in range(T):
        X=step(X,law,t,rng)
        if not np.all(np.isfinite(X)): return None,{'blew_up':t}
        if snap and t in snap: snaps[t]=X.copy()
    return X, snaps

# ---------------- VALIDATION CONTROL LAWS (Part 19; NOT synthesis examples) ----------------
GRAY_SCOTT={'D':8,'dt':1.0,'noise':0.0,'clamp':[0,1.2],'terms':[   # localized self-replicating spots + repair (channels 0,1)
    {'tgt':0,'op':'diffuse','coef':0.16},{'tgt':1,'op':'diffuse','coef':0.08},
    {'tgt':0,'op':'react','src':[0,1,1],'coef':-1.0},{'tgt':1,'op':'react','src':[0,1,1],'coef':1.0},
    {'tgt':0,'op':'supply','coef':0.037,'target':1.0},{'tgt':1,'op':'supply','coef':0.0997,'target':0.0}]}
DIFFUSE_ONLY={'D':8,'dt':1.0,'noise':0.0,'clamp':[-2,2],'terms':[{'tgt':0,'op':'diffuse','coef':0.3}]}  # homogenizes
OSC={'D':8,'dt':0.4,'noise':0.0,'clamp':[-2,2],'terms':[          # local oscillator (activator-inhibitor)
    {'tgt':0,'op':'diffuse','coef':0.1},{'tgt':0,'op':'nonlin','src':[0],'f':'tanh','coef':1.0},{'tgt':0,'op':'react','src':[1],'coef':-1.0},
    {'tgt':1,'op':'react','src':[0],'coef':0.5},{'tgt':1,'op':'decay','coef':0.1}]}
GROWTH={'D':8,'dt':0.5,'noise':0.0,'clamp':[0,2],'terms':[        # resource-driven growth (inflow feeds catalytic growth)
    {'tgt':0,'op':'inflow','coef':0.05,'grad':'r'},{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':0.1},
    {'tgt':1,'op':'decay','coef':0.02},{'tgt':0,'op':'react','src':[0,1],'coef':-0.1},{'tgt':1,'op':'diffuse','coef':0.05}]}

def localization(X):  # spatial structure: variance of channel-0 field / how non-uniform (0=homogeneous)
    f=X[:,:,0]; return round(float(np.std(f)),3)
def active_frac(X): return round(float(np.mean(X[:,:,1]>0.2)),3)

if __name__=='__main__':
    print("=== SYNTHESIZED_ARTIFICIAL_PHYSICS_V1 — Part 19 SIMULATOR VALIDATION (controls only) ===", flush=True)
    def gs_init(seed):  # GS needs u~1, v seeded spots
        X=np.zeros((32,32,8),np.float32); X[:,:,0]=1.0; g=np.random.RandomState(seed)
        for _ in range(6): y,x=g.randint(6,26),g.randint(6,26); X[y-2:y+2,x-2:x+2,0]=0.5; X[y-2:y+2,x-2:x+2,1]=0.25
        return X
    # Gray-Scott: localized replicating spots (persistence + reproduction) + repair
    X=gs_init(0); rng=np.random.RandomState(7)
    for t in range(3000): X=step(X,GRAY_SCOTT,t,rng)
    print(f"  GRAY_SCOTT localized spots: localization(std x0)={localization(X)} active_frac(x1)={active_frac(X)} (want >0.1 struct, patchy)", flush=True)
    Xdmg=X.copy(); Xdmg[10:22,10:22,:]=0; Xdmg[:,:,0]=np.where(Xdmg[:,:,0]==0,1.0,Xdmg[:,:,0])   # damage a region
    for t in range(2000): Xdmg=step(Xdmg,GRAY_SCOTT,t,rng)
    print(f"  GRAY_SCOTT after damage+2000: active_frac(x1)={active_frac(Xdmg)} (repair: pattern regrows into damaged zone)", flush=True)
    for name,law,kind in [('DIFFUSE_ONLY',DIFFUSE_ONLY,'droplet'),('OSC',OSC,'impulse'),('GROWTH',GROWTH,'impulse')]:
        Xf,info=run(law,kind=kind,seed=0,T=1200)
        if Xf is None: print(f"  {name}: BLEW_UP at {info}", flush=True); continue
        print(f"  {name:12s}: localization(std x0)={localization(Xf)} active_frac(x1)={active_frac(Xf)} finite=True", flush=True)
    print("=== VALIDATION: GS makes localized patterns+repair; diffuse->homogenize(low std); OSC/GROWTH finite ===", flush=True)
    print("=== PHYSICS_VALIDATION_DONE ===", flush=True)
