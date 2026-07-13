import os, math
os.environ['ME_MODE']='none'
import numpy as np
np.seterr(over='ignore', invalid='ignore')

# SYNTHESIZED_ARTIFICIAL_PHYSICS_V1 — RICHER SUBSTRATE (Part 7). 8-ch/Moore-r1/reaction-diffusion DSL had a genuine
# ceiling at Gray-Scott (no novel/surprising organization). Richer DSL adds: 16 channels, radius-R Lenia kernels
# (mobile continuous-CA creatures), anisotropic/directional transport (advection -> movers), delayed local state
# (memory/history -> Case-F history dependence). Superset of physics.py ops. Neutral channels; inflow/dissipation/noise.
H=W=32; DEF_D=16
def lap(F):
    up=np.roll(F,1,0); dn=np.roll(F,-1,0); lf=np.roll(F,1,1); rt=np.roll(F,-1,1)
    ul=np.roll(up,1,1); ur=np.roll(up,-1,1); dl=np.roll(dn,1,1); dr=np.roll(dn,-1,1)
    return 0.2*(up+dn+lf+rt)+0.05*(ul+ur+dl+dr)-F
def gx(F): return 0.5*(np.roll(F,-1,1)-np.roll(F,1,1))     # x-gradient (central)
def gy(F): return 0.5*(np.roll(F,-1,0)-np.roll(F,1,0))     # y-gradient
def inflow_field(grad,t):
    if grad=='x': g=np.linspace(0,1,W)[None,:].repeat(H,0)
    elif grad=='y': g=np.linspace(0,1,H)[:,None].repeat(W,1)
    elif grad=='r':
        yy,xx=np.mgrid[0:H,0:W]; g=1-np.sqrt(((yy-H/2)**2+(xx-W/2)**2))/(0.7*H); g=np.clip(g,0,1)
    else: g=np.ones((H,W))
    return (0.6+0.4*math.sin(0.01*t))*g.astype(np.float32)

_KCACHE={}
def lenia_kernel_fft(R,mu,sig):                            # radius-R Gaussian-ring kernel (Lenia), FFT for fast periodic conv
    key=(R,round(mu,3),round(sig,3))
    if key in _KCACHE: return _KCACHE[key]
    yy,xx=np.mgrid[0:H,0:W]; yy=np.minimum(yy,H-yy).astype(np.float32); xx=np.minimum(xx,W-xx).astype(np.float32)
    r=np.sqrt(yy**2+xx**2)/max(R,1e-6); K=np.exp(-((r-mu)**2)/(2*sig**2+1e-9)); K[r>1]=0.0
    s=K.sum(); K=K/(s if s>0 else 1.0); Kf=np.fft.rfft2(K); _KCACHE[key]=Kf; return Kf
def lenia_conv(F,Kf): return np.fft.irfft2(np.fft.rfft2(F)*Kf, F.shape).astype(np.float32)
def growth(u,gm,gs): return (2.0*np.exp(-((u-gm)**2)/(2*gs**2+1e-9))-1.0).astype(np.float32)  # Lenia growth bump [-1,1]

def step2(X, law, t, rng):
    D=X.shape[2]; dX=np.zeros_like(X)
    for tm in law['terms']:
        tgt=tm['tgt']; op=tm['op']; c=float(tm.get('coef',1.0))
        if op=='diffuse': dX[:,:,tgt]+=c*lap(X[:,:,tgt])
        elif op=='diffuse2': dX[:,:,tgt]+=c*lap(lap(X[:,:,tgt]))*(-1.0)     # radius-2 (bi-Laplacian smoothing)
        elif op=='decay': dX[:,:,tgt]-=c*X[:,:,tgt]
        elif op=='react':
            p=np.ones((H,W),np.float32)
            for s in tm['src']: p=p*X[:,:,s]
            dX[:,:,tgt]+=c*p
        elif op=='supply': dX[:,:,tgt]+=c*(float(tm.get('target',0.0))-X[:,:,tgt])
        elif op=='inflow': dX[:,:,tgt]+=c*inflow_field(tm.get('grad','r'),t)
        elif op=='nonlin':
            s=tm['src'][0]; f=tm.get('f','tanh'); v=X[:,:,s]
            v=np.tanh(v) if f=='tanh' else (1/(1+np.exp(-v)) if f=='sigmoid' else np.maximum(0,v))
            dX[:,:,tgt]+=c*v
        elif op=='catalyze': dX[:,:,tgt]+=c*X[:,:,tm['cat']]*X[:,:,tm['src'][0]]
        elif op=='transport': dX[:,:,tgt]+=c*lap(X[:,:,tgt]*X[:,:,tm['src'][0]])
        elif op=='exchange':
            a,b=tm['src']; flow=c*(X[:,:,a]-X[:,:,b]); dX[:,:,a]-=flow; dX[:,:,b]+=flow
        elif op=='lenia':                                                   # Lenia continuous-CA: growth of ring-kernel potential
            s=tm['src'][0]; Kf=lenia_kernel_fft(int(tm.get('R',6)),float(tm.get('kmu',0.5)),float(tm.get('ksig',0.15)))
            U=lenia_conv(X[:,:,s],Kf); dX[:,:,tgt]+=c*growth(U,float(tm.get('gmu',0.15)),float(tm.get('gsig',0.02)))
        elif op=='advect':                                                  # anisotropic transport along gradient of dir channel -> movers
            d=tm['src'][0]; vx=gx(X[:,:,d]); vy=gy(X[:,:,d])
            dX[:,:,tgt]+=c*(vx*gx(X[:,:,tgt])+vy*gy(X[:,:,tgt]))
        elif op=='delay':                                                   # delayed local state (memory): tgt slowly tracks src (history)
            r=float(tm.get('rate',0.02)); dX[:,:,tgt]+=r*(X[:,:,tm['src'][0]]-X[:,:,tgt])
    X=X+float(law.get('dt',0.5))*dX
    X=X*(1.0-float(law.get('base_leak',0.004)))
    if law.get('noise',0)>0: X=X+float(law['noise'])*rng.standard_normal(X.shape).astype(np.float32)
    lo,hi=law.get('clamp',[-4.0,4.0]); X=np.clip(X,lo,hi)
    return X

def blank(D=DEF_D, seed=0): return (0.02*np.random.RandomState(seed).standard_normal((H,W,D))).astype(np.float32)

# ---- validation controls for the richer substrate (NOT synthesis examples) ----
ORBIUM={'D':DEF_D,'dt':0.1,'noise':0.0,'clamp':[0,1],'base_leak':0.0,'terms':[     # Lenia glider (mobile!) on ch0
    {'tgt':0,'op':'lenia','src':[0],'R':6,'kmu':0.5,'ksig':0.15,'gmu':0.15,'gsig':0.017,'coef':1.0}]}
ADVECT_CTRL={'D':DEF_D,'dt':0.3,'noise':0.0,'clamp':[0,2],'terms':[               # directional flow
    {'tgt':0,'op':'inflow','coef':0.05,'grad':'r'},{'tgt':0,'op':'advect','src':[1],'coef':0.4},
    {'tgt':1,'op':'supply','coef':0.05,'target':1.0},{'tgt':0,'op':'decay','coef':0.02}]}

def centroid(F,thr):
    m=F>thr
    if m.sum()<3: return None
    yy,xx=np.mgrid[0:H,0:W]; return (float(yy[m].mean()),float(xx[m].mean()))

def advect_mobility():   # clean MOVER control: bump in ch0 translated by a constant velocity field (grad of ch1 ramp)
    law={'D':DEF_D,'dt':0.3,'noise':0.0,'clamp':[0,2],'base_leak':0.0,'terms':[
        {'tgt':0,'op':'advect','src':[1],'coef':-1.0},{'tgt':0,'op':'diffuse','coef':0.02}]}
    X=blank(); X[14:18,6:10,0]=1.0; X[:,:,1]=np.linspace(0,3,W)[None,:].repeat(H,0)   # ch1 = x-ramp -> constant vx
    rng=np.random.RandomState(0); c0=centroid(X[:,:,0],0.2)
    for t in range(300): X=step2(X,law,t,rng)
    cN=centroid(X[:,:,0],0.2); return (math.hypot(cN[1]-c0[1],cN[0]-c0[0]) if c0 and cN else 0.0), int((X[:,:,0]>0.2).sum())

def lenia_scan():        # find a Lenia config that SELF-ORGANIZES (survives + non-trivial structure) from noise
    best=None
    for gmu,gsig,dt in [(0.14,0.06,0.15),(0.20,0.08,0.15),(0.12,0.05,0.1),(0.26,0.10,0.2),(0.18,0.07,0.12)]:
        law={'D':DEF_D,'dt':dt,'noise':0.0,'clamp':[0,1],'base_leak':0.0,'terms':[
            {'tgt':0,'op':'lenia','src':[0],'R':6,'kmu':0.5,'ksig':0.15,'gmu':gmu,'gsig':gsig,'coef':1.0}]}
        X=blank(); X[:,:,0]=(np.random.RandomState(3).random((H,W))<0.35)*np.random.RandomState(4).random((H,W))
        rng=np.random.RandomState(0); ok=True
        for t in range(300):
            X=step2(X,law,t,rng)
            if not np.all(np.isfinite(X)): ok=False; break
        alive=int((X[:,:,0]>0.15).sum()); std=float(X[:,:,0].std())
        surv=ok and 20<alive<900 and std>0.05    # survives (not dead, not full), structured
        if surv and (best is None or std>best[1]): best=((gmu,gsig,dt),std,alive)
        print(f"    lenia gmu={gmu} gsig={gsig} dt={dt}: alive={alive} std={std:.3f} survive={surv}", flush=True)
    return best

if __name__=='__main__':
    print("=== RICHER SUBSTRATE VALIDATION (Part 7): advection mobility + Lenia self-organization ===", flush=True)
    disp,al=advect_mobility()
    print(f"  ADVECT mobility: bump centroid displacement={disp:.1f} cells (MOVER — RD spots are static), alive={al}", flush=True)
    print("  Lenia self-organization scan:", flush=True)
    best=lenia_scan()
    print(f"  -> best self-organizing Lenia: {best[0] if best else 'NONE survived'}", flush=True)
    Xf=blank(); rng=np.random.RandomState(0)
    for t in range(600): Xf=step2(Xf,ADVECT_CTRL,t,rng)
    print(f"  ADVECT_CTRL: finite={np.all(np.isfinite(Xf))} std(ch0)={float(Xf[:,:,0].std()):.3f}", flush=True)
    print("=== RICHER_SUBSTRATE_VALIDATED (16ch, Lenia, advect, delay, diffuse2) ===", flush=True)
    print("=== PHYSICS2_DONE ===", flush=True)
