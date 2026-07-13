import os, math, random
os.environ['ME_MODE']='none'
import numpy as np
np.seterr(over='ignore', invalid='ignore')
from physics import step, init_field, lap, GRAY_SCOTT, DIFFUSE_ONLY, OSC, GROWTH

# SYNTHESIZED_ARTIFICIAL_PHYSICS_V1 — Parts 9-11: MECHANISM-AGNOSTIC organizational diagnostics + causal interventions.
# Every synthesized world-law is scored by these generic detectors (NOT by any task, NOT by one hand-coded 'organism'
# definition). Pareto vector; artifact gates reject static/homogeneous/chaotic/exploded. NO consciousness/autopoiesis.

# ---- structure field + connected-component detection (Part 10 detector: persistent spatial clusters) ----
def sfield(X):                                   # per-cell 'structure' = L2 deviation from global channel-mean
    mu=X.mean(axis=(0,1),keepdims=True); return np.sqrt(((X-mu)**2).sum(2))
def blobs(mask, smin=4, smax=400):
    H,W=mask.shape; lab=np.zeros((H,W),int); out=[]; cur=0
    for i in range(H):
        for j in range(W):
            if mask[i,j] and lab[i,j]==0:
                cur+=1; st=[(i,j)]; lab[i,j]=cur; cells=[]
                while st:
                    y,x=st.pop(); cells.append((y,x))
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=y+dy,x+dx
                        if 0<=ny<H and 0<=nx<W and mask[ny,nx] and lab[ny,nx]==0: lab[ny,nx]=cur; st.append((ny,nx))
                if smin<=len(cells)<=smax: out.append(cells)
    return out
def detect(X, k=1.2):
    s=sfield(X); thr=s.mean()+k*s.std(); return blobs(s>thr), s
def spatial_autocorr(f):                          # lag-1 spatial autocorr of a field (smooth->high, salt&pepper->~0)
    a=f-f.mean(); v=(a*a).mean()+1e-9
    return float(((a*np.roll(a,1,0)).mean()+(a*np.roll(a,1,1)).mean())/(2*v))

# ---- rollout that captures settled snapshots ----
def capture(law, kind, seed, T=1500, warm=600, every=60):
    D=law.get('D',8); X=init_field(32,32,D,kind,seed); rng=np.random.RandomState(1000+seed); snaps=[]
    for t in range(T):
        X=step(X,law,t,rng)
        if not np.all(np.isfinite(X)): return None, None
        if t>=warm and (t-warm)%every==0: snaps.append(X.copy())
    return snaps, rng

# ---- damage/intervention rollout from a given state (Part 11) ----
def run_from(X0, law, steps, seed=7):
    X=X0.copy(); rng=np.random.RandomState(seed)
    for t in range(steps):
        X=step(X,law,10000+t,rng)
        if not np.all(np.isfinite(X)): return None
    return X

def high_mass(X,k=1.2): s=sfield(X); return float((s>s.mean()+k*s.std()).sum())

def metrics(law, kind='droplet', seeds=(0,1,2), T=1500):
    loc=[]; per=[]; ns=[]; act=[]; het=[]; sac=[]; finite=0; rep=[]; caus=[]
    for sd in seeds:
        snaps,_=capture(law,kind,sd,T=T)
        if snaps is None or len(snaps)<3: continue
        finite+=1
        for X in snaps:
            s=sfield(X); N=s.size
            ipr=1.0-(s.sum()**2)/(N*(s**2).sum()+1e-9)   # inverse participation: localized(spiky)->high
            loc.append(ipr); het.append(float(s.std())); sac.append(spatial_autocorr(s))
            ns.append(len(detect(X)[0]))
        for a,b in zip(snaps[:-1],snaps[1:]):
            fa,fb=sfield(a).ravel(),sfield(b).ravel()
            per.append(float(np.corrcoef(fa,fb)[0,1]) if fa.std()>1e-6 and fb.std()>1e-6 else 0.0)  # pattern persistence
            act.append(float(np.abs(b-a).mean()))
        # --- Part 11 causal probes from a settled snapshot ---
        Xs=snaps[-1]; bl=detect(Xs)[0]
        if bl:
            big=max(bl,key=len); ys=[c[0] for c in big]; xs=[c[1] for c in big]
            y0,y1,x0,x1=min(ys),max(ys)+1,min(xs),max(xs)+1
            pre=high_mass(Xs)
            # (D) REPAIR: zero the blob's bounding box, measure high-mass recovery
            Xd=Xs.copy(); Xd[y0:y1,x0:x1,:]=0.0; dmg=high_mass(Xd)
            Xr=run_from(Xd,law,400)
            if Xr is not None and pre-dmg>1: rep.append(float(np.clip((high_mass(Xr)-dmg)/(pre-dmg),0,1)))
            # (Part11) CAUSAL: scramble-INTERNAL (preserve mass, destroy organization) -> does org self-restore?
            Xc=Xs.copy(); reg=Xc[y0:y1,x0:x1,:].reshape(-1,Xc.shape[2]); idx=np.random.RandomState(sd).permutation(len(reg))
            Xc[y0:y1,x0:x1,:]=reg[idx].reshape(y1-y0,x1-x0,Xc.shape[2])   # spatial shuffle of full state-vectors in region
            org_pre=spatial_autocorr(sfield(Xs)[y0:y1,x0:x1]); org_scr=spatial_autocorr(sfield(Xc)[y0:y1,x0:x1])
            Xcr=run_from(Xc,law,400)
            if Xcr is not None and org_pre-org_scr>0.02:
                org_rec=spatial_autocorr(sfield(Xcr)[y0:y1,x0:x1])
                caus.append(float(np.clip((org_rec-org_scr)/(org_pre-org_scr),0,1)))  # self-restoration of organization
    m=lambda a: round(float(np.mean(a)),3) if a else 0.0
    het_m=m(het); act_m=m(act); per_m=m(per); ns_m=m(ns)
    alive = (finite>0) and (0.0004<act_m<0.6) and (het_m>0.01) and (per_m>0.25) and (0.3<=ns_m<=60)
    return {'finite':finite,'alive':bool(alive),'localization':m(loc),'persistence':per_m,'n_struct':ns_m,
            'activity':act_m,'heterogen':het_m,'sp_autocorr':m(sac),'repair':m(rep),'causal_selfrestore':m(caus)}

# ---- artifact controls to VALIDATE the engine discriminates ----
STATIC={'D':8,'dt':1.0,'noise':0.0,'clamp':[-2,2],'terms':[]}                                   # frozen: fail activity
NOISE ={'D':8,'dt':1.0,'noise':0.15,'clamp':[-2,2],'terms':[{'tgt':0,'op':'decay','coef':0.05}]}# pure noise: fail persistence
CHAOS ={'D':8,'dt':0.9,'noise':0.02,'clamp':[-3,3],'terms':[                                     # turbulent: fail persistence
    {'tgt':0,'op':'react','src':[0,1],'coef':2.2},{'tgt':1,'op':'react','src':[1,0],'coef':-2.2},
    {'tgt':0,'op':'diffuse','coef':0.4},{'tgt':1,'op':'diffuse','coef':0.4}]}
def gs_init(seed):
    X=np.zeros((32,32,8),np.float32); X[:,:,0]=1.0; g=np.random.RandomState(seed)
    for _ in range(6): y,x=g.randint(6,26),g.randint(6,26); X[y-2:y+2,x-2:x+2,0]=0.5; X[y-2:y+2,x-2:x+2,1]=0.25
    return X

if __name__=='__main__':
    print("=== SYNTHESIZED_ARTIFICIAL_PHYSICS_V1 — Parts 9-11 DIAGNOSTICS ENGINE VALIDATION ===", flush=True)
    print("  (engine must score GS high on localization/persistence/repair/causal, and REJECT static/noise/chaos/diffuse)", flush=True)
    # GS needs its own u~1 init: monkey-patch capture via custom snapshots
    def gs_metrics():
        loc=per=None; snaps=[]; rng=np.random.RandomState(7); X=gs_init(0)
        for t in range(3200):
            X=step(X,GRAY_SCOTT,t,rng)
            if t>=1200 and (t-1200)%60==0: snaps.append(X.copy())
        # reuse metric internals by wrapping a fake law with precomputed snaps:
        return snaps
    # generic laws via standard init families:
    print(f"  {'law':14s} {'alive':5s} {'loc':>5s} {'persist':>7s} {'nstr':>5s} {'act':>6s} {'het':>5s} {'repair':>6s} {'causal':>6s}", flush=True)
    for name,law,kind in [('DIFFUSE_ONLY',DIFFUSE_ONLY,'droplet'),('NOISE',NOISE,'uniform'),('CHAOS',CHAOS,'impulse'),
                          ('OSC',OSC,'impulse'),('GROWTH',GROWTH,'impulse')]:
        r=metrics(law,kind=kind,seeds=(0,1,2))
        print(f"  {name:14s} {str(r['alive']):5s} {r['localization']:>5.2f} {r['persistence']:>7.2f} {r['n_struct']:>5.1f} {r['activity']:>6.3f} {r['heterogen']:>5.2f} {r['repair']:>6.2f} {r['causal_selfrestore']:>6.2f}", flush=True)
    # Gray-Scott with proper init, scored by same metric internals:
    import types
    _orig_init=init_field
    def _gsinit(H,W,D,kind,seed): return gs_init(seed) if kind=='gs' else _orig_init(H,W,D,kind,seed)
    import physics; physics.init_field=_gsinit
    globals()['init_field']=_gsinit
    r=metrics(GRAY_SCOTT,kind='gs',seeds=(0,1,2),T=3200)
    print(f"  {'GRAY_SCOTT':14s} {str(r['alive']):5s} {r['localization']:>5.2f} {r['persistence']:>7.2f} {r['n_struct']:>5.1f} {r['activity']:>6.3f} {r['heterogen']:>5.2f} {r['repair']:>6.2f} {r['causal_selfrestore']:>6.2f}", flush=True)
    print("=== ENGINE VALIDATION: GS should be alive=True w/ high loc+persist+repair; static/noise/chaos alive=False ===", flush=True)
    print("=== DIAG_VALIDATION_DONE ===", flush=True)
