import os, math, random
os.environ['ME_MODE']='none'
import numpy as np
np.seterr(over='ignore', invalid='ignore')
from physics import step, init_field, GRAY_SCOTT
from diag import sfield, detect, spatial_autocorr, high_mass, capture, run_from
from worldlaws import CLAUDE, random_law, L, D

# HARDENED, ARTIFACT-CORRECTED diagnostics (Parts 9C/11/14/21). Kills three confounds that made the raw batch
# untrustworthy: (1) frozen attractors scoring persistence~1, (2) generic refill scoring repair, (3) diffusive
# re-smoothing scoring causal. Decisive CAUSAL tests: inflow-cut dependence (Case B vs C), diffusion-NULL-corrected
# organization self-restoration (Part 11), refill-corrected structure repair. Classify Case A/B/C/D per Part 21.
DIFFNULL={'D':D,'dt':0.5,'noise':0.0,'clamp':[-6,6],'terms':[{'tgt':c,'op':'diffuse','coef':0.2} for c in range(D)]}
def strip_drive(law):   # cut ALL nonequilibrium drives: inflow AND supply-toward-nonzero (a reservoir feed). Keep supply->0 (that's decay).
    keep=[t for t in law['terms'] if not (t['op']=='inflow' or (t['op']=='supply' and float(t.get('target',0))>0.05))]
    l=dict(law); l['terms']=keep; return l
def corrf(a,b):
    a=a.ravel()-a.mean(); b=b.ravel()-b.mean()
    if a.std()<1e-6 or b.std()<1e-6: return 0.0
    return float((a*b).mean()/(a.std()*b.std()))

def hardened(law, kind, seeds=(0,1), T=1200, warm=600):
    infdep=[]; orgc=[]; reps=[]; dyn=[]; loc=[]; has=0
    for sd in seeds:
        snaps,_=capture(law,kind,sd,T=T,warm=warm,every=60)
        if not snaps or len(snaps)<3: continue
        Xs=snaps[-1]; s=sfield(Xs); pre=high_mass(Xs); amp0=float(s.mean())
        loc.append(1.0-(s.sum()**2)/(s.size*(s**2).sum()+1e-9))
        if pre<8: continue
        has+=1
        # --- Part 9C active resource coupling: CUT the drive (inflow+supply-feed); does structure AMPLITUDE decay?
        #     Use ABSOLUTE structure amplitude (mean sfield) not the scale-invariant thresholded count. (B frozen vs C dissipative) ---
        Xcut=run_from(Xs,strip_drive(law),700)
        if Xcut is not None: infdep.append(float(np.clip(1-float(sfield(Xcut).mean())/(amp0+1e-6),0,1)))
        act=np.mean([np.abs(b-a).mean() for a,b in zip(snaps[-3:-1],snaps[-2:])]); dyn.append(1.0 if act>0.0008 else 0.0)
        bl=detect(Xs)[0]
        if not bl: continue
        big=max(bl,key=len); ys=[c[0] for c in big]; xs=[c[1] for c in big]
        y0,y1,x0,x1=min(ys),max(ys)+1,min(xs),max(xs)+1; hh,ww=y1-y0,x1-x0
        # --- Part 11 organization causality: scramble internal org (preserve mass), does the LAW restore the SPECIFIC original
        #     pattern MORE than a pure-diffusion null? (identity-restoration, longer window) ---
        reg=Xs[y0:y1,x0:x1,:].reshape(-1,D); perm=np.random.RandomState(sd).permutation(len(reg))
        Xc=Xs.copy(); Xc[y0:y1,x0:x1,:]=reg[perm].reshape(hh,ww,D)
        pre_sf=sfield(Xs)[y0:y1,x0:x1]
        if corrf(sfield(Xc)[y0:y1,x0:x1],pre_sf)<0.9:   # scramble actually disrupted the pattern
            rl=run_from(Xc,law,700); rn=run_from(Xc,DIFFNULL,700)
            if rl is not None and rn is not None:
                c_law=corrf(sfield(rl)[y0:y1,x0:x1],pre_sf); c_null=corrf(sfield(rn)[y0:y1,x0:x1],pre_sf)
                orgc.append(float(np.clip(c_law-c_null,-1,1)))   # restores ORIGINAL pattern beyond diffusion
        # --- Part 9D repair, REFILL corrected: high-s mass re-formed in a damaged STRUCTURE region vs a damaged EMPTY region ---
        if hh<30 and ww<30:
            Xd=Xs.copy(); Xd[y0:y1,x0:x1,:]=0.0
            ey=min(y0+16,32-hh) if 32-hh>0 else 0; ex=min(x0+16,32-ww) if 32-ww>0 else 0
            Xe=Xs.copy(); Xe[ey:ey+hh,ex:ex+ww,:]=0.0
            Rd=run_from(Xd,law,500); Re=run_from(Xe,law,500)
            if Rd is not None and Re is not None:
                def hm(X,a,b,c,d): sf=sfield(X); return float((sf[a:b,c:d]>sf.mean()+1.0*sf.std()).sum())
                struct_rec=hm(Rd,y0,y1,x0,x1)/(hh*ww+1e-6); empty_fill=hm(Re,ey,ey+hh,ex,ex+ww)/(hh*ww+1e-6)
                reps.append(float(np.clip(struct_rec-empty_fill,0,1)))
    m=lambda a: round(float(np.mean(a)),3) if a else 0.0
    r={'struct':has,'localization':m(loc),'inflow_dep':m(infdep),'dynamic':m(dyn),'org_caus':m(orgc),'repair_c':m(reps)}
    if has==0: r['case']='A'
    elif r['inflow_dep']<0.3: r['case']='B'          # persists without throughput -> static/frozen attractor
    elif r['org_caus']<=0.12: r['case']='C'          # dissipative + dynamic, but organization does NOT self-restore
    else: r['case']='D'                               # MILESTONE: dissipative + organization causally self-restores
    r['milestone']=bool(r['case']=='D' and r['localization']>=0.2 and r['inflow_dep']>=0.3 and r['org_caus']>0.12)
    return r

FROZEN_TURING=L([{'tgt':0,'op':'diffuse','coef':0.05},{'tgt':1,'op':'diffuse','coef':0.5},   # pure Turing, NO feed -> Case B
  {'tgt':0,'op':'catalyze','cat':0,'src':[0],'coef':0.4},{'tgt':0,'op':'react','src':[1],'coef':-0.5},
  {'tgt':1,'op':'react','src':[0,0],'coef':0.4},{'tgt':1,'op':'decay','coef':0.1},{'tgt':0,'op':'decay','coef':0.06}],dt=0.25,clamp=(0,3))
STATIC=L([],dt=1.0)
def gs_init(seed):
    X=np.zeros((32,32,8),np.float32); X[:,:,0]=1.0; g=np.random.RandomState(seed)
    for _ in range(6): y,x=g.randint(6,26),g.randint(6,26); X[y-2:y+2,x-2:x+2,0]=0.5; X[y-2:y+2,x-2:x+2,1]=0.25
    return X

if __name__=='__main__':
    import physics; _oi=init_field
    physics.init_field=lambda H,W,Dd,kind,seed:(gs_init(seed) if kind=='gs' else _oi(H,W,Dd,kind,seed))
    import diag; diag.init_field=physics.init_field
    KINDS=['droplet','impulse','uniform']
    def best(law,kinds=KINDS,T=1200):
        rs=[hardened(law,kind=k,T=T) for k in kinds]
        return max(rs,key=lambda r:(r['case']=='D',r['org_caus']+r['inflow_dep']+r['repair_c']))
    print("=== HARDENED artifact-corrected re-score (inflow-cut dep / diffusion-null org-causality / refill-corrected repair) ===", flush=True)
    print(f"  {'law':24s} {'case':4s} {'loc':>4s} {'infdep':>6s} {'dyn':>4s} {'orgC':>5s} {'repC':>5s} {'MILE':>4s}", flush=True)
    print("  -- CONTROLS --", flush=True)
    for nm,law,k in [('GRAY_SCOTT(ctrl)',GRAY_SCOTT,'gs'),('FROZEN_TURING(ctrl)',FROZEN_TURING,'droplet'),('STATIC(ctrl)',STATIC,'droplet')]:
        r=best(law,kinds=[k]) if k=='gs' else best(law)
        print(f"  {nm:24s} {r['case']:4s} {r['localization']:>4.2f} {r['inflow_dep']:>6.2f} {r['dynamic']:>4.1f} {r['org_caus']:>5.2f} {r['repair_c']:>5.2f} {'HIT' if r['milestone'] else '':>4s}", flush=True)
    if os.environ.get('CTRL_ONLY'): print("=== CTRL_ONLY: positive control GS must reach Case D before batch is trusted ===",flush=True); print("=== DIAG2_DONE ===",flush=True); raise SystemExit
    print("  -- CLAUDE batch --", flush=True)
    cmile=0; ccase={}
    for nm,law in CLAUDE.items():
        r=best(law); cmile+=r['milestone']; ccase[r['case']]=ccase.get(r['case'],0)+1
        print(f"  {nm:24s} {r['case']:4s} {r['localization']:>4.2f} {r['inflow_dep']:>6.2f} {r['dynamic']:>4.1f} {r['org_caus']:>5.2f} {r['repair_c']:>5.2f} {'HIT' if r['milestone'] else '':>4s}", flush=True)
    print("  -- RANDOM batch (40) --", flush=True)
    rmile=0; rcase={}
    for s in range(40):
        r=best(random_law(1000+s)); rmile+=r['milestone']; rcase[r['case']]=rcase.get(r['case'],0)+1
        if r['milestone']: print(f"  random_{s:<17d} {r['case']:4s} {r['localization']:>4.2f} {r['inflow_dep']:>6.2f} {r['dynamic']:>4.1f} {r['org_caus']:>5.2f} {r['repair_c']:>5.2f} HIT", flush=True)
    print(f"=== HARDENED ENRICHMENT: Claude Case-D milestone={cmile}/{len(CLAUDE)} cases={ccase} | Random Case-D={rmile}/40 cases={rcase} ===", flush=True)
    print("=== DIAG2_DONE ===", flush=True)
