import os
os.environ['ME_MODE']='none'
import numpy as np
from physics import GRAY_SCOTT
from diag2 import hardened, FROZEN_TURING
from worldlaws2 import STRONG
import diag, physics

# FAIR re-score: channel-agnostic "primordial soup" init families (Part 5: low-amplitude field + resource level +
# sparse impulses; NOT a seeded organism). Fills a CANDIDATE resource channel to operating level (try ch0/1/3 since
# channels are neutral) + sparse nucleation. Score each law BEST-over-families x seeds. GS must reach Case D here to
# confirm the protocol is fair; then ask whether NON-GS strong-drive laws reach Case D.
def soup(rc):   # resource channel rc filled ~1, sparse impulse nucleation on all channels
    def f(H,W,D,kind,seed):
        g=np.random.RandomState(seed); X=(0.02*g.standard_normal((H,W,D))).astype(np.float32)
        X[:,:,rc]+=1.0
        m=(g.random((H,W))<0.10)                      # 10% nucleation sites
        for c in range(D): X[:,:,c]+=0.3*m*g.standard_normal((H,W))
        return X
    return f
FAM={'soup0':soup(0),'soup1':soup(1),'soup3':soup(3)}

def best_fair(law):
    best=None
    for nm,fn in FAM.items():
        physics.init_field=fn; diag.init_field=fn
        r=hardened(law,kind='x',seeds=(0,1),T=1500,warm=900); r['fam']=nm
        if best is None or (r['case']=='D',r['org_caus']+r['inflow_dep']+r['repair_c'])>(best['case']=='D',best['org_caus']+best['inflow_dep']+best['repair_c']): best=r
    return best

if __name__=='__main__':
    print("=== FAIR RE-SCORE: primordial-soup inits (resource-channel + sparse nucleation), best-over-families x seeds ===", flush=True)
    print(f"  {'law':24s} {'case':4s} {'fam':6s} {'loc':>4s} {'infdep':>6s} {'orgC':>5s} {'repC':>5s} {'MILE':>4s}", flush=True)
    print("  -- CONTROLS --", flush=True)
    for nm,law in [('GRAY_SCOTT(ctrl)',GRAY_SCOTT),('FROZEN_TURING(ctrl)',FROZEN_TURING)]:
        r=best_fair(law)
        print(f"  {nm:24s} {r['case']:4s} {r['fam']:6s} {r['localization']:>4.2f} {r['inflow_dep']:>6.2f} {r['org_caus']:>5.2f} {r['repair_c']:>5.2f} {'HIT' if r['milestone'] else '':>4s}", flush=True)
    print("  -- STRONG-DRIVE batch --", flush=True)
    mile=0; cases={}; hits=[]
    for nm,law in STRONG.items():
        r=best_fair(law); mile+=r['milestone']; cases[r['case']]=cases.get(r['case'],0)+1
        if r['milestone']: hits.append(nm)
        print(f"  {nm:24s} {r['case']:4s} {r['fam']:6s} {r['localization']:>4.2f} {r['inflow_dep']:>6.2f} {r['org_caus']:>5.2f} {r['repair_c']:>5.2f} {'HIT' if r['milestone'] else '':>4s}", flush=True)
    print(f"=== FAIR RESULT: Case-D milestone={mile}/{len(STRONG)} cases={cases} hits={hits} ===", flush=True)
    print("=== (GS must be Case-D here for protocol validity; non-GS hits = surprise candidates) ===", flush=True)
    print("=== INITFAIR_DONE ===", flush=True)
