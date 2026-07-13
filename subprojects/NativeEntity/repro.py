import os
os.environ['ME_MODE']='none'
import numpy as np
np.seterr(over='ignore', invalid='ignore')
from physics import step, GRAY_SCOTT
from diag import sfield, detect
from worldlaws import CLAUDE, random_law
from worldlaws2 import STRONG
import physics

# Part 9G (REPRODUCTION/propagation) + 9H (NOVELTY growth): the HARD milestones (Case E/F) that separate GS-like
# REPLICATING/open-ended structures from generic static-dissipative blobs. Case D was generic + no LLM enrichment;
# re-ask enrichment HERE. Reproduction = ONE seed colonizes an empty resource-filled region by making NEW structures
# (not exact copy, Part 9G). Novelty = structure configuration keeps changing late (world not settled, Part 9H).
def seed_init(rc, pc):   # resource channel rc filled + ONE seed cluster of PRODUCT channel pc in a corner (colonization test)
    def f(H,W,D,kind,seed):
        g=np.random.RandomState(seed); X=(0.02*g.standard_normal((H,W,D))).astype(np.float32); X[:,:,rc]+=1.0
        X[3:8,3:8,pc]+=0.3                         # seed only ONE product channel (avoids static specks in unused channels)
        return X
    return f

def run_repro(law, rc, pc, seed=0, T=4000):
    law=dict(law); law['base_leak']=0.0            # leak kills replication (validated: GS colonizes only w/o leak); reproduction test runs leak-free
    physics.init_field=seed_init(rc,pc)
    X=physics.init_field(32,32,law.get('D',8),'x',seed); rng=np.random.RandomState(seed)
    counts=[]; spreads=[]; centroids=[]
    for t in range(T):
        X=step(X,law,t,rng)
        if not np.all(np.isfinite(X)): return None
        if t>=400 and t%300==0:
            bl,s=detect(X); counts.append(len(bl)); spreads.append(float((s>s.mean()+s.std()).mean()))
            centroids.append(tuple(round(float(np.mean([c[k] for b in bl for c in b])),1) if bl else 0 for k in (0,1)))
    if len(counts)<4: return None
    early=np.mean(counts[:2])+1e-6; late=np.mean(counts[-3:])
    spread_early=np.mean(spreads[:2]); spread_late=np.mean(spreads[-3:])
    # novelty proxy: is the structure configuration still CHANGING late (count variance in last half) vs settled?
    late_change=float(np.std(counts[len(counts)//2:]))
    return {'repro_ratio':round(float(late/early),2),'final_count':round(float(late),1),
            'spread_gain':round(float(spread_late-spread_early),3),'late_change':round(late_change,2),
            'colonized':bool(late>=4 and late>early*1.8),   # Case E: one seed multiplies into many structures (validated on GS_REPLICATE)
            'unsettled':bool(late_change>1.0)}                                                 # Case F proxy: still generating motifs

GS_REPLICATE={'D':8,'dt':1.0,'noise':0.0,'clamp':[0,1.3],'base_leak':0.0,'terms':[   # k=0.060 REPLICATING regime (validated: colonizes) — GRAY_SCOTT k=0.0628 makes STABLE non-replicating spots
    {'tgt':0,'op':'diffuse','coef':0.16},{'tgt':1,'op':'diffuse','coef':0.08},
    {'tgt':0,'op':'react','src':[0,1,1],'coef':-1.0},{'tgt':1,'op':'react','src':[0,1,1],'coef':1.0},
    {'tgt':0,'op':'supply','coef':0.037,'target':1.0},{'tgt':1,'op':'supply','coef':0.097,'target':0.0}]}
PAIRS=[(0,1),(1,0),(1,2),(3,0),(0,2),(3,4)]   # (resource,product) seed pairs; GS=(0,1). Applied equally to LLM & random.
def best_repro(law):
    best=None
    for rc,pc in PAIRS:
        r=run_repro(law,rc,pc)
        if r is None: continue
        if best is None or (r['colonized'],r['repro_ratio'])>(best['colonized'],best['repro_ratio']): best=r; best['rc']=rc; best['pc']=pc
    return best or {'repro_ratio':0,'final_count':0,'spread_gain':0,'late_change':0,'colonized':False,'unsettled':False,'rc':-1,'pc':-1}

if __name__=='__main__':
    print("=== Case-E (reproduction/colonization) + Case-F (novelty/unsettled) detectors — does LLM enrich HERE? ===", flush=True)
    gsc=best_repro(GS_REPLICATE)
    print(f"  GS_REPLICATE(ctrl) colonization: repro_ratio={gsc['repro_ratio']:.1f} final={gsc['final_count']:.0f} COLONIZE={gsc['colonized']} rc={gsc['rc']} pc={gsc.get('pc')} (must be True for detector validity)", flush=True)
    if not gsc['colonized']: print("!!! CONTROL FAILED — detector invalid, aborting", flush=True); print("=== REPRO_DONE ===", flush=True); raise SystemExit
    def tally(name, items):
        col=0; uns=0
        for nm,law in items:
            r=best_repro(law)
            if r['colonized'] or r['unsettled']:
                print(f"  [{name}] {nm:22s} repro_ratio={r['repro_ratio']:.1f} final={r['final_count']:.0f} spread_gain={r['spread_gain']:+.2f} late_chg={r['late_change']:.1f} COLONIZE={r['colonized']} UNSETTLED={r['unsettled']} rc={r['rc']}", flush=True)
            col+=r['colonized']; uns+=r['unsettled']
        print(f"  >>> {name}: n={len(items)} Case-E(colonize)={col} Case-F(unsettled)={uns}", flush=True)
        return col,uns,len(items)
    llm=list(CLAUDE.items())+list(STRONG.items())
    rnd=[(f'random_{s}',random_law(2000+s)) for s in range(30)]
    lc,lu,ln=tally('LLM',llm)
    rc_,ru,rn=tally('RAND',rnd)
    print(f"=== Case-E/F ENRICHMENT: LLM colonize={lc}/{ln} unsettled={lu}/{ln} | RAND colonize={rc_}/{rn} unsettled={ru}/{rn} ===", flush=True)
    print("=== REPRO_DONE ===", flush=True)
