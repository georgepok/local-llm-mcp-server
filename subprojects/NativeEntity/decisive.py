import os, random
os.environ['ME_MODE']='none'
from chem import World, Obj, C, TOK, T, OPS, show
import chem

# PART 20 FIRST DECISIVE TEST: does >=2-class causal closure ARISE from NON-COMPLETE functional fragments (not seeded
# replicators)? Seed a soup of partial functional fragments in a BOOTSTRAPPING-DECAY economy (idle net slightly
# negative -> soup dies unless organized; ~100-tick window to organize). Run across interpreter families + inner
# mutation. For any persistent population: intervention-based closure detection (Part 8) — remove a program class,
# does the organization collapse (dependency) and can the remainder reconstruct it?
ECON=dict(HARVEST_IDLE=0.26, DECAY=0.20, HARVEST_ACTIVE=0.50)   # idle net ~-0.03/tick (dies ~100 ticks); active +0.21

FRAGS=[['BIND','ACTIVATE'],['BIND','COPY','SPAWN'],['BIND','TRANSFER'],['ACTIVATE','NOP'],['BIND','COPY'],
       ['MATCH','S0','ACTIVATE'],['BIND','ACTIVATE','TRANSFER'],['COPY','SPAWN'],['BIND','EXEC']]
def seed_soup(w,n,g):
    for _ in range(n):
        y,x=g.randrange(w.H),g.randrange(w.W)
        if g.random()<0.6: instr=[T[o] for o in g.choice(FRAGS)]                    # partial functional fragment
        else: instr=[g.randrange(len(TOK)) for _ in range(g.randint(2,6))]          # random short
        o=Obj(instr,y,x,C['START_RES'],'seed'); o.active=(g.random()<0.5); o.act_ttl=6 if o.active else 0; w.add(o)

def sig(o): return tuple(o.instr)                                                    # program class = exact instruction motif
def classes(w, minmem=4):
    from collections import Counter
    c=Counter(sig(o) for o in w.live()); return [(s,n) for s,n in c.most_common() if n>=minmem]

def run_world(family, seedv, seed_n=300, ticks=6000, H=26):
    g=random.Random(seedv); w=World(H=H,Wd=H,seed=seedv,inflow=0.9,family=family); seed_soup(w,seed_n,g)
    n0=w.count(); peak=n0
    for t in range(ticks):
        w.tick(); peak=max(peak,w.count())
    return w, n0, peak

def closure_test(w, ticks=250):
    """For the surviving population: remove each major class, measure collapse vs control-random-removal."""
    base_final=w.count()
    if base_final<8: return None
    cls=classes(w, minmem=max(4,base_final//12))
    if len(cls)<2: return {'n_classes':len(cls),'verdict':'single/low-diversity (no >=2-class structure)'}
    results=[]
    import copy
    def snapshot(): return [(o.instr[:],o.regs[:],o.res,o.ip,o.active,o.act_ttl,o.bound,o.y,o.x,o.age,o.cls,o.alive) for o in w.objs]
    def restore(snap):
        for o,s in zip(w.objs,snap): o.instr,o.regs,o.res,o.ip,o.active,o.act_ttl,o.bound,o.y,o.x,o.age,o.cls,o.alive=s[0][:],s[1][:],s[2],s[3],s[4],s[5],s[6],s[7],s[8],s[9],s[10],s[11]
    snap=snapshot()
    for s,n in cls[:3]:
        restore(snap)
        for o in w.live():
            if sig(o)==s: o.alive=False               # remove class
        for _ in range(ticks): w.tick()
        after_kill=w.count()
        restore(snap)                                  # control: remove n RANDOM objects
        live=w.live(); kill=w.rng.sample(live,min(n,len(live)))
        for o in kill: o.alive=False
        for _ in range(ticks): w.tick()
        after_ctrl=w.count()
        results.append((show(list(s))[:28], n, after_kill, after_ctrl))
    restore(snap)
    essential=[r for r in results if r[2] < 0.5*r[3] and r[2] < 0.5*base_final]   # class-removal collapses vs control survives
    return {'n_classes':len(cls),'base_final':base_final,'interventions':results,'n_essential':len(essential),
            'verdict':('>=2-class CLOSURE (multiple essential classes)' if len(essential)>=2 else
                       ('single keystone class' if len(essential)==1 else 'no essential class (persistence not organization-dependent)'))}

if __name__=='__main__':
    C.update(ECON)
    print("=== PART 20 DECISIVE TEST: does >=2-class causal closure ARISE from partial fragments? ===", flush=True)
    print(f"  economy: idle {ECON['HARVEST_IDLE']}/decay {ECON['DECAY']}/active {ECON['HARVEST_ACTIVE']} (idle net<0 -> organization required)", flush=True)
    for family in ['base','broadcast','autocat']:
        best=None
        for sv in (0,1,2):
            w,n0,peak=run_world(family,sv)
            fin=w.count()
            if best is None or fin>best[0]: best=(fin,peak,n0,w)
        fin,peak,n0,w=best
        reg='L0 collapse' if fin==0 else ('persist' if fin>=8 else 'transient')
        line=f"  family={family:10s} n0={n0} peak={peak} final={fin} motifs={len(set(sig(o) for o in w.live()))} children={w.count('child')} -> {reg}"
        print(line, flush=True)
        if fin>=8:
            ct=closure_test(w)
            print(f"      CLOSURE: {ct['verdict']} (classes={ct.get('n_classes')})", flush=True)
            for iv in ct.get('interventions',[]): print(f"        remove [{iv[0]}] x{iv[1]}: pop {iv[2]} vs control {iv[3]} (base {ct['base_final']})", flush=True)
    print("=== >=2 essential classes w/ control-survives = primitive executable causal closure (first milestone) ===", flush=True)
    print("=== DECISIVE_DONE ===", flush=True)
