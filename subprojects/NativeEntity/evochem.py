import os, random
os.environ['ME_MODE']='none'
from chem import World, Obj, C, TOK, T, show
from decisive import closure_test, sig, classes

# INNER EVOLUTION (Part 10): reproduction MUST occur for selection to explore program space. Broadcast family
# (activation-sets form) + reproduction-enabling economy + mutation-on-spawn + dense activator/copier/template
# seeding + long run. Track whether a REPRODUCING organization develops and reaches >=2-class causal closure.
ECON=dict(HARVEST_IDLE=0.24, DECAY=0.20, HARVEST_ACTIVE=0.55, MUT_RATE=0.03, SPAWN_COST=0.5, SPAWN_ENDOW=1.5)

SEED_FRAGS=[['ACTIVATE','NOP'],['ACTIVATE','ACTIVATE'],['BIND','COPY','SPAWN'],['BIND','COPY','SPAWN'],
            ['COPY','SPAWN'],['BIND','ACTIVATE'],['S0','S1','S2'],['MATCH','S0','COPY','SPAWN']]
def seed(w,n,g):
    for _ in range(n):
        y,x=g.randrange(w.H),g.randrange(w.W)
        instr=[T[o] for o in g.choice(SEED_FRAGS)] if g.random()<0.7 else [g.randrange(len(TOK)) for _ in range(g.randint(2,6))]
        o=Obj(instr,y,x,C['START_RES'],'seed'); o.active=(g.random()<0.5); o.act_ttl=6 if o.active else 0; w.add(o)

def run_evo(seedv, ticks=15000, H=24, seed_n=280):
    g=random.Random(seedv); w=World(H=H,Wd=H,seed=seedv,inflow=1.1,family='broadcast'); seed(w,seed_n,g)
    n0=w.count(); traj=[]; repro_total=0; prev_children=0
    for t in range(ticks):
        w.tick()
        if t%1500==0 or t==ticks-1:
            live=w.count(); ch=w.count('child'); mots=len(set(sig(o) for o in w.live()))
            traj.append((t,live,mots,ch));
    return w, n0, traj

if __name__=='__main__':
    C.update(ECON)
    print("=== INNER-EVOLUTION run (broadcast family + reproduction + mutation) — does organization develop? ===", flush=True)
    print(f"  economy: idle {ECON['HARVEST_IDLE']}/decay {ECON['DECAY']}/active {ECON['HARVEST_ACTIVE']} mut {ECON['MUT_RATE']} spawn_cost {ECON['SPAWN_COST']}", flush=True)
    best=None
    for sv in (0,1,2):
        w,n0,traj=run_evo(sv)
        fin=w.count()
        print(f"  seed{sv}: n0={n0} traj[(t,live,motifs,children)]={traj}", flush=True)
        if best is None or fin>best[0]: best=(fin,w,n0)
    fin,w,n0=best
    print(f"  BEST final={fin} motifs={len(set(sig(o) for o in w.live()))} children={w.count('child')}", flush=True)
    # top classes
    cls=classes(w, minmem=max(3,fin//15))
    print(f"  major classes (>=minmem): {[(show(list(s))[:24],n) for s,n in cls[:6]]}", flush=True)
    if fin>=8:
        ct=closure_test(w)
        print(f"  CLOSURE: {ct['verdict']} (classes={ct.get('n_classes')}, essential={ct.get('n_essential')})", flush=True)
        for iv in ct.get('interventions',[]): print(f"    remove [{iv[0]}] x{iv[1]}: pop {iv[2]} vs control {iv[3]} (base {ct.get('base_final')})", flush=True)
    print("=== does population GROW + reproduce + reach >=2-class closure over 15k ticks? ===", flush=True)
    print("=== EVOCHEM_DONE ===", flush=True)
