import os, random
os.environ['ME_MODE']='none'
from chem import World, Obj, C, TOK, T, show
from decisive import closure_test, sig, classes

# ECOLOGY/NICHE PRESSURE (Part 13) testing the diagnosis: make the trivial keystone NON-VIABLE via COSTLY ACTIVATE
# -> a pure activator starves unless REPAID (TRANSFER) by objects it activates -> reciprocal >=2-class dependency
# is the only viable organization. Set the PRESSURE only (per Part 16 — don't seed the farmer); seed GENERIC
# functional fragments + reproduction + mutation, and test whether >=2-class closure ARISES + is intervention-validated.
ECON=dict(HARVEST_IDLE=0.20, DECAY=0.20, HARVEST_ACTIVE=0.60, MUT_RATE=0.03, SPAWN_COST=0.5, SPAWN_ENDOW=1.5, ACTIVATE_COST=0.35)

FRAGS=[['BIND','ACTIVATE'],['BIND','TRANSFER'],['BIND','ACTIVATE','TRANSFER'],['ACTIVATE','BIND','TRANSFER'],
       ['BIND','COPY','SPAWN'],['ACTIVATE','NOP'],['TRANSFER','NOP'],['BIND','ACTIVATE','BIND','TRANSFER'],
       ['MATCH','S0','TRANSFER'],['COPY','SPAWN']]
def seed(w,n,g):
    for _ in range(n):
        y,x=g.randrange(w.H),g.randrange(w.W)
        instr=[T[o] for o in g.choice(FRAGS)] if g.random()<0.7 else [g.randrange(len(TOK)) for _ in range(g.randint(2,6))]
        o=Obj(instr,y,x,C['START_RES'],'seed'); o.active=(g.random()<0.5); o.act_ttl=6 if o.active else 0; w.add(o)

def run_eco(seedv, family='broadcast', ticks=12000, H=24, seed_n=280):
    g=random.Random(seedv); w=World(H=H,Wd=H,seed=seedv,inflow=1.2,family=family); seed(w,seed_n,g)
    n0=w.count(); traj=[]
    for t in range(ticks):
        w.tick()
        if t%2000==0 or t==ticks-1: traj.append((t,w.count(),len(set(sig(o) for o in w.live())),w.count('child')))
    return w,n0,traj

if __name__=='__main__':
    C.update(ECON)
    print("=== ECOLOGY TEST: costly ACTIVATE -> does reciprocal >=2-class closure ARISE? ===", flush=True)
    print(f"  economy: idle {ECON['HARVEST_IDLE']}/decay {ECON['DECAY']}/active {ECON['HARVEST_ACTIVE']} ACTIVATE_COST {ECON['ACTIVATE_COST']} (pure activator starves w/o repayment)", flush=True)
    for family in ['broadcast','base']:
        best=None
        for sv in (0,1,2):
            w,n0,traj=run_eco(sv,family=family)
            fin=w.count()
            if best is None or fin>best[0]: best=(fin,w,n0,traj)
        fin,w,n0,traj=best
        print(f"  -- family={family}: best final={fin} n0={n0} traj={traj} --", flush=True)
        if fin>=8:
            cls=classes(w, minmem=max(3,fin//15))
            print(f"     major classes: {[(show(list(s))[:22],n) for s,n in cls[:6]]}", flush=True)
            ct=closure_test(w)
            print(f"     CLOSURE: {ct['verdict']} (classes={ct.get('n_classes')}, essential={ct.get('n_essential')})", flush=True)
            for iv in ct.get('interventions',[]): print(f"       remove [{iv[0]}] x{iv[1]}: pop {iv[2]} vs control {iv[3]} (base {ct.get('base_final')})", flush=True)
    print("=== >=2 essential classes (each removal collapses vs control survives) = primitive causal closure (L2/L3 first milestone) ===", flush=True)
    print("=== ECOLOGY_DONE ===", flush=True)
