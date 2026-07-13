import os, random
os.environ['ME_MODE']='none'
from chem import World, Obj, C, TOK, show, OPS
import chem

# FIRST EMERGENCE PROBE (baseline for Part 20): seed RANDOM non-complete fragments (Part-4 families) into the base
# interpreter; observe population dynamics + degeneracy controls (Part 12). Question: does anything PERSIST beyond
# transient execution (Level 1+), or collapse (Level 0)? NOT the decisive test yet — establishes the substrate baseline.
def rand_instr(g, lo=3, hi=14): return [g.randrange(len(TOK)) for _ in range(g.randint(lo,hi))]

def seed(w, family, n, g):
    for _ in range(n):
        y,x=g.randrange(w.H),g.randrange(w.W)
        if family=='random': instr=rand_instr(g,4,16)
        elif family=='shortops': instr=rand_instr(g,1,3)
        elif family=='grammar':   # bias toward op-rich (executable) sequences
            instr=[g.choice([chem.T[o] for o in OPS]) for _ in range(g.randint(4,12))]
        elif family=='clusters':
            for _ in range(g.randint(2,4)):
                o=Obj(rand_instr(g,3,10),y,x,C['START_RES'],'seed'); o.active=(g.random()<0.5); o.act_ttl=8 if o.active else 0; w.add(o)
            continue
        else: instr=rand_instr(g,3,12)
        o=Obj(instr,y,x,C['START_RES'],'seed'); o.active=(g.random()<0.4); o.act_ttl=8 if o.active else 0; w.add(o)

def motifs(w):  # distinct instruction sequences among live objects
    return len(set(tuple(o.instr) for o in w.live()))

def probe(family, seed_n=200, ticks=3000, H=28, seedv=0, inflow=0.5):
    g=random.Random(seedv); w=World(H=H,Wd=H,seed=seedv,inflow=inflow); seed(w,family,seed_n,g)
    n0=w.count(); traj=[]; peak=n0; spawns0=0
    for t in range(ticks):
        w.tick()
        if t%300==0 or t==ticks-1:
            live=w.count(); traj.append(live); peak=max(peak,live)
    live=w.count(); mot=motifs(w); tot_res=sum(o.res for o in w.live())
    ages=[o.age for o in w.live()]; maxage=max(ages) if ages else 0
    children=w.count('child')
    return dict(family=family, n0=n0, final=live, peak=peak, motifs=mot, maxage=maxage, children=children,
                tot_res=round(tot_res,1), traj=traj[-6:])

ECONOMIES={
 'harsh(default)': dict(HARVEST_IDLE=0.10, DECAY=0.25, HARVEST_ACTIVE=0.45),
 'permissive':     dict(HARVEST_IDLE=0.28, DECAY=0.20, HARVEST_ACTIVE=0.55),   # persistent soup + strong active orgs
 'very-permissive':dict(HARVEST_IDLE=0.34, DECAY=0.18, HARVEST_ACTIVE=0.60),
}
if __name__=='__main__':
    print("=== EMERGENCE PROBE (base interpreter, random Part-4 fragments) — persistence or collapse across economies? ===", flush=True)
    base={k:C[k] for k in ('HARVEST_IDLE','DECAY','HARVEST_ACTIVE')}
    for econ,vals in ECONOMIES.items():
        C.update(vals)
        print(f"  -- economy: {econ} (idle {vals['HARVEST_IDLE']}/decay {vals['DECAY']}/active {vals['HARVEST_ACTIVE']}) --", flush=True)
        print(f"  {'family':10s} {'n0':>4s} {'final':>5s} {'peak':>5s} {'motifs':>6s} {'maxage':>6s} {'children':>8s} {'totres':>7s}  regime", flush=True)
        for fam in ['random','grammar','clusters','mixed']:
            rs=[probe(fam, seedv=s) for s in (0,1)]
            r=max(rs,key=lambda r:r['final'])
            if r['final']==0: reg='L0 collapse'
            elif r['final']>r['n0']*5: reg='RUNAWAY (degeneracy?)'
            elif r['final']>=3 and r['maxage']>1500: reg='PERSISTENCE (L1?)'
            else: reg='transient/decaying'
            print(f"  {fam:10s} {r['n0']:>4d} {r['final']:>5d} {r['peak']:>5d} {r['motifs']:>6d} {r['maxage']:>6d} {r['children']:>8d} {r['tot_res']:>7.1f}  {reg}", flush=True)
        C.update(base)
    print("=== persistence in ANY economy -> proceed to closure detection; all L0 -> interpreter-family search (Part 5/10) ===", flush=True)
    print("=== EMERGE_DONE ===", flush=True)
