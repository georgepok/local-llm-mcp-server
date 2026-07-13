import os, random
os.environ['ME_MODE']='none'
from chem import World, Obj, C, T, show

# SUPPORTABILITY_CONTROL (Part 9) — SUBSTRATE SUPPORTABILITY ONLY. Question: if a nearly-complete reciprocal
# assembly is SUPPLIED, can the current chemistry MAINTAIN, REPAIR, and PROPAGATE it? This does NOT test spontaneous
# origin / construction / bootstrapping / abiogenesis / emergence. Bounded budget. Label result "substrate supportability".
ECON=dict(HARVEST_IDLE=0.24, DECAY=0.20, HARVEST_ACTIVE=0.55, MUT_RATE=0.02, SPAWN_COST=0.5, SPAWN_ENDOW=1.5)
COPIER=[T[o] for o in ('BIND','COPY','SPAWN')]        # class A: reproduces co-cell templates (both classes)
ACTIV =[T[o] for o in ('ACTIVATE','NOP')]             # class B: broadcast-activates the cell (A needs activation to work)

def seed_assembly(w, cells, per_cell, g, with_copier=True, with_activ=True):
    for (y,x) in cells:
        for _ in range(per_cell):
            if with_copier: a=Obj(list(COPIER),y,x,3.0,'A'); a.active=True; a.act_ttl=6; w.add(a)
            if with_activ:  b=Obj(list(ACTIV),y,x,3.0,'B'); b.active=True; b.act_ttl=6; w.add(b)

def run(w,n):
    for _ in range(n): w.tick()

if __name__=='__main__':
    C.update(ECON)
    print("=== SUPPORTABILITY_CONTROL (substrate supportability ONLY — NOT emergence/origin/bootstrapping) ===", flush=True)
    print(f"  supplied assembly: A=COPIER[{show(COPIER)}] + B=ACTIVATOR[{show(ACTIV)}] (broadcast family)", flush=True)
    def counts(w): return w.count('A'), w.count('B'), w.count('child')
    # 1 MAINTENANCE
    for sv in (0,1):
        w=World(H=10,Wd=10,seed=sv,inflow=1.2,family='broadcast'); g=random.Random(sv)
        seed_assembly(w,[(y,x) for y in (4,5,6) for x in (4,5,6)],2,g); a0,b0,_=counts(w); run(w,4000); a1,b1,c1=counts(w)
        print(f"  [MAINTENANCE seed{sv}] supplied A={a0} B={b0} -> after 4000: A={a1} B={b1} child={c1}  {'MAINTAINED' if a1>=3 and b1>=3 else 'lost'}", flush=True)
    # 2 REPAIR (knock down A by ~80% mid-run, does A recover?)
    w=World(H=10,Wd=10,seed=0,inflow=1.2,family='broadcast'); g=random.Random(0)
    seed_assembly(w,[(y,x) for y in (4,5,6) for x in (4,5,6)],2,g); run(w,2000); aA=w.count('A')
    kill=[o for o in w.live() if o.cls=='A']; g.shuffle(kill)
    for o in kill[:int(0.8*len(kill))]: o.alive=False
    aMid=w.count('A'); run(w,2000); aEnd=w.count('A')
    print(f"  [REPAIR] A {aA} -> knocked to {aMid} -> after 2000: A={aEnd}  {'REPAIRED' if aEnd>=0.5*aA else 'not repaired'}", flush=True)
    # 3 PROPAGATION (seed one cell of a big grid, does occupied-cell count grow?)
    w=World(H=16,Wd=16,seed=0,inflow=1.2,family='broadcast'); g=random.Random(0)
    seed_assembly(w,[(8,8),(8,9),(9,8)],2,g)
    def occ(w): return len(set((o.y,o.x) for o in w.live()))
    o0=occ(w); run(w,4000); o1=occ(w); tot=w.count()
    print(f"  [PROPAGATION] occupied cells {o0} -> after 4000: {o1} (total objs {tot})  {'PROPAGATED' if o1>o0 else 'stayed local'}", flush=True)
    # 4 COMPLETION boundary (supply ONLY B, no copier — does A ever appear? expected NO = construction, not supportability)
    w=World(H=10,Wd=10,seed=0,inflow=1.2,family='broadcast'); g=random.Random(0)
    seed_assembly(w,[(y,x) for y in (4,5,6) for x in (4,5,6)],2,g,with_copier=False,with_activ=True); run(w,3000)
    print(f"  [COMPLETION boundary] supplied ONLY B (no copier) -> A appeared? {w.count('A')>0} (expected False = origin, outside supportability)", flush=True)
    print("=== RESULT LABEL: SUBSTRATE SUPPORTABILITY (maintain/repair/propagate SUPPLIED closure) — NOT spontaneous construction ===", flush=True)
    print("=== SUPPORT_DONE ===", flush=True)
