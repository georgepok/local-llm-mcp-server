import os, random, math, copy
os.environ['ME_MODE']='none'
from collections import Counter, defaultdict
import statistics as st

# FAMILY A — MUTABLE PARTIAL REACTION GRAPH. Minimal graph-theoretic test of whether CLOSURE-BLIND selection can
# CREATE reconstructive closure absent at t=0. PARTIAL producer graph (not a total map -> genuinely acyclic
# possible). Types 0..NT-1 executable + WASTE sink. Edge X->Y = X catalyzes production of Y from primitives.
# Mutations are UNBIASED (no operator biased toward closing a cycle). Viability = GENERIC physics (material
# surviving+regrowing through a bottleneck); the environment NEVER sees cycles/closure. Certificate verifies no
# >=2-type cycle at t=0. Reconstruction matrix (leave-one-type-out, cloned RNG) measures OPERATIONAL closure.
NT=10; FOOD=0; WASTE=NT
SRC=0.06; DEATH=0.10; COST=1.0

def rand_acyclic(seed):
    g=random.Random(seed); E=defaultdict(set)
    for i in range(NT):
        for j in range(i+1,NT):                          # forward edges only -> acyclic among executables
            if g.random()<0.22: E[i].add(j)
        if g.random()<0.4: E[i].add(WASTE)               # sinks allowed
    return E
def cert_acyclic(E):                                     # SCC/cycle certificate over executable types (Part 3)
    col={}; oncyc=set()
    def dfs(u,st_):
        col[u]=1
        for v in E.get(u,()):
            if v==WASTE: continue
            if col.get(v,0)==1: oncyc.update(st_+[v])    # back-edge in current stack => cycle
            elif col.get(v,0)==0: dfs(v,st_+[v])
        col[u]=2
    for u in range(NT):
        if col.get(u,0)==0: dfs(u,[u])
    selfloops=[u for u in range(NT) if u in E.get(u,())]
    return dict(acyclic=(len(oncyc)==0 and not selfloops), cycle_nodes=oncyc, selfloops=selfloops)
def cycle_types(E):
    onc=set()
    for s in range(NT):
        stack=[(s,[s])]
        while stack:
            u,path=stack.pop()
            for v in E.get(u,()):
                if v==WASTE: continue
                if v==s and len(path)>=2: onc.update(path)
                elif v not in path and len(path)<NT: stack.append((v,path+[v]))
    return onc

def react(comp, R, E, steps, rng):
    comp=Counter(comp)
    for _ in range(steps):
        if R<COST or not comp: break
        if rng.random()<SRC: comp[FOOD]+=1; R-=COST; continue
        tot=sum(comp.values()); r=rng.random()*tot; c=0; X=None
        for t,w in comp.items():
            c+=w
            if r<=c: X=t; break
        if X is None: break
        outs=list(E.get(X,()))
        if outs:
            Y=rng.choice(outs)
            if Y==WASTE:
                comp[X]-=1
                if comp[X]<=0: del comp[X]
            else: comp[Y]+=1; R-=COST
        if comp and rng.random()<DEATH:
            k=rng.choice(list(comp)); comp[k]-=1
            if comp[k]<=0: del comp[k]
    return comp,R
def viability(comp0, E, seedv):                          # CLOSURE-BLIND: executable material surviving bottleneck+regrow
    rng=random.Random(seedv)
    comp,_=react(comp0,600.0,E,1200,rng)
    comp2=Counter({t:sum(1 for _ in range(c) if rng.random()<0.2) for t,c in comp.items()})  # BOTTLENECK
    comp2=Counter({t:c for t,c in comp2.items() if c>0})
    comp3,_=react(comp2,400.0,E,800,rng)
    return sum(comp3.values()), comp3

def cos(a,b):
    ka=set(a)|set(b); na=math.sqrt(sum(v*v for v in a.values()))or 1; nb=math.sqrt(sum(v*v for v in b.values()))or 1
    return sum(a.get(k,0)*b.get(k,0) for k in ka)/(na*nb)
def op_closure(comp, E, seedv):                          # leave-one-type-out restoration gain; is a CYCLE-type selectively regenerated?
    onc=cycle_types(E); ab=[t for t in comp if comp[t]>=2 and t!=WASTE]
    if not ab: return False, 0.0
    def regen(j):
        post0=Counter(comp); post0.pop(j,None); S0=cos(comp,post0)
        rng=random.Random(seedv*7+j); post1,_=react(post0,600.0,E,600,rng); S1=cos(comp,post1)
        return S1-S0, post1.get(j,0)/(sum(post1.values())or 1)
    cyc=[t for t in ab if t in onc]; non=[t for t in ab if t not in onc]
    if not cyc: return False, 0.0
    gc=[regen(j) for j in cyc]; gn=[regen(j) for j in non] if non else [(0,0)]
    cyc_regen=st.mean(r for _,r in gc); non_regen=st.mean(r for _,r in gn); cyc_G=st.mean(g for g,_ in gc)
    return (cyc_G>0.02 and cyc_regen>non_regen+0.03), cyc_G

def mutate_topology(E, g, one_step=False):
    E=copy.deepcopy(E); op=g.random()
    if one_step:                                         # C10 diagnostic: allowed to close a cycle in one edit (add a back-edge)
        i=g.randrange(1,NT); j=g.randrange(0,i); E[i].add(j); return E
    if op<0.5:                                           # add edge (UNBIASED: uniform over all (X,Y), no cycle bias)
        X=g.randrange(NT); Y=g.choice([t for t in range(NT) if t!=X]+[WASTE]); E[X].add(Y)
    elif op<0.75 and any(E.values()):                    # delete edge
        X=g.choice([k for k in E if E[k]]); E[X].discard(g.choice(list(E[X])))
    else:                                                # redirect edge
        cand=[k for k in E if E[k]]
        if cand:
            X=g.choice(cand); old=g.choice(list(E[X])); E[X].discard(old); E[X].add(g.choice([t for t in range(NT) if t!=X]+[WASTE]))
    return E

def evolve(cond, seed, P=12, gens=20):
    g=random.Random(seed)
    pop=[]
    for i in range(P):
        E=rand_acyclic(seed*100+i); assert cert_acyclic(E)['acyclic']   # certified acyclic at t=0
        pop.append([E, Counter({FOOD:12})])
    cyc_traj=[]; opc_traj=[]; first_cycle=None; first_opc=None
    for gen in range(gens):
        viab=[viability(comp,E,seed*999+gen*P+i)[0] for i,(E,comp) in enumerate(pop)]
        cyc=[1 if cycle_types(E) else 0 for (E,_) in pop]
        opc=[1 if op_closure(comp,E,seed*13+gen*P+i)[0] else 0 for i,(E,comp) in enumerate(pop)]
        cyc_traj.append(st.mean(cyc)); opc_traj.append(st.mean(opc))
        if first_cycle is None and sum(cyc)>0: first_cycle=gen
        if first_opc is None and sum(opc)>0: first_opc=gen
        # SELECTION (closure-blind)
        idx=list(range(P))
        if cond=='C2': gg=viab[:]; g.shuffle(gg); order=sorted(idx,key=lambda i:-gg[i])     # shuffled fitness
        elif cond=='C6': order=idx[:]; g.shuffle(order)                                       # no selection (random)
        else: order=sorted(idx,key=lambda i:-viab[i])                                         # C1/C5/C10 real viability
        survivors=order[:max(2,P//2)]
        newpop=[]
        for i in survivors:
            E,comp=pop[i]; newpop.append([copy.deepcopy(E), Counter(comp)])
            child_E = E if cond=='C5' else mutate_topology(E,g, one_step=(cond=='C10'))       # C5: topology frozen
            newpop.append([child_E, Counter({FOOD:12})])
        pop=newpop[:P]
    # final measures
    final_cyc=st.mean(1 if cycle_types(E) else 0 for (E,_) in pop)
    final_opc=st.mean(1 if op_closure(comp,E,seed*99+i)[0] else 0 for i,(E,comp) in enumerate(pop))
    return dict(cond=cond, first_cycle=first_cycle, first_opc=first_opc, final_cyc=final_cyc, final_opc=final_opc,
                cyc_end=st.mean(cyc_traj[-3:]), opc_end=st.mean(opc_traj[-3:]))

def ci(xs,B=2000):
    r=random.Random(0); ms=sorted(st.mean(r.choice(xs) for _ in range(len(xs))) for _ in range(B))
    return st.mean(xs), ms[int(0.025*B)], ms[int(0.975*B)]

if __name__=='__main__':
    print("=== FAMILY A smoke — mutable PARTIAL reaction graph; can closure-blind selection CREATE closure? ===", flush=True)
    # M0: certificate + reachability positive control
    E=rand_acyclic(0); c=cert_acyclic(E); print(f"  [cert] initial graph acyclic={c['acyclic']} selfloops={c['selfloops']}", flush=True)
    Epos=copy.deepcopy(E); Epos[FOOD].add(1); Epos[1].add(2); Epos[2].add(3); Epos[3].add(1)  # FED 3-cycle 0->1->2->3->1 (hidden pos ctrl)
    comp,_=react(Counter({FOOD:12}),3000.0,Epos,4000,random.Random(0))
    ok,gG=op_closure(comp,Epos,0)
    onc=cycle_types(Epos); ab={t:comp.get(t,0) for t in range(NT) if comp.get(t,0)>=2}
    print(f"  [pos-ctrl] FED 3-cycle: cycle_types={sorted(onc)} abundant={ab} -> operational closure={ok} (cyc restoration G={gG:+.3f})", flush=True)
    if not ok: print("  !!! M0 FAIL — harness cannot detect injected closure; do NOT trust smoke opclosure numbers", flush=True)
    SEEDS=list(range(10))
    print("  condition   first_cycle(gen)  first_opclosure(gen)  cyc_rate_end  opclosure_rate_end [95% CI]", flush=True)
    store={}
    for cond in ['C1','C2','C5','C6','C10']:
        rs=[evolve(cond,s) for s in SEEDS]; store[cond]=rs
        fc=[r['first_cycle'] for r in rs if r['first_cycle'] is not None]
        fo=[r['first_opc'] for r in rs if r['first_opc'] is not None]
        mo,lo,ho=ci([r['opc_end'] for r in rs]); mc=st.mean(r['cyc_end'] for r in rs)
        print(f"    {cond:5s}      {(st.mean(fc) if fc else float('nan')):>6.1f}           {(st.mean(fo) if fo else float('nan')):>6.1f}            {mc:>5.2f}        {mo:.2f} [{lo:.2f},{ho:.2f}]", flush=True)
    # M4: operational closure MORE under real selection (C1) than shuffled-fitness (C2) and no-selection (C6)?
    def paired(a,b): d=[x-y for x,y in zip([r['opc_end'] for r in store[a]],[r['opc_end'] for r in store[b]])]; return ci(d)
    for b in ('C2','C6','C5'):
        m,l,h=paired('C1',b); print(f"  M4 test: opclosure C1 - {b} = {m:+.3f} [{l:+.3f},{h:+.3f}]  {'(C1 higher, CI>0)' if l>0 else '(not significant)'}", flush=True)
    print("  Success ladder: M2=endogenous cycle arises; M3=operational closure(+selective); M4=M3 more under real selection than controls.", flush=True)
    print("=== FAMA_SMOKE_DONE ===", flush=True)
